"""The fields: each upstream block's weak write (all but the strongest quarter of its activations) and strong write (the kept quarter) have sixteen principal directions each, found on held-out text (e696, e703, e706). A field summarises the crowd's contribution to a neuron's input: the crowd's write, projected on the field, read through the neuron's read row."""
import torch
from .model import DEV, Capture, run_base, write_w
def thresholds(M, blocks, ids, q=0.25, batch=8):
    """per-neuron thresholds: the (1 - q) quantile of |activation| on the windows ids, per block"""
    DFF = M["DFF"]; out = {}
    with Capture(M, list(blocks), ["act"]) as C:
        acts = {b: [] for b in blocks}
        for s0 in range(0, ids.shape[0], batch):
            run_base(M, ids[s0:s0 + batch])
            for b in blocks: acts[b].append(C.cap[("act", b)].reshape(-1, DFF).half())
    for b in blocks:
        a = torch.cat(acts[b]); out[b] = torch.cat([a[:, c0:c0 + 512].float().abs().quantile(1 - q, dim=0) for c0 in range(0, DFF, 512)])
    return out, acts
def field_directions(M, blocks, ids, q=0.25, k=16, batch=8):
    """for each block in blocks: the k principal directions of its weak write and of its strong write on the windows ids; returns dict block -> (weak [D, k], strong [D, k]) and the thresholds"""
    D = M["D"]; tau, acts = thresholds(M, blocks, ids, q, batch); out = {}
    for b in blocks:
        a = torch.cat(acts[b]); W = write_w(M, b); cw = torch.zeros(D, D, device=DEV); cs = torch.zeros(D, D, device=DEV)
        for s_ in range(0, a.shape[0], 2048):
            af = a[s_:s_ + 2048].float(); wk = af.abs() < tau[b][None]; dw = (af * wk) @ W.T; ds = (af * ~wk) @ W.T; cw += dw.T @ dw; cs += ds.T @ ds
        out[b] = (torch.linalg.eigh(cw)[1][:, -k:].contiguous(), torch.linalg.eigh(cs)[1][:, -k:].contiguous())
    return out, tau
def union(dirs, which="weak"):
    """an orthonormal basis of the union of the chosen field directions over blocks (the bus)"""
    i = 0 if which == "weak" else 1; return torch.linalg.qr(torch.cat([d[i] for d in dirs.values()], 1))[0]
def crowd_on_field(L, Pm):
    """a term for Ledger.simulate's extra: the crowd's write (the input minus the sixteen wires' writes, the attention and the embedding where they are wires) projected on the field Pm [D, m] and read through the neuron's read row; returns fn(k, key) -> [N]"""
    def fn(k, key):
        r_ = L.RT[key][k]; pr = Pm @ (Pm.T @ r_); out = torch.empty(L.N, device=DEV); NMLP, NATT, D = L.NMLP, L.NATT, L.M["D"]
        for s in range(0, L.N, 2048):
            Pp = torch.arange(s, min(s + 2048, L.N), device=DEV); c, _ = L.contributions(k, Pp, key); top = c.abs().topk(16, dim=1).indices; isM = (top < NMLP).float(); isA = (top >= NMLP) & (top < NMLP + NATT); isE = (top >= NMLP + NATT).any(1).float()
            ti = top.clamp(max=max(NMLP - 1, 0)); a = L.UP[Pp].gather(1, ti).float() * isM; xm = torch.einsum("nk,nkd->nd", a, L.WUP[ti]) if NMLP else torch.zeros(len(Pp), D, device=DEV)
            ai_ = (top - NMLP).clamp(0, NATT - 1); xa = (L.AT[Pp].float().gather(1, ai_[:, :, None].expand(-1, -1, D)) * isA[:, :, None]).sum(1)
            xc = (L.XM[Pp] - L.BIAS[None] - xm - xa - L.EM[Pp].float() * isE[:, None]) / L.SIG[Pp][:, None]; out[Pp] = xc @ pr
        return out
    return fn
