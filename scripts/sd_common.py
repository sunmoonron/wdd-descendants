"""Shared helpers for the self-description chain (e399 onward): states at the output of a block and a splice that returns
per-token losses; k-sparse descriptions of the centred states over a vocabulary, by OMP under the Euclidean metric or
under a metric x -> x S; loss recovered with a bootstrap interval over sequences; and the control vocabularies
(rotations, Gaussian vocabularies with a given second moment, sparse mixtures of the vocabulary's own words, reader
directions of the MLPs)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p3_common import *

def unitr(A): return A / A.norm(dim=-1, keepdim=True).clamp_min(1e-8)

class Level:
    """States at the output of block L for ids (positions 1:), their mean, and a splice replacing them."""
    def __init__(self, model, arch, ids, L, dimcentre=False):
        self.model, self.arch, self.ids, self.L, self.B, self.T = model, arch, ids, L, ids.shape[0], ids.shape[1]
        out = {}
        h = arch.layers[L].register_forward_hook(lambda m, i, o: out.__setitem__("x", (o[0] if isinstance(o, tuple) else o).detach().float()))
        try:
            with torch.no_grad(): self.lc = token_loss(model(ids).logits.float(), ids).view(self.B, -1)
        finally: h.remove()
        self.x = out["x"][:, 1:].reshape(-1, out["x"].shape[-1]); self.mu = self.x.mean(0, keepdim=True); self.Xc = self.x - self.mu
        self.P = None
        if dimcentre:   # describe only the part a centring norm can see (the all-ones component is invisible to every reader)
            self.Xc = self.Xc - self.Xc.mean(-1, keepdim=True); self.P = True
        self.lm = self.splice(self.mu.expand(self.x.shape[0], -1))
        self.gap = (self.lm.mean() - self.lc.mean()).item()
    def splice(self, Xh):
        B, T = self.B, self.T
        def hk(m, i, o):
            xo = o[0] if isinstance(o, tuple) else o; y = xo.clone(); y[:, 1:] = Xh.to(xo.dtype).view(B, T - 1, -1)
            return (y,) + tuple(o[1:]) if isinstance(o, tuple) else y
        hh = self.arch.layers[self.L].register_forward_hook(hk)
        try:
            with torch.no_grad(): return token_loss(self.model(self.ids).logits.float(), self.ids).view(B, -1)
        finally: hh.remove()
    def recovered(self, ls, nboot=2000, seed=0):
        lc, lm, l = self.lc.sum(1), self.lm.sum(1), ls.sum(1)
        pooled = ((lm - l).sum() / (lm - lc).sum().clamp_min(1e-9)).item()
        g = torch.Generator().manual_seed(seed); idx = torch.randint(0, self.B, (nboot, self.B), generator=g).to(l.device)
        bs = ((lm - l)[idx].sum(1) / (lm - lc)[idx].sum(1).clamp_min(1e-9)).sort().values
        return pooled, bs[int(0.025 * nboot)].item(), bs[int(0.975 * nboot) - 1].item()

def describe(lv, A, ks, S=None, batch=256):
    """k-sparse descriptions of lv's centred states over the rows of A (unit-normalised here). S: None for Euclidean OMP,
    or a [D, D] matrix; then the atoms and states are mapped by x -> x S, the atoms renormalised there, and the
    reconstruction taken back to the original coordinates. Returns {k: dict(rec, lo, hi, fvu)}."""
    A = unitr(A)
    if lv.P: A = unitr(A - A.mean(-1, keepdim=True))
    if S is None: An, Xs, nrm = A, lv.Xc, None
    else:
        As = A @ S; nrm = As.norm(dim=-1).clamp_min(1e-8); An = As / nrm[:, None]; Xs = lv.Xc @ S
    sel, _, _ = omp(Xs, An, max(ks), batch=batch, record_err=False); out = {}; tot = lv.Xc.pow(2).sum().item()
    for k in ks:
        s = sel[:, :k]; cof, _ = refit(Xs, An, s)
        Xh = torch.einsum("nk,nkd->nd", cof, An[s]) if S is None else torch.einsum("nk,nkd->nd", cof / nrm[s], A[s])
        fv = (lv.Xc - Xh).pow(2).sum().item() / tot; r, lo, hi = lv.recovered(lv.splice(lv.mu + Xh))
        out[str(k)] = dict(rec=r, lo=lo, hi=hi, fvu=fv); del Xh
    del sel; return out

def gauss_like(n, C, seed, power=0.5):
    """n unit rows drawn from N(0, C^(2*power)) (power 0.5: second moment C; 0.25: its square root)."""
    ev, U = torch.linalg.eigh(C.double()); R = (U * ev.clamp_min(0).pow(power)) @ U.T
    g = torch.Generator(device=DEV).manual_seed(seed); Z = torch.randn(n, C.shape[0], device=DEV, generator=g)
    return unitr(Z @ R.float())

def mixtures(A, groups, m=8, seed=0):
    """each word replaced by a random signed sum of m words of its own family (same span, no individual words)"""
    g = torch.Generator(device=DEV).manual_seed(seed); out = torch.empty_like(A)
    for idx in groups:
        idx = idx.to(DEV); n = idx.numel()
        for s0 in range(0, n, 8192):
            nn_ = min(8192, n - s0); pick = idx[torch.randint(0, n, (nn_, m), device=DEV, generator=g)]
            sg = torch.randint(0, 2, (nn_, m, 1), device=DEV, generator=g).float() * 2 - 1
            out[idx[s0:s0 + nn_]] = unitr((A[pick] * sg).sum(1))
    return out

def reader_rows(arch, blocks):
    """MLP input rows of the given blocks as residual-space read directions: gamma * w, centred when the norm centres"""
    rows = []
    for b in blocks:
        l = arch.layers[b]; ln = l.ln_2 if arch.fam == "gpt2" else l.post_attention_layernorm
        g = ln.weight.detach().float(); W = arch.rdir(b).to(g.device) * g[None]
        if "rms" not in type(ln).__name__.lower(): W = W - W.mean(-1, keepdim=True)
        rows.append(W)
    return unitr(torch.cat(rows))

def fmt(cell, ks):
    s = " ".join(f"{cell[str(k)]['rec']:.2f}" for k in ks) + " | fvu " + " ".join(f"{cell[str(k)]['fvu']:.2f}" for k in ks)
    if "16" in cell: s += f" | k16 CI {cell['16']['lo']:.2f}-{cell['16']['hi']:.2f}"
    return s
