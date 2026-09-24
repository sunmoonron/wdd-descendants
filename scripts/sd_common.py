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
        w = getattr(ln, "weight", None); g = w.detach().float() if w is not None else torch.ones(arch.D, device=DEV); W = arch.rdir(b).to(g.device) * g[None]
        if "rms" not in type(ln).__name__.lower(): W = W - W.mean(-1, keepdim=True)
        rows.append(W)
    return unitr(torch.cat(rows))

def fmt(cell, ks):
    s = " ".join(f"{cell[str(k)]['rec']:.2f}" for k in ks) + " | fvu " + " ".join(f"{cell[str(k)]['fvu']:.2f}" for k in ks)
    if "16" in cell: s += f" | k16 CI {cell['16']['lo']:.2f}-{cell['16']['hi']:.2f}"
    return s

def reader_gram(model, arch, blocks, unembed=True):
    """Weights only: the sum over the modules that read the state downstream (each given block's attention input and
    MLP input, and the unembedding) of W^T W for their residual-space read directions (norm gain folded in, centred when
    the norm centres), each module normalised to unit trace (one vote per module). Returns (all readers, unembedding)."""
    mods = []
    def add(W, ln):
        w = getattr(ln, "weight", None); g = w.detach().float() if w is not None else torch.ones(W.shape[-1], device=DEV); W = W.detach().float() * g[None]
        if "rms" not in type(ln).__name__.lower(): W = W - W.mean(-1, keepdim=True)
        Gm = (W.T @ W).double(); mods.append(Gm / Gm.trace())
    for b in blocks:
        l = arch.layers[b]
        if arch.fam == "gpt2": add(l.attn.c_attn.weight.T, l.ln_1); add(arch.rdir(b), l.ln_2)
        elif arch.fam == "neox": add(l.attention.query_key_value.weight, l.input_layernorm); add(arch.rdir(b), l.post_attention_layernorm)
        else:
            a = l.self_attn; add(torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight]), l.input_layernorm)
            add(torch.cat([l.mlp.up_proj.weight, l.mlp.gate_proj.weight]), l.post_attention_layernorm)
    Gu = None
    if unembed:
        fl = model.transformer.ln_f if arch.fam == "gpt2" else (model.gpt_neox.final_layer_norm if arch.fam == "neox" else model.model.norm)
        add(model.get_output_embeddings().weight, fl); Gu = mods[-1].float()
    return sum(mods).float(), Gu

def fisher_gram(model, arch, ids, L, samples=2, seed=0):
    """Fisher information of the model's own next-token distribution with respect to the output of block L
    (positions 1:), E[g g^T] with g the gradient of the log-likelihood of a token sampled from the model's prediction."""
    for p in model.parameters(): p.requires_grad_(False)
    torch.set_grad_enabled(True); G = torch.zeros(arch.D, arch.D, device=DEV, dtype=torch.float64); n = 0; gen = torch.Generator(device=DEV).manual_seed(seed)
    try:
        for s0 in range(0, ids.shape[0], 2):
            x_ids = ids[s0:s0 + 2]; leaf = {}
            def hk(m, i, o):
                xo = o[0] if isinstance(o, tuple) else o; y = xo.detach().requires_grad_(True); leaf["x"] = y
                return (y,) + tuple(o[1:]) if isinstance(o, tuple) else y
            for smp in range(samples):
                h = arch.layers[L].register_forward_hook(hk)
                try: lg = model(x_ids).logits.float()
                finally: h.remove()
                lp = torch.log_softmax(lg[:, :-1], -1)
                with torch.no_grad(): y = torch.multinomial(lp.exp().reshape(-1, lp.shape[-1]), 1, generator=gen).view(lp.shape[0], -1)
                (-lp.gather(2, y[..., None]).sum()).backward()
                gx = leaf["x"].grad[:, 1:].reshape(-1, arch.D).double(); G += gx.T @ gx; n += gx.shape[0]; del lg, lp, gx, leaf["x"]
    finally: torch.set_grad_enabled(False)
    return (G / n).float()

def metric_sqrt(G, ridge=0.01):
    """square root of the metric G normalised to mean eigenvalue 1, plus a ridge"""
    D = G.shape[0]; M = G / (G.trace() / D) + ridge * torch.eye(D, device=G.device)
    ev, U = torch.linalg.eigh(M.double()); return ((U * ev.clamp_min(0).sqrt()) @ U.T).float()

def share_on(G, U):
    """share of the trace of G on the subspace spanned by the orthonormal columns of U"""
    return ((U.T @ G @ U).trace() / G.trace()).item()
