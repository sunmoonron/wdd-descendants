"""Read energy: the Fisher information of the model's own output over the next tokens, with respect to the state after a block, sampled at chosen positions (e707); a vector's read energy is its quadratic form, which predicts the finite KL of removing the vector from the state. Also the blended metric for pursuit in that grade (e714)."""
import numpy as np, torch
from .model import DEV, logprobs_at
def sampled_fisher(M, block, ids, positions, n_samples=2048, horizon=64, batch=8, seed=707):
    """positions: list of (window, t). Returns F [P, D, D] (the mean outer product of sampled log-probability gradients over output positions within the horizon) and the sample counts [P]"""
    T = ids.shape[1]; D = M["D"]; SW = torch.tensor([w for w, t in positions], device=DEV); ST = torch.tensor([t for w, t in positions], device=DEV); P = SW.numel(); F = torch.zeros(P, D, D, device=DEV); CNT = torch.zeros(P, device=DEV); LEAF = {}; gen = torch.Generator(device=DEV).manual_seed(seed)
    def leaf_hook(m, a, o):
        x = o[0] if isinstance(o, tuple) else o; l = x.detach().requires_grad_(True); LEAF["x"] = l; return ((l,) + tuple(o[1:])) if isinstance(o, tuple) else l
    hd = M["layers"][block].register_forward_hook(leaf_hook)
    try:
        for s0 in range(0, ids.shape[0], batch):
            nb = min(batch, ids.shape[0] - s0); inb = (SW >= s0) & (SW < s0 + nb); idx = inb.nonzero()[:, 0]
            if idx.numel() == 0: continue
            wl = SW[idx] - s0; tl = ST[idx]; bw = torch.arange(nb, device=DEV)
            with torch.enable_grad():
                o = M["base"](input_ids=ids[s0:s0 + nb], use_cache=False); lp = logprobs_at(M, o.last_hidden_state); Xl = LEAF["x"]; Pr = lp.detach().exp()
                for s in range(n_samples):
                    sw = torch.randint(0, T, (nb,), generator=gen, device=DEV); y = torch.multinomial(Pr[bw, sw], 1, generator=gen)[:, 0]
                    g, = torch.autograd.grad(lp[bw, sw, y].sum(), Xl, retain_graph=True); m = (sw[wl] >= tl) & (sw[wl] < tl + horizon)
                    if m.any(): G = g[wl[m], tl[m]]; F[idx[m]] += torch.einsum("pi,pj->pij", G, G); CNT[idx[m]] += 1
            del o, lp, Pr, Xl; LEAF.clear(); torch.cuda.empty_cache()
    finally: hd.remove()
    return F / CNT.clamp_min(1)[:, None, None], CNT
def read_energy(F, V): return torch.einsum("pd,pde,pe->p", V, F, V)
def grade(F, V):
    """read energy per unit of Euclidean energy, against a random direction's (trace(F)/D): above 1 means the output reads this vector more than its energy"""
    D = F.shape[-1]; return read_energy(F, V) / V.pow(2).sum(-1).clamp_min(1e-12) / (F.diagonal(dim1=1, dim2=2).sum(-1) / D).clamp_min(1e-30)
def blended_sqrt(F, lam=1.0):
    """L [P, D, D] with |L v|^2 = v^T (F_n + lam I) v, F_n the Fisher at unit mean eigenvalue: the metric of the blended pursuit (lam 0 is the pure Fisher metric, which blows up in the directions the sampled Fisher cannot see)"""
    D = F.shape[-1]; FN = F * (D / F.diagonal(dim1=1, dim2=2).sum(-1).clamp_min(1e-30))[:, None, None]; EV, U = torch.linalg.eigh(FN); return torch.einsum("pk,pdk->pkd", (EV.clamp_min(0) + lam).sqrt(), U)
