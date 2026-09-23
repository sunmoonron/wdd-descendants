"""e388: WDD judged the way sparse autoencoders are judged. At three depths (a quarter, half and three quarters) the
residual state after the block is replaced, at every position but the first, by a k-sparse WDD reconstruction (OMP over
the model's own write atoms up to that block, the centring mean added back), for k = 8 to 256, and the model is run on
to the loss. Reported: the fraction of variance unexplained, and the loss recovered, (L_mean - L_splice) / (L_mean -
L_clean), where L_mean replaces the state by its mean. Baselines at the same k: the same OMP over a randomly rotated copy
of the dictionary (the same Gram matrix, no provenance), OMP over random unit atoms of the same number, and PCA-k
fitted on other sequences (the best linear k-dimensional code)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p3_common import *
tag = sys.argv[1]; c = Cache(tag); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NB = arch.NB; torch.manual_seed(0)
ev = c.s["eval_ids"][:3].to(DEV); fit = c.s["eval_ids"][4:8].to(DEV); KS = [8, 16, 32, 64, 128]
def states(ids, L):
    out = {}
    h = arch.layers[L].register_forward_hook(lambda m, i, o: out.__setitem__("x", (o[0] if isinstance(o, tuple) else o).detach().float()))
    try:
        with torch.no_grad(): lg = model(ids).logits.float()
    finally: h.remove()
    return out["x"], lg
def loss_with(ids, L, Xnew):
    def hk(m, i, o):
        x = o[0] if isinstance(o, tuple) else o; y = x.clone(); y[:, 1:] = Xnew.to(x.dtype).view(x.shape[0], x.shape[1] - 1, -1)
        return (y,) + tuple(o[1:]) if isinstance(o, tuple) else y
    h = arch.layers[L].register_forward_hook(hk)
    try:
        with torch.no_grad(): return token_loss(model(ids).logits.float(), ids).mean().item()
    finally: h.remove()
res = dict(model=tag, levels={})
for L in (NB // 4, NB // 2, (3 * NB) // 4):
    Xe, lg = states(ev, L); Lc = token_loss(lg, ev).mean().item(); B, T, D = Xe.shape; Xp = Xe[:, 1:].reshape(-1, D); mu = c.s["mu"][L + 1].to(DEV).float(); Xc = Xp - mu[None]
    Lm = loss_with(ev, L, mu[None].expand(Xp.shape[0], -1)); A, lab = c.dictionary(L); Ar = rotate(A); g = torch.Generator().manual_seed(5); Au = torch.randn(A.shape[0], D, generator=g).to(DEV); Au = Au / Au.norm(dim=-1, keepdim=True)
    Xf, _ = states(fit, L); Xfc = Xf[:, 1:].reshape(-1, D) - mu[None]; U = torch.linalg.svd(Xfc - Xfc.mean(0, keepdim=True), full_matrices=False)[2]
    row = dict(clean=Lc, mean_ablate=Lm, n_atoms=int(A.shape[0]))
    for nm, Dct in (("wdd", A), ("rotated", Ar), ("random", Au)):
        sel, _, _ = omp(Xc, Dct, max(KS), batch=256, record_err=False)
        for k in KS:
            cof, err = refit(Xc, Dct, sel[:, :k]); Xh = mu[None] + torch.einsum("nk,nkd->nd", cof, Dct[sel[:, :k]])
            Ls = loss_with(ev, L, Xh); row[f"{nm}_{k}"] = dict(fvu=(err.sum() / Xc.pow(2).sum()).item(), loss=Ls, recovered=(Lm - Ls) / max(Lm - Lc, 1e-9))
        del sel
    for k in KS:
        Pk = U[:k].T; Xh = mu[None] + Xfc.mean(0, keepdim=True) + (Xc - Xfc.mean(0, keepdim=True)) @ Pk @ Pk.T; Ls = loss_with(ev, L, Xh)
        row[f"pca_{k}"] = dict(fvu=((Xc - (Xh - mu[None])).pow(2).sum() / Xc.pow(2).sum()).item(), loss=Ls, recovered=(Lm - Ls) / max(Lm - Lc, 1e-9))
    res["levels"][str(L)] = row
    log(f"{tag} L{L} (clean {Lc:.3f}, mean-ablated {Lm:.3f}, {A.shape[0]} atoms): loss recovered / FVU by k: " + " | ".join(f"k{k}: WDD {row[f'wdd_{k}']['recovered']:.2f}/{row[f'wdd_{k}']['fvu']:.2f}, rotated {row[f'rotated_{k}']['recovered']:.2f}/{row[f'rotated_{k}']['fvu']:.2f}, random {row[f'random_{k}']['recovered']:.2f}, PCA {row[f'pca_{k}']['recovered']:.2f}/{row[f'pca_{k}']['fvu']:.2f}" for k in KS))
mid = res["levels"][str(NB // 2)]
record(f"e388_wddsae_{tag}", res, "mid level loss recovered: " + " ".join(f"k{k} WDD {mid[f'wdd_{k}']['recovered']:.2f} rot {mid[f'rotated_{k}']['recovered']:.2f} PCA {mid[f'pca_{k}']['recovered']:.2f}" for k in (8, 16, 32, 64)))
