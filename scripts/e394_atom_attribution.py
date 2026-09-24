"""e394: are WDD codes usable as attribution-graph nodes? At the middle depth the state is replaced by its 32-atom WDD
code (as e388). For every position, each of its 32 atoms is removed in turn (its coefficient set to zero, one atom slot
at a time across all positions, 32 forward passes) and the change in that position's next-token loss is recorded.
Reported per position: how concentrated the effects are (participation ratio of |effects| and how many atoms carry 80%),
whether the linear attribution (gradient of the loss with respect to each coefficient times the coefficient, one
backward pass) predicts the ablation effects (Spearman over the 32 atoms), and which atom types carry the effect (MLP
rows, head bases, embeddings)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p3_common import *
tag = sys.argv[1]; c = Cache(tag); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NB = arch.NB; L = NB // 2; K = 32; torch.manual_seed(0)
for p in model.parameters(): p.requires_grad_(False)
ev = c.s["eval_ids"][:2].to(DEV); B, T = ev.shape
out = {}
h = arch.layers[L].register_forward_hook(lambda m, i, o: out.__setitem__("x", (o[0] if isinstance(o, tuple) else o).detach().float()))
with torch.no_grad(): model(ev)
h.remove()
x = out["x"][:, 1:].reshape(-1, out["x"].shape[-1]); mu = c.s["mu"][L + 1].to(DEV).float(); A, lab = c.dictionary(L); sel, cof, _ = omp(x - mu[None], A, K, batch=256, record_err=False); At = A[sel]   # [P, K, D]
typ = lab["type"].to(DEV)[sel]
def per_token_loss(coef):
    Xh = mu[None] + torch.einsum("pk,pkd->pd", coef, At)
    def hk(m, i, o):
        xo = o[0] if isinstance(o, tuple) else o; y = xo.clone(); y[:, 1:] = Xh.to(xo.dtype).view(B, T - 1, -1)
        return (y,) + tuple(o[1:]) if isinstance(o, tuple) else y
    hh = arch.layers[L].register_forward_hook(hk)
    try: lg = model(ev).logits.float()
    finally: hh.remove()
    return token_loss(lg, ev).view(B, T - 1)[:, :].reshape(-1)   # loss predicting token t+1 from position t; positions 1.. align with x rows shifted by one
with torch.no_grad(): base = per_token_loss(cof)
# position n of the code is sequence position n+1 (row-major over the batch); its next-token loss is entry (b, n+1) of the per-token loss
idx = torch.arange(B * (T - 1), device=DEV).view(B, T - 1)[:, 1:].reshape(-1); rows = torch.arange(B * (T - 1), device=DEV).view(B, T - 1)[:, :-1].reshape(-1)
eff = torch.zeros(len(rows), K, device=DEV)
with torch.no_grad():
    for j in range(K):
        cj = cof.clone(); cj[:, j] = 0; eff[:, j] = (per_token_loss(cj) - base)[idx]
cg = cof.clone().requires_grad_(True)
with torch.enable_grad():
    lt = per_token_loss(cg); grad = torch.autograd.grad(lt[idx].sum(), cg)[0]
lin = -(grad * cof)[rows]                     # linear estimate of the loss change from removing each atom
E = eff.abs(); pr = E.sum(1) ** 2 / E.pow(2).sum(1).clamp_min(1e-12); srt = (E / E.sum(1, keepdim=True).clamp_min(1e-12)).sort(1, descending=True).values.cumsum(1); n80 = ((srt < 0.8).sum(1) + 1).float()
def spear_rows(a, b):
    ra = a.argsort(1).argsort(1).float(); rb = b.argsort(1).argsort(1).float(); ra = ra - ra.mean(1, keepdim=True); rb = rb - rb.mean(1, keepdim=True)
    return (ra * rb).sum(1) / (ra.norm(dim=1) * rb.norm(dim=1)).clamp_min(1e-9)
rho = spear_rows(eff, lin); ty = typ[rows]; share = {nm: (E * (ty == t)).sum().item() / E.sum().item() for nm, t in (("mlp", T_MLP), ("heads", T_ATT), ("token_emb", T_TOK), ("pos_emb", T_POS), ("bias", T_BIAS))}
res = dict(model=tag, level=L, K=K, effect_participation_ratio_median=pr.median().item(), atoms_for_80pct_median=n80.median().item(), linear_vs_ablation_spearman_median=rho.median().item(), effect_share_by_type=share, atom_share_by_type={nm: (ty == t).float().mean().item() for nm, t in (("mlp", T_MLP), ("heads", T_ATT), ("token_emb", T_TOK), ("pos_emb", T_POS), ("bias", T_BIAS))}, mean_abs_effect=E.mean().item(), sum_effects_vs_full_removal=None)
log(f"{tag} L{L}, 32-atom WDD code: per position the ablation effects of its atoms have participation ratio {res['effect_participation_ratio_median']:.1f} and {res['atoms_for_80pct_median']:.0f} atoms carry 80% (median); linear attribution vs ablation Spearman {res['linear_vs_ablation_spearman_median']:+.2f} (median); effect share by atom type " + " ".join(f"{k} {v:.2f}" for k, v in share.items()) + " (atom share " + " ".join(f"{k} {v:.2f}" for k, v in res["atom_share_by_type"].items()) + ")")
record(f"e394_atomattr_{tag}", res, f"PR {res['effect_participation_ratio_median']:.1f} n80 {res['atoms_for_80pct_median']:.0f} lin-vs-abl rho {res['linear_vs_ablation_spearman_median']:+.2f} mlp effect share {share['mlp']:.2f}")
