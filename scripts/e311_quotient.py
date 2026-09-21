"""e311: the quotient of physical perturbations by function. 150 perturbations (50 WDD writes of block 2, 50 random
directions, 50 attention-head writes) injected at the block-3 input; per-perturbation centroid images at six levels
and the centroid logit change F. Pairwise physical distance at every level against pairwise functional distance:
Spearman by level (does physical geometry come to match functional geometry?), and the memory of the injected
geometry. A per-level functional distance from a data-free read-out fitted at each level (ridge from random
injections at that block to the logit footprint PCA) gives the contraction ratio r = D_func(l+1) / D_func(l) per
pair, its median and the fraction below one. Multiplicity: pairs functionally close (bottom 5% of functional
distance) but physically far (top 50% at injection), and the ratio of physical to functional participation rank."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from func_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; NB = c.NB; b = 2; D = c.D; lab = c.d["lab"]; A = c.d["A"]; levels = [lv for lv in sorted({b + 1, b + 2, b + 4, L, L + 2, NB - 2}) if lv < NB]
led, tn, tc = dominant(model, arch, c, ids_seq, b); run = make_runner_logits(model, arch, c, ids_seq, levels, NT, b); S0 = run(); typ = typical_mask(S0[L]); pool = torch.nonzero(typ)[:, 0]; s_inj = tc.abs().median(); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20)
torch.manual_seed(0); Rm = A[(lab["type"] == T_MLP) & (lab["block"] == b)].float().to(DEV); Ratt = A[(lab["type"] == T_ATT) & (lab["block"] <= b)].float().to(DEV); V = torch.cat([unit(Rm[torch.randperm(len(Rm), device=DEV)[:50]]), unit(torch.randn(50, D, device=DEV)), unit(Ratt[torch.randperm(len(Ratt), device=DEV)[:50]])]); N = len(V); fam_id = torch.repeat_interleave(torch.arange(3, device=DEV), 50)
base = run(positions=pool); lg0 = base["lg"]; imgs = {lv: torch.zeros(N, D, device=DEV) for lv in levels}; Fl = torch.zeros(N, lg0.shape[1], device=DEV); cnt = torch.zeros(N, device=DEV)
for p in range(3):
    a = torch.randint(0, N, (len(pool),), device=DEV); inj = torch.zeros(NT, D, device=DEV); inj[pool] = s_inj * V[a]; r = run(positions=pool, inject=inj, inject_block=b + 1); dlr = r["lg"] - lg0; dlr = dlr - dlr.mean(1, keepdim=True)
    for lv in levels: imgs[lv].index_add_(0, a, (r[lv] - S0[lv])[pool])
    Fl.index_add_(0, a, dlr); cnt.index_add_(0, a, torch.ones(len(pool), device=DEV))
for lv in levels: imgs[lv] = imgs[lv] / cnt[:, None].clamp_min(1)
Fl = Fl / cnt[:, None].clamp_min(1); iu = torch.triu_indices(N, N, 1, device=DEV); pd = lambda X: torch.cdist(unit(X), unit(X))[iu[0], iu[1]]; Dfunc = pd(Fl); D0 = pd(V)
def spearman(a_, b_):
    ra = a_.argsort().argsort().float(); rb = b_.argsort().argsort().float(); return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
Ud, Sd, Vd = torch.linalg.svd(Fl, full_matrices=False); B = Vd[:128].T; geo = {}; func_lv = {}
for lv in levels:
    Dp = pd(imgs[lv]); torch.manual_seed(10 + lv); Vr = unit(torch.randn(512, D, device=DEV)); X = []; Y = []
    a = torch.randint(0, 512, (len(pool),), device=DEV); inj = torch.zeros(NT, D, device=DEV); inj[pool] = s_inj * Vr[a]; rr = run(positions=pool, inject=inj, inject_block=lv + 1) if lv + 1 < NB else None
    if rr is not None:
        dlr = rr["lg"] - lg0; dlr = dlr - dlr.mean(1, keepdim=True); X = Vr[a]; Y = (dlr @ B) / s_inj; Gx = X.T @ X; W = torch.linalg.solve(Gx + 1e-2 * Gx.diagonal().mean() * torch.eye(D, device=DEV), X.T @ Y); func_lv[lv] = pd(imgs[lv] @ W)
    geo[lv] = dict(spearman_phys_vs_func=spearman(Dp, Dfunc), spearman_phys_vs_injected=spearman(Dp, D0), spearman_readout_vs_func=(spearman(func_lv[lv], Dfunc) if lv in func_lv else None))
contr = {}; lvs = [lv for lv in levels if lv in func_lv]
for i in range(len(lvs) - 1):
    r_ = func_lv[lvs[i + 1]] / func_lv[lvs[i]].clamp_min(1e-9); contr[f"{lvs[i]}->{lvs[i + 1]}"] = dict(median=r_.median().item(), frac_below_one=(r_ < 1).float().mean().item())
close = Dfunc <= Dfunc.quantile(0.05); far0 = D0 >= D0.quantile(0.5); farL = pd(imgs[L]) >= pd(imgs[L]).quantile(0.5); cross = fam_id[iu[0]] != fam_id[iu[1]]
def prank(X):
    ev = torch.linalg.eigvalsh((X - X.mean(0, keepdim=True)).T @ (X - X.mean(0, keepdim=True)) / len(X)).clamp_min(0); return (ev.sum() ** 2 / (ev ** 2).sum()).item()
mult = dict(functionally_close_pairs=int(close.sum()), of_which_physically_far_at_injection=(far0[close].float().mean()).item(), of_which_physically_far_at_L=(farL[close].float().mean()).item(), of_which_cross_family=(cross[close].float().mean()).item(), cross_family_baseline=cross.float().mean().item(), prank_physical_injected=prank(V), prank_physical_L=prank(imgs[L]), prank_functional=prank(unit(Fl) @ B))
log(f"{tag} (N {N}): Spearman(physical distance at level, functional distance) by level: " + " ".join(f"{lv}:{v['spearman_phys_vs_func']:.2f}" for lv, v in geo.items()) + " | memory of injected geometry: " + " ".join(f"{lv}:{v['spearman_phys_vs_injected']:.2f}" for lv, v in geo.items()) + " | level read-out geometry vs final functional: " + " ".join(f"{lv}:{v['spearman_readout_vs_func']:.2f}" for lv, v in geo.items() if v['spearman_readout_vs_func'] is not None) + " | contraction ratio of functional distances between levels (median, fraction < 1): " + " ".join(f"{k} {v['median']:.2f} ({v['frac_below_one']:.2f})" for k, v in contr.items()) + f" | multiplicity: of the {mult['functionally_close_pairs']} functionally closest pairs, {mult['of_which_physically_far_at_injection']:.2f} are physically far at injection, {mult['of_which_physically_far_at_L']:.2f} at L, {mult['of_which_cross_family']:.2f} cross-family (baseline {mult['cross_family_baseline']:.2f}); participation rank physical injected {mult['prank_physical_injected']:.1f}, physical at L {mult['prank_physical_L']:.1f}, functional {mult['prank_functional']:.1f}")
record(f"e311_quotient_{tag}", dict(model=tag, b=b, L=L, N=N, geometry={str(k): v for k, v in geo.items()}, contraction=contr, multiplicity=mult), "phys-vs-func by level " + " ".join(f"{lv}:{v['spearman_phys_vs_func']:.2f}" for lv, v in geo.items()) + " | contraction " + " ".join(f"{k} {v['median']:.2f}" for k, v in contr.items()) + f" | close pairs far-at-injection {mult['of_which_physically_far_at_injection']:.2f} cross-family {mult['of_which_cross_family']:.2f}/{mult['cross_family_baseline']:.2f} | pranks {mult['prank_physical_injected']:.0f}/{mult['prank_physical_L']:.0f}/{mult['prank_functional']:.0f}")
