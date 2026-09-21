"""e306: do physically unrelated perturbation families share one causal coordinate system? Five families (WDD writes,
random, covariance-matched random, attention-head writes, later-MLP writes) of 30 vectors injected at the block-3
input; at +2 blocks, at L and near the end, each family's 16-dimensional function core (PLS of its images to its own
logit changes) and identity core (between-vector scatter). Reported per level: the mean pairwise cross-family
overlap of the function cores and of the identity cores, the within-family split-half reliability (ceiling), and
chance 16/D."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from func_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; NB = c.NB; b = 2; D = c.D; lab = c.d["lab"]; A = c.d["A"]; Kf = 30; d = 16; levels = sorted({b + 2, L, NB - 2})
led, tn, tc = dominant(model, arch, c, ids_seq, b); run = make_runner_logits(model, arch, c, ids_seq, [b] + levels, NT, b); S0 = run(); typ = typical_mask(S0[L]); pool = torch.nonzero(typ)[:, 0]; s_inj = tc.abs().median(); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20)
torch.manual_seed(0); Rm = A[(lab["type"] == T_MLP) & (lab["block"] == b)].float().to(DEV); Ratt = A[(lab["type"] == T_ATT) & (lab["block"] <= b)].float().to(DEV); R5 = A[(lab["type"] == T_MLP) & (lab["block"] == min(5, NB - 2))].float().to(DEV); Sc = S0[b][typ] - S0[b][typ].mean(0, keepdim=True)
fams = {"wdd": unit(Rm[keep[:Kf]] if len(keep) >= Kf else Rm[torch.randperm(len(Rm), device=DEV)[:Kf]]), "random": unit(torch.randn(Kf, D, device=DEV)), "cov_random": unit((torch.randn(Kf, len(Sc), device=DEV) / len(Sc) ** 0.5) @ Sc), "attention": unit(Ratt[torch.randperm(len(Ratt), device=DEV)[:Kf]]), "later_mlp": unit(R5[torch.randperm(len(R5), device=DEV)[:Kf]])}
base = run(positions=pool); lg0 = base["lg"]; data = {}
for nm, V in fams.items():
    a = torch.randint(0, Kf, (len(pool),), device=DEV); inj = torch.zeros(NT, D, device=DEV); inj[pool] = s_inj * V[a]; r = run(positions=pool, inject=inj, inject_block=b + 1); dlr = r["lg"] - lg0; dlr = dlr - dlr.mean(1, keepdim=True); data[nm] = dict(a=a, dl=dlr, F={lv: (r[lv] - S0[lv])[pool] for lv in levels})
def cores(F, dlr, a, rows):
    Fc = F[rows] - F[rows].mean(0, keepdim=True); dd = dlr[rows]; G = dd @ dd.T; evg, Vg = torch.linalg.eigh(G); Z = Vg.flip(1)[:, :128] * evg.flip(0)[:128].clamp_min(0).sqrt()[None]; Z = Z - Z.mean(0, keepdim=True); S_fn = torch.linalg.svd(Fc.T @ Z, full_matrices=False)[0][:, :d]
    cents = torch.stack([Fc[a[rows] == k].mean(0) if (a[rows] == k).any() else torch.zeros(D, device=DEV) for k in range(Kf)]); wv = torch.bincount(a[rows], minlength=Kf).float(); Sb = (cents * wv[:, None]).T @ cents / wv.sum(); return S_fn, torch.linalg.eigh(Sb)[1].flip(1)[:, :d]
inside = lambda A_, B_: ((B_.T @ A_) ** 2).sum().item() / A_.shape[1]; names = list(fams); out = {}
for lv in levels:
    C = {nm: cores(data[nm]["F"][lv], data[nm]["dl"], data[nm]["a"], torch.arange(len(pool), device=DEV)) for nm in names}; half = torch.arange(len(pool), device=DEV) % 2 == 0; rel = {nm: cores(data[nm]["F"][lv], data[nm]["dl"], data[nm]["a"], torch.nonzero(half)[:, 0]) for nm in names}; rel2 = {nm: cores(data[nm]["F"][lv], data[nm]["dl"], data[nm]["a"], torch.nonzero(~half)[:, 0]) for nm in names}
    cross_fn = [inside(C[x][0], C[y][0]) for i, x in enumerate(names) for y in names[i + 1:]]; cross_id = [inside(C[x][1], C[y][1]) for i, x in enumerate(names) for y in names[i + 1:]]; within_fn = [inside(rel[x][0], rel2[x][0]) for x in names]; within_id = [inside(rel[x][1], rel2[x][1]) for x in names]; wdd_vs = {y: inside(C["wdd"][0], C[y][0]) for y in names if y != "wdd"}
    out[lv] = dict(cross_function_mean=sum(cross_fn) / len(cross_fn), cross_function_min=min(cross_fn), cross_identity_mean=sum(cross_id) / len(cross_id), within_function_mean=sum(within_fn) / len(within_fn), within_identity_mean=sum(within_id) / len(within_id), wdd_function_vs=wdd_vs, chance=d / D)
log(f"{tag} (chance {d / D:.3f}): " + " | ".join(f"level {lv}: function cores cross-family mean {v['cross_function_mean']:.2f} (min {v['cross_function_min']:.2f}) vs within-family reliability {v['within_function_mean']:.2f}; identity cores cross-family {v['cross_identity_mean']:.2f} vs within {v['within_identity_mean']:.2f}; WDD function core vs " + ", ".join(f"{y} {vv:.2f}" for y, vv in v['wdd_function_vs'].items()) for lv, v in out.items()))
record(f"e306_familycores_{tag}", dict(model=tag, b=b, L=L, per_level={str(k): v for k, v in out.items()}), " | ".join(f"lv{lv}: fn cross {v['cross_function_mean']:.2f} within {v['within_function_mean']:.2f}; id cross {v['cross_identity_mean']:.2f} within {v['within_identity_mean']:.2f}" for lv, v in out.items()) + f" (chance {d / D:.3f})")
