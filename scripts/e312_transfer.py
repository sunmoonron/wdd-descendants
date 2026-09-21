"""e312: does the converged coordinate system transfer across perturbation distributions? Five families of 30
vectors injected at block 3 (WDD, random, covariance-matched, attention, later-MLP), images at L and near the end.
(7) Core transfer: the 16-dimensional function core fitted on family A decodes family B's function and future (kNN)
and identity, against B's own core and a random subspace; reported as the mean off-diagonal score over family pairs
relative to the own-core score. (8) Metric transfer: the causal metric fitted on A (ridge to the footprint PCA)
scores B's centroid geometry against B's functional geometry, against the Euclidean metric. (12) Cross-family
superposition: pairs (a from WDD, b from another family) with high functional similarity and low physical similarity,
injected together: cos(F(a+b), F(a)+F(b)) and relative error, against pairs with low functional similarity."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from func_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; NB = c.NB; b = 2; D = c.D; lab = c.d["lab"]; A = c.d["A"]; Kf = 30; d = 16; levels = sorted({L, NB - 2})
led, tn, tc = dominant(model, arch, c, ids_seq, b); run = make_runner_logits(model, arch, c, ids_seq, [b] + levels, NT, b); S0 = run(); typ = typical_mask(S0[L]); pool = torch.nonzero(typ)[:, 0]; s_inj = tc.abs().median(); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20)
torch.manual_seed(0); Rm = A[(lab["type"] == T_MLP) & (lab["block"] == b)].float().to(DEV); Ratt = A[(lab["type"] == T_ATT) & (lab["block"] <= b)].float().to(DEV); R5 = A[(lab["type"] == T_MLP) & (lab["block"] == min(5, NB - 2))].float().to(DEV); Sc = S0[b][typ] - S0[b][typ].mean(0, keepdim=True)
fams = {"wdd": unit(Rm[keep[:Kf]] if len(keep) >= Kf else Rm[torch.randperm(len(Rm), device=DEV)[:Kf]]), "random": unit(torch.randn(Kf, D, device=DEV)), "cov_random": unit((torch.randn(Kf, len(Sc), device=DEV) / len(Sc) ** 0.5) @ Sc), "attention": unit(Ratt[torch.randperm(len(Ratt), device=DEV)[:Kf]]), "later_mlp": unit(R5[torch.randperm(len(R5), device=DEV)[:Kf]])}
base = run(positions=pool); lg0 = base["lg"]; data = {}
for nm, V in fams.items():
    a = torch.randint(0, Kf, (len(pool),), device=DEV); inj = torch.zeros(NT, D, device=DEV); inj[pool] = s_inj * V[a]; r = run(positions=pool, inject=inj, inject_block=b + 1); dlr = r["lg"] - lg0; dlr = dlr - dlr.mean(1, keepdim=True); data[nm] = dict(a=a, dl=dlr, F={lv: (r[lv] - S0[lv])[pool] for lv in levels}, V=V)
Xd = torch.nan_to_num(torch.cat([data[nm]["dl"][::4] for nm in fams])); evd, Ued = torch.linalg.eigh((Xd @ Xd.T).double()); Ued, evd = Ued.flip(1)[:, :256].float(), evd.flip(0)[:256].clamp_min(1e-6).float(); B = Xd.T @ (Ued / evd.sqrt()[None])
def knn(Ptr, Pte, target, tr, k=5):
    nn = torch.cdist(Pte, Ptr).topk(k, dim=1, largest=False).indices; return target[tr][nn].mean(1)
def spearman(a_, b_):
    ra = a_.argsort().argsort().float(); rb = b_.argsort().argsort().float(); return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
names = list(fams); out = {}
for lv in levels:
    cores = {}; metrics = {}
    for nm in names:
        F = data[nm]["F"][lv]; Fc = F - F.mean(0, keepdim=True); Z = data[nm]["dl"] @ B; Z = Z - Z.mean(0, keepdim=True); cores[nm] = torch.linalg.svd(Fc.T @ Z, full_matrices=False)[0][:, :d]; Gx = Fc.T @ Fc; W = torch.linalg.solve(Gx + 1e-1 * Gx.diagonal().mean() * torch.eye(D, device=DEV), Fc.T @ Z); metrics[nm] = W @ W.T
    torch.manual_seed(2); S_rand = torch.linalg.qr(torch.randn(D, d, device=DEV))[0]; tr_sc = {}; own = {}; rnd = {}; met_own = {}; met_cross = {}; met_euc = {}
    for bnm in names:
        F = data[bnm]["F"][lv]; Fc = F - F.mean(0, keepdim=True); a = data[bnm]["a"]; dln = unit(data[bnm]["dl"]); split = torch.rand(len(pool), device=DEV) < 0.5; tr, te = torch.nonzero(split)[:, 0], torch.nonzero(~split)[:, 0]
        def sc(P):
            Ptr, Pte = (Fc @ P)[tr], (Fc @ P)[te]; return dict(function=((unit(knn(Ptr, Pte, dln, tr)) * dln[te]).sum(1)).median().item(), identity=accuracy(Pte, centroids(Ptr, a[tr], Kf), a[te]))
        own[bnm] = sc(cores[bnm]); rnd[bnm] = sc(S_rand); tr_sc[bnm] = {anm: sc(cores[anm]) for anm in names if anm != bnm}
        Cd = torch.stack([Fc[a == k].mean(0) for k in range(Kf)]); Cf = unit(torch.stack([dln[a == k].mean(0) for k in range(Kf)])); iu = torch.triu_indices(Kf, Kf, 1, device=DEV); GF = (Cf @ Cf.T)[iu[0], iu[1]]; GE = (unit(Cd) @ unit(Cd).T)[iu[0], iu[1]]
        def mcos(M): CM = Cd @ M; return ((CM @ Cd.T) / torch.sqrt(torch.outer((CM * Cd).sum(1), (CM * Cd).sum(1))).clamp_min(1e-9))[iu[0], iu[1]]
        met_euc[bnm] = spearman(GE, GF); met_own[bnm] = spearman(mcos(metrics[bnm]), GF); met_cross[bnm] = sum(spearman(mcos(metrics[anm]), GF) for anm in names if anm != bnm) / (len(names) - 1)
    cross_fn = sum(tr_sc[bn][an]["function"] for bn in names for an in names if an != bn) / (len(names) * (len(names) - 1)); own_fn = sum(own[bn]["function"] for bn in names) / len(names); rnd_fn = sum(rnd[bn]["function"] for bn in names) / len(names); cross_id = sum(tr_sc[bn][an]["identity"] for bn in names for an in names if an != bn) / (len(names) * (len(names) - 1)); own_id = sum(own[bn]["identity"] for bn in names) / len(names); rnd_id = sum(rnd[bn]["identity"] for bn in names) / len(names)
    out[lv] = dict(core_transfer=dict(function_cross=cross_fn, function_own=own_fn, function_random=rnd_fn, identity_cross=cross_id, identity_own=own_id, identity_random=rnd_id), metric_transfer=dict(euclid=sum(met_euc.values()) / len(names), own_metric=sum(met_own.values()) / len(names), cross_metric=sum(met_cross.values()) / len(names)))
Fa = {nm: unit(torch.stack([data[nm]["dl"][data[nm]["a"] == k].mean(0) for k in range(Kf)])) for nm in names}; pairs_hi, pairs_lo = [], []
for onm in ("random", "attention", "later_mlp"):
    fs = Fa["wdd"] @ Fa[onm].T; ps = fams["wdd"] @ fams[onm].T; lowphys = ps.abs() < ps.abs().median(); hi = (fs >= fs.quantile(0.9)) & lowphys; lo = (fs <= fs.quantile(0.1)) & lowphys; pairs_hi += [(onm, int(i), int(j)) for i, j in torch.nonzero(hi).tolist()][:4]; pairs_lo += [(onm, int(i), int(j)) for i, j in torch.nonzero(lo).tolist()][:4]
sub = pool[::2][:1024]; base2 = run(positions=sub); lg2 = base2["lg"]
def eff(v):
    inj = torch.zeros(NT, D, device=DEV); inj[sub] = s_inj * v[None]; r = run(positions=sub, inject=inj, inject_block=b + 1); dlr = r["lg"] - lg2; return dlr - dlr.mean(1, keepdim=True)
def superpose(pairs):
    cs, errs = [], []
    for onm, i, j in pairs:
        Ea, Eb, Eab = eff(fams["wdd"][i]), eff(fams[onm][j]), eff(fams["wdd"][i] + fams[onm][j]); lin = Ea + Eb; cs.append(((unit(Eab) * unit(lin)).sum(1)).median().item()); errs.append(((Eab - lin).norm(dim=1) / Eab.norm(dim=1).clamp_min(1e-9)).median().item())
    return dict(n=len(pairs), cos=sum(cs) / max(len(cs), 1), rel_error=sum(errs) / max(len(errs), 1))
sup = dict(functionally_similar=superpose(pairs_hi), functionally_dissimilar=superpose(pairs_lo))
log(f"{tag}: " + " | ".join(f"level {lv}: core transfer function cross/own/random {v['core_transfer']['function_cross']:.2f}/{v['core_transfer']['function_own']:.2f}/{v['core_transfer']['function_random']:.2f}, identity {v['core_transfer']['identity_cross']:.2f}/{v['core_transfer']['identity_own']:.2f}/{v['core_transfer']['identity_random']:.2f}; metric transfer Spearman Euclid/own/cross {v['metric_transfer']['euclid']:.2f}/{v['metric_transfer']['own_metric']:.2f}/{v['metric_transfer']['cross_metric']:.2f}" for lv, v in out.items()) + f" | cross-family superposition (physically dissimilar pairs): functionally similar cos {sup['functionally_similar']['cos']:.2f} error {sup['functionally_similar']['rel_error']:.2f} (n {sup['functionally_similar']['n']}), dissimilar cos {sup['functionally_dissimilar']['cos']:.2f} error {sup['functionally_dissimilar']['rel_error']:.2f} (n {sup['functionally_dissimilar']['n']})")
record(f"e312_transfer_{tag}", dict(model=tag, b=b, L=L, per_level={str(k): v for k, v in out.items()}, superposition=sup), " | ".join(f"lv{lv}: fn cross/own/rand {v['core_transfer']['function_cross']:.2f}/{v['core_transfer']['function_own']:.2f}/{v['core_transfer']['function_random']:.2f} id {v['core_transfer']['identity_cross']:.2f}/{v['core_transfer']['identity_own']:.2f}/{v['core_transfer']['identity_random']:.2f} metric E/own/cross {v['metric_transfer']['euclid']:.2f}/{v['metric_transfer']['own_metric']:.2f}/{v['metric_transfer']['cross_metric']:.2f}" for lv, v in out.items()) + f" | superposition similar cos {sup['functionally_similar']['cos']:.2f} err {sup['functionally_similar']['rel_error']:.2f} vs dissimilar {sup['functionally_dissimilar']['cos']:.2f} err {sup['functionally_dissimilar']['rel_error']:.2f}")
