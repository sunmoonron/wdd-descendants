"""e282: under which metric is transport closest to an isometry? The block-2 candidate writes and as many random unit
directions are injected at the block-3 input (natural median amplitude), their image centroids read at +2, L and
NB-2. In the Euclidean geometry and in the whitened geometry of the residual stream (Mahalanobis under the state
covariance of the injection level and of the reading level): the spread of the gain (coefficient of variation over
directions, writes and random separately) and the Spearman between input and image pair cosines for the writes
(random directions have no input geometry to preserve). Also the singular-value spread of the fitted operator
T(3->L) in both geometries. The geometry with the smaller gain spread and the higher Gram preservation is the one
in which the transport is an isometry; a smaller spread for writes than for random directions means the isometry
holds on the manifold of actual writes."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from desc_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; NB = c.NB; b = 2; D = c.D; lab = c.d["lab"]; A = c.d["A"]; R = A[(lab["type"] == T_MLP) & (lab["block"] == b)].float().to(DEV); levels = sorted({b, b + 2, L, NB - 2}); run = make_runner(model, arch, c, ids_seq, levels, NT)
led, tn, tc = dominant(model, arch, c, ids_seq, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep)
pool = torch.nonzero(typ)[:, 0]; s_inj = tc.abs().median(); torch.manual_seed(0); W = unit(R[keep]); Rn = unit(torch.randn(K, D, device=DEV)); dirs = torch.cat([W, Rn]); ND = 2 * K
imgs = {lv: torch.zeros(ND, D, device=DEV) for lv in levels if lv > b}; cnt = torch.zeros(ND, device=DEV)
for p in range(3):
    a = torch.randint(0, ND, (len(pool),), device=DEV); inj = torch.zeros(NT, D, device=DEV); inj[pool] = s_inj * dirs[a]; S2 = run(inject=inj, inject_block=b + 1)
    for lv in imgs: imgs[lv].index_add_(0, a, (S2[lv] - S0[lv])[pool] / s_inj)
    cnt.index_add_(0, a, torch.ones(len(pool), device=DEV))
for lv in imgs: imgs[lv] = imgs[lv] / cnt[:, None].clamp_min(1)
def whitener(S):
    Sc = S - S.mean(0, keepdim=True); C = Sc.T @ Sc / len(Sc); ev, V = torch.linalg.eigh(C); ev = ev.clamp_min(ev.max() * 1e-4); return V @ torch.diag(ev.rsqrt()) @ V.T, V @ torch.diag(ev.sqrt()) @ V.T
Wb, Wb_inv = whitener(S0[b][typ])
def spearman(a_, b_):
    ra = a_.argsort().argsort().float(); rb = b_.argsort().argsort().float(); return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
iu = torch.triu_indices(K, K, 1, device=DEV); gram_in_e = (W @ W.T)[iu[0], iu[1]]; Wt = W @ Wb; gram_in_w = (unit(Wt) @ unit(Wt).T)[iu[0], iu[1]]
X = []; Y = []; torch.manual_seed(5); Vr = unit(torch.randn(1024, D, device=DEV))
for p in range(2):
    a = torch.randint(0, 1024, (len(pool),), device=DEV); inj = torch.zeros(NT, D, device=DEV); inj[pool] = s_inj * Vr[a]; S2 = run(inject=inj, inject_block=b + 1); X.append(Vr[a]); Y.append((S2[L] - S0[L])[pool] / s_inj)
X, Y = torch.cat(X), torch.cat(Y); G = X.T @ X; T = torch.linalg.solve(G + 1e-2 * G.diagonal().mean() * torch.eye(D, device=DEV), X.T @ Y)
out = {}
for lv in imgs:
    Wl, Wl_inv = whitener(S0[lv][typ]); Ye = imgs[lv]; Yw = Ye @ Wl; Xw_norm = (dirs @ Wb).norm(dim=1)
    ge = Ye.norm(dim=1); gw = Yw.norm(dim=1) / Xw_norm; cv = lambda g: (g.std() / g.mean()).item()
    rec = dict(euclid_gain_cv_writes=cv(ge[:K]), euclid_gain_cv_random=cv(ge[K:]), whitened_gain_cv_writes=cv(gw[:K]), whitened_gain_cv_random=cv(gw[K:]), euclid_gain_writes=ge[:K].median().item(), euclid_gain_random=ge[K:].median().item(), gram_spearman_euclid=spearman(gram_in_e, (unit(Ye[:K]) @ unit(Ye[:K]).T)[iu[0], iu[1]]), gram_spearman_whitened=spearman(gram_in_w, (unit(Yw[:K]) @ unit(Yw[:K]).T)[iu[0], iu[1]]), gram_spearman_cross_euclid_in_whitened_out=spearman(gram_in_e, (unit(Yw[:K]) @ unit(Yw[:K]).T)[iu[0], iu[1]]))
    if lv == L:
        sv = torch.linalg.svdvals(T); svw = torch.linalg.svdvals(Wb_inv @ T @ Wl); rec["operator_sv_spread_euclid"] = (sv.quantile(0.9) / sv.quantile(0.1)).item(); rec["operator_sv_spread_whitened"] = (svw.quantile(0.9) / svw.quantile(0.1)).item()
    out[lv] = rec; log(f"{tag} level {lv}: gain CV writes/random Euclidean {rec['euclid_gain_cv_writes']:.2f}/{rec['euclid_gain_cv_random']:.2f}, whitened {rec['whitened_gain_cv_writes']:.2f}/{rec['whitened_gain_cv_random']:.2f}; Gram Spearman (writes) Euclidean {rec['gram_spearman_euclid']:.2f}, whitened {rec['gram_spearman_whitened']:.2f}, Euclidean-in/whitened-out {rec['gram_spearman_cross_euclid_in_whitened_out']:.2f}" + (f"; operator singular-value spread q90/q10 Euclidean {rec['operator_sv_spread_euclid']:.1f}, whitened {rec['operator_sv_spread_whitened']:.1f}" if lv == L else ""))
record(f"e282_metric_{tag}", dict(model=tag, b=b, L=L, K=K, per_level={str(k): v for k, v in out.items()}), " | ".join(f"lv{lv}: gain CV E {v['euclid_gain_cv_writes']:.2f}/{v['euclid_gain_cv_random']:.2f} W {v['whitened_gain_cv_writes']:.2f}/{v['whitened_gain_cv_random']:.2f}; Gram E {v['gram_spearman_euclid']:.2f} W {v['gram_spearman_whitened']:.2f}" + (f"; sv spread E {v['operator_sv_spread_euclid']:.1f} W {v['operator_sv_spread_whitened']:.1f}" if 'operator_sv_spread_euclid' in v else "") for lv, v in out.items()))
