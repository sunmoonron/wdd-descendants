"""e09: residual of the residual (meta). After OMP with the weight dictionary at k, is what remains still aligned
with the model's coordinates, or is alignment exhausted? Decompose the k=64 residual with weight / rotated / random
dictionaries; the reverse order (rotated first, then weight); the frame-covariance alignment of the residual;
and the weight-vs-rotated FVU gap as a function of k on the original state."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); N = int(os.environ.get("WDD_N", 8192)); ids = sub(c.NT, N)
X = c.X(L)[ids]; typ = typical_mask(c.X(L, center=False)[ids]); A, lab = c.dictionary(L); Ar = rotate(A); R = random_dict(A.shape[0], c.D)
def cosF(P, Q): return (P * Q).sum().item() / (P.norm() * Q.norm()).item()
S = A.T @ A; Sr = Ar.T @ Ar
def stage(Xin, Ad, k):
    sel, cof, err = omp(Xin, Ad, k); rec = torch.einsum("nk,nkd->nd", cof, Ad[sel]); return Xin - rec, err
res = dict(model=tag, L=L, N=N)
r_w, err_w = stage(X, A, 64); r_r, err_r = stage(X, Ar, 64); r_n, err_n = stage(X, R, 64)
res["stage1_fvu64"] = dict(weight=fvu(err_w[:, 63], X, typ), rot=fvu(err_r[:, 63], X, typ), random=fvu(err_n[:, 63], X, typ))
res["gap_curve"] = {k: dict(weight=fvu(err_w[:, k - 1], X, typ), rot=fvu(err_r[:, k - 1], X, typ), random=fvu(err_n[:, k - 1], X, typ)) for k in (1, 2, 4, 8, 16, 32, 64)}
# second stage on the weight residual: relative FVU of the residual itself
out = {}
for nm, Ad in (("weight", A), ("rot", Ar), ("random", R)):
    _, e2 = stage(r_w, Ad, 32); out[nm] = fvu(e2[:, 31], r_w, typ)
res["stage2_on_weight_residual_fvu32"] = out
out = {}
for nm, Ad in (("weight", A), ("rot", Ar)):
    _, e2 = stage(r_r, Ad, 32); out[nm] = fvu(e2[:, 31], r_r, typ)
res["stage2_on_rot_residual_fvu32"] = out
Sig = lambda V: (V[typ].T @ V[typ]) / typ.sum()
res["align_S_Sigma"] = dict(state=dict(weight=cosF(S, Sig(X)), rot=cosF(Sr, Sig(X))), weight_residual=dict(weight=cosF(S, Sig(r_w)), rot=cosF(Sr, Sig(r_w))), rot_residual=dict(weight=cosF(S, Sig(r_r)), rot=cosF(Sr, Sig(r_r))))
# how much of the alignment advantage sits in the first few atoms: energy explained by step
ex_w = -(err_w[:, 1:] - err_w[:, :-1]); ex_r = -(err_r[:, 1:] - err_r[:, :-1])
tot = (X ** 2).sum(1)
res["per_step_gain_share"] = dict(weight=[(ex_w[typ][:, i].sum() / tot[typ].sum()).item() for i in (0, 1, 3, 7, 15, 31, 62)], rot=[(ex_r[typ][:, i].sum() / tot[typ].sum()).item() for i in (0, 1, 3, 7, 15, 31, 62)])
record(f"e09_resid_{tag}", res, f"stage1 w/r/n {res['stage1_fvu64']['weight']:.3f}/{res['stage1_fvu64']['rot']:.3f}/{res['stage1_fvu64']['random']:.3f} | stage2 on w-residual w/r/n {out.get('weight', 0):.3f}/{res['stage2_on_weight_residual_fvu32']['rot']:.3f}/{res['stage2_on_weight_residual_fvu32']['random']:.3f} (weight {res['stage2_on_weight_residual_fvu32']['weight']:.3f}) | align state w {res['align_S_Sigma']['state']['weight']:.3f} r {res['align_S_Sigma']['state']['rot']:.3f}; residual w {res['align_S_Sigma']['weight_residual']['weight']:.3f} r {res['align_S_Sigma']['weight_residual']['rot']:.3f} | gap@1 {res['gap_curve'][1]['rot']-res['gap_curve'][1]['weight']:.3f} @8 {res['gap_curve'][8]['rot']-res['gap_curve'][8]['weight']:.3f} @64 {res['gap_curve'][64]['rot']-res['gap_curve'][64]['weight']:.3f}")
