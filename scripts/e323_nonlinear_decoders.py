"""e323: is the quotient a tangent-space approximation to a nonlinear map? The logit footprint decoded from the
descendant with a linear ridge, kNN (k = 1, 5, 20) and kernel ridge (RBF, closed form) on inputs of increasing
dimension: the 16-dimensional linear quotient, the 64-dimensional quotient, the top-64 PCs, the full descendant.
If nonlinear decoders on the 16-dimensional coordinate beat linear ones, the map from coordinate to function is
nonlinear; if nonlinear decoders on higher-dimensional inputs beat the 16-dimensional ones by a wide margin, the
coordinate is only a tangent approximation."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quot_common import *
tag = sys.argv[1]; S = Setup(tag, levels=[0]); L = S.L; S = Setup(tag, levels=[L]); nat = S.natural(); F = nat["F"][L]; tr, te = S.halves(len(S.idx)); Fc = F - F[tr].mean(0, keepdim=True); Z, _ = logit_scores(nat["dl"], tr); dln = unit(nat["dl"]); Q16 = pls(Fc[tr], Z[tr], 16); Q64 = pls(Fc[tr], Z[tr], 64); U64 = torch.linalg.svd(Fc[tr], full_matrices=False)[2][:64].T
def lin(X): W = ridge(X[tr], Z[tr], 1e-1); return knn_cos_pred(X[te] @ W)
def knn_cos_pred(predZ): return ((unit(predZ @ Bt) * dln[te]).sum(1)).median().item()
_, Bv = logit_scores(nat["dl"], tr); Bt = Bv.T
def krr(X, gamma_scale=1.0):
    Xtr, Xte = X[tr], X[te]; d2 = torch.cdist(Xtr, Xtr) ** 2; med = d2[d2 > 0].median(); Kt = torch.exp(-d2 / (gamma_scale * med)); Kte = torch.exp(-(torch.cdist(Xte, Xtr) ** 2) / (gamma_scale * med)); alpha = torch.linalg.solve(Kt + 1e-2 * torch.eye(len(tr), device=DEV), Z[tr] - Z[tr].mean(0, keepdim=True)); return knn_cos_pred(Kte @ alpha + Z[tr].mean(0, keepdim=True))
inputs = {"quotient16": Fc @ Q16, "quotient64": Fc @ Q64, "pca64": Fc @ U64, "full": Fc}; out = {}
for nm, X in inputs.items(): out[nm] = dict(linear=lin(X), knn1=knn_cos(X, dln, tr, te, 1), knn5=knn_cos(X, dln, tr, te, 5), knn20=knn_cos(X, dln, tr, te, 20), kernel_ridge=krr(X))
log(f"{tag} (K {S.K}): logit footprint cosine by decoder (linear / kNN-1 / kNN-5 / kNN-20 / kernel ridge): " + " | ".join(f"{nm}: {v['linear']:.2f}/{v['knn1']:.2f}/{v['knn5']:.2f}/{v['knn20']:.2f}/{v['kernel_ridge']:.2f}" for nm, v in out.items()))
record(f"e323_nonlinear_{tag}", dict(model=tag, L=L, K=S.K, per_input=out), " | ".join(f"{nm} lin {v['linear']:.2f} knn5 {v['knn5']:.2f} krr {v['kernel_ridge']:.2f}" for nm, v in out.items()))
