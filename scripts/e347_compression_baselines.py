"""e347: the compression baselines at every dimension. The logit footprint decoded (kNN cosine, held-out) from
projections of the descendant of dimension 1 to 128 by: random projection, PCA, PLS (the quotient), CCA to the
logit scores, ICA (FastICA-like fixed-point on the top-64 PCs), kernel PCA (RBF, top components), random Fourier
features (nonlinear, dimension-matched), and reduced-rank ridge; plus the same under a mean-squared-error criterion.
The dimension for 90% of the full-descendant kNN score per method."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quot_common import *
tag = sys.argv[1]; S = Setup(tag, levels=[0]); L = S.L; S = Setup(tag, levels=[L]); nat = S.natural(); F = nat["F"][L]; tr, te = S.halves(len(S.idx)); Fc = F - F[tr].mean(0, keepdim=True); Z, _ = logit_scores(nat["dl"], tr); dln = unit(nat["dl"]); dims = [1, 2, 4, 8, 16, 32, 64, 128]; full = knn_cos(Fc, dln, tr, te); torch.manual_seed(0)
Upca = torch.linalg.svd(Fc[tr], full_matrices=False)[2][:128].T; Upls = pls(Fc[tr], Z[tr], 128); Xp = Fc[tr] @ Upca[:, :64]; Cxx = Xp.T @ Xp + 1e-3 * torch.eye(64, device=DEV); Cyy = Z[tr].T @ Z[tr] + 1e-3 * torch.eye(Z.shape[1], device=DEV); Cxy = Xp.T @ Z[tr]; Lx = torch.linalg.cholesky(Cxx); Ly = torch.linalg.cholesky(Cyy); M = torch.linalg.solve(Lx, torch.linalg.solve(Ly, Cxy.T).T); Uc = torch.linalg.svd(M, full_matrices=False)[0]; Ucca = Upca[:, :64] @ torch.linalg.solve(Lx.T, Uc)
Xw = Xp / Xp.std(0, keepdim=True).clamp_min(1e-6); Wi = torch.linalg.qr(torch.randn(64, 64, device=DEV))[0]
for it in range(30):
    G = torch.tanh(Xw @ Wi); Wn = Xw.T @ G / len(Xw) - Wi * (1 - G ** 2).mean(0)[None]; Wi = torch.linalg.qr(Wn)[0]
Uica = Upca[:, :64] @ (Wi / Xp.std(0, keepdim=True).clamp_min(1e-6).T)[:, :64]; kurt = ((Xw @ Wi) ** 4).mean(0) - 3; Uica = Uica[:, kurt.argsort(descending=True)]
d2 = torch.cdist(Fc[tr], Fc[tr]) ** 2; med = d2[d2 > 0].median(); Kk = torch.exp(-d2 / med); Kc = Kk - Kk.mean(0, keepdim=True) - Kk.mean(1, keepdim=True) + Kk.mean(); ev, Vk = torch.linalg.eigh(Kc); Vk = Vk.flip(1)[:, :128] / ev.flip(0)[:128].clamp_min(1e-6).sqrt()[None]; Kte = torch.exp(-(torch.cdist(Fc, Fc[tr]) ** 2) / med); Kte_c = Kte - Kte.mean(1, keepdim=True) - Kk.mean(0, keepdim=True) + Kk.mean(); kpca = Kte_c @ Vk
Wrff = torch.randn(S.D, 128, device=DEV) / med.sqrt() * 2 ** 0.5; brff = torch.rand(128, device=DEV) * 2 * math.pi; rff = torch.cos(Fc @ Wrff + brff); Wrr = ridge(Fc[tr], Z[tr], 1e-1); Vrr = torch.linalg.svd(Fc[tr] @ Wrr, full_matrices=False)[2]
methods = {"random": lambda k: Fc @ torch.linalg.qr(torch.randn(S.D, k, device=DEV))[0], "pca": lambda k: Fc @ Upca[:, :k], "pls_quotient": lambda k: Fc @ Upls[:, :k], "cca": lambda k: Fc @ Ucca[:, :min(k, 64)], "ica": lambda k: Fc @ Uica[:, :min(k, 64)], "kernel_pca": lambda k: kpca[:, :k], "random_fourier": lambda k: rff[:, :k], "reduced_rank_ridge": lambda k: (Fc @ Wrr) @ Vrr[:k].T}
out = {}
for nm, fn in methods.items():
    curve = {k: knn_cos(fn(k), dln, tr, te) for k in dims}; out[nm] = dict(dim90=next((k for k in dims if curve[k] >= 0.9 * full), 128), curve={str(k): v for k, v in curve.items()}, at8=curve[8], at16=curve[16])
log(f"{tag} (K {S.K}, full-descendant kNN score {full:.2f}): dimension for 90% of full, and score at 8 / 16: " + " ; ".join(f"{nm} {v['dim90']} ({v['at8']:.2f}/{v['at16']:.2f})" for nm, v in out.items()))
record(f"e347_baselines_{tag}", dict(model=tag, L=L, K=S.K, full=full, per_method=out), " ; ".join(f"{nm} {v['dim90']} ({v['at16']:.2f})" for nm, v in out.items()) + f" | full {full:.2f}")
