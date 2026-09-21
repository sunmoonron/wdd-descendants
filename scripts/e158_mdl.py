"""e158: minimum description length (Rissanen 1978). Bits to describe a typical state at distortion FVU = 0.30
and 0.15 with (a) the weight dictionary: k atoms x (log2 m + b) bits, (b) PCA of held-out states: k components x b
bits (basis known), (c) random atoms: k x (log2 m + b), with b = 8 bits per coefficient. Which code is shorter at
equal distortion, and what overcompleteness premium does the model's own dictionary pay?"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); N = 4096; ids = sub(c.NT, N); X = c.X(L)[ids]; typ = typical_mask(c.X(L, center=False)[ids]); A, lab = c.dictionary(L)
seq = (torch.arange(c.NT)[ids] // CTX).to(DEV); fit = typ & (seq % 2 == 1); ev = typ & (seq % 2 == 0)
U = torch.linalg.eigh((X[fit].T @ X[fit]) / fit.sum()).eigenvectors; ks = [4, 8, 16, 32, 48, 64, 96, 128, 192, 256]
sel, cof, err = omp(X[ev], A, 256); errr = omp(X[ev], rotate(A), 256)[2]
fv_w = {k: fvu(err[:, k - 1], X[ev]) for k in ks}; fv_r = {k: fvu(errr[:, k - 1], X[ev]) for k in ks}
fv_p = {k: (((X[ev] - (X[ev] @ U[:, -k:]) @ U[:, -k:].T) ** 2).sum() / (X[ev] ** 2).sum()).item() for k in ks}
b = 8; m = A.shape[0]; bits_w = {k: k * (math.log2(m) + b) for k in ks}; bits_p = {k: k * b for k in ks}
def k_at(fv, target):
    kk = [k for k in ks if fv[k] <= target]; return kk[0] if kk else None
res = dict(model=tag, L=L, m=m, d=c.D, fvu=dict(weight=fv_w, rotated=fv_r, pca=fv_p), bits_at_fvu={})
for target in (0.5, 0.3, 0.15):
    kw, kp, kr = k_at(fv_w, target), k_at(fv_p, target), k_at(fv_r, target)
    res["bits_at_fvu"][target] = dict(k_weight=kw, k_pca=kp, k_rotated=kr, bits_weight=(bits_w[kw] if kw else None), bits_pca=(bits_p[kp] if kp else None), bits_rotated=(bits_w[kr] if kr else None))
record(f"e158_mdl_{tag}", res, " | ".join(f"FVU<={t}: weight k {v['k_weight']} ({v['bits_weight']} bits) vs PCA k {v['k_pca']} ({v['bits_pca']} bits) vs rotated k {v['k_rotated']}" for t, v in res["bits_at_fvu"].items()))
