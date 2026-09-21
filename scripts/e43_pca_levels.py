"""e43: per level, the data-derived baselines and the shape of the state: PCA (from the centering slice) FVU at
k=32/64, the state's effective rank, the top-5 coordinate share, and the weight dictionary's FVU on the same
sample. Reads the final-layer FVU drop and where the dictionary beats or loses to PCA with depth."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); N = int(os.environ.get("WDD_N", 4096)); ids = sub(c.NT, N); rows = []
for L in range(c.NB):
    X = c.X(L)[ids]; Xraw = c.X(L, center=False)[ids]; typ = typical_mask(Xraw); A, lab = c.dictionary(L)
    # PCA basis from the centering slice of the same level: we only stored the mean, so use the eval typical states of OTHER sequences (odd) and evaluate on even
    seq = (torch.arange(c.NT)[ids] // CTX).to(DEV); fit = typ & (seq % 2 == 1); ev = typ & (seq % 2 == 0)
    U = torch.linalg.eigh((X[fit].T @ X[fit]) / fit.sum()).eigenvectors
    pca = {k: (((X[ev] - (X[ev] @ U[:, -k:]) @ U[:, -k:].T) ** 2).sum() / (X[ev] ** 2).sum()).item() for k in (32, 64)}
    sel, cof, err = omp(X, A, 64); w = {k: fvu(err[:, k - 1], X, ev) for k in (32, 64)}
    evals = torch.linalg.eigvalsh((X[typ].T @ X[typ]) / typ.sum()); p = evals / evals.sum(); effr = math.exp(-(p * (p + 1e-12).log()).sum().item())
    coord = (Xraw[typ] ** 2).mean(0); top5 = (coord.topk(5).values.sum() / coord.sum()).item()
    rows.append(dict(L=L, pca32=pca[32], pca64=pca[64], wdd32=w[32], wdd64=w[64], eff_rank=effr, top5_coord_share=top5, norm_med=Xraw[typ].norm(dim=1).median().item(), sink_share=((X[~typ] ** 2).sum() / (X ** 2).sum()).item()))
    log(f"{tag} L{L}: pca32 {pca[32]:.3f} wdd32 {w[32]:.3f} pca64 {pca[64]:.3f} wdd64 {w[64]:.3f} effrank {effr:.0f} top5 {top5:.2f} norm {rows[-1]['norm_med']:.1f}")
record(f"e43_pca_{tag}", dict(model=tag, N=N, rows=rows), " | ".join(f"L{r['L']}: pca {r['pca32']:.2f} wdd {r['wdd32']:.2f} rank {r['eff_rank']:.0f}" for r in rows))
