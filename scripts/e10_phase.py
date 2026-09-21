"""e10: a Donoho-Tanner style map. Identification and FVU as a function of the budget k/d and the overcompleteness
m/d, by subsampling the non-embedding atoms (keeping the true dominant atom always present) and sweeping k.
Also the rotated dictionary at each cell so alignment can be read as a function of (k, m)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); N = int(os.environ.get("WDD_N", 4096)); ids = sub(c.NT, N)
X = c.X(L)[ids]; typ = typical_mask(c.X(L, center=False)[ids]); A, lab = c.dictionary(L)
tb, tn, tc = c.top_writes(L, 1); row = c.atom_index(L, tb[ids, 0], tn[ids, 0]).to(DEV)
blk = lab["block"].to(DEV) >= 0; NA = A.shape[0]; g = torch.Generator().manual_seed(3)
cells = []
for frac in (0.0625, 0.125, 0.25, 0.5, 1.0):
    keep = torch.ones(NA, dtype=torch.bool, device=DEV)
    if frac < 1:
        r = torch.rand(NA, generator=g).to(DEV); keep = (~blk) | (r < frac)      # keep all embeddings, subsample block atoms
        keep[row.unique()] = True                                                # the true dominant atoms stay
    Asub = A[keep]; newrow = (torch.cumsum(keep.long(), 0) - 1)[row]; Ar = rotate(Asub)
    K = 128 if Asub.shape[0] > 128 else Asub.shape[0] - 1
    sel, cof, err = omp(X, Asub, K); selr, cofr, errr = omp(X, Ar, K)
    for k in (4, 8, 16, 32, 64, 128):
        hit = (sel[:, :k] == newrow[:, None]).any(1)[typ].float().mean().item()
        cells.append(dict(frac=frac, m=Asub.shape[0], m_over_d=Asub.shape[0] / c.D, k=k, k_over_d=k / c.D, recall=hit, fvu=fvu(err[:, k - 1], X, typ), fvu_rot=fvu(errr[:, k - 1], X, typ)))
    log(f"{tag} frac {frac} m {Asub.shape[0]}: " + " ".join(f"k{k}: rec {cl['recall']:.2f} fvu {cl['fvu']:.2f}/{cl['fvu_rot']:.2f}" for k in (8, 32, 128) for cl in [x for x in cells if x['frac'] == frac and x['k'] == k]))
record(f"e10_phase_{tag}", dict(model=tag, L=L, N=N, D=c.D, cells=cells), " | ".join(f"m/d {cl['m_over_d']:.0f} k{cl['k']}: rec {cl['recall']:.2f} fvu {cl['fvu']:.2f} rot {cl['fvu_rot']:.2f}" for cl in cells if cl['k'] in (16, 64)))
