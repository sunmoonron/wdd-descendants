"""e49: the depth map with the better identifier. Per level: dominant-write recall under the dual-frame one-shot
(k=64) and plain one-shot, next to OMP from e01, plus the fraction of tokens whose dominant write was born in the
last two blocks. Is the late-depth collapse a solver artifact or a state property?"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); N = int(os.environ.get("WDD_N", 8192)); ids = sub(c.NT, N); rows = []
for L in range(c.NB):
    X = c.X(L)[ids]; typ = typical_mask(c.X(L, center=False)[ids]); A, lab = c.dictionary(L)
    tb, tn, tc = c.top_writes(L, 1); row = c.atom_index(L, tb[ids, 0], tn[ids, 0]).to(DEV)
    S = A.T @ A; ev, V = torch.linalg.eigh(S); Winv = V @ torch.diag(1 / (ev + 1e-2 * ev[-1])) @ V.T
    sd, _, _ = oneshot(X, A, 64, whiten=Winv); s0, _, _ = oneshot(X, A, 64)
    born_late = (tb[ids, 0].to(DEV) >= L - 1)[typ].float().mean().item()
    prom = ((X * A[row]).sum(1).abs() / X.norm(dim=1))[typ].median().item()
    rows.append(dict(L=L, dual=(sd == row[:, None]).any(1)[typ].float().mean().item(), oneshot=(s0 == row[:, None]).any(1)[typ].float().mean().item(), born_last2=born_late, prominence_med=prom))
    log(f"{tag} L{L}: dual {rows[-1]['dual']:.3f} oneshot {rows[-1]['oneshot']:.3f} born-last2 {born_late:.2f} prom {prom:.3f}")
record(f"e49_duallayers_{tag}", dict(model=tag, N=N, rows=rows), " | ".join(f"L{r['L']}: dual {r['dual']:.2f} os {r['oneshot']:.2f} late {r['born_last2']:.2f} prom {r['prominence_med']:.2f}" for r in rows))
