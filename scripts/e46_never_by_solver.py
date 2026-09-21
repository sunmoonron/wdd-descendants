"""e46: is the never-identified class solver-limited or state-limited? Per-neuron recall (dominant writes) under
OMP, dual-frame one-shot and LASSO(+refit) on the same states; the fate of OMP's never-neurons under the other
solvers; the union recall; and the ICC of each solver's per-neuron recall."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); K = 64; N = int(os.environ.get("WDD_N", 4096)); ids = sub(c.NT, N)
X = c.X(L)[ids]; typ = typical_mask(c.X(L, center=False)[ids]); A, lab = c.dictionary(L)
tb, tn, tc = c.top_writes(L, 1); row = c.atom_index(L, tb[ids, 0], tn[ids, 0]).to(DEV)
S = A.T @ A; ev, V = torch.linalg.eigh(S); Winv = V @ torch.diag(1 / (ev + 1e-2 * ev[-1])) @ V.T
hits = {}
sel, cof, err = omp(X, A, K); hits["omp"] = (sel == row[:, None]).any(1)
s1, _, _ = oneshot(X, A, K, whiten=Winv); hits["dual_oneshot"] = (s1 == row[:, None]).any(1)
s0, _, _ = oneshot(X, A, K); hits["oneshot"] = (s0 == row[:, None]).any(1)
lam = None
try: lam = json.load(open(os.path.join(RESULTS, f"e04_solvers_{tag}.json")))["solvers"]["lasso"]["lambda"]
except Exception: lam = 0.3
C = fista_l1(X, A, lam, iters=300, L=ev[-1].item()).float().to(DEV); sl = C.abs().topk(K, dim=1).indices; del C; hits["lasso"] = (sl == row[:, None]).any(1)
hits["union"] = hits["omp"] | hits["dual_oneshot"] | hits["lasso"]
key = row[typ]; uk, inv, cnt = key.unique(return_inverse=True, return_counts=True); m = cnt >= 20
per = {k: torch.zeros(len(uk), device=DEV).index_add_(0, inv, v[typ].float()) / cnt for k, v in hits.items()}
never = m & (per["omp"] < 0.1); always = m & (per["omp"] > 0.9)
def icc(p, n):
    pbar = (p * n).sum() / n.sum(); vb = p.var().item(); vbin = (pbar * (1 - pbar) / n).mean().item(); return (vb - vbin) / max(vb, 1e-9)
res = dict(model=tag, L=L, N=N, recall={k: v[typ].float().mean().item() for k, v in hits.items()}, n_never=int(never.sum()), n_always=int(always.sum()),
           never_under={k: per[k][never].mean().item() for k in per}, always_under={k: per[k][always].mean().item() for k in per},
           icc={k: icc(per[k][m], cnt[m].float()) for k in per}, frac_never={k: (per[k][m] < 0.1).float().mean().item() for k in per},
           never_rescued_frac={k: (per[k][never] > 0.5).float().mean().item() for k in per})
record(f"e46_never_{tag}", res, "recall " + " ".join(f"{k} {v:.3f}" for k, v in res["recall"].items()) + f" | OMP-never neurons (n={res['n_never']}) under: " + " ".join(f"{k} {v:.2f}" for k, v in res["never_under"].items()) + " | ICC " + " ".join(f"{k} {v:.2f}" for k, v in res["icc"].items()) + " | frac never " + " ".join(f"{k} {v:.2f}" for k, v in res["frac_never"].items()))
