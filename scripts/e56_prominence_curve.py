"""e56: the recall-vs-prominence law in the real models (from the cached mid-layer OMP): identification rate of the
dominant write binned by its prominence |x_c . d| / ||x_c|| (0.1-wide bins), for OMP and one-shot, so the
synthetic S1 curves can be compared to it."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); X = c.X(L); typ = typical_mask(c.X(L, center=False)); A, lab = c.dictionary(L)
sel, cof, err = get_omp(c, L, A=A, X=X); tb, tn, tc = c.top_writes(L, 1); row = c.atom_index(L, tb[:, 0], tn[:, 0]).to(DEV)
hit = (sel == row[:, None]).any(1); s1, _, _ = oneshot(X, A, 64); hit1 = (s1 == row[:, None]).any(1)
prom = (X * A[row]).sum(1).abs() / X.norm(dim=1); bins = {}
for lo in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    m = typ & (prom >= lo) & (prom < lo + 0.1)
    if m.sum() > 20: bins[f"{lo:.1f}"] = dict(n=int(m.sum()), omp=hit[m].float().mean().item(), oneshot=hit1[m].float().mean().item())
res = dict(model=tag, L=L, m_over_d=A.shape[0] / c.D, prom_med=prom[typ].median().item(), bins=bins)
record(f"e56_promcurve_{tag}", res, f"m/d {res['m_over_d']:.0f} prom med {res['prom_med']:.2f} | " + " ".join(f"{k}: omp {v['omp']:.2f} os {v['oneshot']:.2f} (n{v['n']})" for k, v in bins.items()))
