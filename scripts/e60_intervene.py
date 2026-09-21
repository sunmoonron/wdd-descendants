"""e60: is prominence causal for identification at the token level? Intervene on the state only along the dominant
write's own direction: for tokens whose dominant write is cancelled (raw survival < 0.5), add back alpha * c * d
(alpha = 0.25, 0.5, 1.0) and re-run OMP and one-shot; for tokens whose write is intact (survival > 0.9), subtract
beta * c * d (beta = 0.5, 0.9). Everything else in the state is untouched. If identification follows the resulting
prominence along the law's curve, prominence is the operative variable."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); N = int(os.environ.get("WDD_N", 4096)); ids = sub(c.NT, N)
Xraw = c.X(L, center=False)[ids]; mu = c.s["mu"][L + 1].to(DEV); X = Xraw - mu; typ = typical_mask(Xraw); A, lab = c.dictionary(L)
tb, tn, tc = c.top_writes(L, 1); row = c.atom_index(L, tb[ids, 0], tn[ids, 0]).to(DEV); ct = tc[ids, 0].to(DEV); d = A[row]
surv = (Xraw * d).sum(1) / ct; floor = math.sqrt(2 * math.log(A.shape[0]) / c.D)
def run(V, mask):
    sel, _, _ = omp(V, A, 64); s1, _, _ = oneshot(V, A, 64); prom = (V * d).sum(1).abs() / V.norm(dim=1)
    return dict(n=int(mask.sum()), recall_omp=(sel == row[:, None]).any(1)[mask].float().mean().item(), recall_oneshot=(s1 == row[:, None]).any(1)[mask].float().mean().item(), prom_med=prom[mask].median().item(), frac_above_floor=(prom[mask] > floor).float().mean().item())
res = dict(model=tag, L=L, floor=floor, restore={}, cancel={})
erased = typ & (surv < 0.5); intact = typ & (surv > 0.9)
res["restore"]["0"] = run(X, erased)
for a in (0.25, 0.5, 1.0): res["restore"][str(a)] = run(X + (a * ct)[:, None] * d, erased)
res["cancel"]["0"] = run(X, intact)
for b in (0.5, 0.9): res["cancel"][str(b)] = run(X - (b * ct)[:, None] * d, intact)
record(f"e60_intervene_{tag}", res, "RESTORE erased writes (n=%d): " % res["restore"]["0"]["n"] + " ".join(f"+{k}c: omp {v['recall_omp']:.2f} os {v['recall_oneshot']:.2f} prom {v['prom_med']:.2f} above-floor {v['frac_above_floor']:.2f}" for k, v in res["restore"].items()) + " | CANCEL intact writes (n=%d): " % res["cancel"]["0"]["n"] + " ".join(f"-{k}c: omp {v['recall_omp']:.2f} os {v['recall_oneshot']:.2f} prom {v['prom_med']:.2f} above-floor {v['frac_above_floor']:.2f}" for k, v in res["cancel"].items()))
