"""e58: out-of-sample test of the prominence law. For every level of a model (and for the Pythia checkpoints),
predict the one-shot identification rate with NO fitting as the fraction of dominant writes whose prominence
|x_c . d| / ||x_c|| exceeds the dictionary noise floor sqrt(2 ln m_L / d), and compare with the measured
one-shot recall of e01 (or a fresh one-shot run). Also the OMP prediction. Reports per-level pairs, the
correlation across levels and the mean absolute error of the prediction."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); N = int(os.environ.get("WDD_N", 4096)); ids = sub(c.NT, N)
try: e01 = {r["L"]: r for r in json.load(open(os.path.join(RESULTS, f"e01_layers_{tag}.json")))["rows"]}
except Exception: e01 = {}
rows = []
for L in range(c.NB):
    X = c.X(L)[ids]; typ = typical_mask(c.X(L, center=False)[ids]); A, lab = c.dictionary(L)
    tb, tn, tc = c.top_writes(L, 1); row = c.atom_index(L, tb[ids, 0], tn[ids, 0]).to(DEV); d = A[row]
    prom = (X * d).sum(1).abs() / X.norm(dim=1); floor = math.sqrt(2 * math.log(A.shape[0]) / c.D)
    pred = (prom[typ] > floor).float().mean().item()
    if L in e01: os_, omp_ = e01[L]["oneshot64"], e01[L]["recall"]["64"]
    else:
        s1, _, _ = oneshot(X, A, 64); os_ = (s1 == row[:, None]).any(1)[typ].float().mean().item(); omp_ = None
    rows.append(dict(L=L, floor=floor, pred=pred, oneshot=os_, omp=omp_, prom_med=prom[typ].median().item()))
    log(f"{tag} L{L}: floor {floor:.3f} predicted {pred:.3f} one-shot {os_:.3f} omp {omp_}")
P = torch.tensor([r["pred"] for r in rows]); O = torch.tensor([r["oneshot"] for r in rows])
res = dict(model=tag, rows=rows, corr_pred_oneshot=torch.corrcoef(torch.stack([P, O]))[0, 1].item(), mae_pred_oneshot=(P - O).abs().mean().item(), bias=(P - O).mean().item())
if all(r["omp"] is not None for r in rows):
    Om = torch.tensor([r["omp"] for r in rows]); res["corr_pred_omp"] = torch.corrcoef(torch.stack([P, Om]))[0, 1].item(); res["mae_pred_omp"] = (P - Om).abs().mean().item()
record(f"e58_lawtest_{tag}", res, f"levels {len(rows)} | corr(pred, one-shot) {res['corr_pred_oneshot']:.2f} MAE {res['mae_pred_oneshot']:.3f} bias {res['bias']:+.3f}" + (f" | corr(pred, omp) {res['corr_pred_omp']:.2f} MAE {res['mae_pred_omp']:.3f}" if 'corr_pred_omp' in res else "") + " | " + " ".join(f"L{r['L']}:{r['pred']:.2f}/{r['oneshot']:.2f}" for r in rows))
