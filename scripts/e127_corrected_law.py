"""e127 (closes the audit's framing): per level, the covariance-matched 64th-order competitor level as the
threshold (random directions drawn with the typical-state covariance of THAT level), predicted one-shot@64
identification = P(prominence > that level), against e01's measured one-shot. Bias should vanish if the
competitor level is the state covariance's doing."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); N = 4096; ids = sub(c.NT, N)
try: e01 = {r["L"]: r for r in json.load(open(os.path.join(RESULTS, f"e01_layers_{tag}.json")))["rows"]}
except Exception: e01 = {}
rows = []; g = torch.Generator(device=DEV).manual_seed(0)
for L in range(c.NB):
    X = c.X(L)[ids]; typ = typical_mask(c.X(L, center=False)[ids]); A, lab = c.dictionary(L); Xt = X[typ]
    tb, tn, tc = c.top_writes(L, 1); row = c.atom_index(L, tb[ids, 0], tn[ids, 0]).to(DEV); prom = (X * A[row]).sum(1).abs() / X.norm(dim=1)
    Sig = (Xt.T @ Xt) / Xt.shape[0]; ev, V = torch.linalg.eigh(Sig); half = V @ torch.diag(ev.clamp_min(0).sqrt()) @ V.T
    z = torch.randn(1024, c.D, generator=g, device=DEV) @ half; z = z / z.norm(dim=1, keepdim=True)
    k64 = torch.cat([(z[s:s + 256] @ A.T).abs().topk(64, dim=1).values[:, -1] for s in range(0, 1024, 256)]).mean().item(); iso = math.sqrt(2 * math.log(A.shape[0]) / c.D)
    if L in e01: os_ = e01[L]["oneshot64"]
    else: s1, _, _ = oneshot(X, A, 64); os_ = (s1 == row[:, None]).any(1)[typ].float().mean().item()
    rows.append(dict(L=L, cov64=k64, iso_floor=iso, pred_cov64=(prom[typ] > k64).float().mean().item(), pred_iso=(prom[typ] > iso).float().mean().item(), oneshot=os_))
    log(f"{tag} L{L}: cov-64th {k64:.3f} iso {iso:.3f} | pred cov {rows[-1]['pred_cov64']:.3f} pred iso {rows[-1]['pred_iso']:.3f} | one-shot {os_:.3f}")
P = torch.tensor([r["pred_cov64"] for r in rows]); Pi = torch.tensor([r["pred_iso"] for r in rows]); O = torch.tensor([r["oneshot"] for r in rows])
res = dict(model=tag, rows=rows, cov64=dict(corr=torch.corrcoef(torch.stack([P, O]))[0, 1].item(), mae=(P - O).abs().mean().item(), bias=(P - O).mean().item()), iso=dict(corr=torch.corrcoef(torch.stack([Pi, O]))[0, 1].item(), mae=(Pi - O).abs().mean().item(), bias=(Pi - O).mean().item()))
record(f"e127_corrlaw_{tag}", res, f"covariance-matched 64th-order threshold: corr {res['cov64']['corr']:.2f} MAE {res['cov64']['mae']:.3f} bias {res['cov64']['bias']:+.3f} | isotropic k=1 floor: corr {res['iso']['corr']:.2f} MAE {res['iso']['mae']:.3f} bias {res['iso']['bias']:+.3f}")
