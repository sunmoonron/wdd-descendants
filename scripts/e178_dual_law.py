"""e178: the law for the best decoder. The dual-frame projection scores atoms by (S^-1 x) . a; define the dual
prominence of a write as |(S^-1 x_c) . d| / ||S^-1 x_c|| and the dual competitor level as the 64th-order statistic
of |(S^-1 z) . a| for covariance-matched random z. Predict dual@64 identification per level with zero parameters;
compare with the raw-prominence prediction of the dual reading."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); N = 4096; ids = sub(c.NT, N); g = torch.Generator(device=DEV).manual_seed(0); rows = []
for L in range(c.NB):
    X = c.X(L)[ids]; typ = typical_mask(c.X(L, center=False)[ids]); A, lab = c.dictionary(L); tb, tn, tc = c.top_writes(L, 1); row = c.atom_index(L, tb[ids, 0], tn[ids, 0]).to(DEV)
    S = A.T @ A; ev, V = torch.linalg.eigh(S); Winv = V @ torch.diag(1 / (ev + 1e-2 * ev[-1])) @ V.T; Xw = X @ Winv.T
    sd, _, _ = oneshot(X, A, 64, whiten=Winv); hit = (sd == row[:, None]).any(1)
    prom_raw = (X * A[row]).sum(1).abs() / X.norm(dim=1); prom_dual = (Xw * A[row]).sum(1).abs() / Xw.norm(dim=1)
    Xt = X[typ]; Sig = (Xt.T @ Xt) / Xt.shape[0]; ev2, V2 = torch.linalg.eigh(Sig); half = V2 @ torch.diag(ev2.clamp_min(0).sqrt()) @ V2.T; z = torch.randn(1024, c.D, generator=g, device=DEV) @ half
    zw = z @ Winv.T; zw = zw / zw.norm(dim=1, keepdim=True); z = z / z.norm(dim=1, keepdim=True)
    lvl_dual = torch.cat([(zw[s:s + 256] @ A.T).abs().topk(64, dim=1).values[:, -1] for s in range(0, 1024, 256)]).mean().item(); lvl_raw = torch.cat([(z[s:s + 256] @ A.T).abs().topk(64, dim=1).values[:, -1] for s in range(0, 1024, 256)]).mean().item()
    rows.append(dict(L=L, dual_recall=hit[typ].float().mean().item(), pred_dual=(prom_dual[typ] > lvl_dual).float().mean().item(), pred_raw=(prom_raw[typ] > lvl_raw).float().mean().item(), rule_acc=((prom_dual > lvl_dual) == hit)[typ].float().mean().item()))
    log(f"{tag} L{L}: dual recall {rows[-1]['dual_recall']:.3f} | predicted from dual prominence {rows[-1]['pred_dual']:.3f} (rule accuracy {rows[-1]['rule_acc']:.2f}) | from raw prominence {rows[-1]['pred_raw']:.3f}")
P, Pr, O = (torch.tensor([r[k] for r in rows]) for k in ("pred_dual", "pred_raw", "dual_recall"))
res = dict(model=tag, rows=rows, dual_law=dict(corr=torch.corrcoef(torch.stack([P, O]))[0, 1].item(), mae=(P - O).abs().mean().item(), bias=(P - O).mean().item()), raw_law_for_dual=dict(corr=torch.corrcoef(torch.stack([Pr, O]))[0, 1].item(), mae=(Pr - O).abs().mean().item(), bias=(Pr - O).mean().item()), rule_acc_mean=float(np.mean([r["rule_acc"] for r in rows])))
record(f"e178_duallaw_{tag}", res, f"dual reading predicted from dual prominence: corr {res['dual_law']['corr']:.2f} MAE {res['dual_law']['mae']:.3f} bias {res['dual_law']['bias']:+.3f} (per-token rule accuracy {res['rule_acc_mean']:.2f}) | from raw prominence: corr {res['raw_law_for_dual']['corr']:.2f} MAE {res['raw_law_for_dual']['mae']:.3f} bias {res['raw_law_for_dual']['bias']:+.3f}")
