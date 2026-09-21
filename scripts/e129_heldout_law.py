"""e129: is the covariance-matched competitor level a law or a description? Zero-parameter, held-out versions:
(a) the competitor level (64th-order statistic of covariance-matched random directions) estimated from the
ODD evaluation sequences, predicting identification on the EVEN sequences; (b) estimated from the disjoint
centering slice (train split; we only cached its mean, so we use the odd/even split as the held-out proxy and,
for Pythia, checkpoint transfer: the step-64000 covariance predicting step-143000 identification);
(c) a per-token decision rule 'identified iff prominence > level' scored against actual OMP and dual hits
(accuracy, AUC). Per level, no fitting anywhere."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); N = 4096; ids = sub(c.NT, N); seq = (torch.arange(c.NT)[ids] // CTX).to(DEV); g = torch.Generator(device=DEV).manual_seed(0)
def level64(Xfit, A):
    Sig = (Xfit.T @ Xfit) / Xfit.shape[0]; ev, V = torch.linalg.eigh(Sig); half = V @ torch.diag(ev.clamp_min(0).sqrt()) @ V.T
    z = torch.randn(1024, c.D, generator=g, device=DEV) @ half; z = z / z.norm(dim=1, keepdim=True)
    return torch.cat([(z[s:s + 256] @ A.T).abs().topk(64, dim=1).values[:, -1] for s in range(0, 1024, 256)]).mean().item()
rows = []; cross = None
if tag == "pythia410":
    ce = Cache("pythia410_step64000")
for L in range(c.NB):
    X = c.X(L)[ids]; typ = typical_mask(c.X(L, center=False)[ids]); A, lab = c.dictionary(L)
    tb, tn, tc = c.top_writes(L, 1); row = c.atom_index(L, tb[ids, 0], tn[ids, 0]).to(DEV); prom = (X * A[row]).sum(1).abs() / X.norm(dim=1)
    odd, even = typ & (seq % 2 == 1), typ & (seq % 2 == 0)
    S = A.T @ A; ev_, V = torch.linalg.eigh(S); Winv = V @ torch.diag(1 / (ev_ + 1e-2 * ev_[-1])) @ V.T
    so, _, _ = omp(X, A, 64); sd, _, _ = oneshot(X, A, 64, whiten=Winv); s1, _, _ = oneshot(X, A, 64)
    ho, hd, h1 = (so == row[:, None]).any(1), (sd == row[:, None]).any(1), (s1 == row[:, None]).any(1)
    lvl_odd = level64(X[odd], A); lvl_own = level64(X[even], A)
    r = dict(L=L, level_heldout=lvl_odd, level_own=lvl_own, pred_heldout=(prom[even] > lvl_odd).float().mean().item(), pred_own=(prom[even] > lvl_own).float().mean().item(),
             oneshot=h1[even].float().mean().item(), dual=hd[even].float().mean().item(), omp=ho[even].float().mean().item(),
             rule_acc_oneshot=((prom[even] > lvl_odd) == h1[even]).float().mean().item(), rule_acc_omp=((prom[even] > lvl_odd) == ho[even]).float().mean().item(), rule_acc_dual=((prom[even] > lvl_odd) == hd[even]).float().mean().item())
    if tag == "pythia410":
        Xe = ce.X(L)[ids]; te = typical_mask(ce.X(L, center=False)[ids]); Ae, _ = ce.dictionary(L); lvl_ck = level64(Xe[te], Ae)
        r["level_from_step64000"] = lvl_ck; r["pred_from_step64000"] = (prom[even] > lvl_ck).float().mean().item()
    rows.append(r); log(f"{tag} L{L}: level held-out {lvl_odd:.3f} (own {lvl_own:.3f}" + (f", ckpt {r['level_from_step64000']:.3f}" if 'level_from_step64000' in r else "") + f") | pred {r['pred_heldout']:.3f} vs one-shot {r['oneshot']:.3f} dual {r['dual']:.3f} omp {r['omp']:.3f} | rule acc os {r['rule_acc_oneshot']:.2f} omp {r['rule_acc_omp']:.2f} dual {r['rule_acc_dual']:.2f}")
P = torch.tensor([r["pred_heldout"] for r in rows]); O1 = torch.tensor([r["oneshot"] for r in rows]); Od = torch.tensor([r["dual"] for r in rows]); Oo = torch.tensor([r["omp"] for r in rows])
res = dict(model=tag, rows=rows, heldout_vs_oneshot=dict(corr=torch.corrcoef(torch.stack([P, O1]))[0, 1].item(), mae=(P - O1).abs().mean().item(), bias=(P - O1).mean().item()),
           heldout_vs_dual=dict(corr=torch.corrcoef(torch.stack([P, Od]))[0, 1].item(), mae=(P - Od).abs().mean().item()), heldout_vs_omp=dict(corr=torch.corrcoef(torch.stack([P, Oo]))[0, 1].item(), mae=(P - Oo).abs().mean().item(), bias=(P - Oo).mean().item()),
           rule_acc_mean=dict(oneshot=float(np.mean([r["rule_acc_oneshot"] for r in rows])), omp=float(np.mean([r["rule_acc_omp"] for r in rows])), dual=float(np.mean([r["rule_acc_dual"] for r in rows]))))
if tag == "pythia410":
    Pc = torch.tensor([r["pred_from_step64000"] for r in rows]); res["ckpt_vs_oneshot"] = dict(corr=torch.corrcoef(torch.stack([Pc, O1]))[0, 1].item(), mae=(Pc - O1).abs().mean().item(), bias=(Pc - O1).mean().item())
record(f"e129_heldout_{tag}", res, f"held-out level: vs one-shot corr {res['heldout_vs_oneshot']['corr']:.2f} MAE {res['heldout_vs_oneshot']['mae']:.3f} bias {res['heldout_vs_oneshot']['bias']:+.3f} | vs dual corr {res['heldout_vs_dual']['corr']:.2f} MAE {res['heldout_vs_dual']['mae']:.3f} | vs OMP corr {res['heldout_vs_omp']['corr']:.2f} MAE {res['heldout_vs_omp']['mae']:.3f} bias {res['heldout_vs_omp']['bias']:+.3f} | per-token rule accuracy: one-shot {res['rule_acc_mean']['oneshot']:.2f} dual {res['rule_acc_mean']['dual']:.2f} omp {res['rule_acc_mean']['omp']:.2f}" + (f" | ckpt-64k level vs one-shot corr {res['ckpt_vs_oneshot']['corr']:.2f} MAE {res['ckpt_vs_oneshot']['mae']:.3f} bias {res['ckpt_vs_oneshot']['bias']:+.3f}" if 'ckpt_vs_oneshot' in res else ""))
