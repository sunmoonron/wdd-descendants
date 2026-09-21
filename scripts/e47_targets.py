"""e47: the target problem. Identification has been scored against the ledger's largest write, but the largest
write is often a cancelled one. Alternative ground-truth targets per token: (a) ledger top-1 |c|; (b) net-present
write: largest |c| * raw survival (what remains of it along its own direction); (c) most prominent true write:
largest |x_c . d| among the true MLP writes with |c| >= 5% of the largest. Recall of each under OMP, dual one-shot
and LASSO. Also: how often (a), (b), (c) coincide."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); K = 64; N = int(os.environ.get("WDD_N", 4096)); ids = sub(c.NT, N)
Xraw = c.X(L, center=False)[ids]; X = c.X(L)[ids]; typ = typical_mask(Xraw); A, lab = c.dictionary(L)
led = c.ledger(L); C = torch.cat([led[b][ids] for b in range(L + 1)], 1).to(DEV)                    # [N, (L+1)*DFF]
mlp_rows = torch.cat([c.atom_index(L, torch.full((c.DFF,), b), torch.arange(c.DFF)) for b in range(L + 1)]).to(DEV)   # row of each (b, j)
Am = A[mlp_rows]                                                                                    # [(L+1)*DFF, D] unit dirs
proj_c = X @ Am.T; proj_r = Xraw @ Am.T                                                             # [N, M]
big = C.abs() >= 0.05 * C.abs().max(1, keepdim=True).values
t_a = C.abs().argmax(1)
net = (C.abs() * (proj_r / C).clamp(min=0)).masked_fill(~big, -1); t_b = net.argmax(1)
prom = proj_c.abs().masked_fill(~big, -1); t_c = prom.argmax(1)
targets = dict(ledger_top1=mlp_rows[t_a], net_present=mlp_rows[t_b], most_prominent=mlp_rows[t_c])
S = A.T @ A; ev, V = torch.linalg.eigh(S); Winv = V @ torch.diag(1 / (ev + 1e-2 * ev[-1])) @ V.T
sel_omp, _, _ = omp(X, A, K); sel_dual, _, _ = oneshot(X, A, K, whiten=Winv)
lam = 0.3
try: lam = json.load(open(os.path.join(RESULTS, f"e04_solvers_{tag}.json")))["solvers"]["lasso"]["lambda"]
except Exception: pass
Cl = fista_l1(X, A, lam, iters=300, L=ev[-1].item()).float().to(DEV); sel_lasso = Cl.abs().topk(K, dim=1).indices; del Cl
res = dict(model=tag, L=L, N=N, recall={}, coincide=dict(a_eq_b=(t_a == t_b)[typ].float().mean().item(), a_eq_c=(t_a == t_c)[typ].float().mean().item(), b_eq_c=(t_b == t_c)[typ].float().mean().item()))
for tn_, rows in targets.items():
    res["recall"][tn_] = {sn: (s == rows[:, None]).any(1)[typ].float().mean().item() for sn, s in (("omp", sel_omp), ("dual", sel_dual), ("lasso", sel_lasso))}
# how cancelled is the ledger top-1 vs the most prominent: raw survival medians
sv = lambda t: (proj_r.gather(1, t[:, None])[:, 0] / C.gather(1, t[:, None])[:, 0])[typ].median().item()
res["raw_survival_med"] = dict(ledger_top1=sv(t_a), most_prominent=sv(t_c))
res["prominence_share"] = dict(ledger_top1=(proj_c.gather(1, t_a[:, None])[:, 0].abs() / X.norm(dim=1))[typ].median().item(), most_prominent=(proj_c.gather(1, t_c[:, None])[:, 0].abs() / X.norm(dim=1))[typ].median().item())
record(f"e47_targets_{tag}", res, " | ".join(f"{k}: omp {v['omp']:.3f} dual {v['dual']:.3f} lasso {v['lasso']:.3f}" for k, v in res["recall"].items()) + f" | coincide a=b {res['coincide']['a_eq_b']:.2f} a=c {res['coincide']['a_eq_c']:.2f} | raw surv top1 {res['raw_survival_med']['ledger_top1']:.2f} prominent {res['raw_survival_med']['most_prominent']:.2f} | prominence top1 {res['prominence_share']['ledger_top1']:.2f} prominent {res['prominence_share']['most_prominent']:.2f}")
