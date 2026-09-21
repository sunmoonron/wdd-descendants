"""e119 (audit follow-up): what is the 'most prominent true write'? Its ledger rank and |c|/|c_max| distribution,
and a non-circular target: the largest-|c| true write among those with raw survival > 0.75 (uses the ledger and
raw projections, not the solver's centered correlation ranking). Recall of that target under OMP / dual / LASSO."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); K = 64; N = int(os.environ.get("WDD_N", 4096)); ids = sub(c.NT, N)
Xraw = c.X(L, center=False)[ids]; X = c.X(L)[ids]; typ = typical_mask(Xraw); A, lab = c.dictionary(L)
led = c.ledger(L); C = torch.cat([led[b][ids] for b in range(L + 1)], 1).to(DEV); mlp_rows = torch.cat([c.atom_index(L, torch.full((c.DFF,), b), torch.arange(c.DFF)) for b in range(L + 1)]).to(DEV); Am = A[mlp_rows]
big = C.abs() >= 0.05 * C.abs().max(1, keepdim=True).values; proj_c = X @ Am.T; proj_r = Xraw @ Am.T
t_prom = proj_c.abs().masked_fill(~big, -1).argmax(1)
surv = proj_r / C.where(C != 0, torch.ones_like(C)); intact = big & (surv > 0.75); has = intact.any(1)
t_int = C.abs().masked_fill(~intact, -1).argmax(1)
rank = (C.abs() > C.abs().gather(1, t_prom[:, None])).sum(1); relmag = C.abs().gather(1, t_prom[:, None])[:, 0] / C.abs().max(1).values
S = A.T @ A; ev, V = torch.linalg.eigh(S); Winv = V @ torch.diag(1 / (ev + 1e-2 * ev[-1])) @ V.T
so, _, _ = omp(X, A, K); sd, _, _ = oneshot(X, A, K, whiten=Winv)
lam = 0.3
try: lam = json.load(open(os.path.join(RESULTS, f"e04_solvers_{tag}.json")))["solvers"]["lasso"]["lambda"]
except Exception: pass
Cl = fista_l1(X, A, lam, iters=300, L=ev[-1].item()).float().to(DEV); sl = Cl.abs().topk(K, dim=1).indices; del Cl
def rec(t, m): return {nm: (s == mlp_rows[t][:, None]).any(1)[m].float().mean().item() for nm, s in (("omp", so), ("dual", sd), ("lasso", sl))}
res = dict(model=tag, L=L, most_prominent=dict(rank_median=rank[typ].float().median().item(), rank_q90=rank[typ].float().quantile(0.9).item(), frac_rank0=(rank[typ] == 0).float().mean().item(), frac_rank_lt3=(rank[typ] < 3).float().mean().item(), relmag_median=relmag[typ].median().item(), recall=rec(t_prom, typ)),
           largest_intact=dict(frac_tokens_with_intact_write=has[typ].float().mean().item(), recall=rec(t_int, typ & has), relmag_median=(C.abs().gather(1, t_int[:, None])[:, 0] / C.abs().max(1).values)[typ & has].median().item(), frac_rank0=((C.abs().gather(1, t_int[:, None])[:, 0] == C.abs().max(1).values))[typ & has].float().mean().item()))
record(f"e119_target_{tag}", res, f"most-prominent true write: ledger rank median {res['most_prominent']['rank_median']:.0f} q90 {res['most_prominent']['rank_q90']:.0f}, is the largest {res['most_prominent']['frac_rank0']:.2f}, top-3 {res['most_prominent']['frac_rank_lt3']:.2f}, |c|/|c_max| median {res['most_prominent']['relmag_median']:.2f}, recall omp {res['most_prominent']['recall']['omp']:.2f} dual {res['most_prominent']['recall']['dual']:.2f} lasso {res['most_prominent']['recall']['lasso']:.2f} | largest INTACT write (raw survival > 0.75; {res['largest_intact']['frac_tokens_with_intact_write']:.2f} of tokens have one): is the largest {res['largest_intact']['frac_rank0']:.2f}, |c|/|c_max| {res['largest_intact']['relmag_median']:.2f}, recall omp {res['largest_intact']['recall']['omp']:.2f} dual {res['largest_intact']['recall']['dual']:.2f} lasso {res['largest_intact']['recall']['lasso']:.2f}")
