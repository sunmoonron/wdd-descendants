"""e52: does the recovered coefficient track the true one WITHIN a neuron? For neurons identified as dominant on
>= 50 typical tokens: Pearson and Spearman of c_hat vs c across those tokens, the within-neuron slope, and the
spread of true coefficients (CV). Separates 'estimation' from 'predicting the neuron's typical magnitude'."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); X = c.X(L); typ = typical_mask(c.X(L, center=False)); A, lab = c.dictionary(L)
sel, cof, err = get_omp(c, L, A=A, X=X); tb, tn, tc = c.top_writes(L, 1); row = c.atom_index(L, tb[:, 0], tn[:, 0]).to(DEV); ct = tc[:, 0].to(DEV)
hit = sel == row[:, None]; idn = hit.any(1) & typ; chat = (cof * hit).sum(1)
key = row[idn]; uk, inv, cnt = key.unique(return_inverse=True, return_counts=True); big = torch.nonzero(cnt >= 50)[:, 0]
rows = []
for i in big.tolist():
    m = inv == i; a = chat[idn][m]; b = ct[idn][m]
    pr = torch.corrcoef(torch.stack([a, b]))[0, 1].item(); ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float(); sp = torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
    slope = (((a - a.mean()) * (b - b.mean())).sum() / ((a - a.mean()) ** 2).sum()).item()
    rows.append(dict(n=int(m.sum()), pearson=pr, spearman=sp, slope_true_on_est=slope, cv_true=(b.std() / b.abs().mean()).item(), cv_est=(a.std() / a.abs().mean()).item(), ratio_med=(a / b).median().item()))
P = torch.tensor([r["pearson"] for r in rows]); S_ = torch.tensor([r["spearman"] for r in rows]); CV = torch.tensor([r["cv_true"] for r in rows])
# pooled: correlation across all identified tokens, and after removing per-neuron means (within-neuron pooled)
a_all, b_all = chat[idn], ct[idn]; mean_a = torch.zeros(len(uk), device=DEV).index_add_(0, inv, a_all) / cnt; mean_b = torch.zeros(len(uk), device=DEV).index_add_(0, inv, b_all) / cnt
wa, wb = a_all - mean_a[inv], b_all - mean_b[inv]
res = dict(model=tag, L=L, n_neurons_ge50=len(rows), within_pearson_median=P.median().item(), within_spearman_median=S_.median().item(), frac_within_pearson_above_0p5=(P > 0.5).float().mean().item(),
           cv_true_median=CV.median().item(), pooled_pearson=torch.corrcoef(torch.stack([a_all, b_all]))[0, 1].item(), pooled_within_pearson=torch.corrcoef(torch.stack([wa, wb]))[0, 1].item(),
           between_neuron_pearson=torch.corrcoef(torch.stack([mean_a, mean_b]))[0, 1].item(), var_share_between=(mean_b[inv].var() / b_all.var()).item())
record(f"e52_within_{tag}", res, f"neurons {len(rows)} | within-neuron Pearson med {res['within_pearson_median']:.2f} Spearman {res['within_spearman_median']:.2f} (>0.5: {res['frac_within_pearson_above_0p5']:.2f}) | CV(true) med {res['cv_true_median']:.2f} | pooled r {res['pooled_pearson']:.2f} within-pooled r {res['pooled_within_pearson']:.2f} between-neuron r {res['between_neuron_pearson']:.2f} | between share of variance {res['var_share_between']:.2f}")
