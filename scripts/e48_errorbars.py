"""e48: error bars from the support alone (Gauss / Cramer-Rao 1945). For an identified dominant write, the
least-squares theory predicts its coefficient's standard error as ||r|| / sqrt(d - k) * sqrt((G_S^-1)_jj), with G_S
the Gram of the selected atoms and r the residual. Does this predict the actual error |c_hat - c|? Rank
correlation, AUC for 'error above median', and coverage of 1- and 2-sigma intervals."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); X = c.X(L); typ = typical_mask(c.X(L, center=False)); A, lab = c.dictionary(L)
sel, cof, err = get_omp(c, L, A=A, X=X); tb, tn, tc = c.top_writes(L, 1); row = c.atom_index(L, tb[:, 0], tn[:, 0]).to(DEV); ct = tc[:, 0].to(DEV)
hit = sel == row[:, None]; idn = hit.any(1) & typ; ii = torch.nonzero(idn)[:, 0]
chat = (cof * hit).sum(1)[ii]; slot = hit[ii].float().argmax(1); k = sel.shape[1]
se = torch.zeros(len(ii), device=DEV); lev = torch.zeros(len(ii), device=DEV)
for s in range(0, len(ii), 1024):
    j = ii[s:s + 1024]; As = A[sel[j]]; G = As @ As.transpose(1, 2) + 1e-5 * torch.eye(k, device=DEV); Gi = torch.linalg.inv(G)
    d = torch.diagonal(Gi, dim1=1, dim2=2); g = d.gather(1, slot[s:s + 1024][:, None])[:, 0]
    sig = (err[j, k - 1] / (c.D - k)).sqrt(); se[s:s + 1024] = sig * g.sqrt(); lev[s:s + 1024] = g
ae = (chat - ct[ii]).abs(); z = (chat - ct[ii]) / se
def spearman(a, b):
    ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float(); return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
res = dict(model=tag, L=L, n=len(ii), spearman_se_vs_abs_err=spearman(se, ae), spearman_lev_vs_abs_err=spearman(lev, ae), spearman_se_vs_rel_err=spearman(se / chat.abs(), ae / ct[ii].abs()),
           coverage_1sig=(z.abs() < 1).float().mean().item(), coverage_2sig=(z.abs() < 2).float().mean().item(), z_median=z.median().item(), z_mad=(z - z.median()).abs().median().item(),
           se_median=se.median().item(), abs_err_median=ae.median().item(), bias_ratio_median=(chat / ct[ii]).median().item())
# after removing the known multiplicative bias (median ratio), do the intervals cover?
zc = (chat / res["bias_ratio_median"] - ct[ii]) / (se / res["bias_ratio_median"]); res["coverage_2sig_debiased"] = (zc.abs() < 2).float().mean().item(); res["coverage_1sig_debiased"] = (zc.abs() < 1).float().mean().item()
record(f"e48_errbars_{tag}", res, f"n {len(ii)} | Spearman(se, |err|) {res['spearman_se_vs_abs_err']:.2f} (leverage {res['spearman_lev_vs_abs_err']:.2f}) rel {res['spearman_se_vs_rel_err']:.2f} | coverage 1s {res['coverage_1sig']:.2f} 2s {res['coverage_2sig']:.2f} (debiased {res['coverage_1sig_debiased']:.2f}/{res['coverage_2sig_debiased']:.2f}) | z med {res['z_median']:.2f} mad {res['z_mad']:.2f} | se med {res['se_median']:.2f} vs |err| med {res['abs_err_median']:.2f}")
