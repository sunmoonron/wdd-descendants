"""e57: the identification threshold vs the dictionary's noise floor. Fine-binned recall vs prominence
(one-shot and OMP, k=64), the prominence at which recall crosses 50%, compared with (i) sqrt(2 ln m / d) (Gaussian
extreme-value floor for m random atoms) and (ii) the measured floor E max_i |<z, a_i>| for random unit z over THIS
dictionary (data-free). If the crossing matches the floor, identification is generic sparse recovery: a write is
readable iff it stands above the largest spurious correlation the dictionary can produce."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); X = c.X(L); typ = typical_mask(c.X(L, center=False)); A, lab = c.dictionary(L)
sel, cof, err = get_omp(c, L, A=A, X=X); tb, tn, tc = c.top_writes(L, 1); row = c.atom_index(L, tb[:, 0], tn[:, 0]).to(DEV)
hit = (sel == row[:, None]).any(1); s1, _, _ = oneshot(X, A, 64); hit1 = (s1 == row[:, None]).any(1); s16, _, _ = oneshot(X, A, 16); hit16 = (s16 == row[:, None]).any(1)
prom = (X * A[row]).sum(1).abs() / X.norm(dim=1)
def crossing(h):
    edges = torch.arange(0, 0.6, 0.02, device=DEV); xs, ys = [], []
    for lo in edges:
        m = typ & (prom >= lo) & (prom < lo + 0.02)
        if m.sum() >= 30: xs.append(lo.item() + 0.01); ys.append(h[m].float().mean().item())
    for i in range(1, len(xs)):
        if ys[i - 1] < 0.5 <= ys[i]: return xs[i - 1] + (0.5 - ys[i - 1]) / (ys[i] - ys[i - 1]) * (xs[i] - xs[i - 1]), list(zip(xs, ys))
    return None, list(zip(xs, ys))
c50_os, curve_os = crossing(hit1); c50_omp, curve_omp = crossing(hit); c50_16, _ = crossing(hit16)
m, d = A.shape; gauss = math.sqrt(2 * math.log(m) / d)
z = torch.randn(2048, d, device=DEV); z = z / z.norm(dim=1, keepdim=True); floor = torch.cat([(z[i:i + 512] @ A.T).abs().max(1).values for i in range(0, 2048, 512)])
# the same floor measured with typical-state directions: the max spurious correlation among atoms OTHER than the true one
sc = torch.zeros(4096, device=DEV); ids = sub(c.NT, 4096).to(DEV)
for s in range(0, 4096, 512):
    ii = ids[s:s + 512]; G = (X[ii] / X[ii].norm(dim=1, keepdim=True)) @ A.T; G[torch.arange(len(ii)), row[ii]] = 0; sc[s:s + 512] = G.abs().max(1).values
res = dict(model=tag, L=L, m=m, d=d, gauss_floor=gauss, measured_floor_random_z_mean=floor.mean().item(), measured_floor_random_z_q90=floor.quantile(0.9).item(),
           max_competitor_corr_real_states_median=sc.median().item(), crossing50=dict(oneshot64=c50_os, oneshot16=c50_16, omp64=c50_omp), prom_median=prom[typ].median().item(),
           frac_above_gauss=(prom[typ] > gauss).float().mean().item(), recall_os=hit1[typ].float().mean().item(), recall_omp=hit[typ].float().mean().item(), curve_oneshot=curve_os, curve_omp=curve_omp)
record(f"e57_threshold_{tag}", res, f"m {m} d {d} | floor: gauss {gauss:.3f} measured(random z) {res['measured_floor_random_z_mean']:.3f} real-state competitor {res['max_competitor_corr_real_states_median']:.3f} | 50% crossing: one-shot64 {c50_os} one-shot16 {c50_16} omp64 {c50_omp} | prom med {res['prom_median']:.3f} frac above floor {res['frac_above_gauss']:.3f} vs recall os {res['recall_os']:.3f} omp {res['recall_omp']:.3f}")
