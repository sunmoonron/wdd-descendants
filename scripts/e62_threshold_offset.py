"""e62: the threshold offset. At the mid layer, compare candidate thresholds for the identification step against
the measured one-shot(k=64) and OMP recalls: (a) the Gaussian floor sqrt(2 ln m/d); (b) the measured floor for
random unit vectors against THIS dictionary; (c) the 64th-largest competitor correlation of real states (what a
k=64 support must beat); (d) the largest competitor. Reports P(prominence > t) for each, and the multiplier
c* of the Gaussian floor at which P(prom > c* floor) equals the measured one-shot and OMP recalls."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); X = c.X(L); typ = typical_mask(c.X(L, center=False)); A, lab = c.dictionary(L)
sel, cof, err = get_omp(c, L, A=A, X=X); tb, tn, tc = c.top_writes(L, 1); row = c.atom_index(L, tb[:, 0], tn[:, 0]).to(DEV)
hit = (sel == row[:, None]).any(1); s1, _, _ = oneshot(X, A, 64); hit1 = (s1 == row[:, None]).any(1)
prom = (X * A[row]).sum(1).abs() / X.norm(dim=1); m, d = A.shape; gauss = math.sqrt(2 * math.log(m) / d)
z = torch.randn(1024, d, device=DEV); z = z / z.norm(dim=1, keepdim=True); zf = (z @ A.T).abs()
floor_rand_max = zf.max(1).values.mean().item(); floor_rand_64 = zf.topk(64, dim=1).values[:, -1].mean().item()
ids = sub(c.NT, 4096).to(DEV); comp_max = torch.zeros(4096, device=DEV); comp_64 = torch.zeros(4096, device=DEV)
for s in range(0, 4096, 512):
    ii = ids[s:s + 512]; G = (X[ii] / X[ii].norm(dim=1, keepdim=True)) @ A.T; G[torch.arange(len(ii)), row[ii]] = 0; G = G.abs()
    comp_max[s:s + 512] = G.max(1).values; comp_64[s:s + 512] = G.topk(64, dim=1).values[:, -1]
t_real_max, t_real_64 = comp_max.median().item(), comp_64.median().item()
r_os, r_omp = hit1[typ].float().mean().item(), hit[typ].float().mean().item()
P = lambda t: (prom[typ] > t).float().mean().item()
def cstar(target):
    lo, hi = 0.2, 3.0
    for _ in range(40):
        mid_ = (lo + hi) / 2
        if P(mid_ * gauss) > target: lo = mid_
        else: hi = mid_
    return (lo + hi) / 2
res = dict(model=tag, L=L, m=m, d=d, recall_oneshot=r_os, recall_omp=r_omp, thresholds=dict(gauss=gauss, random_max=floor_rand_max, random_64th=floor_rand_64, real_competitor_max=t_real_max, real_competitor_64th=t_real_64),
           predicted=dict(gauss=P(gauss), random_max=P(floor_rand_max), random_64th=P(floor_rand_64), real_max=P(t_real_max), real_64th=P(t_real_64)), cstar_oneshot=cstar(r_os), cstar_omp=cstar(r_omp))
record(f"e62_offset_{tag}", res, f"recall os {r_os:.3f} omp {r_omp:.3f} | thresholds gauss {gauss:.3f} rand-max {floor_rand_max:.3f} rand-64th {floor_rand_64:.3f} real-max {t_real_max:.3f} real-64th {t_real_64:.3f} | P(prom>t): gauss {res['predicted']['gauss']:.3f} rand-max {res['predicted']['random_max']:.3f} rand-64th {res['predicted']['random_64th']:.3f} real-max {res['predicted']['real_max']:.3f} real-64th {res['predicted']['real_64th']:.3f} | c* (multiple of gauss floor matching recall): one-shot {res['cstar_oneshot']:.2f} omp {res['cstar_omp']:.2f}")
