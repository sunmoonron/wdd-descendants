"""e30: why are some neurons never identified? Take neurons that are (almost) never identified when dominant
and neurons that (almost) always are, inject their atom into random typical states at controlled magnitudes
(multiples of the state norm), and measure recall vs dose. If never-neurons stay unrecoverable at high dose the
failure is geometric (another atom explains them); if they recover, it is about their natural magnitude/survival."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); X = c.X(L); typ = typical_mask(c.X(L, center=False)); A, lab = c.dictionary(L)
sel, cof, err = get_omp(c, L, A=A, X=X); tb, tn, tc = c.top_writes(L, 1); row = c.atom_index(L, tb[:, 0], tn[:, 0]).to(DEV); hit = (sel == row[:, None]).any(1)
key = row[typ]; h = hit[typ].float(); uk, inv, cnt = key.unique(return_inverse=True, return_counts=True)
rec = torch.zeros(len(uk), device=DEV).index_add_(0, inv, h) / cnt; m = cnt >= 20
never = uk[m & (rec < 0.1)]; always = uk[m & (rec > 0.9)]; mid_ = uk[m & (rec > 0.4) & (rec < 0.6)]
g = torch.Generator().manual_seed(0); base = X[typ][sub(int(typ.sum()), 512, seed=1)]; xn = base.norm(dim=1, keepdim=True)
res = dict(model=tag, L=L, n_never=len(never), n_always=len(always), n_mid=len(mid_), doses={})
# also the atoms' own competition: max coherence and how the state already projects on them
def run(group, name, dose):
    if len(group) == 0: return None
    gi = group[torch.randint(0, len(group), (64,), generator=g).to(DEV)]; hits = []; ratios = []
    for a in gi:
        d = A[a]; Y = base + dose * xn * d
        s, cf, e = omp(Y, A, 64); hh = (s == a).any(1); hits.append(hh.float().mean().item())
        est = (cf * (s == a)).sum(1)[hh]; ratios.append((est / (dose * xn[:, 0][hh] + (base @ d)[hh])).median().item() if hh.any() else float("nan"))
    return dict(recall=float(np.mean(hits)), coef_ratio_med=float(np.nanmedian(ratios)))
for dose in (0.1, 0.25, 0.5, 1.0, 2.0):
    res["doses"][dose] = dict(never=run(never, "never", dose), always=run(always, "always", dose), mid=run(mid_, "mid", dose))
    log(f"{tag} dose {dose}: " + " ".join(f"{k}: {v}" for k, v in res["doses"][dose].items()))
# geometry of the two classes
def geo(group):
    if len(group) == 0: return None
    G = (A[group] @ A.T).abs(); G[torch.arange(len(group)), group] = 0
    return dict(maxcoh_med=G.max(1).values.median().item(), n_above_0p5_med=(G > 0.5).sum(1).float().median().item(), n_above_0p3_med=(G > 0.3).sum(1).float().median().item(),
                natural_prominence_med=None)
res["geometry"] = dict(never=geo(never), always=geo(always), mid=geo(mid_))
record(f"e30_inject_{tag}", res, f"never n={len(never)} always n={len(always)} | " + " | ".join(f"dose {d}: never {v['never']['recall'] if v['never'] else None} always {v['always']['recall'] if v['always'] else None}" for d, v in res["doses"].items()) + f" | maxcoh never {res['geometry']['never']['maxcoh_med'] if res['geometry']['never'] else None} always {res['geometry']['always']['maxcoh_med'] if res['geometry']['always'] else None}")
