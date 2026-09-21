"""e130: what does 'identifiability is a neuron property' mean? For neurons dominant on >= 30 typical tokens:
(a) within-neuron: does token-level prominence predict hit/miss inside a neuron (AUC pooled within neurons)?
(b) between-neuron: variance of per-neuron recall explained by the neuron's mean prominence alone (amplitude
phenotype), by mean prominence + data-free geometry (max coherence, atom norm), and by geometry alone;
(c) matched comparison: pairs of tokens from the same neuron with matched |c| (within 10%): does the one with
the higher prominence get identified more often? (d) the reverse: different neurons, matched prominence bins:
residual between-neuron variance after conditioning on prominence."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); X = c.X(L); Xraw = c.X(L, center=False); typ = typical_mask(Xraw); A, lab = c.dictionary(L)
sel, cof, err = get_omp(c, L, A=A, X=X); tb, tn, tc = c.top_writes(L, 1); row = c.atom_index(L, tb[:, 0], tn[:, 0]).to(DEV); ct = tc[:, 0].to(DEV); hit = (sel == row[:, None]).any(1)
d = A[row]; prom = (X * d).sum(1).abs() / X.norm(dim=1); key = row[typ]; uk, inv, cnt = key.unique(return_inverse=True, return_counts=True); m = cnt >= 30
agg = lambda v: torch.zeros(len(uk), device=DEV).index_add_(0, inv, v[typ].float()) / cnt
rec, pm, cm = agg(hit), agg(prom), agg(ct.abs())
G = torch.zeros(len(uk), device=DEV)
for s in range(0, len(uk), 1024):
    u = uk[s:s + 1024]; Gm = (A[u] @ A.T).abs(); Gm[torch.arange(len(u)), u] = 0; G[s:s + 1024] = Gm.max(1).values
wn = torch.stack([c.d["WN"][b] for b in range(L + 1)]).to(DEV)[lab["block"].to(DEV)[uk], lab["index"].to(DEV)[uk]]
# (a) within-neuron AUC of prominence, pooled over neurons with both hits and misses
h, p, k = hit[typ], prom[typ], inv; num = den = 0.0
for i in torch.nonzero(m)[:, 0].tolist():
    mm = k == i; hh, pp = h[mm], p[mm]
    if hh.any() and (~hh).any(): pos, neg = pp[hh], pp[~hh]; num += (pos[:, None] > neg[None, :]).float().sum().item(); den += len(pos) * len(neg)
within_auc = num / max(den, 1)
# (b) between-neuron R2 via least squares on standardized predictors
def r2(cols):
    Z = torch.stack([torch.ones(int(m.sum()), device=DEV)] + [((v - v[m].mean()) / v[m].std())[m] for v in cols], 1); y = rec[m]
    beta = torch.linalg.lstsq(Z, y[:, None]).solution[:, 0]; return (1 - ((y - Z @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()).item()
res = dict(model=tag, L=L, n_neurons=int(m.sum()), within_neuron_auc_prominence=within_auc, between_r2=dict(mean_prominence=r2([pm]), mean_abs_coef=r2([cm]), coherence=r2([G]), write_norm=r2([wn]), prominence_plus_geometry=r2([pm, G, wn]), geometry_only=r2([G, wn])))
# (c) matched pairs within a neuron: |c| within 10%, the higher-prominence token identified more often?
wins = ties = tot = 0
for i in torch.nonzero(m)[:, 0].tolist():
    mm = torch.nonzero(k == i)[:, 0][:200]; cc = ct.abs()[typ][mm]; pp = p[mm]; hh = h[mm]
    close = (cc[:, None] / cc[None, :] - 1).abs() < 0.1; iu = torch.triu_indices(len(mm), len(mm), 1, device=DEV); sel_ = close[iu[0], iu[1]]
    a, b = iu[0][sel_], iu[1][sel_]; hi = torch.where(pp[a] > pp[b], a, b); lo = torch.where(pp[a] > pp[b], b, a); diff = hh[hi].float() - hh[lo].float()
    wins += (diff > 0).sum().item(); ties += (diff == 0).sum().item(); tot += len(diff)
res["matched_pairs_same_neuron"] = dict(n_pairs=tot, higher_prominence_wins=wins / max(tot, 1), ties=ties / max(tot, 1), loses=(tot - wins - ties) / max(tot, 1))
# (d) between-neuron variance after conditioning on prominence bins (token level): ICC of residual hit
edges = prom[typ].quantile(torch.linspace(0, 1, 11, device=DEV)); binid = torch.bucketize(p, edges[1:-1]); bin_mean = torch.zeros(10, device=DEV).index_add_(0, binid, h.float()) / torch.bincount(binid, minlength=10).clamp_min(1)
resid = h.float() - bin_mean[binid]; rn = torch.zeros(len(uk), device=DEV).index_add_(0, k, resid) / cnt
res["residual_between_neuron_sd_after_prominence"] = rn[m].std().item(); res["between_neuron_sd_raw"] = rec[m].std().item()
record(f"e130_phenotype_{tag}", res, f"n {res['n_neurons']} | within-neuron AUC(prominence) {within_auc:.2f} | between-neuron R2: mean prominence {res['between_r2']['mean_prominence']:.2f} |c| {res['between_r2']['mean_abs_coef']:.2f} coherence {res['between_r2']['coherence']:.2f} norm {res['between_r2']['write_norm']:.2f} prom+geom {res['between_r2']['prominence_plus_geometry']:.2f} geom only {res['between_r2']['geometry_only']:.2f} | same-neuron matched-|c| pairs: higher prominence wins {res['matched_pairs_same_neuron']['higher_prominence_wins']:.2f} ties {res['matched_pairs_same_neuron']['ties']:.2f} | between-neuron sd raw {res['between_neuron_sd_raw']:.2f} -> after prominence {res['residual_between_neuron_sd_after_prominence']:.2f}")
