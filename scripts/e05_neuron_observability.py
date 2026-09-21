"""e05: is identifiability a property of the NEURON or of the TOKEN? Per-neuron recall when it is the dominant
write, against data-free predictors (atom coherence, write norm, block age) and data predictors (survival,
prominence, firing frequency). Also the between-neuron variance of recall vs the binomial expectation."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); K = 64
X = c.X(L); Xraw = c.X(L, center=False); typ = typical_mask(Xraw); A, lab = c.dictionary(L)
sel, cof, err = get_omp(c, L, A=A, X=X)
tb, tn, tc = c.top_writes(L, 1); tb, tn, tc = tb[:, 0], tn[:, 0], tc[:, 0].to(DEV)
row = c.atom_index(L, tb, tn).to(DEV); hit = (sel == row[:, None]).any(1)
d = A[row]; surv = (Xraw * d).sum(1) / tc; prom = tc.abs() / Xraw.norm(dim=1)
# data-free predictors of the true atom: max coherence with any other atom (all types), and with MLP atoms only
uniq, inv = row.unique(return_inverse=True)
mlpmask = (lab["type"].to(DEV) == T_MLP); maxc_all = torch.zeros(len(uniq), device=DEV); maxc_mlp = torch.zeros(len(uniq), device=DEV)
for s_ in range(0, len(uniq), 1024):
    u = uniq[s_:s_ + 1024]; G = (A[u] @ A.T).abs_(); G[torch.arange(len(u)), u] = 0
    maxc_all[s_:s_ + 1024] = G.max(1).values; G.masked_fill_(~mlpmask[None], 0); maxc_mlp[s_:s_ + 1024] = G.max(1).values; del G
coh_all, coh_mlp = maxc_all[inv], maxc_mlp[inv]
wn = torch.stack([c.d["WN"][b] for b in range(L + 1)]).to(DEV)[tb.to(DEV), tn.to(DEV)]
led = c.ledger(L); freq = torch.stack([(led[b].abs() > 0.05 * led[b].abs().max(1, keepdim=True).values).float().mean(0) for b in range(L + 1)]).to(DEV)[tb.to(DEV), tn.to(DEV)]
age = (L - tb).float().to(DEV)
def auc(score, y):
    pos, neg = score[y], score[~y]; i = sub(len(pos), 4000).to(DEV); j = sub(len(neg), 4000).to(DEV)
    return (pos[i][:, None] > neg[j][None, :]).float().mean().item()
y = hit[typ]
res = dict(model=tag, L=L, recall=y.float().mean().item(), auc=dict(
    survival=auc(surv[typ], y), prominence=auc(prom[typ], y), abs_coef=auc(tc.abs()[typ], y), coherence_all=auc(-coh_all[typ], y),
    coherence_mlp=auc(-coh_mlp[typ], y), write_norm=auc(wn[typ], y), firing_freq=auc(-freq[typ], y), age=auc(-age[typ], y)))
# per-neuron table
key = tb.to(DEV) * c.DFF + tn.to(DEV); key_t = key[typ]
uk, cnt = key_t.unique(return_counts=True)
rec_n = torch.zeros(len(uk), device=DEV).index_add_(0, torch.searchsorted(uk, key_t), hit[typ].float()) / cnt
surv_n = torch.zeros(len(uk), device=DEV).index_add_(0, torch.searchsorted(uk, key_t), surv[typ]) / cnt
coh_n = torch.zeros(len(uk), device=DEV).index_add_(0, torch.searchsorted(uk, key_t), coh_all[typ]) / cnt
m = cnt >= 20
p = rec_n[m]; n = cnt[m].float(); pbar = (p * n).sum() / n.sum()
var_between = p.var().item(); var_binom = (pbar * (1 - pbar) / n).mean().item()
res["neuron_level"] = dict(n_neurons_ge20=int(m.sum()), n_neurons_total=len(uk), var_between=var_between, var_binomial=var_binom,
                           icc=(var_between - var_binom) / max(var_between, 1e-9), frac_always=(p > 0.95).float().mean().item(), frac_never=(p < 0.05).float().mean().item())
def spearman(a, b):
    ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float(); return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
res["neuron_level"]["spearman_recall_vs"] = dict(survival=spearman(p, surv_n[m]), coherence=spearman(p, -coh_n[m]), count=spearman(p, n))
order = p.argsort()
def row_(i): k = uk[m][i].item(); return dict(block=k // c.DFF, neuron=k % c.DFF, n=int(n[i]), recall=p[i].item(), survival=surv_n[m][i].item(), coherence=coh_n[m][i].item())
res["least_identifiable"] = [row_(i) for i in order[:8].tolist()]; res["most_identifiable"] = [row_(i) for i in order[-8:].tolist()]
res["top1_concentration"] = dict(most_frequent_share=(cnt.max() / cnt.sum()).item(), n_distinct=len(uk))
record(f"e05_neuron_{tag}", res, f"recall {res['recall']:.3f} | AUC surv {res['auc']['survival']:.2f} prom {res['auc']['prominence']:.2f} coh {res['auc']['coherence_all']:.2f} norm {res['auc']['write_norm']:.2f} freq {res['auc']['firing_freq']:.2f} age {res['auc']['age']:.2f} | neuron ICC {res['neuron_level']['icc']:.2f} always {res['neuron_level']['frac_always']:.2f} never {res['neuron_level']['frac_never']:.2f} rho(surv) {res['neuron_level']['spearman_recall_vs']['survival']:.2f} rho(coh) {res['neuron_level']['spearman_recall_vs']['coherence']:.2f}")
