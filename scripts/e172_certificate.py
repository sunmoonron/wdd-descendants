"""e172: the observability certificate table (GPT-2 and SmolLM2). Per neuron dominant on >= 20 typical tokens at
the mid layer: state-level recall (OMP, dual), increment-level recall (OMP@16), mean prominence, mean raw
survival, dominance count, max coherence (data-free), write norm, mean |c|, birth block. Written as CSV."""
import sys, os, csv; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); X = c.X(L); Xraw = c.X(L, center=False); typ = typical_mask(Xraw); A, lab = c.dictionary(L)
sel, cof, err = get_omp(c, L, A=A, X=X); tb, tn, tc = c.top_writes(L, 1); tb, tn, ct = tb[:, 0], tn[:, 0], tc[:, 0].to(DEV); row = c.atom_index(L, tb, tn).to(DEV); d = A[row]
S = A.T @ A; ev, V = torch.linalg.eigh(S); Winv = V @ torch.diag(1 / (ev + 1e-2 * ev[-1])) @ V.T; sd, _, _ = oneshot(X, A, 64, whiten=Winv)
hit_o, hit_d = (sel == row[:, None]).any(1), (sd == row[:, None]).any(1); prom = (X * d).sum(1).abs() / X.norm(dim=1); surv = (Xraw * d).sum(1) / ct
hit_i = torch.zeros(c.NT, dtype=torch.bool, device=DEV)
for b in range(L + 1):
    idx = torch.nonzero(tb == b)[:, 0]
    if len(idx) == 0: continue
    D_ = c.s["H"][b + 1][idx].float().to(DEV) - c.s["H"][b][idx].float().to(DEV); D_ = D_ - D_.mean(0); Ab, labb = c.dictionary(b, blocks=[b], types=(T_MLP, T_ATT, T_BIAS)); mlp_rows = torch.nonzero(labb["type"] == T_MLP)[:, 0].to(DEV)
    so, _, _ = omp(D_, Ab, 16); hit_i[idx.to(DEV)] = (so == mlp_rows[tn[idx].to(DEV)][:, None]).any(1)
key = (tb * c.DFF + tn).to(DEV)[typ]; uk, inv, cnt = key.unique(return_inverse=True, return_counts=True); m = cnt >= 20
agg = lambda v: (torch.zeros(len(uk), device=DEV).index_add_(0, inv, v[typ].float()) / cnt)
coh = torch.zeros(len(uk), device=DEV); rows_u = c.atom_index(L, uk.cpu() // c.DFF, uk.cpu() % c.DFF).to(DEV)
for s in range(0, len(uk), 1024):
    u = rows_u[s:s + 1024]; G = (A[u] @ A.T).abs(); G[torch.arange(len(u)), u] = 0; coh[s:s + 1024] = G.max(1).values
WN = torch.stack([c.d["WN"][b] for b in range(L + 1)]).to(DEV)
table = dict(block=(uk // c.DFF), neuron=(uk % c.DFF), n_dominant=cnt, recall_omp=agg(hit_o), recall_dual=agg(hit_d), recall_increment=agg(hit_i), mean_prominence=agg(prom), mean_raw_survival=agg(surv.clamp(-2, 3)), mean_abs_coef=agg(ct.abs()), max_coherence=coh, write_norm=WN[uk // c.DFF, uk % c.DFF])
path = os.path.join(RESULTS, f"e172_certificate_{tag}.csv")
with open(path, "w") as f:
    w = csv.writer(f); keys = list(table); w.writerow(keys)
    for i in torch.nonzero(m)[:, 0].tolist(): w.writerow([round(float(table[k][i]), 4) if k not in ("block", "neuron", "n_dominant") else int(table[k][i]) for k in keys])
record(f"e172_certificate_{tag}", dict(model=tag, L=L, n=int(m.sum()), path=path), f"certificate table with {int(m.sum())} neurons written to {path}")
