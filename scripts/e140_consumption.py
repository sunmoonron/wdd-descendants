"""e140: consumption vs broadcast. For each token's dominant MLP write (born in block b*, direction d): its read
strength by the next blocks' MLP input weights, ||R_b d|| / sqrt(DFF) for b = b*+1 .. b*+3 (data-free geometry;
for gated models the gate rows; also the attention QKV input weights of the next block), against (i) raw survival
at level L, (ii) OMP identification, (iii) the neuron's per-neuron recall. Hypothesis: erased/unreadable writes
are the strongly-read (consumed) ones; readable ones are broadcast."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); X = c.X(L); Xraw = c.X(L, center=False); typ = typical_mask(Xraw); A, lab = c.dictionary(L)
sel, cof, err = get_omp(c, L, A=A, X=X); tb, tn, tc = c.top_writes(L, 1); tb, tn, ct = tb[:, 0], tn[:, 0], tc[:, 0].to(DEV); row = c.atom_index(L, tb, tn).to(DEV); hit = (sel == row[:, None]).any(1); d = A[row]
surv = (Xraw * d).sum(1) / ct; prom = (X * d).sum(1).abs() / X.norm(dim=1)
RD = c.d["RD"].to(DEV); RG = c.d["RG"].to(DEV) if c.d["RG"] is not None else None                                          # [NB, DFF, D]
def readstr(bdelta):
    out = torch.zeros(c.NT, device=DEV); bt = (tb.to(DEV) + bdelta).clamp(max=c.NB - 1)
    for b in range(c.NB):
        m = bt == b
        if not m.any(): continue
        Rb = RD[b]; out[m] = (Rb @ d[m].T).norm(dim=0) / Rb.norm() * math.sqrt(c.D)                                        # read strength relative to an average direction
    return out
rs1, rs2, rs3 = readstr(1), readstr(2), readstr(3); rs_own = readstr(0)
rand_d = torch.randn(c.NT, c.D, device=DEV); rand_d = rand_d / rand_d.norm(dim=1, keepdim=True); d_save = d; d = rand_d; rs1_rand = readstr(1); d = d_save
def auc(score, y):
    pos, neg = score[y], score[~y]; i = sub(len(pos), 3000).to(DEV); j = sub(len(neg), 3000).to(DEV); return (pos[i][:, None] > neg[j][None, :]).float().mean().item()
def spearman(a, b):
    ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float(); return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
t = typ & (tb.to(DEV) < L - 1); erased = surv < 0.5
key = row[t]; uk, inv, cnt = key.unique(return_inverse=True, return_counts=True); m = cnt >= 20
agg = lambda v: torch.zeros(len(uk), device=DEV).index_add_(0, inv, v[t].float()) / cnt
rec_n, rs_n, surv_n = agg(hit), agg(rs1), agg(surv.clamp(-2, 3))
res = dict(model=tag, L=L, read_strength_next_block=dict(median=rs1[t].median().item(), random_direction_median=rs1_rand[t].median().item(), own_block=rs_own[t].median().item(), plus2=rs2[t].median().item(), plus3=rs3[t].median().item()),
           auc_read_next_for=dict(erased=auc(rs1[t], erased[t]), identified=auc(-rs1[t], hit[t])), spearman_read_next_vs=dict(raw_survival=spearman(rs1[t], surv.clamp(-2, 3)[t]), prominence=spearman(rs1[t], prom[t]), abs_coef=spearman(rs1[t], ct.abs()[t])),
           neuron_level=dict(n=int(m.sum()), spearman_recall_vs_read=spearman(rec_n[m], rs_n[m]), spearman_survival_vs_read=spearman(surv_n[m], rs_n[m])))
if RG is not None:
    RD_save = RD; RD = RG; rg1 = readstr(1); RD = RD_save; res["gate_read_next"] = dict(median=rg1[t].median().item(), auc_erased=auc(rg1[t], erased[t]), auc_identified=auc(-rg1[t], hit[t]))
record(f"e140_consume_{tag}", res, f"read strength by next block's MLP inputs (rel.): dominant writes {res['read_strength_next_block']['median']:.2f} vs random dirs {res['read_strength_next_block']['random_direction_median']:.2f} | AUC(read -> erased) {res['auc_read_next_for']['erased']:.2f}, AUC(low read -> identified) {res['auc_read_next_for']['identified']:.2f} | Spearman read vs survival {res['spearman_read_next_vs']['raw_survival']:+.2f} prominence {res['spearman_read_next_vs']['prominence']:+.2f} | neuron level: recall vs read {res['neuron_level']['spearman_recall_vs_read']:+.2f}, survival vs read {res['neuron_level']['spearman_survival_vs_read']:+.2f}" + (f" | gate rows: AUC erased {res['gate_read_next']['auc_erased']:.2f} identified {res['gate_read_next']['auc_identified']:.2f}" if 'gate_read_next' in res else ""))
