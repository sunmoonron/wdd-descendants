"""e173: who cancels whom. For dominant MLP writes born in block b (all b <= L), the mean contribution of each
block b' (attention and MLP separately) to the write's raw survival at level L: a (birth block x contributing
block) matrix per model, split into MLP and attention contributions; plus the same for the reinforcement (positive
part) and cancellation (negative part) separately."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); N = 8192; ids = sub(c.NT, N); Xraw = c.X(L, center=False)[ids]; typ = typical_mask(Xraw); A, lab = c.dictionary(L)
tb, tn, tc = c.top_writes(L, 1); tb, tn, ct = tb[ids, 0], tn[ids, 0], tc[ids, 0].to(DEV); row = c.atom_index(L, tb, tn).to(DEV); d = A[row]; tbd = tb.to(DEV)
M_mlp = torch.zeros(L + 1, L + 1); M_att = torch.zeros(L + 1, L + 1); M_neg = torch.zeros(L + 1, L + 1); M_pos = torch.zeros(L + 1, L + 1); counts = torch.zeros(L + 1)
for bp in range(L + 1):
    att = c.s["ATT"][bp][ids].float().to(DEV); mlp = c.acts[bp][ids].float().to(DEV) @ c.wdir_cpu(bp).to(DEV)
    if c.d["mlp_bias"][bp] is not None: mlp = mlp + c.d["mlp_bias"][bp].to(DEV)
    pa, pm = (att * d).sum(1) / ct, (mlp * d).sum(1) / ct
    for b in range(L + 1):
        m = typ & (tbd == b)
        if m.sum() < 30: continue
        pmm = pm[m] - (1.0 if bp == b else 0.0)                                                      # remove the write itself
        M_mlp[b, bp] = pmm.mean().item(); M_att[b, bp] = pa[m].mean().item(); M_neg[b, bp] = (pmm.clamp(max=0) + pa[m].clamp(max=0)).mean().item(); M_pos[b, bp] = (pmm.clamp(min=0) + pa[m].clamp(min=0)).mean().item(); counts[b] = m.sum().item()
res = dict(model=tag, L=L, counts=counts.tolist(), mlp=M_mlp.tolist(), attention=M_att.tolist(), cancellation=M_neg.tolist(), reinforcement=M_pos.tolist(),
           summary=dict(strongest_canceller_per_birth={int(b): (int(M_neg[b].argmin()), round(M_neg[b].min().item(), 2)) for b in range(L + 1) if counts[b] > 0}, strongest_reinforcer_per_birth={int(b): (int(M_pos[b].argmax()), round(M_pos[b].max().item(), 2)) for b in range(L + 1) if counts[b] > 0},
                        total_net_by_birth={int(b): round((M_mlp[b] + M_att[b]).sum().item(), 2) for b in range(L + 1) if counts[b] > 0}))
record(f"e173_erasure_{tag}", res, "net later contribution by birth block: " + " ".join(f"b{b}:{v:+.2f}" for b, v in res["summary"]["total_net_by_birth"].items()) + " | strongest canceller (block, mean): " + " ".join(f"b{b}->b{v[0]} {v[1]:+.2f}" for b, v in res["summary"]["strongest_canceller_per_birth"].items()) + " | strongest reinforcer: " + " ".join(f"b{b}->b{v[0]} {v[1]:+.2f}" for b, v in res["summary"]["strongest_reinforcer_per_birth"].items()))
