"""e199: are canceller blocks the unreadable blocks? Per block: the total cancellation it delivers to other blocks'
dominant writes (column sums of the negative part of the e173 matrix) and the total reinforcement, vs the
identification rate of its OWN dominant writes at level L (OMP@64 and dual@64). Spearman across blocks."""
import sys, os, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); r = json.load(open(os.path.join(RESULTS, f"e173_erasure_{tag}.json"))); Mneg = torch.tensor(r["cancellation"]); Mpos = torch.tensor(r["reinforcement"]); cnt = torch.tensor(r["counts"])
deliver_neg = -(Mneg * (cnt[:, None] > 0)).sum(0); deliver_pos = (Mpos * (cnt[:, None] > 0)).sum(0); own_neg = -torch.tensor([Mneg[b, b] for b in range(L + 1)])
N = 8192; ids = sub(c.NT, N); Xraw = c.X(L, center=False)[ids]; typ = typical_mask(Xraw); X = Xraw - c.s["mu"][L + 1].to(DEV); A, lab = c.dictionary(L)
tb, tn, tc = c.top_writes(L, 1); tb, tn = tb[ids, 0], tn[ids, 0]; row = c.atom_index(L, tb, tn).to(DEV); tbd = tb.to(DEV)
sel_o, _, _ = get_omp(c, L); sel_o = sel_o[ids]; S = A.T @ A; ev, V = torch.linalg.eigh(S); Winv = V @ torch.diag(1 / (ev + 1e-2 * ev[-1])) @ V.T; sel_d, _, _ = oneshot(X, A, 64, whiten=Winv)
hit_o = (sel_o == row[:, None]).any(1); hit_d = (sel_d == row[:, None]).any(1); rows = []
for b in range(L + 1):
    m = typ & (tbd == b)
    if m.sum() < 30: continue
    rows.append(dict(block=b, n=int(m.sum()), recall_omp=hit_o[m].float().mean().item(), recall_dual=hit_d[m].float().mean().item(), cancellation_delivered=deliver_neg[b].item(), reinforcement_delivered=deliver_pos[b].item(), own_cancellation=own_neg[b].item()))
def spearman(a, b):
    a, b = torch.tensor(a), torch.tensor(b); ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float(); return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
rho_o = spearman([x["cancellation_delivered"] for x in rows], [x["recall_omp"] for x in rows]); rho_d = spearman([x["cancellation_delivered"] for x in rows], [x["recall_dual"] for x in rows]); rho_p = spearman([x["reinforcement_delivered"] for x in rows], [x["recall_dual"] for x in rows])
for x in rows: log(f"{tag} b{x['block']}: cancellation delivered {x['cancellation_delivered']:.2f} reinforcement {x['reinforcement_delivered']:.2f} | own dominant writes read: omp {x['recall_omp']:.2f} dual {x['recall_dual']:.2f}")
record(f"e199_cancread_{tag}", dict(model=tag, L=L, rows=rows, spearman=dict(cancel_omp=rho_o, cancel_dual=rho_d, reinforce_dual=rho_p)), f"Spearman across {len(rows)} blocks: cancellation delivered vs own readability omp {rho_o:+.2f} dual {rho_d:+.2f}; reinforcement delivered vs dual readability {rho_p:+.2f} | top cancellers: " + " ".join(f"b{x['block']}({x['cancellation_delivered']:.1f}, dual {x['recall_dual']:.2f})" for x in sorted(rows, key=lambda x: -x['cancellation_delivered'])[:3]) + " | least cancelling: " + " ".join(f"b{x['block']}({x['cancellation_delivered']:.1f}, dual {x['recall_dual']:.2f})" for x in sorted(rows, key=lambda x: x['cancellation_delivered'])[:3]))
