"""e141: is cancellation concentrated? For dominant writes born in block b0 (larger half) that end up with raw
survival < 0.5 at level L: the per-level survival trajectory (which level has the largest drop), the share of the
total cancellation carried by the single largest opposing MLP write and by the top-5, and whether the largest
opposer is the same neuron across tokens (an eraser)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); N = 4096; ids = sub(c.NT, N); b0 = 1
led0 = c.ledger(b0, blocks=[b0])[b0][ids]; tcv, tnv = led0.abs().max(1); tcv = torch.gather(led0, 1, tnv[:, None])[:, 0].to(DEV); big = tcv.abs() >= tcv.abs().quantile(0.5)
A, lab = c.dictionary(L); row = c.atom_index(L, torch.full_like(tnv, b0), tnv).to(DEV); d = A[row]
survs = torch.stack([(c.X(l, center=False)[ids] * d).sum(1) / tcv for l in range(b0, L + 1)], 1)                          # [N, L-b0+1]
erased = big & typical_mask(c.X(L, center=False)[ids]) & (survs[:, -1] < 0.5)
drops = survs[:, :-1] - survs[:, 1:]; lvl = drops.argmax(1) + b0 + 1                                                        # level whose block did the largest drop
# opposing MLP writes: projections of every later neuron's write onto d, per token
led = c.ledger(L); opp = []; who = []
for b in range(b0 + 1, L + 1):
    W = c.wdir_cpu(b).to(DEV); Wn = W / W.norm(dim=1, keepdim=True); proj = (led[b][ids].to(DEV) * (Wn @ d.T).T)                # [N, DFF]: coef_j * cos(d_j, d) = write's projection on d
    opp.append(proj); who.append(torch.full((c.DFF,), b, device=DEV))
P = torch.cat(opp, 1); blocks = torch.cat(who); tot_cancel = (-P.clamp(max=0)).sum(1)                                        # total opposing projection
top = (-P).topk(5, dim=1); share1 = top.values[:, 0] / tot_cancel.clamp_min(1e-6); share5 = top.values.sum(1) / tot_cancel.clamp_min(1e-6)
tot_re = P.clamp(min=0).sum(1); net = (P.sum(1)) / tcv
top_id = top.indices[:, 0]; u, cnt = top_id[erased].unique(return_counts=True)
res = dict(model=tag, b0=b0, L=L, n_erased=int(erased.sum()), frac_erased=(erased.float().sum() / big.float().sum()).item(),
           largest_drop_level_hist={int(l): float(((lvl == l) & erased).sum() / erased.sum().clamp_min(1)) for l in range(b0 + 1, L + 1)},
           share_of_cancellation_top1_median=share1[erased].median().item(), share_top5_median=share5[erased].median().item(), top_opposer_block_hist={int(b): float(((blocks[top_id] == b) & erased).sum() / erased.sum().clamp_min(1)) for b in range(b0 + 1, L + 1)},
           most_common_opposer=dict(block=int(blocks[u[cnt.argmax()]]), neuron=int(u[cnt.argmax()] % c.DFF), share_of_erased_tokens=(cnt.max() / erased.sum().clamp_min(1)).item()),
           opposition_vs_reinforcement_median=dict(total_opposing_over_c=(tot_cancel / tcv.abs())[erased].median().item(), total_reinforcing_over_c=(tot_re / tcv.abs())[erased].median().item(), net_later_mlp_over_c=net[erased].median().item()))
record(f"e141_eraser_{tag}", res, f"born b{b0}, erased at L{L}: {res['frac_erased']:.2f} of the larger half (n={res['n_erased']}) | largest-drop level hist " + " ".join(f"L{k}:{v:.2f}" for k, v in res['largest_drop_level_hist'].items()) + f" | share of cancellation by the top-1 opposing neuron {res['share_of_cancellation_top1_median']:.2f}, top-5 {res['share_top5_median']:.2f} | most common top opposer b{res['most_common_opposer']['block']}#{res['most_common_opposer']['neuron']} in {res['most_common_opposer']['share_of_erased_tokens']:.2f} of erased tokens | opposing/c {res['opposition_vs_reinforcement_median']['total_opposing_over_c']:.2f} reinforcing/c {res['opposition_vs_reinforcement_median']['total_reinforcing_over_c']:.2f}")
