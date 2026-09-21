"""e145 (data-free): do MLP write directions live inside single attention-head output subspaces? Per block: for
each MLP atom, the energy inside the best single head's W_O row space (rank HD) of the SAME block, of any block,
vs the same for random rank-HD subspaces; the fraction of atoms with > 50% energy in one head's subspace."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); A, lab = c.dictionary(c.NB - 1); typA, blkA, idxA, rkA = (lab[k].to(DEV) for k in ("type", "block", "index", "rank")); rows = []; g = torch.Generator(device=DEV).manual_seed(0)
heads = {(b, h): A[(typA == T_ATT) & (blkA == b) & (idxA == h)] for b in range(c.NB) for h in range(c.NH)}                    # orthonormal [HD, D]
Rq = [torch.linalg.qr(torch.randn(c.D, c.HD, generator=g, device=DEV))[0].T for _ in range(16)]
for b in range(c.NB):
    M = A[(typA == T_MLP) & (blkA == b)]
    same = torch.stack([((M @ heads[(b, h)].T) ** 2).sum(1) for h in range(c.NH)], 1).max(1).values
    any_ = torch.stack([((M @ heads[(bb, h)].T) ** 2).sum(1) for bb in range(c.NB) for h in range(c.NH)], 1).max(1).values
    rnd = torch.stack([((M @ R.T) ** 2).sum(1) for R in Rq], 1).max(1).values
    rows.append(dict(b=b, same_block_best_head_med=same.median().item(), any_block_best_head_med=any_.median().item(), random_best_of_16_med=rnd.median().item(), frac_above_half_same=(same > 0.5).float().mean().item(), frac_above_half_any=(any_ > 0.5).float().mean().item(), frac_above_half_random=(rnd > 0.5).float().mean().item()))
    log(f"{tag} b{b}: energy in best single head subspace: same block {rows[-1]['same_block_best_head_med']:.2f} any block {rows[-1]['any_block_best_head_med']:.2f} random rank-{c.HD} {rows[-1]['random_best_of_16_med']:.2f} | >0.5: same {rows[-1]['frac_above_half_same']:.2f} any {rows[-1]['frac_above_half_any']:.2f} random {rows[-1]['frac_above_half_random']:.2f}")
record(f"e145_headsub_{tag}", dict(model=tag, HD=c.HD, rows=rows), " | ".join(f"b{r['b']}: same {r['same_block_best_head_med']:.2f} any {r['any_block_best_head_med']:.2f} rnd {r['random_best_of_16_med']:.2f} (>0.5 any {r['frac_above_half_any']:.2f})" for r in rows))
