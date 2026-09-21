"""e25: is the ledger sparse, and is that learned? Per level: the effective number of active MLP writes per token
(participation ratio of c^2 over all neurons of blocks 0..L), the energy share of the top-1 / top-3 / top-64 true
writes, and the share of tokens whose top write exceeds 10x the median write. Trained vs random init vs training
checkpoints. WDD can only identify what stands out; this measures whether anything does."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); N = 8192; ids = sub(c.NT, N); rows = []
for L in range(c.NB):
    led = c.ledger(L); C = torch.cat([led[b][ids] for b in range(L + 1)], 1).to(DEV) ** 2      # [N, (L+1)*DFF]
    typ = typical_mask(c.X(L, center=False)[ids]); C = C[typ]
    tot = C.sum(1); pr = tot ** 2 / (C ** 2).sum(1)
    srt = C.topk(64, dim=1).values
    r = dict(L=L, n_writes=C.shape[1], pr_med=pr.median().item(), pr_q10=pr.quantile(0.1).item(), pr_q90=pr.quantile(0.9).item(),
             top1_share_med=(srt[:, 0] / tot).median().item(), top3_share_med=(srt[:, :3].sum(1) / tot).median().item(), top64_share_med=(srt[:, :64].sum(1) / tot).median().item(),
             active_frac=(C > 1e-4 * srt[:, :1]).float().mean().item())
    rows.append(r); log(f"{tag} L{L}: PR {r['pr_med']:.0f} of {r['n_writes']} | top1 {r['top1_share_med']:.3f} top3 {r['top3_share_med']:.3f} top64 {r['top64_share_med']:.3f}")
record(f"e25_sparsity_{tag}", dict(model=tag, rows=rows), " | ".join(f"L{r['L']}: PR {r['pr_med']:.0f} top1 {r['top1_share_med']:.2f} top64 {r['top64_share_med']:.2f}" for r in rows))
