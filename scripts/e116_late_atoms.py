"""e116 (data-free): where do MLP write directions live? Per block: the fraction of MLP atoms whose max-coherence
partner is an attention atom (any block / same block), the median energy of MLP atoms inside the span of the SAME
block's attention output subspaces (projection onto the union of its heads' W_O row spaces), and inside earlier
blocks' MLP atoms' span (top-256 PCs). Do late blocks write inside their own attention subspaces?"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); A, lab = c.dictionary(c.NB - 1); typA, blkA = lab["type"].to(DEV), lab["block"].to(DEV); rows = []
for b in range(c.NB):
    M = A[(typA == T_MLP) & (blkA == b)]; Att_b = A[(typA == T_ATT) & (blkA == b)]                                   # [NH*HD, D] orthonormal per head, not across heads
    Q, _ = torch.linalg.qr(Att_b.T); inA = ((M @ Q) ** 2).sum(1)                                                   # energy of each MLP atom inside the block's attention write span
    G = (M @ A.T).abs(); own = torch.nonzero((typA == T_MLP) & (blkA == b))[:, 0]; G[torch.arange(len(own)), own] = 0
    part = G.argmax(1); pt = typA[part]; pb = blkA[part]
    Wo_rank = Q.shape[1]
    rows.append(dict(b=b, attn_span_rank=Wo_rank, energy_in_own_attention_span_med=inA.median().item(), partner_is_attention=(pt == T_ATT).float().mean().item(), partner_is_same_block_attention=((pt == T_ATT) & (pb == b)).float().mean().item(), partner_is_earlier_mlp=((pt == T_MLP) & (pb < b)).float().mean().item(), partner_is_later_mlp=((pt == T_MLP) & (pb > b)).float().mean().item(), partner_is_token=(pt == T_TOK).float().mean().item(), maxcoh_med=G.max(1).values.median().item()))
    log(f"{tag} b{b}: energy in own attn span {rows[-1]['energy_in_own_attention_span_med']:.2f} (rank {Wo_rank}/{c.D}) | partner attn {rows[-1]['partner_is_attention']:.2f} (same block {rows[-1]['partner_is_same_block_attention']:.2f}) earlier mlp {rows[-1]['partner_is_earlier_mlp']:.2f} later mlp {rows[-1]['partner_is_later_mlp']:.2f} token {rows[-1]['partner_is_token']:.2f} | maxcoh {rows[-1]['maxcoh_med']:.2f}")
record(f"e116_lateatoms_{tag}", dict(model=tag, rows=rows), " | ".join(f"b{r['b']}: attn-span {r['energy_in_own_attention_span_med']:.2f} partner-attn {r['partner_is_attention']:.2f} earlier-mlp {r['partner_is_earlier_mlp']:.2f} tok {r['partner_is_token']:.2f}" for r in rows))
