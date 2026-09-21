"""e77: targeted dictionaries on the FULL state. To read block b's writes, use only block b's atoms (+ embeddings
+ the token atom) against the full state H[L+1]: recall of block-b-born dominant writes vs the full dictionary and
vs the increment reading. Is the competition from other blocks' atoms the cost, or the cancellation in the state?"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); N = int(os.environ.get("WDD_N", 4096)); ids = sub(c.NT, N)
X = c.X(L)[ids]; typ = typical_mask(c.X(L, center=False)[ids]); Afull, lab = c.dictionary(L); typA, blkA = lab["type"].to(DEV), lab["block"].to(DEV)
tb, tn, tc = c.top_writes(L, 1); tb, tn = tb[ids, 0], tn[ids, 0]; rowfull = c.atom_index(L, tb, tn).to(DEV)
selF, _, _ = omp(X, Afull, 64); hitF = (selF == rowfull[:, None]).any(1); rows = []
for b in range(L + 1):
    m = typ & (tb.to(DEV) == b)
    if m.sum() < 50: continue
    keep = (blkA == b) | (typA <= T_POS); A = Afull[keep]; newrow = (torch.cumsum(keep.long(), 0) - 1)[rowfull]
    selT, _, _ = omp(X[m], A, 32); hitT = (selT == newrow[m][:, None]).any(1)
    keepM = (blkA == b) & (typA == T_MLP); Am = Afull[keepM]; newrowM = (torch.cumsum(keepM.long(), 0) - 1)[rowfull]
    selM, _, _ = omp(X[m], Am, 16); hitM = (selM == newrowM[m][:, None]).any(1)
    D_ = (c.s["H"][b + 1][ids].float().to(DEV) - c.s["H"][b][ids].float().to(DEV))[m]; Ab, labb = c.dictionary(b, blocks=[b], types=(T_MLP, T_ATT, T_BIAS)); mlp_rows = torch.nonzero(labb["type"] == T_MLP)[:, 0].to(DEV)
    selD, _, _ = omp(D_, Ab, 16); hitD = (selD == mlp_rows[tn.to(DEV)[m]][:, None]).any(1)
    rows.append(dict(block=b, n=int(m.sum()), full_dict=hitF[m].float().mean().item(), targeted_block_plus_emb_k32=hitT.float().mean().item(), targeted_block_mlp_only_k16=hitM.float().mean().item(), increment_k16=hitD.float().mean().item()))
    log(f"{tag} b{b}: full {rows[-1]['full_dict']:.2f} targeted+emb {rows[-1]['targeted_block_plus_emb_k32']:.2f} targeted-mlp {rows[-1]['targeted_block_mlp_only_k16']:.2f} increment {rows[-1]['increment_k16']:.2f}")
record(f"e77_targeted_{tag}", dict(model=tag, L=L, rows=rows), " | ".join(f"b{r['block']}: full {r['full_dict']:.2f} targ {r['targeted_block_plus_emb_k32']:.2f} mlp {r['targeted_block_mlp_only_k16']:.2f} inc {r['increment_k16']:.2f}" for r in rows))
