"""e122: what is in the tail? A second OMP pass (64 more atoms) on the residual of the first pass with the same
dictionary: the real-write share of the MLP atoms selected in pass 1 vs pass 2, the FVU after each pass, and how
many of the token's top-8 true writes are found in pass 1 vs pass 2."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); N = 4096; ids = sub(c.NT, N); X = c.X(L)[ids]; typ = typical_mask(c.X(L, center=False)[ids]); A, lab = c.dictionary(L)
typA, blkA, idxA = lab["type"].to(DEV), lab["block"].to(DEV), lab["index"].to(DEV); led = c.ledger(L); C = torch.cat([led[b][ids] for b in range(L + 1)], 1).to(DEV); thr = 0.05 * C.abs().max(1, keepdim=True).values
tb, tn, tc = c.top_writes(L, 8); rows8 = torch.stack([c.atom_index(L, tb[ids, j], tn[ids, j]) for j in range(8)], 1).to(DEV)
def real_share(sel):
    ismlp = typA[sel] == T_MLP; key = torch.where(ismlp, blkA[sel] * c.DFF + idxA[sel], torch.zeros_like(sel)); real = ismlp & (C.gather(1, key).abs() >= thr)
    return (real[typ].sum() / ismlp[typ].sum()).item(), ismlp[typ].float().mean().item()
s1, c1, e1 = omp(X, A, 64); R = X - torch.einsum("nk,nkd->nd", c1, A[s1])
# second pass: exclude pass-1 atoms by zeroing their columns is costly; instead run OMP on R and drop repeats when scoring
s2, c2, e2 = omp(R, A, 64)
rs1, m1 = real_share(s1); rs2, m2 = real_share(s2)
top8_in1 = (s1[:, :, None] == rows8[:, None, :]).any(1).float().sum(1); top8_in2 = ((s2[:, :, None] == rows8[:, None, :]).any(1) & ~(s1[:, :, None] == rows8[:, None, :]).any(1)).float().sum(1)
res = dict(model=tag, L=L, pass1=dict(fvu64=fvu(e1[:, 63], X, typ), real_share=rs1, mlp_share=m1, top8_found=top8_in1[typ].mean().item()), pass2=dict(fvu_after=fvu(e2[:, 63], X, typ), real_share=rs2, mlp_share=m2, top8_found_new=top8_in2[typ].mean().item(), repeat_frac=(s2[:, :, None] == s1[:, None, :]).any(2).float().mean().item()))
record(f"e122_pass2_{tag}", res, f"pass1: fvu {res['pass1']['fvu64']:.3f} real share {rs1:.2f} top-8 found {res['pass1']['top8_found']:.2f}/8 | pass2: fvu {res['pass2']['fvu_after']:.3f} real share {rs2:.2f} new top-8 found {res['pass2']['top8_found_new']:.2f} repeats {res['pass2']['repeat_frac']:.2f}")
