"""e87: geometry of co-active writes. For each token, the cosines among its top-3 (and top-8) true MLP writes'
directions, signed by coefficient sign (so negative = the pair cancels along the shared component): distribution,
fraction of cancelling pairs, and the same for the top-3 writes vs the attention sum. Superposition-style
interference measured directly on the ledger, no OMP."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); N = 8192; ids = sub(c.NT, N); typ = typical_mask(c.X(L, center=False)[ids]); A, lab = c.dictionary(L)
tb, tn, tc = c.top_writes(L, 8); rows = torch.stack([c.atom_index(L, tb[ids, j], tn[ids, j]) for j in range(8)], 1).to(DEV); cf = tc[ids].to(DEV)
D_ = A[rows]; G = torch.einsum("nid,njd->nij", D_, D_); S = torch.sign(cf)[:, :, None] * torch.sign(cf)[:, None, :]; Gs = G * S    # signed: <0 means the pair opposes
iu3 = torch.triu_indices(3, 3, 1); iu8 = torch.triu_indices(8, 8, 1)
p3 = Gs[:, iu3[0], iu3[1]][typ]; p8 = Gs[:, iu8[0], iu8[1]][typ]
att = sum(c.s["ATT"][b][ids].float().to(DEV) for b in range(L + 1)); ca = ((D_[:, 0] * att).sum(1) / att.norm(dim=1)) * torch.sign(cf[:, 0])
res = dict(model=tag, L=L, top3_signed_cos=dict(median=p3.median().item(), mean=p3.mean().item(), frac_cancel_below_m0p3=(p3 < -0.3).float().mean().item(), frac_reinforce_above_0p3=(p3 > 0.3).float().mean().item(), frac_abs_below_0p1=(p3.abs() < 0.1).float().mean().item()),
           top8_signed_cos=dict(median=p8.median().item(), frac_cancel_below_m0p3=(p8 < -0.3).float().mean().item(), frac_reinforce_above_0p3=(p8 > 0.3).float().mean().item()),
           top1_vs_attention_signed_cos=dict(median=ca[typ].median().item(), frac_cancel=(ca[typ] < -0.3).float().mean().item(), frac_reinforce=(ca[typ] > 0.3).float().mean().item()),
           top12_pair_signed_cos_median=Gs[:, 0, 1][typ].median().item())
record(f"e87_coactive_{tag}", res, f"top-3 signed cos median {res['top3_signed_cos']['median']:+.2f} cancel(<-0.3) {res['top3_signed_cos']['frac_cancel_below_m0p3']:.2f} reinforce(>0.3) {res['top3_signed_cos']['frac_reinforce_above_0p3']:.2f} ~orthogonal(|c|<0.1) {res['top3_signed_cos']['frac_abs_below_0p1']:.2f} | top-1/top-2 pair {res['top12_pair_signed_cos_median']:+.2f} | top-8 cancel {res['top8_signed_cos']['frac_cancel_below_m0p3']:.2f} reinforce {res['top8_signed_cos']['frac_reinforce_above_0p3']:.2f} | top-1 vs attention sum median {res['top1_vs_attention_signed_cos']['median']:+.2f} cancel {res['top1_vs_attention_signed_cos']['frac_cancel']:.2f} reinforce {res['top1_vs_attention_signed_cos']['frac_reinforce']:.2f}")
