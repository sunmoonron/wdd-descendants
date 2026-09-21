"""e181: alias or swamp? For dominant writes born in block 1 that are MISSED at level L (cached OMP@64): the cosine
between the write's direction and the closest atom in the support (alias if > 0.5), the write's projection on its
direction relative to its birth (survival), and the state norm relative to level 1; per model, at L = mid and the
last level. Distinguishes 'a twin took the direction' from 'the state grew along other directions'."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); N = 4096; ids = sub(c.NT, N); b0 = 1; led = c.ledger(b0, blocks=[b0])[b0][ids]; tcv, tnv = led.abs().max(1); tcv = torch.gather(led, 1, tnv[:, None])[:, 0].to(DEV); big = tcv.abs() >= tcv.abs().quantile(0.5)
norm1 = c.X(b0, center=False)[ids].norm(dim=1); out = {}
for L in sorted(set([mid(c), c.NB - 1])):
    X = c.X(L)[ids]; Xraw = c.X(L, center=False)[ids]; typ = typical_mask(Xraw) & big; A, lab = c.dictionary(L); row = c.atom_index(L, torch.full_like(tnv, b0), tnv).to(DEV); d = A[row]
    sel, cof, err = omp(X, A, 64); hit = (sel == row[:, None]).any(1); miss = typ & ~hit
    cosv = torch.einsum("nkd,nd->nk", A[sel], d).abs(); closest = cosv.max(1).values; cb = lab["block"].to(DEV)[sel.gather(1, cosv.argmax(1)[:, None])[:, 0]]; ct_ = lab["type"].to(DEV)[sel.gather(1, cosv.argmax(1)[:, None])[:, 0]]
    surv = (Xraw * d).sum(1) / tcv; prom = (X * d).sum(1).abs() / X.norm(dim=1); norm_ratio = Xraw.norm(dim=1) / norm1
    out[L] = dict(n_miss=int(miss.sum()), recall=hit[typ].float().mean().item(), closest_cos_med=closest[miss].median().item(), frac_alias_above_0p5=(closest[miss] > 0.5).float().mean().item(), frac_alias_above_0p8=(closest[miss] > 0.8).float().mean().item(),
                  closest_is_attention=(ct_[miss] == T_ATT).float().mean().item(), closest_block_med=cb[miss].float().median().item(), survival_med_miss=surv[miss].median().item(), survival_med_hit=surv[typ & hit].median().item() if (typ & hit).any() else None,
                  prominence_med_miss=prom[miss].median().item(), prominence_med_hit=prom[typ & hit].median().item() if (typ & hit).any() else None, state_norm_ratio_med=norm_ratio[typ].median().item())
    log(f"{tag} L{L}: born-b1 writes recall {out[L]['recall']:.2f}; misses: closest support atom cos {out[L]['closest_cos_med']:.2f} (>0.5: {out[L]['frac_alias_above_0p5']:.2f}, >0.8: {out[L]['frac_alias_above_0p8']:.2f}; attention {out[L]['closest_is_attention']:.2f}) | survival miss {out[L]['survival_med_miss']:.2f} vs hit {out[L]['survival_med_hit']} | prominence miss {out[L]['prominence_med_miss']:.2f} vs hit {out[L]['prominence_med_hit']} | state norm / level-1 norm {out[L]['state_norm_ratio_med']:.2f}")
record(f"e181_missanat_{tag}", dict(model=tag, b0=b0, levels=out), " | ".join(f"L{L}: recall {v['recall']:.2f}, misses alias>0.5 {v['frac_alias_above_0p5']:.2f} (>0.8 {v['frac_alias_above_0p8']:.2f}), surv miss {v['survival_med_miss']:.2f} hit {v['survival_med_hit']}, norm x{v['state_norm_ratio_med']:.1f}" for L, v in out.items()))
