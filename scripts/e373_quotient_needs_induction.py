"""e373: is the quotient's growth at the induction transition carried by the induction circuit? Pythia-410m at one
checkpoint: the block-2 quotient at level 12 (as e365) measured three times on the same tokens and candidates: with
the network intact, with the checkpoint's top-8 induction heads (e365's ablation ranking) zero-ablated, and with 8
random heads zero-ablated. Reported per condition: function dimension (PLS curve to 90% of the full score), the full
kNN score, the participation rank of the descendant cloud, and the top-16 subspace overlap with the intact quotient."""
import sys, os, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pc_common import *
from quot_common import Setup, pls, logit_scores, knn_cos, inside, unit, prank
rev = sys.argv[1]; j = json.load(open(os.path.join(RESULTS, f"e365_inddev_{rev}.json"))); r = j.get("result", j); top = [tuple(x) for x in r["top_heads"]]
S = Setup("pythia410", levels=[12], revision=rev); NB, NH = S.arch.NB, S.arch.NH; rng = random.Random(0)
rnd = rng.sample([(l, h) for l in range(NB) for h in range(NH) if (l, h) not in top], 8)
def zero_heads(hs):
    byl = {}
    for l, h in hs: byl.setdefault(l, []).append(h)
    out = []
    for l, items in byl.items():
        def pre(m, a, items=items):
            x = a[0].clone(); B, T, W = x.shape; xv = x.reshape(B, T, NH, W // NH)
            for h in items: xv[:, :, h, :] = 0
            return (xv.reshape(B, T, W),) + tuple(a[1:])
        out.append(S.arch.attn_lin(l).register_forward_pre_hook(pre))
    return out
res = dict(rev=rev, induction_heads=[list(x) for x in top], random_heads=[list(x) for x in rnd]); Q0 = None
for cond, hs in (("intact", []), ("induction_top8_ablated", top), ("random8_ablated", rnd)):
    hh = zero_heads(hs)
    try:
        nat = S.natural(); F = nat["F"][12]; tr, te = S.halves(len(S.idx)); Fc = F - F[tr].mean(0, keepdim=True); Z, _ = logit_scores(nat["dl"], tr); dln = unit(nat["dl"]); Q = pls(Fc[tr], Z[tr], 64)
        curve = {q: knn_cos(Fc @ Q[:, :q], dln, tr, te) for q in (1, 2, 4, 8, 16, 32, 64)}; full = knn_cos(Fc, dln, tr, te)
    finally: [h.remove() for h in hh]
    if Q0 is None: Q0 = Q[:, :16]
    res[cond] = dict(function_dim=next((q for q in curve if curve[q] >= 0.9 * full), 64), full=full, curve={str(k): v for k, v in curve.items()}, participation_rank=prank(Fc), overlap_with_intact16=inside(Q[:, :16], Q0), descendant_norm=F.norm(dim=1).median().item())
log(f"{rev}: " + " | ".join(f"{c}: dim {res[c]['function_dim']} full {res[c]['full']:.2f} prank {res[c]['participation_rank']:.1f} overlap {res[c]['overlap_with_intact16']:.2f} norm {res[c]['descendant_norm']:.3g}" for c in ("intact", "induction_top8_ablated", "random8_ablated")))
record(f"e373_qneedsind_{rev}", res, " | ".join(f"{c}: dim {res[c]['function_dim']} full {res[c]['full']:.2f} prank {res[c]['participation_rank']:.1f}" for c in ("intact", "induction_top8_ablated", "random8_ablated")))
