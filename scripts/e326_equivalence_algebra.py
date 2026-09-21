"""e326: does functional equivalence form an algebra? From 512 exact atom coordinates: pairs a ~ a' and b ~ b' (top
decile of coordinate cosine, physically dissimilar). Test: a + b ~ a' + b', measured by injecting both sums and
comparing their coordinates and their logit effects, against random re-pairings (a + b vs a' + b'' with b'' not
equivalent to b) and against the pair baseline (a vs a')."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quot_common import *
tag = sys.argv[1]; S = Setup(tag, levels=[0]); L = S.L; S = Setup(tag, levels=[L]); d = 16; nat = S.natural(); F = nat["F"][L]; tr, te = S.halves(len(S.idx)); Fc = F - F[tr].mean(0, keepdim=True); Z, Bv = logit_scores(nat["dl"], tr); P = pls(Fc[tr], Z[tr], d)
torch.manual_seed(0); atoms = torch.randperm(S.DFF, device=DEV)[:512]; V = S.W2[atoms]; img = torch.zeros(512, S.D, device=DEV); cnt = torch.zeros(512, device=DEV)
for p in range(4):
    r = S.inject_family(V, S.b + 1, seed=p); img.index_add_(0, r["a"], r["F"][L]); cnt.index_add_(0, r["a"], torch.ones(len(r["a"]), device=DEV))
img = img / cnt[:, None].clamp_min(1); z = img @ P; zc = unit(z) @ unit(z).T; pc = (V @ V.T).abs(); zc.fill_diagonal_(-1); equiv = [(i, int(zc[i].argmax())) for i in range(512) if zc[i].max() >= zc.max(1).values.quantile(0.9) and pc[i, int(zc[i].argmax())] < 0.3][:10]
sub = S.pool[::2][:1024]
def coord_effect(v):
    r = S.inject_family(unit(v)[None], S.b + 1, positions=sub, seed=11); return r["F"][L].mean(0) @ P, (r["dl"] @ Bv).mean(0)
res = {"equivalent_sums": [], "repaired_sums": [], "single_pairs": []}
for t in range(min(6, len(equiv) // 2)):
    (a, a2), (b_, b2) = equiv[2 * t], equiv[2 * t + 1]; bb = int(torch.randint(0, 512, (1,)).item())
    za, fa = coord_effect(V[a] + V[b_]); zb, fb = coord_effect(V[a2] + V[b2]); zr, fr = coord_effect(V[a2] + V[bb]); zs1, fs1 = coord_effect(V[a]); zs2, fs2 = coord_effect(V[a2])
    res["equivalent_sums"].append(((unit(za[None]) @ unit(zb[None]).T).item(), (unit(fa[None]) @ unit(fb[None]).T).item())); res["repaired_sums"].append(((unit(za[None]) @ unit(zr[None]).T).item(), (unit(fa[None]) @ unit(fr[None]).T).item())); res["single_pairs"].append(((unit(zs1[None]) @ unit(zs2[None]).T).item(), (unit(fs1[None]) @ unit(fs2[None]).T).item()))
summ = {nm: dict(n=len(v), coordinate_cos=float(torch.tensor([x[0] for x in v]).median()) if v else float("nan"), effect_cos=float(torch.tensor([x[1] for x in v]).median()) if v else float("nan")) for nm, v in res.items()}
log(f"{tag} ({len(equiv)} equivalent pairs found among 512 atoms): coordinate / effect cosine: a+b vs a'+b' {summ['equivalent_sums']['coordinate_cos']:.2f} / {summ['equivalent_sums']['effect_cos']:.2f}; a+b vs a'+b'' (re-paired) {summ['repaired_sums']['coordinate_cos']:.2f} / {summ['repaired_sums']['effect_cos']:.2f}; single a vs a' {summ['single_pairs']['coordinate_cos']:.2f} / {summ['single_pairs']['effect_cos']:.2f} (n {summ['equivalent_sums']['n']})")
record(f"e326_algebra_{tag}", dict(model=tag, L=L, n_equivalent=len(equiv), summary=summ), f"a+b~a'+b' {summ['equivalent_sums']['coordinate_cos']:.2f}/{summ['equivalent_sums']['effect_cos']:.2f} repaired {summ['repaired_sums']['coordinate_cos']:.2f}/{summ['repaired_sums']['effect_cos']:.2f} singles {summ['single_pairs']['coordinate_cos']:.2f}/{summ['single_pairs']['effect_cos']:.2f}")
