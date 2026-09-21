"""e328: is equivalence preserved by transport? Pairs of atoms equivalent at L (top decile of coordinate cosine,
physically dissimilar) and their coordinates at L+2 and at the last level (quotients refitted there): the pair
cosine by level, against random pairs. Equivalence that survives transport marks a genuine quotient; equivalence
that decays marks a level-local coincidence."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quot_common import *
tag = sys.argv[1]; S0_ = Setup(tag, levels=[0]); L = S0_.L; NB = S0_.NB; S = Setup(tag, levels=[L, L + 2, NB - 2]); d = 16; nat = S.natural(); tr, te = S.halves(len(S.idx)); Z, _ = logit_scores(nat["dl"], tr); Ps = {lv: pls((nat["F"][lv] - nat["F"][lv][tr].mean(0, keepdim=True))[tr], Z[tr], d) for lv in S.levels}
torch.manual_seed(0); atoms = torch.randperm(S.DFF, device=DEV)[:384]; V = S.W2[atoms]; img = {lv: torch.zeros(384, S.D, device=DEV) for lv in S.levels}; cnt = torch.zeros(384, device=DEV)
for p in range(3):
    r = S.inject_family(V, S.b + 1, seed=p)
    for lv in S.levels: img[lv].index_add_(0, r["a"], r["F"][lv])
    cnt.index_add_(0, r["a"], torch.ones(len(r["a"]), device=DEV))
z = {lv: (img[lv] / cnt[:, None].clamp_min(1)) @ Ps[lv] for lv in S.levels}; zc = unit(z[L]) @ unit(z[L]).T; pc = (V @ V.T).abs(); zc.fill_diagonal_(-1); pairs = [(i, int(zc[i].argmax())) for i in range(384) if zc[i].max() >= zc.max(1).values.quantile(0.9) and pc[i, int(zc[i].argmax())] < 0.3]; rnd = [(i, int(torch.randint(0, 384, (1,)).item())) for i in range(len(pairs))]
out = {lv: dict(equivalent=float(torch.tensor([(unit(z[lv][i][None]) @ unit(z[lv][j][None]).T).item() for i, j in pairs]).median()), random=float(torch.tensor([(unit(z[lv][i][None]) @ unit(z[lv][j][None]).T).item() for i, j in rnd]).median())) for lv in S.levels}
log(f"{tag} ({len(pairs)} pairs equivalent at {L}): coordinate cosine of the pairs by level (equivalent / random): " + " ; ".join(f"{lv}: {v['equivalent']:.2f} / {v['random']:.2f}" for lv, v in out.items()))
record(f"e328_transport_{tag}", dict(model=tag, L=L, n_pairs=len(pairs), per_level={str(k): v for k, v in out.items()}), " ".join(f"{lv}:{v['equivalent']:.2f}/{v['random']:.2f}" for lv, v in out.items()))
