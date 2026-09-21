"""e327: scalar and sign structure of equivalence. For equivalent atom pairs (a ~ a'): is 0.5a ~ 0.5a', 2a ~ 2a', and
-a ~ -a' (coordinate and effect cosines of the scaled pairs); and is the coordinate of alpha a equal to alpha times
the coordinate of a (homogeneity of the quotient) for alpha in 0.5, 2, -1, 4."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quot_common import *
tag = sys.argv[1]; S = Setup(tag, levels=[0]); L = S.L; S = Setup(tag, levels=[L]); d = 16; nat = S.natural(); F = nat["F"][L]; tr, te = S.halves(len(S.idx)); Fc = F - F[tr].mean(0, keepdim=True); Z, Bv = logit_scores(nat["dl"], tr); P = pls(Fc[tr], Z[tr], d)
torch.manual_seed(0); atoms = torch.randperm(S.DFF, device=DEV)[:256]; V = S.W2[atoms]; img = torch.zeros(256, S.D, device=DEV); cnt = torch.zeros(256, device=DEV)
for p in range(3):
    r = S.inject_family(V, S.b + 1, seed=p); img.index_add_(0, r["a"], r["F"][L]); cnt.index_add_(0, r["a"], torch.ones(len(r["a"]), device=DEV))
img = img / cnt[:, None].clamp_min(1); z = img @ P; zc = unit(z) @ unit(z).T; zc.fill_diagonal_(-1); pairs = [(i, int(zc[i].argmax())) for i in range(256) if zc[i].max() >= zc.max(1).values.quantile(0.9)][:8]; sub = S.pool[::2][:1024]
def ce(v, amp):
    r = S.inject_family(unit(v)[None], S.b + 1, positions=sub, amp=amp * S.s_inj, seed=13); return r["F"][L].mean(0) @ P, (r["dl"] @ Bv).mean(0)
out = {}
for al in (0.5, 2.0, -1.0, 4.0):
    cz, cf, hom = [], [], []
    for i, j in pairs:
        zi, fi = ce(V[i], al); zj, fj = ce(V[j], al); z1, _ = ce(V[i], 1.0); cz.append((unit(zi[None]) @ unit(zj[None]).T).item()); cf.append((unit(fi[None]) @ unit(fj[None]).T).item()); hom.append((unit(zi[None]) @ unit((al * z1)[None]).T).item())
    out[al] = dict(pair_coordinate_cos=float(torch.tensor(cz).median()), pair_effect_cos=float(torch.tensor(cf).median()), homogeneity_cos=float(torch.tensor(hom).median()))
base = [((unit(z[i][None]) @ unit(z[j][None]).T).item()) for i, j in pairs]
log(f"{tag} ({len(pairs)} equivalent pairs, base coordinate cos {float(torch.tensor(base).median()):.2f}): alpha -> pair coordinate cos / pair effect cos / homogeneity z(alpha a) vs alpha z(a): " + " ; ".join(f"{al}: {v['pair_coordinate_cos']:.2f}/{v['pair_effect_cos']:.2f}/{v['homogeneity_cos']:+.2f}" for al, v in out.items()))
record(f"e327_scalar_{tag}", dict(model=tag, L=L, n_pairs=len(pairs), per_alpha={str(k): v for k, v in out.items()}), " ".join(f"a{al}:{v['pair_coordinate_cos']:.2f}/{v['pair_effect_cos']:.2f}/{v['homogeneity_cos']:+.2f}" for al, v in out.items()))
