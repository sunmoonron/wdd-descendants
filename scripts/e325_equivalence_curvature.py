"""e325: are the equivalence classes flat? Pairs of block-2 atoms at increasing physical distance but similar
functional coordinate (from exact per-atom coordinates, 512 atoms), and pairs with dissimilar coordinate. For each
pair the midpoint perturbation (a + b)/2 is injected; its coordinate against the average of the endpoint coordinates,
and its logit effect against the average of the endpoint effects, by physical distance and by functional similarity.
Flat classes: midpoints stay on the average regardless of physical distance."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quot_common import *
tag = sys.argv[1]; S = Setup(tag, levels=[0]); L = S.L; S = Setup(tag, levels=[L]); d = 16; nat = S.natural(); F = nat["F"][L]; tr, te = S.halves(len(S.idx)); Fc = F - F[tr].mean(0, keepdim=True); Z, Bv = logit_scores(nat["dl"], tr); P = pls(Fc[tr], Z[tr], d)
torch.manual_seed(0); atoms = torch.randperm(S.DFF, device=DEV)[:512]; V = S.W2[atoms]; img = torch.zeros(512, S.D, device=DEV); Fl = torch.zeros(512, Bv.shape[1], device=DEV); cnt = torch.zeros(512, device=DEV)
for p in range(4):
    r = S.inject_family(V, S.b + 1, seed=p); img.index_add_(0, r["a"], r["F"][L]); Fl.index_add_(0, r["a"], r["dl"] @ Bv); cnt.index_add_(0, r["a"], torch.ones(len(r["a"]), device=DEV))
img = img / cnt[:, None].clamp_min(1); Fl = Fl / cnt[:, None].clamp_min(1); z = img @ P; zc = unit(z) @ unit(z).T; pc = (V @ V.T).abs(); iu = torch.triu_indices(512, 512, 1, device=DEV); zs, ps = zc[iu[0], iu[1]], pc[iu[0], iu[1]]
def pick(mask, n=12): idx_ = torch.nonzero(mask)[:, 0]; idx_ = idx_[torch.randperm(len(idx_), device=DEV)[:n]]; return [(int(iu[0][i]), int(iu[1][i])) for i in idx_.tolist()]
groups = {"similar_far": pick((zs >= zs.quantile(0.95)) & (ps <= ps.quantile(0.5))), "similar_near": pick((zs >= zs.quantile(0.95)) & (ps >= ps.quantile(0.9))), "dissimilar_far": pick((zs <= zs.quantile(0.05)) & (ps <= ps.quantile(0.5)))}; out = {}
sub = S.pool[::2][:1024]
for nm, pairs in groups.items():
    czs, cfs = [], []
    for i, j in pairs:
        r = S.inject_family(unit((V[i] + V[j]))[None], S.b + 1, positions=sub, seed=7); mimg = r["F"][L].mean(0); mF = (r["dl"] @ Bv).mean(0); scale = unit(V[i] + V[j]) @ (V[i] + V[j]) / 2; zm = (mimg @ P); zavg = (z[i] + z[j]) / 2; czs.append((unit(zm[None]) @ unit(zavg[None]).T).item()); cfs.append((unit(mF[None]) @ unit(((Fl[i] + Fl[j]) / 2)[None]).T).item())
    out[nm] = dict(n=len(pairs), midpoint_coordinate_cos=float(torch.tensor(czs).median()) if czs else float("nan"), midpoint_effect_cos=float(torch.tensor(cfs).median()) if cfs else float("nan"))
log(f"{tag} (512 atoms): midpoint of a pair vs the average of the endpoints, coordinate / effect cosine: " + " | ".join(f"{nm} (n {v['n']}): {v['midpoint_coordinate_cos']:.2f} / {v['midpoint_effect_cos']:.2f}" for nm, v in out.items()))
record(f"e325_curvature_{tag}", dict(model=tag, L=L, groups=out), " | ".join(f"{nm} {v['midpoint_coordinate_cos']:.2f}/{v['midpoint_effect_cos']:.2f}" for nm, v in out.items()))
