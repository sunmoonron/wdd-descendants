"""e249: (1) minimum-dimensional provenance: natural footprints and centered states at +2 and L projected to d
dimensions (random Gaussian projections, and the top-d principal components of the footprint cloud fitted on the
train half) for d = 1..D, then nearest-centroid source identification among K candidates. (2) Location of the
coordinate change: for blocks 3..6 the state is read at the block input, after the attention residual add
(input + attention output) and at the block output; at each point the along-direction survival of the dominant
block-2 write, the K-way identification of the footprint, and the native K-way read of the state (nearest native
atom). For the parallel-residual model (Pythia) the middle point is the attention-only partial sum."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from desc_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 16; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; NB = c.NB; b = 2; D = c.D; lab = c.d["lab"]; A = c.d["A"]; R = A[(lab["type"] == T_MLP) & (lab["block"] == b)].float().to(DEV); blocks = [j for j in (3, 4, 5, 6) if j <= L]
def run(neuron=None):
    st = {"att": {}}; hs = [arch.layers[j].register_forward_hook((lambda j_: lambda m, i, o: st.__setitem__(j_, (o[0] if isinstance(o, tuple) else o).detach().float().reshape(-1, D)))(j)) for j in set(blocks + [b, L, b + 2])]
    for j in blocks: hs.append(arch.attn_lin(j).register_forward_hook((lambda j_: lambda m, i, o: st["att"].__setitem__(j_, o.detach().float().reshape(-1, D)))(j)))
    if neuron is not None:
        def pre(m, inp):
            x = inp[0].clone(); flat = x.reshape(-1, c.DFF); flat[torch.arange(NT, device=DEV), neuron] = 0; return (flat.reshape(x.shape),)
        hs.append(arch.mlp_lin(b).register_forward_pre_hook(pre))
    model(ids_seq); [h.remove() for h in hs]; return st
led, tn, tc = dominant(model, arch, c, ids_seq, b); S0 = run(); S1 = run(tn); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep); d_w = R[tn]; torch.manual_seed(0); split = torch.rand(len(idx), device=DEV) < 0.5; tr, te = idx[split], idx[~split]; ltr, lte = lab_i[split], lab_i[~split]
def nc_acc(Ftr, Fte):
    cents = centroids(Ftr, ltr, K); return accuracy(Fte, cents, lte)
dims = [d for d in (1, 2, 4, 8, 16, 32, 64, 128, 256, D) if d <= D]; part1 = {}
for lv in sorted({b + 2, L}):
    F = S0[lv] - S1[lv]; X = S0[lv] - c.s["mu"][lv + 1].to(DEV); rec = {}
    Fc = F[tr] - F[tr].mean(0, keepdim=True); Ufp = torch.linalg.svd(Fc, full_matrices=False)[2]                      # PCs of the footprint cloud [min(n,D), D]
    for d in dims:
        torch.manual_seed(d); P = torch.randn(D, d, device=DEV) / math.sqrt(d); Pp = Ufp[:d].T
        rec[d] = dict(footprint_random=nc_acc(F[tr] @ P, F[te] @ P), footprint_pca=nc_acc(F[tr] @ Pp, F[te] @ Pp), state_random=nc_acc(X[tr] @ P, X[te] @ P), state_pca=nc_acc(X[tr] @ Pp, X[te] @ Pp))
    part1[lv] = rec
    log(f"{tag} level {lv} (K {K}, chance {1 / K:.2f}) d -> footprint random/pca, state random/pca: " + " ".join(f"{d}:{v['footprint_random']:.2f}/{v['footprint_pca']:.2f},{v['state_random']:.2f}/{v['state_pca']:.2f}" for d, v in rec.items()))
part2 = {}
for j in blocks:
    pts = {"in": (S0[j - 1], S1[j - 1]), "mid": (S0[j - 1] + S0["att"][j], S1[j - 1] + S1["att"][j]), "out": (S0[j], S1[j])}; rec = {}
    for nm, (H0, H1) in pts.items():
        F = H0 - H1; typj = typical_mask(H0); m = typj & big; along = ((F * d_w).sum(1) / tc)[m].median().item(); Xc = H0 - H0[typj].mean(0)
        rec[nm] = dict(along=along, footprint_id=nc_acc(F[tr], F[te]), state_id=nc_acc(Xc[tr], Xc[te]), native_argmax=((unit(Xc[te]) @ unit(R[keep]).T).argmax(1) == lte).float().mean().item())
    part2[j] = rec
    log(f"{tag} block {j}: " + " | ".join(f"{nm}: along {v['along']:.2f}, footprint id {v['footprint_id']:.2f}, state id {v['state_id']:.2f}, native argmax {v['native_argmax']:.2f}" for nm, v in rec.items()))
def first_d(rec, key, frac=0.9):
    full = rec[dims[-1]][key]
    for d in dims:
        if rec[d][key] >= frac * full: return d
    return dims[-1]
record(f"e249_dimloc_{tag}", dict(model=tag, b=b, L=L, K=K, dims={str(lv): {str(d): v for d, v in rec.items()} for lv, rec in part1.items()}, location={str(j): v for j, v in part2.items()}), " | ".join(f"lv{lv}: dims for 90% of full accuracy: footprint random {first_d(rec, 'footprint_random')} pca {first_d(rec, 'footprint_pca')}, state random {first_d(rec, 'state_random')} pca {first_d(rec, 'state_pca')} (full {rec[dims[-1]]['footprint_random']:.2f} / {rec[dims[-1]]['state_random']:.2f})" for lv, rec in part1.items()) + " | along in/mid/out by block: " + " ".join(f"b{j}:{v['in']['along']:.2f}/{v['mid']['along']:.2f}/{v['out']['along']:.2f}" for j, v in part2.items()) + " | footprint id in/mid/out: " + " ".join(f"b{j}:{v['in']['footprint_id']:.2f}/{v['mid']['footprint_id']:.2f}/{v['out']['footprint_id']:.2f}" for j, v in part2.items()))
