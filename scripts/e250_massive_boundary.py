"""e250: are the massive-activation neurons the boundary of vector-determined transport? Block 2 candidates with
>= 20 natural tokens; M = the candidate with the largest median |coefficient|, N1-N3 = candidates nearest the median.
Injections at foreign tokens: (i) homogeneity error between 0.5x and 1x of each neuron's own magnitude; (ii) cross-
magnitude: the normal neurons at M's magnitude and M at the normal magnitude; (iii) angle preservation of a pair with
initial cosine 0.75 built on M's vector and on N1's; (iv) coherent fraction of transport per neuron and the cosine of
the transplant image with the natural centroid; (v) native dual@64 readability of the injected write at 1x; all at +2
and L."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from desc_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; b = 2; levels = sorted({b + 2, L}); run = make_runner(model, arch, c, ids_seq, levels, NT); lab = c.d["lab"]; A = c.d["A"]; R = A[(lab["type"] == T_MLP) & (lab["block"] == b)].float().to(DEV); D = c.D
led, tn, tc = dominant(model, arch, c, ids_seq, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); S1 = run(b, tn); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep); med = torch.stack([tc[idx][lab_i == k].median() for k in range(K)])
order = med.abs().argsort(descending=True); M = order[0]; mid_val = med.abs().median(); N = (med.abs() - mid_val).abs().argsort()[:3]; sel = torch.cat([M[None], N]); names = ["massive", "normal1", "normal2", "normal3"]
foreign = torch.nonzero(typ & ~torch.isin(tn, keep))[:, 0]; NF = len(foreign); torch.manual_seed(0); assign = torch.randint(0, 4, (NF,), device=DEV); vec = R[keep[sel]]; mag = med[sel]
dicts = {}; rows = {}
for lv in levels:
    Alv, lablv = c.dictionary(lv); S = Alv.T @ Alv; ev, V = torch.linalg.eigh(S); dicts[lv] = (Alv, V @ torch.diag(1 / (ev + 1e-2 * ev[-1])) @ V.T, c.s["mu"][lv + 1].to(DEV)); rows[lv] = c.atom_index(lv, torch.full((4,), b, dtype=torch.long), keep[sel].cpu()).to(DEV)
def inject(amp_per_token, dirs_per_token):
    inj = torch.zeros(NT, D, device=DEV); inj[foreign] = amp_per_token[:, None] * dirs_per_token; return run(inject=inj, inject_block=b + 1)
S_1 = inject(mag[assign], vec[assign]); S_h = inject(0.5 * mag[assign], vec[assign]); cross_mag = torch.where(assign == 0, med[N[0]].abs() * torch.sign(mag[0]), mag[0].abs() * torch.sign(mag[assign])); S_x = inject(cross_mag, vec[assign])
u = torch.randn(NF, D, device=DEV); u = unit(u - (u * vec[assign]).sum(1, keepdim=True) * vec[assign]); S_r = inject(mag[assign], 0.75 * vec[assign] + math.sqrt(1 - 0.75 ** 2) * u)
S_2 = inject(mag[assign], vec[assign]); S_3 = inject(mag[assign], vec[assign])  # same injection twice more? contexts are the same, so use different assignments instead
torch.manual_seed(1); assign2 = torch.randint(0, 4, (NF,), device=DEV); S_2 = inject(mag[assign2], vec[assign2]); torch.manual_seed(2); assign3 = torch.randint(0, 4, (NF,), device=DEV); S_3 = inject(mag[assign3], vec[assign3])
out = {}
for lv in levels:
    F1 = (S_1[lv] - S0[lv])[foreign]; Fh = (S_h[lv] - S0[lv])[foreign]; Fx = (S_x[lv] - S0[lv])[foreign]; Fr = (S_r[lv] - S0[lv])[foreign]; Fn = S0[lv] - S1[lv]; nat = centroids(Fn[idx], lab_i, K); rec = {}
    Alv, Winv, mu = dicts[lv]; sd = oneshot((S_1[lv] - mu)[foreign], Alv, 64, whiten=Winv)[0]
    for j, nm in enumerate(names):
        m = assign == j
        imgs = torch.cat([F1[m] / mag[j], (S_2[lv] - S0[lv])[foreign][assign2 == j] / mag[j], (S_3[lv] - S0[lv])[foreign][assign3 == j] / mag[j]]); meanimg = imgs.mean(0)
        rec[nm] = dict(median_coef=mag[j].abs().item(), homogeneity=((Fh[m] - 0.5 * F1[m]).norm(dim=1) / (0.5 * F1[m]).norm(dim=1).clamp_min(1e-6)).median().item(), cross_magnitude_homogeneity=((Fx[m] / cross_mag[m][:, None] - F1[m] / mag[j]).norm(dim=1) / (F1[m] / mag[j]).norm(dim=1).clamp_min(1e-6)).median().item(), pair_cos_0p75=((F1[m] * Fr[m]).sum(1) / (F1[m].norm(dim=1) * Fr[m].norm(dim=1)).clamp_min(1e-9)).median().item(), coherent_fraction=((meanimg ** 2).sum() / (imgs ** 2).sum(1).mean()).item(), image_vs_natural_cos=(unit(meanimg[None]) @ nat[sel[j]][:, None])[0, 0].item(), natural_identification=accuracy(Fn[idx][lab_i == sel[j]], nat, lab_i[lab_i == sel[j]]) if (lab_i == sel[j]).any() else float("nan"), native_dual_injected=(sd[m] == rows[lv][j]).any(1).float().mean().item())
    out[lv] = rec
    log(f"{tag} level {lv}: " + " | ".join(f"{nm} (|coef| {v['median_coef']:.1f}): homogeneity {v['homogeneity']:.2f}, cross-magnitude {v['cross_magnitude_homogeneity']:.2f}, pair cos {v['pair_cos_0p75']:.2f}, coherent {v['coherent_fraction']:.2f}, image-vs-natural cos {v['image_vs_natural_cos']:.2f}, natural id {v['natural_identification']:.2f}, native dual of injected {v['native_dual_injected']:.2f}" for nm, v in rec.items()))
def avg(lv, key): return sum(out[lv][n][key] for n in names[1:]) / 3
record(f"e250_massive_{tag}", dict(model=tag, b=b, L=L, neurons={nm: int(keep[sel[j]]) for j, nm in enumerate(names)}, per_level={str(k): v for k, v in out.items()}), " | ".join(f"lv{lv}: massive (|coef| {out[lv]['massive']['median_coef']:.1f} vs normal {avg(lv, 'median_coef'):.1f}): homogeneity {out[lv]['massive']['homogeneity']:.2f} vs {avg(lv, 'homogeneity'):.2f}, pair cos {out[lv]['massive']['pair_cos_0p75']:.2f} vs {avg(lv, 'pair_cos_0p75'):.2f}, coherent {out[lv]['massive']['coherent_fraction']:.2f} vs {avg(lv, 'coherent_fraction'):.2f}, image-natural cos {out[lv]['massive']['image_vs_natural_cos']:.2f} vs {avg(lv, 'image_vs_natural_cos'):.2f}, natural id {out[lv]['massive']['natural_identification']:.2f} vs {avg(lv, 'natural_identification'):.2f}, native dual {out[lv]['massive']['native_dual_injected']:.2f} vs {avg(lv, 'native_dual_injected'):.2f}" for lv in levels))
