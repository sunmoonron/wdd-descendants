"""e242: can an unseen neuron's descendant be predicted from its write vector alone? Blocks b = 2 and 3; candidate
neurons (>= 20 natural tokens). The transplant image of each candidate (its median write vector injected at foreign
tokens, averaged) is computed WITHOUT any natural footprint of that neuron. Natural footprints (all tokens) are
classified against (i) the transplant images (zero-shot) and (ii) natural centroids fitted on a train half
(standard), at levels b+2 and L; plus the cosine between each neuron's transplant image and its natural centroid.
Kill: zero-shot accuracy below twice chance."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from desc_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 16; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; lab = c.d["lab"]; A = c.d["A"]; out = {}
for b in (2, 3):
    levels = sorted({b + 2, L}); run = make_runner(model, arch, c, ids_seq, levels, NT); R = A[(lab["type"] == T_MLP) & (lab["block"] == b)].float().to(DEV); led, tn, tc = dominant(model, arch, c, ids_seq, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); S1 = run(b, tn); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep)
    med = torch.stack([tc[idx][lab_i == k].median() for k in range(K)]); foreign = torch.nonzero(typ & ~torch.isin(torch.arange(NT, device=DEV), idx))[:, 0]; acc_img = {lv: torch.zeros(K, c.D, device=DEV) for lv in levels}; cnt = torch.zeros(K, device=DEV); torch.manual_seed(0)
    for p in range(3):
        assign = torch.randint(0, K, (len(foreign),), device=DEV); act = (led[foreign].abs() >= 0.05 * led[foreign].abs().max(1, keepdim=True).values); ok = ~act[torch.arange(len(foreign), device=DEV), keep[assign]]; f2, a2 = foreign[ok], assign[ok]
        inj = torch.zeros(NT, c.D, device=DEV); inj[f2] = med[a2][:, None] * R[keep[a2]]; S2 = run(inject=inj, inject_block=b + 1)
        for lv in levels: acc_img[lv].index_add_(0, a2, (S2[lv] - S0[lv])[f2])
        cnt.index_add_(0, a2, torch.ones(len(a2), device=DEV))
    split = torch.rand(len(idx), device=DEV) < 0.5; rec = {}
    for lv in levels:
        F = S0[lv] - S1[lv]; timg = unit(acc_img[lv] / cnt.clamp_min(1)[:, None]); nat = centroids(F[idx[split]], lab_i[split], K)
        rec[lv] = dict(K=K, chance=1.0 / K, zero_shot_all=accuracy(F[idx], timg, lab_i), zero_shot_heldout=accuracy(F[idx[~split]], timg, lab_i[~split]), standard_heldout=accuracy(F[idx[~split]], nat, lab_i[~split]), image_vs_natural_cos=((timg * nat).sum(1)).median().item())
        log(f"{tag} born b{b} level {lv} (K {K}, chance {1 / K:.2f}): zero-shot from transplant images {rec[lv]['zero_shot_heldout']:.2f} (all tokens {rec[lv]['zero_shot_all']:.2f}) vs standard natural centroids {rec[lv]['standard_heldout']:.2f} | cos(transplant image, natural centroid) {rec[lv]['image_vs_natural_cos']:.2f}")
    out[b] = rec
record(f"e242_zeroshot_{tag}", dict(model=tag, L=L, per_birth={str(k): {str(kk): vv for kk, vv in v.items()} for k, v in out.items()}), " | ".join(f"b{b} lv{lv}: zero-shot {v['zero_shot_heldout']:.2f} vs standard {v['standard_heldout']:.2f} (chance {v['chance']:.2f}), image-centroid cos {v['image_vs_natural_cos']:.2f}" for b, rec in out.items() for lv, v in rec.items()))
