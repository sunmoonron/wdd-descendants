"""e251: is the downstream descendant a change of coding basis? At birth (the write itself) and at +2, +4 and L, the
descendants of the dominant block-2 writes (natural footprints) are coded with an equal budget of k coefficients
(k = 1, 8, 32, 128) in four bases: (a) the native sparse dictionary (OMP over blocks 0..level, k atoms per token);
(b) the descendant basis (top-k principal directions of the footprint cloud, fitted on a train half, dense
coefficients); (c) PCA of the state itself (top-k components of the level's state covariance); (d) k random
orthogonal directions. Reported: fraction of variance unexplained per basis and k, and provenance identification
(nearest centroid in the k-dimensional code) under (b), (c), (d) and under the native code (nearest native atom
among the k selected). A sparse-to-distributed recoding shows as the native code winning at birth and the dense
bases winning downstream."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from desc_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 16; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; b = 2; D = c.D; levels = sorted({b, b + 2, b + 4, L}); run = make_runner(model, arch, c, ids_seq, levels, NT); lab = c.d["lab"]; A = c.d["A"]; R = A[(lab["type"] == T_MLP) & (lab["block"] == b)].float().to(DEV)
led, tn, tc = dominant(model, arch, c, ids_seq, b); S0 = run(); S1 = run(b, tn); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep); torch.manual_seed(0); split = torch.rand(len(idx), device=DEV) < 0.5; tr, te = idx[split], idx[~split]; ltr, lte = lab_i[split], lab_i[~split]
out = {}
for lv in levels:
    F = S0[lv] - S1[lv] if lv > b else tc[:, None] * R[tn]                                                   # at birth the object is the write itself
    Alv, lablv = c.dictionary(lv); Xs = c.X(lv)[sub(c.NT, 8192, seed=3)]; Sig = Xs.T @ Xs / len(Xs); evS, VS = torch.linalg.eigh(Sig); Fc = F[tr] - F[tr].mean(0, keepdim=True); Uf = torch.linalg.svd(Fc, full_matrices=False)[2]; rec = {}
    row_map = c.atom_index(lv, torch.full_like(tn.cpu(), b), tn.cpu()).to(DEV)
    for k in (1, 8, 32, 128):
        sel, cof, err = omp(F[te], Alv, k); fvu_native = fvu(err[:, -1], F[te]); native_id = (sel == row_map[te][:, None]).any(1).float().mean().item()
        r = dict(fvu_native=fvu_native, id_native=native_id)
        for nm, B in (("descendant_pca", Uf[:k].T), ("state_pca", VS[:, -k:]), ("random", torch.linalg.qr(torch.randn(D, k, device=DEV))[0])):
            Ptr, Pte = F[tr] @ B, F[te] @ B; r["fvu_" + nm] = 1 - (Pte ** 2).sum().item() / (F[te] ** 2).sum().item(); r["id_" + nm] = accuracy(Pte, centroids(Ptr, ltr, K), lte)
        rec[k] = r
    out[lv] = rec
    log(f"{tag} level {lv}{' (birth)' if lv == b else ''} (K {K}): k -> FVU native/descPCA/statePCA/random, id native/descPCA/statePCA/random: " + " ".join(f"{k}: {v['fvu_native']:.2f}/{v['fvu_descendant_pca']:.2f}/{v['fvu_state_pca']:.2f}/{v['fvu_random']:.2f}, {v['id_native']:.2f}/{v['id_descendant_pca']:.2f}/{v['id_state_pca']:.2f}/{v['id_random']:.2f}" for k, v in rec.items()))
record(f"e251_coding_{tag}", dict(model=tag, b=b, L=L, K=K, per_level={str(lv): {str(k): v for k, v in rec.items()} for lv, rec in out.items()}), " | ".join(f"lv{lv}: k=8 FVU native {rec[8]['fvu_native']:.2f} vs descPCA {rec[8]['fvu_descendant_pca']:.2f} vs statePCA {rec[8]['fvu_state_pca']:.2f}; k=32 id native {rec[32]['id_native']:.2f} vs descPCA {rec[32]['id_descendant_pca']:.2f} vs random {rec[32]['id_random']:.2f}" for lv, rec in out.items()))
