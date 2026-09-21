"""e279: coordinate transition or continuous drift? At every block output after the write: the overlap between the
identity subspace (top-d eigenvectors of the between-neuron scatter of descendants) and the function subspace (top-d
PLS directions to the logit footprints), the rotation of each subspace between consecutive blocks (energy of the
block-j subspace inside the block-(j+1) subspace), and the held-out identification. Chance d/D. A sharp drop in the
consecutive-block overlap at one depth is a transition; a steady value is drift."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from func_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 16; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; NB = c.NB; b = 2; D = c.D; levels = list(range(b + 1, NB))
led, tn, tc = dominant(model, arch, c, ids_seq, b); run = make_runner_logits(model, arch, c, ids_seq, levels, NT, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep)
S0 = run(positions=idx); S1 = run(tn, positions=idx); dl = S0["lg"] - S1["lg"]; dl = dl - dl.mean(1, keepdim=True); G = dl @ dl.T; evg, Vg = torch.linalg.eigh(G); Z = Vg.flip(1)[:, :256] * evg.flip(0)[:256].clamp_min(0).sqrt()[None]; Z = Z - Z.mean(0, keepdim=True)
torch.manual_seed(0); split = torch.rand(len(idx), device=DEV) < 0.5
def inside(Ua, Ub): return ((Ub.T @ Ua) ** 2).sum().item() / Ua.shape[1]
prev = {}; out = {}
for lv in levels:
    F = (S0[lv] - S1[lv])[idx]; Fc = F - F.mean(0, keepdim=True); cents = torch.stack([Fc[lab_i == k].mean(0) for k in range(K)]); w = torch.bincount(lab_i, minlength=K).float(); Sb = (cents * w[:, None]).T @ cents / w.sum(); ev, V = torch.linalg.eigh(Sb); U_id = V.flip(1); U_fn = torch.linalg.svd(Fc.T @ Z, full_matrices=False)[0]
    rec = dict(chance8=8 / D, chance32=32 / D, identification=accuracy(F[~split], centroids(F[split], lab_i[split], K), lab_i[~split]))
    for d in (8, 32):
        rec[f"fn_in_id_{d}"] = inside(U_fn[:, :d], U_id[:, :d]); rec[f"id_rot_{d}"] = inside(U_id[:, :d], prev[f"id{d}"]) if f"id{d}" in prev else None; rec[f"fn_rot_{d}"] = inside(U_fn[:, :d], prev[f"fn{d}"]) if f"fn{d}" in prev else None; prev[f"id{d}"] = U_id[:, :d]; prev[f"fn{d}"] = U_fn[:, :d]
    out[lv] = rec; log(f"{tag} block {lv}: identification {rec['identification']:.2f} | function-in-identity d8 {rec['fn_in_id_8']:.2f} d32 {rec['fn_in_id_32']:.2f} | identity kept from previous block d8 {rec['id_rot_8'] if rec['id_rot_8'] is None else round(rec['id_rot_8'], 2)} d32 {rec['id_rot_32'] if rec['id_rot_32'] is None else round(rec['id_rot_32'], 2)} | function kept d8 {rec['fn_rot_8'] if rec['fn_rot_8'] is None else round(rec['fn_rot_8'], 2)} d32 {rec['fn_rot_32'] if rec['fn_rot_32'] is None else round(rec['fn_rot_32'], 2)} (chance {8 / D:.3f}, {32 / D:.3f})")
record(f"e279_coredepth_{tag}", dict(model=tag, b=b, L=L, K=K, per_block={str(k): v for k, v in out.items()}), "block: id | fn-in-id d8/d32 | id kept d8/d32 | fn kept d8/d32 :: " + " ; ".join(f"{lv}: {v['identification']:.2f} | {v['fn_in_id_8']:.2f}/{v['fn_in_id_32']:.2f} | " + ("-" if v['id_rot_8'] is None else f"{v['id_rot_8']:.2f}/{v['id_rot_32']:.2f}") + " | " + ("-" if v['fn_rot_8'] is None else f"{v['fn_rot_8']:.2f}/{v['fn_rot_32']:.2f}") for lv, v in out.items()))
