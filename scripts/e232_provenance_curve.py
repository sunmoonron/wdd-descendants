"""e232: the provenance transport curve. Block b = 2, neurons dominant at >= 12 typical tokens (more classes). At
levels b+1, b+2, b+4, b+6, L, NB-2: source identification by (1) the native atom read (dual@64 at that level),
(2) nearest descendant centroid on the centered state, (3) nearest centroid on the footprint; and the mutual
information between source and prediction (bits) for (2) and (3) against log2 K. Split: a crossover where the
native read gives way to the descendant signature vs no regime where the state-level signature is informative."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from desc_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 16; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; NB = c.NB; b = 2; levels = sorted({b + 1, b + 2, b + 4, b + 6, L, NB - 2}); levels = [lv for lv in levels if lv < NB]; run = make_runner(model, arch, c, ids_seq, levels, NT)
led, tn, tc = dominant(model, arch, c, ids_seq, b); S0 = run(); S1 = run(b, tn); big = tc.abs() >= tc.abs().quantile(0.5); out = {}
for lv in levels:
    typ = typical_mask(S0[lv]); idx, lab_i, keep = classes(tn, typ & big, 12); K = len(keep)
    if K < 3: continue
    torch.manual_seed(0); split = torch.rand(len(idx), device=DEV) < 0.5; tr, te = idx[split], idx[~split]; ltr, lte = lab_i[split], lab_i[~split]; F = S0[lv] - S1[lv]; cents = centroids(F[tr], ltr, K); X = S0[lv] - c.s["mu"][lv + 1].to(DEV)
    Alv, lablv = c.dictionary(lv); S = Alv.T @ Alv; ev, V = torch.linalg.eigh(S); Winv = V @ torch.diag(1 / (ev + 1e-2 * ev[-1])) @ V.T; sel = oneshot(X[te], Alv, 64, whiten=Winv)[0]; row = c.atom_index(lv, torch.full_like(tn.cpu(), b), tn.cpu()).to(DEV)[te]; native = (sel == row[:, None]).any(1).float().mean().item()
    out[lv] = dict(K=K, n_test=int(len(te)), native_read=native, state_centroid=accuracy(X[te], cents, lte), footprint_centroid=accuracy(F[te], cents, lte), mi_state_bits=mutual_info_bits(X[te], cents, lte, K), mi_footprint_bits=mutual_info_bits(F[te], cents, lte, K), log2K=math.log2(K), chance=1.0 / K)
    log(f"{tag} level {lv} (K {K}, chance {1 / K:.2f}, log2 K {math.log2(K):.1f} bits): native atom read {native:.2f} | state via descendant centroid {out[lv]['state_centroid']:.2f} ({out[lv]['mi_state_bits']:.2f} bits) | footprint {out[lv]['footprint_centroid']:.2f} ({out[lv]['mi_footprint_bits']:.2f} bits)")
record(f"e232_curve_{tag}", dict(model=tag, b=b, L=L, per_level={str(k): v for k, v in out.items()}), "level: native / state-centroid / footprint (MI bits of log2K) -> " + " | ".join(f"{lv}: {v['native_read']:.2f} / {v['state_centroid']:.2f} / {v['footprint_centroid']:.2f} ({v['mi_state_bits']:.1f}, {v['mi_footprint_bits']:.1f} of {v['log2K']:.1f})" for lv, v in out.items()))
