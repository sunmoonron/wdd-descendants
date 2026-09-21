"""e237: provenance capacity vs depth. Block b = 2, candidate sets of the K most frequent dominant neurons
(K = 8, 16, 32, 64, 128, 256, each with >= 8 tokens), levels b+1, b+2, b+4, b+6, L. Source identification from the
state (descendant centroids) and from the footprint, and the mutual information between source and prediction in
bits, against log2 K. Looks for the depth at which added candidates stop adding bits."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from desc_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 32; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; NB = c.NB; b = 2; levels = [lv for lv in sorted({b + 1, b + 2, b + 4, b + 6, L}) if lv < NB]; run = make_runner(model, arch, c, ids_seq, levels, NT)
led, tn, tc = dominant(model, arch, c, ids_seq, b); S0 = run(); S1 = run(b, tn); big = tc.abs() >= tc.abs().quantile(0.5); out = {}
for lv in levels:
    typ = typical_mask(S0[lv]); idx0 = torch.nonzero(typ & big)[:, 0]; neur0 = tn[idx0]; u, cnt = torch.unique(neur0, return_counts=True); order = cnt.argsort(descending=True); u, cnt = u[order], cnt[order]; u = u[cnt >= 8]; F = S0[lv] - S1[lv]; X = S0[lv] - c.s["mu"][lv + 1].to(DEV); rec = {}
    for K in (8, 16, 32, 64, 128, 256):
        if K > len(u): break
        keep = u[:K]; m = torch.isin(neur0, keep); idx = idx0[m]; lab_i = (neur0[m][:, None] == keep[None, :]).float().argmax(1); torch.manual_seed(0); split = torch.rand(len(idx), device=DEV) < 0.5; tr, te = idx[split], idx[~split]; ltr, lte = lab_i[split], lab_i[~split]; cents = centroids(F[tr], ltr, K)
        rec[K] = dict(state_acc=accuracy(X[te], cents, lte), footprint_acc=accuracy(F[te], cents, lte), state_bits=mutual_info_bits(X[te], cents, lte, K), footprint_bits=mutual_info_bits(F[te], cents, lte, K), n_test=int(len(te)))
    out[lv] = rec
    log(f"{tag} level {lv}: K -> state acc (bits) / footprint acc (bits): " + " ".join(f"{K}: {v['state_acc']:.2f} ({v['state_bits']:.1f}) / {v['footprint_acc']:.2f} ({v['footprint_bits']:.1f})" for K, v in rec.items()))
record(f"e237_capacity_{tag}", dict(model=tag, b=b, L=L, per_level={str(k): {str(kk): vv for kk, vv in v.items()} for k, v in out.items()}), " | ".join(f"lv{lv}: state bits " + " ".join(f"K{K}:{v['state_bits']:.1f}" for K, v in rec.items()) + "; footprint bits " + " ".join(f"K{K}:{v['footprint_bits']:.1f}" for K, v in rec.items()) for lv, rec in out.items()))
