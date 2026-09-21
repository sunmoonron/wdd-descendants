"""e264: functional vs causal equivalence. For pairs of block-2 candidates (the 6 most descendant-similar pairs among
low-write-similarity pairs, and 6 control pairs of low descendant similarity), at tokens where BOTH neurons are active:
KL of removing A, of removing B, of removing both; interaction = KL(AB) - KL(A) - KL(B) normalised by KL(A) + KL(B)
(negative = redundant, positive = synergistic, ~0 = independent); and the cosine between the single-removal logit
footprints. Also, why substitution fails: at B's own tokens, the per-token cosine between the descendant of A's vector
injected there and B's natural footprint, compared with the centroid-level descendant similarity of the pair."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from func_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 16; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; b = 2; D = c.D; lab = c.d["lab"]; A = c.d["A"]; R = A[(lab["type"] == T_MLP) & (lab["block"] == b)].float().to(DEV)
led, tn, tc = dominant(model, arch, c, ids_seq, b); run = make_runner_logits(model, arch, c, ids_seq, [L], NT, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep)
S0i = run(positions=idx); S1i = run(tn, positions=idx); F = S0i[L] - S1i[L]; Cd = unit(centroids(F[idx], lab_i, K)); Rw = unit(R[keep]); Gd, Gw = Cd @ Cd.T, Rw @ Rw.T; iu = torch.triu_indices(K, K, 1, device=DEV); gw, gd = Gw[iu[0], iu[1]], Gd[iu[0], iu[1]]; cand = torch.nonzero(gw < gw.median())[:, 0]
pairs = {"similar": cand[gd[cand].argsort(descending=True)[:6]], "control": cand[gd[cand].argsort()[:6]]}; active = led.abs() >= 0.05 * led.abs().max(1, keepdim=True).values; out = {}
def kl_from(lg0, lg1):
    p0 = torch.log_softmax(lg0, -1); p1 = torch.log_softmax(lg1, -1); return (p0.exp() * (p0 - p1)).sum(1)
for nm, ps in pairs.items():
    recs = []
    for pidx in ps.tolist():
        a_, b_ = int(iu[0][pidx]), int(iu[1][pidx]); na, nb = keep[a_], keep[b_]; both = torch.nonzero(typ & active[:, na] & active[:, nb])[:, 0]
        if len(both) < 10: continue
        neur_a = torch.full((NT,), int(na), device=DEV); neur_b = torch.full((NT,), int(nb), device=DEV); Sc = run(positions=both); Sa = run(neur_a, positions=both, ablate_positions=both); Sb = run(neur_b, positions=both, ablate_positions=both)
        # both: ablate A then B via two pre-hooks is not supported by the runner; emulate by zeroing both neurons through a combined neuron tensor trick: run twice is not equal; use a custom hook
        st = {}; hs = [arch.layers[L].register_forward_hook(lambda m, i, o: st.__setitem__(L, (o[0] if isinstance(o, tuple) else o).detach().float().reshape(-1, D)))]
        def pre(m, inp):
            x = inp[0].clone(); flat = x.reshape(-1, c.DFF); flat[both, na] = 0; flat[both, nb] = 0; return (flat.reshape(x.shape),)
        hs.append(arch.mlp_lin(b).register_forward_pre_hook(pre)); o = model(ids_seq); [h.remove() for h in hs]; lg_ab = o.logits.reshape(NT, -1).float()[both]; del o
        kla, klb, klab = kl_from(Sc["lg"], Sa["lg"]), kl_from(Sc["lg"], Sb["lg"]), kl_from(Sc["lg"], lg_ab); inter = ((klab - kla - klb) / (kla + klb).clamp_min(1e-6)); fa, fb = Sc["lg"] - Sa["lg"], Sc["lg"] - Sb["lg"]; fa = fa - fa.mean(1, keepdim=True); fb = fb - fb.mean(1, keepdim=True); cosf = ((fa * fb).sum(1) / (fa.norm(dim=1) * fb.norm(dim=1)).clamp_min(1e-9))
        # substitution divergence at B's tokens: inject A's vector with B's coefficient, compare with B's natural footprint per token
        tb = idx[lab_i == b_]; inj = torch.zeros(NT, D, device=DEV); inj[tb] = tc[tb][:, None] * R[na]; Ss = run(positions=tb, inject=inj, inject_block=b + 1); Fs = (Ss[L] - S0[L])[tb]; Fb_nat = F[tb]; per_tok = ((Fs * Fb_nat).sum(1) / (Fs.norm(dim=1) * Fb_nat.norm(dim=1)).clamp_min(1e-9))
        recs.append(dict(pair=(int(na), int(nb)), n_both=int(len(both)), desc_sim=gd[pidx].item(), write_sim=gw[pidx].item(), kl_a=kla.median().item(), kl_b=klb.median().item(), kl_ab=klab.median().item(), interaction=inter.median().item(), footprint_cos=cosf.median().item(), substitution_pertoken_cos=per_tok.median().item()))
    out[nm] = recs
    import numpy as np
    log(f"{tag} {nm} pairs ({len(recs)}): descendant sim {np.mean([r['desc_sim'] for r in recs]):.2f} | single-removal logit-footprint cos {np.mean([r['footprint_cos'] for r in recs]):.2f} | joint-removal interaction (KL(AB)-KL(A)-KL(B))/(KL(A)+KL(B)) {np.mean([r['interaction'] for r in recs]):+.2f} (per pair " + " ".join(f"{r['interaction']:+.2f}" for r in recs) + f") | per-token cos of A-injected-at-B's-tokens with B's natural footprint {np.mean([r['substitution_pertoken_cos'] for r in recs]):.2f} vs centroid sim {np.mean([r['desc_sim'] for r in recs]):.2f}")
import numpy as np
summ = {nm: dict(desc_sim=float(np.mean([r['desc_sim'] for r in recs])), footprint_cos=float(np.mean([r['footprint_cos'] for r in recs])), interaction=float(np.mean([r['interaction'] for r in recs])), substitution_cos=float(np.mean([r['substitution_pertoken_cos'] for r in recs]))) for nm, recs in out.items() if recs}
record(f"e264_interact_{tag}", dict(model=tag, b=b, L=L, K=K, pairs=out, summary=summ), " | ".join(f"{nm}: desc sim {v['desc_sim']:.2f}, footprint cos {v['footprint_cos']:.2f}, interaction {v['interaction']:+.2f}, per-token substitution cos {v['substitution_cos']:.2f}" for nm, v in summ.items()))
