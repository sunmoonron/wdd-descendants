"""e271: does the descendant coordinate parameterise the intervention effect continuously? For 8 pairs of candidates
(4 high-descendant-similarity / low-write-similarity, 4 low/low), inject alpha*D_A + (1-alpha)*D_B (unit centroids
at L, scaled to the median natural footprint norm) at the block-(L+1) input at foreign tokens for alpha = 0, 0.25,
0.5, 0.75, 1; the logit effect of the interpolant vs the linear interpolation of the endpoint effects (cosine and
relative error), and the monotonic drift of cos(effect, effect(D_A)) with alpha. Also: relative effect distance vs
relative descendant distance across all candidate pairs (slope and Spearman)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from func_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; b = 2; D = c.D; lab = c.d["lab"]; A = c.d["A"]; R = A[(lab["type"] == T_MLP) & (lab["block"] == b)].float().to(DEV)
led, tn, tc = dominant(model, arch, c, ids_seq, b); run = make_runner_logits(model, arch, c, ids_seq, [L], NT, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep)
S0i = run(positions=idx); S1i = run(tn, positions=idx); F = S0i[L] - S1i[L]; Cd = unit(centroids(F[idx], lab_i, K)); fnorm = F[idx].norm(dim=1).median(); Rw = unit(R[keep]); Gd, Gw = Cd @ Cd.T, Rw @ Rw.T; iu = torch.triu_indices(K, K, 1, device=DEV); gw, gd = Gw[iu[0], iu[1]], Gd[iu[0], iu[1]]; cand = torch.nonzero(gw < gw.median())[:, 0]
pairs = torch.cat([cand[gd[cand].argsort(descending=True)[:4]], cand[gd[cand].argsort()[:4]]]); foreign = torch.nonzero(typ & ~torch.isin(tn, keep))[:, 0][:1536]; NF = len(foreign); base = run(positions=foreign)["lg"]
def effect(vec):
    inj = torch.zeros(NT, D, device=DEV); inj[foreign] = fnorm * vec[None]; d = run(positions=foreign, inject=inj, inject_block=L + 1)["lg"] - base; d = d - d.mean(1, keepdim=True); return d
alphas = (0.0, 0.25, 0.5, 0.75, 1.0); out = []
for pidx in pairs.tolist():
    a_, b_ = int(iu[0][pidx]), int(iu[1][pidx]); EA, EB = effect(Cd[a_]), effect(Cd[b_]); rec = dict(pair=(int(keep[a_]), int(keep[b_])), desc_sim=gd[pidx].item(), write_sim=gw[pidx].item(), endpoint_effect_cos=((EA * EB).sum(1) / (EA.norm(dim=1) * EB.norm(dim=1)).clamp_min(1e-9)).median().item(), interp=[])
    for al in alphas[1:-1]:
        E = effect(unit(al * Cd[a_] + (1 - al) * Cd[b_])[0] if False else (al * Cd[a_] + (1 - al) * Cd[b_]) / (al * Cd[a_] + (1 - al) * Cd[b_]).norm()); lin = al * EA + (1 - al) * EB; rec["interp"].append(dict(alpha=al, cos_to_linear=((E * lin).sum(1) / (E.norm(dim=1) * lin.norm(dim=1)).clamp_min(1e-9)).median().item(), rel_err=((E - lin).norm(dim=1) / lin.norm(dim=1).clamp_min(1e-9)).median().item(), cos_to_A=((E * EA).sum(1) / (E.norm(dim=1) * EA.norm(dim=1)).clamp_min(1e-9)).median().item()))
    out.append(rec)
    log(f"{tag} pair {rec['pair']} (desc sim {rec['desc_sim']:.2f}, write sim {rec['write_sim']:.2f}): endpoint effect cos {rec['endpoint_effect_cos']:.2f} | interpolants: cos to linear " + " ".join(f"{r['cos_to_linear']:.2f}" for r in rec['interp']) + ", rel err " + " ".join(f"{r['rel_err']:.2f}" for r in rec['interp']) + ", cos to A " + " ".join(f"{r['cos_to_A']:.2f}" for r in rec['interp']))
# global parameterisation: effect distance vs descendant distance over all candidate pairs (from single-candidate effect centroids)
Eff = torch.stack([unit(effect(Cd[k]).mean(0, keepdim=True))[0] for k in range(K)]); Ge = Eff @ Eff.T; ge = Ge[iu[0], iu[1]]; de = (2 - 2 * ge).clamp_min(0).sqrt(); dd = (2 - 2 * gd).clamp_min(0).sqrt(); ra = de.argsort().argsort().float(); rb = dd.argsort().argsort().float(); rho = torch.corrcoef(torch.stack([ra, rb]))[0, 1].item(); slope = ((de - de.mean()) * (dd - dd.mean())).sum().item() / ((dd - dd.mean()) ** 2).sum().item()
import numpy as np
hi = [r for r in out[:4]]; lo = [r for r in out[4:]]
record(f"e271_interp_{tag}", dict(model=tag, b=b, L=L, K=K, pairs=out, global_spearman_effectdist_vs_descdist=rho, global_slope=slope), f"K {K}: effect distance vs descendant distance over all pairs Spearman {rho:+.2f}, slope {slope:.2f} | high-D pairs: endpoint effect cos {np.mean([r['endpoint_effect_cos'] for r in hi]):.2f}, interpolant cos to linear {np.mean([x['cos_to_linear'] for r in hi for x in r['interp']]):.2f}, rel err {np.mean([x['rel_err'] for r in hi for x in r['interp']]):.2f} | low-D pairs: endpoint cos {np.mean([r['endpoint_effect_cos'] for r in lo]):.2f}, interpolant cos to linear {np.mean([x['cos_to_linear'] for r in lo for x in r['interp']]):.2f}, rel err {np.mean([x['rel_err'] for r in lo for x in r['interp']]):.2f}")
