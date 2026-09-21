"""e280: the counterfactual descendant as a trajectory. For four pairs (A, B) of candidates with the most different
descendants, the interpolated descendant (1-a) d_A + a d_B (a = 0, 1/4, 1/2, 3/4, 1) is injected at the block-(L+1)
input of foreign tokens (where neither A nor B is written) at the natural footprint norm, and the network runs on.
At each later level and at the logits the response is compared with the interpolation of the endpoint responses:
cosine to the interpolant, the fitted position a-hat from a two-endpoint least-squares fit (snapping to a regime
shows as a-hat pushed toward 0 or 1), and the fraction of the response outside the endpoint plane."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from func_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; b = 2; D = c.D; NB = c.NB; levels = sorted({lv for lv in (L + 2, L + 4, (L + NB - 1) // 2, NB - 2) if L + 1 < lv < NB})
led, tn, tc = dominant(model, arch, c, ids_seq, b); run = make_runner_logits(model, arch, c, ids_seq, [L] + levels, NT, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep)
S1 = run(tn); F = (S0[L] - S1[L])[idx]; Cd = centroids(F, lab_i, K); fnorm = F.norm(dim=1).median(); Gm = Cd @ Cd.T; pairs = []; used = set()
for i, j in [(int(p // K), int(p % K)) for p in Gm.flatten().argsort()]:
    if i < j and i not in used and j not in used: pairs.append((i, j)); used |= {i, j}
    if len(pairs) == 4: break
foreign = torch.nonzero(typ & ~torch.isin(tn, keep))[:, 0][:1536]; NF = len(foreign); base = run(positions=foreign); lg0 = base["lg"]; alphas = [0.0, 0.25, 0.5, 0.75, 1.0]; out = {}
def fit(Ra, R0, R1):
    # per-token least squares of Ra on [R0, R1]: coefficients c0, c1
    a00 = (R0 * R0).sum(1); a01 = (R0 * R1).sum(1); a11 = (R1 * R1).sum(1); b0 = (Ra * R0).sum(1); b1 = (Ra * R1).sum(1); det = (a00 * a11 - a01 ** 2).clamp_min(1e-9); c0 = (a11 * b0 - a01 * b1) / det; c1 = (a00 * b1 - a01 * b0) / det; fitv = c0[:, None] * R0 + c1[:, None] * R1; return c0, c1, 1 - ((Ra - fitv) ** 2).sum(1) / (Ra ** 2).sum(1).clamp_min(1e-9)
for (i, j) in pairs:
    resp = {}
    for al in alphas:
        v = (1 - al) * Cd[i] + al * Cd[j]; inj = torch.zeros(NT, D, device=DEV); inj[foreign] = fnorm * v[None]; r = run(positions=foreign, inject=inj, inject_block=L + 1); dl = r["lg"] - lg0; resp[al] = {lv: (r[lv] - S0[lv])[foreign] for lv in levels}; resp[al]["lg"] = dl - dl.mean(1, keepdim=True)
    rec = {}
    for key in levels + ["lg"]:
        R0, R1 = resp[0.0][key], resp[1.0][key]; rec[str(key)] = {}
        for al in (0.25, 0.5, 0.75):
            Ra = resp[al][key]; lin = (1 - al) * R0 + al * R1; cos = ((Ra * lin).sum(1) / (Ra.norm(dim=1) * lin.norm(dim=1)).clamp_min(1e-9)); c0, c1, r2 = fit(Ra, R0, R1); ahat = c1 / (c0 + c1).clamp_min(1e-9)
            rec[str(key)][str(al)] = dict(cos_to_interpolant=cos.median().item(), alpha_hat=ahat.median().item(), alpha_hat_q25=ahat.quantile(0.25).item(), alpha_hat_q75=ahat.quantile(0.75).item(), in_plane_r2=r2.median().item(), endpoint_cos=((R0 * R1).sum(1) / (R0.norm(dim=1) * R1.norm(dim=1)).clamp_min(1e-9)).median().item())
    out[f"{int(keep[i])}-{int(keep[j])}"] = rec
    log(f"{tag} pair {int(keep[i])}-{int(keep[j])} (descendant cos {Gm[i, j]:.2f}): " + " | ".join(f"{key}: cos-to-interpolant " + "/".join(f"{rec[str(key)][str(al)]['cos_to_interpolant']:.2f}" for al in (0.25, 0.5, 0.75)) + ", alpha-hat " + "/".join(f"{rec[str(key)][str(al)]['alpha_hat']:.2f}" for al in (0.25, 0.5, 0.75)) + f", in-plane R2 {rec[str(key)]['0.5']['in_plane_r2']:.2f}, endpoint cos {rec[str(key)]['0.5']['endpoint_cos']:.2f}" for key in levels + ["lg"]))
import numpy as np
agg = {str(key): {str(al): {m: float(np.mean([out[p][str(key)][str(al)][m] for p in out])) for m in ("cos_to_interpolant", "alpha_hat", "in_plane_r2", "endpoint_cos")} for al in (0.25, 0.5, 0.75)} for key in levels + ["lg"]}
record(f"e280_trajectory_{tag}", dict(model=tag, b=b, L=L, levels=levels, pairs=list(out.keys()), per_pair=out, mean=agg), " | ".join(f"{key}: cos " + "/".join(f"{agg[str(key)][str(al)]['cos_to_interpolant']:.2f}" for al in (0.25, 0.5, 0.75)) + ", alpha-hat " + "/".join(f"{agg[str(key)][str(al)]['alpha_hat']:.2f}" for al in (0.25, 0.5, 0.75)) + f", in-plane R2 {agg[str(key)]['0.5']['in_plane_r2']:.2f}, endpoint cos {agg[str(key)]['0.5']['endpoint_cos']:.2f}" for key in levels + ["lg"]))
