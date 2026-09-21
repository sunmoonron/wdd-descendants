"""e273: is descendant-to-function a quotient map? For three descendant centroids (the largest-coefficient candidate
and two typical ones), inject unit(d + 0.5 u) scaled to the natural footprint norm at the block-(L+1) input, for
directions u drawn from five classes (top-8, middle-8 and bottom-8 principal components of the descendant cloud, 8
random directions, 8 unit differences to other descendants), one direction class per pass with the 8 directions
spread over the foreign tokens. Reported per class: the relative change of the logit effect, ||E(d+0.5u) - E(d)|| /
||E(d)||, at a constant descendant displacement (the injected unit vectors differ from unit(d) by ~0.45 for every u).
A quotient map shows as classes with near-zero functional response; isotropy shows as equal responses."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from func_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; b = 2; D = c.D
led, tn, tc = dominant(model, arch, c, ids_seq, b); run = make_runner_logits(model, arch, c, ids_seq, [L], NT, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep)
S0i = run(positions=idx); S1i = run(tn, positions=idx); F = S0i[L] - S1i[L]; Cd = unit(centroids(F[idx], lab_i, K)); fnorm = F[idx].norm(dim=1).median(); med = torch.stack([tc[idx][lab_i == k].median().abs() for k in range(K)]); cnts = torch.bincount(lab_i, minlength=K); picks = [int(med.argmax()), int(cnts.argmax()), int(cnts.argsort(descending=True)[1])]
Fc = F[idx] - F[idx].mean(0, keepdim=True); U = torch.linalg.svd(Fc, full_matrices=False)[2]; r = U.shape[0]; torch.manual_seed(0)
foreign = torch.nonzero(typ & ~torch.isin(tn, keep))[:, 0][:1536]; NF = len(foreign); base = run(positions=foreign)["lg"]
def effect(vecs_per_token):
    inj = torch.zeros(NT, D, device=DEV); inj[foreign] = fnorm * vecs_per_token; d = run(positions=foreign, inject=inj, inject_block=L + 1)["lg"] - base; return d - d.mean(1, keepdim=True)
out = {}
for k in picks:
    d = Cd[k]; E0 = effect(d[None].expand(NF, -1)); n0 = E0.norm(dim=1); classes_u = {"pc_top8": U[:8], "pc_mid8": U[r // 2: r // 2 + 8], "pc_bottom8": U[-8:], "random8": unit(torch.randn(8, D, device=DEV)), "desc_diff8": unit(Cd[torch.randperm(K, device=DEV)[:8]] - d[None])}; rec = {}
    for nm, Us in classes_u.items():
        a = torch.randint(0, 8, (NF,), device=DEV); u = Us[a]; v = unit(d[None] + 0.5 * u); E1 = effect(v); rec[nm] = dict(response=((E1 - E0).norm(dim=1) / n0.clamp_min(1e-9)).median().item(), desc_displacement=(v - d[None]).norm(dim=1).median().item())
    out[int(keep[k])] = rec
    log(f"{tag} descendant of neuron {int(keep[k])} (|coef| {med[k]:.1f}): relative functional response per direction class (descendant displacement ~{rec['random8']['desc_displacement']:.2f}): " + " ".join(f"{nm} {v['response']:.2f}" for nm, v in rec.items()))
import numpy as np
agg = {nm: float(np.mean([out[n][nm]["response"] for n in out])) for nm in ("pc_top8", "pc_mid8", "pc_bottom8", "random8", "desc_diff8")}
record(f"e273_funcnull_{tag}", dict(model=tag, b=b, L=L, K=K, per_descendant={str(k): v for k, v in out.items()}, mean=agg), "relative functional response to a fixed descendant displacement, by direction class: " + " ".join(f"{nm} {v:.2f}" for nm, v in agg.items()) + f" | max/min ratio {max(agg.values()) / max(min(agg.values()), 1e-6):.1f}")
