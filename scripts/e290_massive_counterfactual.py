"""e290: the theory's untested prediction for the massive neuron. If its own tokens form a context class with a
different Jacobian (a coherent reaction), then adding more of the massive direction on top of the existing write at
its own tokens must produce a response aligned with the natural (removal) footprint and anti-parallel to the
transplant image, while the same addition at foreign tokens gives the transplant. For the largest-coefficient
candidate and a typical one: cosines of the own-token addition response (at 0.5x and 1x the natural coefficient)
with the natural footprint centroid, with the foreign-token response, and with the linear transport."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from desc_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; b = 2; D = c.D; lab = c.d["lab"]; A = c.d["A"]; R = A[(lab["type"] == T_MLP) & (lab["block"] == b)].float().to(DEV); run = make_runner(model, arch, c, ids_seq, [L], NT)
led, tn, tc = dominant(model, arch, c, ids_seq, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); S1 = run(b, tn); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep); F = S0[L] - S1[L]; smed = torch.stack([tc[idx][lab_i == k].median() for k in range(K)]); med = smed.abs(); sgn = torch.sign(smed); order = med.argsort(); picks = {"largest": int(med.argmax()), "typical": int(order[len(order) // 2])}
pool = torch.nonzero(typ)[:, 0]; s_inj = tc.abs().median(); torch.manual_seed(0); Vr = unit(torch.randn(1024, D, device=DEV)); X = []; Y = []
for p in range(2):
    a = torch.randint(0, 1024, (len(pool),), device=DEV); inj = torch.zeros(NT, D, device=DEV); inj[pool] = s_inj * Vr[a]; S2 = run(inject=inj, inject_block=b + 1); X.append(Vr[a]); Y.append((S2[L] - S0[L])[pool] / s_inj)
X, Y = torch.cat(X), torch.cat(Y); G = X.T @ X; T = torch.linalg.solve(G + 1e-2 * G.diagonal().mean() * torch.eye(D, device=DEV), X.T @ Y)
foreign = torch.nonzero(typ & ~torch.isin(tn, keep))[:, 0]; out = {}
for nm, k in picks.items():
    own = idx[lab_i == k]; w = unit(R[keep[k]][None])[0]; nat = unit(F[own].mean(0, keepdim=True))[0]; tw = unit((sgn[k] * (R[keep[k]] @ T))[None])[0]; rec = dict(neuron=int(keep[k]), coef=med[k].item(), n_own=int(len(own)))
    for al in (0.5, 1.0):
        amp = al * sgn[k] * med[k]; inj = torch.zeros(NT, D, device=DEV); inj[own] = amp * w[None]; Sa = run(inject=inj, inject_block=b + 1); r_own = unit((Sa[L] - S0[L])[own].mean(0, keepdim=True))[0]
        inj = torch.zeros(NT, D, device=DEV); inj[foreign] = amp * w[None]; Sb = run(inject=inj, inject_block=b + 1); r_for = unit((Sb[L] - S0[L])[foreign].mean(0, keepdim=True))[0]
        rec[str(al)] = dict(own_vs_natural=(r_own @ nat).item(), own_vs_foreign=(r_own @ r_for).item(), foreign_vs_natural=(r_for @ nat).item(), own_vs_linear=(r_own @ tw).item(), foreign_vs_linear=(r_for @ tw).item(), own_norm_per_coef=((Sa[L] - S0[L])[own].norm(dim=1).median() / (al * med[k])).item(), foreign_norm_per_coef=((Sb[L] - S0[L])[foreign].norm(dim=1).median() / (al * med[k])).item())
    out[nm] = rec; log(f"{tag} {nm} neuron {int(keep[k])} (|coef| {med[k]:.1f}, {len(own)} own tokens): " + " | ".join(f"{al}x: own-token addition vs natural footprint {v['own_vs_natural']:+.2f}, vs foreign addition {v['own_vs_foreign']:+.2f}, foreign vs natural {v['foreign_vs_natural']:+.2f}, own vs linear {v['own_vs_linear']:+.2f}, foreign vs linear {v['foreign_vs_linear']:+.2f}, norm/coef own {v['own_norm_per_coef']:.2f} foreign {v['foreign_norm_per_coef']:.2f}" for al, v in ((a_, rec[str(a_)]) for a_ in (0.5, 1.0))))
record(f"e290_counterfactual_{tag}", dict(model=tag, b=b, L=L, K=K, per_neuron=out), " | ".join(f"{nm} ({v['coef']:.1f}): 1x own vs natural {v['1.0']['own_vs_natural']:+.2f}, own vs foreign {v['1.0']['own_vs_foreign']:+.2f}, foreign vs natural {v['1.0']['foreign_vs_natural']:+.2f}, own vs linear {v['1.0']['own_vs_linear']:+.2f}, foreign vs linear {v['1.0']['foreign_vs_linear']:+.2f}" for nm, v in out.items()))
