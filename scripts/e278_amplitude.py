"""e278: is the massive neuron beyond the linear range of the transport law, or off it at every amplitude? The
largest-coefficient candidate and two typical candidates are transplanted at foreign typical tokens at 0.1, 0.3,
1, 3 and 10 times their natural median coefficient (sign kept). The image centroid at L is compared with the linear
transport sign*T w, with the natural descendant centroid, and with the image at 0.1x (self-consistency); the
operating-point shift is the norm ratio of the block-3 input with and without the injection. A typical neuron
pushed to massive amplitude and the massive neuron pulled to typical amplitude test whether the anomaly is one of
amplitude."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from desc_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; b = 2; D = c.D; lab = c.d["lab"]; A = c.d["A"]; R = A[(lab["type"] == T_MLP) & (lab["block"] == b)].float().to(DEV); run = make_runner(model, arch, c, ids_seq, [b, L], NT)
led, tn, tc = dominant(model, arch, c, ids_seq, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); S1 = run(b, tn); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep); F = S0[L] - S1[L]; Cd = unit(centroids(F[idx], lab_i, K)); smed = torch.stack([tc[idx][lab_i == k].median() for k in range(K)]); med = smed.abs(); sgn = torch.sign(smed)
order = med.argsort(); picks = [int(med.argmax()), int(order[len(order) // 2]), int(order[len(order) // 2 - 1])]; names = ["largest", "typical_a", "typical_b"]
pool = torch.nonzero(typ)[:, 0]; s_inj = tc.abs().median(); torch.manual_seed(0); Vr = unit(torch.randn(1024, D, device=DEV)); X = []; Y = []
for p in range(2):
    a = torch.randint(0, 1024, (len(pool),), device=DEV); inj = torch.zeros(NT, D, device=DEV); inj[pool] = s_inj * Vr[a]; S2 = run(inject=inj, inject_block=b + 1); X.append(Vr[a]); Y.append((S2[L] - S0[L])[pool] / s_inj)
X, Y = torch.cat(X), torch.cat(Y); G = X.T @ X; T = torch.linalg.solve(G + 1e-2 * G.diagonal().mean() * torch.eye(D, device=DEV), X.T @ Y)
foreign = torch.nonzero(typ & ~torch.isin(tn, keep))[:, 0]; NF = len(foreign); assign = torch.randint(0, 3, (NF,), device=DEV); vec = unit(R[keep[picks]]); alphas = [0.1, 0.3, 1.0, 3.0, 10.0]; out = {nm: {} for nm in names}; img0 = {}
for al in alphas:
    amp = al * (sgn[picks] * med[picks])[assign]; inj = torch.zeros(NT, D, device=DEV); inj[foreign] = amp[:, None] * vec[assign]; S2 = run(inject=inj, inject_block=b + 1); img = S2[L] - S0[L]; shift = (S0[b][foreign] + inj[foreign]).norm(dim=1) / S0[b][foreign].norm(dim=1)
    for j, nm in enumerate(names):
        m = assign == j; k = picks[j]; cimg = unit(img[foreign[m]].mean(0, keepdim=True))[0]; tw = unit((sgn[k] * (R[keep[k]] @ T))[None])[0]
        if al == alphas[0]: img0[nm] = cimg
        out[nm][al] = dict(neuron=int(keep[k]), coef=(al * med[k]).item(), cos_linear=(cimg @ tw).item(), cos_natural=(cimg @ Cd[k]).item(), cos_first=(cimg @ img0[nm]).item(), operating_point_shift=shift[m].median().item(), image_norm_per_coef=(img[foreign[m]].norm(dim=1).median() / (al * med[k])).item())
for nm in names:
    log(f"{tag} {nm} neuron {out[nm][1.0]['neuron']} (natural |coef| {out[nm][1.0]['coef']:.1f}, median {med.median():.1f}): " + " | ".join(f"{al}x: cos linear {v['cos_linear']:+.2f}, natural {v['cos_natural']:+.2f}, first {v['cos_first']:+.2f}, shift {v['operating_point_shift']:.2f}, norm/coef {v['image_norm_per_coef']:.2f}" for al, v in out[nm].items()))
record(f"e278_amplitude_{tag}", dict(model=tag, b=b, L=L, K=K, per_neuron={nm: {str(al): v for al, v in d.items()} for nm, d in out.items()}), " | ".join(f"{nm} ({out[nm][1.0]['coef']:.1f}): linear " + "/".join(f"{v['cos_linear']:+.2f}" for v in out[nm].values()) + ", natural " + "/".join(f"{v['cos_natural']:+.2f}" for v in out[nm].values()) + ", self " + "/".join(f"{v['cos_first']:+.2f}" for v in out[nm].values()) + ", shift " + "/".join(f"{v['operating_point_shift']:.2f}" for v in out[nm].values()) for nm in names))
