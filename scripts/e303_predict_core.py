"""e303: can the causal core be predicted from the write without constructing the descendant? The core coordinate of
each candidate is z_i = P^T d_i (P the 16-dimensional function basis, d_i the unit natural centroid at L). Predictors:
the write itself P^T w_i (the core is in the atom), the data-free operator P^T (sign * T w_i) (the core is in the
averaged Jacobian), the transplant image P^T (image of w_i injected at foreign tokens) (the core is in the
context-free transport), and random. Reported: median cosine in the core and nearest-neighbour identification of
the neuron among the K candidates, for each predictor."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from func_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; b = 2; D = c.D; lab = c.d["lab"]; A = c.d["A"]; R = A[(lab["type"] == T_MLP) & (lab["block"] == b)].float().to(DEV)
led, tn, tc = dominant(model, arch, c, ids_seq, b); run = make_runner_logits(model, arch, c, ids_seq, [L], NT, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep)
S0i = run(positions=idx); S1i = run(tn, positions=idx); dl = S0i["lg"] - S1i["lg"]; dl = dl - dl.mean(1, keepdim=True); F = (S0i[L] - S1i[L])[idx]; Fc = F - F.mean(0, keepdim=True); G = dl @ dl.T; evg, Vg = torch.linalg.eigh(G); Zt = Vg.flip(1)[:, :256] * evg.flip(0)[:256].clamp_min(0).sqrt()[None]; Zt = Zt - Zt.mean(0, keepdim=True); P = torch.linalg.svd(Fc.T @ Zt, full_matrices=False)[0][:, :16]
Cd = centroids(F, lab_i, K); z = unit(Cd @ P); smed = torch.stack([tc[idx][lab_i == k].median() for k in range(K)]); med = smed.abs(); sgn = torch.sign(smed)
pool = torch.nonzero(typ)[:, 0]; s_inj = tc.abs().median(); torch.manual_seed(0); Vr = unit(torch.randn(1024, D, device=DEV)); X = []; Y = []
for p in range(2):
    a = torch.randint(0, 1024, (len(pool),), device=DEV); inj = torch.zeros(NT, D, device=DEV); inj[pool] = s_inj * Vr[a]; S2 = run(inject=inj, inject_block=b + 1); X.append(Vr[a]); Y.append((S2[L] - S0[L])[pool] / s_inj)
X, Y = torch.cat(X), torch.cat(Y); Gx = X.T @ X; T = torch.linalg.solve(Gx + 1e-2 * Gx.diagonal().mean() * torch.eye(D, device=DEV), X.T @ Y)
foreign = torch.nonzero(typ & ~torch.isin(tn, keep))[:, 0]; a2 = torch.randint(0, K, (len(foreign),), device=DEV); inj = torch.zeros(NT, D, device=DEV); inj[foreign] = (sgn * med)[a2][:, None] * unit(R[keep])[a2]; S2 = run(inject=inj, inject_block=b + 1); timg = torch.stack([(S2[L] - S0[L])[foreign][a2 == k].mean(0) for k in range(K)])
preds = {"write_itself": unit(sgn[:, None] * R[keep]) @ P, "operator_transport": (sgn[:, None] * (R[keep] @ T)) @ P, "transplant_image": timg @ P, "random": torch.randn(K, D, device=DEV) @ P}; out = {}
for nm, zh in preds.items():
    zh = unit(zh); cs = (zh * z).sum(1); nn = (zh @ z.T).argmax(1); out[nm] = dict(median_cos=cs.median().item(), identification=(nn == torch.arange(K, device=DEV)).float().mean().item())
log(f"{tag} (K {K}, chance {1 / K:.2f}): predicting the core coordinate: " + " | ".join(f"{nm}: cos {v['median_cos']:.2f}, identification {v['identification']:.2f}" for nm, v in out.items()))
record(f"e303_predictcore_{tag}", dict(model=tag, b=b, L=L, K=K, per_predictor=out), " | ".join(f"{nm} cos {v['median_cos']:.2f} id {v['identification']:.2f}" for nm, v in out.items()) + f" (chance {1 / K:.2f})")
