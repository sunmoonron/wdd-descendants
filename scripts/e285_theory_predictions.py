"""e285: three predictions derived from the first-order transport theory (descendant = Jacobian image of the write,
with a generic, non-isometric, token-dependent Jacobian), tested on the actual network. From the fitted operator
T(3->L) = U S V^T: (1) the gain of the actual image for input directions along the top-8, middle-8 and bottom-8
left singular vectors and for random directions must order top > random ~ middle > bottom, with the random gain
near the root-mean-square singular value (concentration of measure); (2) for pairs (u, 0.75 u + 0.66 r) with r
random orthogonal, the image-pair cosine must follow cos_out = 0.75 rho / sqrt(0.75^2 rho^2 + 0.66^2), rho =
gain(u)/gain(r): equal to 0.75 for random u (the e238 result), above it for top-singular u, below it for
bottom-singular u; (3) the additivity error of m superposed random writes must depend on the total injected norm,
not on m: flat in m at constant total norm, growing at constant per-write norm (the e241 pattern)."""
import sys, os, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from desc_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; b = 2; D = c.D; run = make_runner(model, arch, c, ids_seq, [L], NT)
led, tn, tc = dominant(model, arch, c, ids_seq, b); S0 = run(); typ = typical_mask(S0[L]); pool = torch.nonzero(typ)[:, 0]; s_inj = tc.abs().median(); torch.manual_seed(0); Vr = unit(torch.randn(1024, D, device=DEV)); X = []; Y = []
for p in range(2):
    a = torch.randint(0, 1024, (len(pool),), device=DEV); inj = torch.zeros(NT, D, device=DEV); inj[pool] = s_inj * Vr[a]; S2 = run(inject=inj, inject_block=b + 1); X.append(Vr[a]); Y.append((S2[L] - S0[L])[pool] / s_inj)
X, Y = torch.cat(X), torch.cat(Y); G = X.T @ X; T = torch.linalg.solve(G + 1e-2 * G.diagonal().mean() * torch.eye(D, device=DEV), X.T @ Y); Ut, S, Vt = torch.linalg.svd(T); r_ = len(S)
def images(dirs, scale):
    # inject each direction at 1/len(dirs) of the pool tokens; return image centroid per direction
    a = torch.randint(0, len(dirs), (len(pool),), device=DEV); inj = torch.zeros(NT, D, device=DEV); inj[pool] = scale * dirs[a]; S2 = run(inject=inj, inject_block=b + 1); img = (S2[L] - S0[L])[pool]; return torch.stack([img[a == q].mean(0) for q in range(len(dirs))]) / scale
classes_u = {"top8": Ut[:, :8].T, "mid8": Ut[:, r_ // 2: r_ // 2 + 8].T, "bottom8": Ut[:, -8:].T, "random8": unit(torch.randn(8, D, device=DEV))}
allu = torch.cat(list(classes_u.values())); img = images(allu, s_inj); gain = img.norm(dim=1); pred_dir = torch.cat([Vt[:8], Vt[r_ // 2: r_ // 2 + 8], Vt[-8:], unit(allu[-8:] @ T)]); cos_pred = (unit(img) * pred_dir).sum(1)
t1 = {nm: dict(gain=gain[8 * i: 8 * i + 8].median().item(), cos_to_predicted_output=cos_pred[8 * i: 8 * i + 8].median().item(), operator_sv=(S[:8].median().item() if nm == "top8" else S[r_ // 2: r_ // 2 + 8].median().item() if nm == "mid8" else S[-8:].median().item() if nm == "bottom8" else (S ** 2).mean().sqrt().item())) for i, nm in enumerate(classes_u)}
log(f"{tag} (1) gain of the actual image by input direction: " + " ".join(f"{nm} {v['gain']:.2f} (operator {v['operator_sv']:.2f}, cos to predicted output {v['cos_to_predicted_output']:.2f})" for nm, v in t1.items()) + f"; rms singular value {(S ** 2).mean().sqrt():.2f}")
# (2) pair cosines: u from each class, partner 0.75 u + 0.66 r
ct, st = 0.75, math.sqrt(1 - 0.75 ** 2); torch.manual_seed(1); R8 = unit(torch.randn(24, D, device=DEV)); U24 = torch.cat([classes_u["top8"], classes_u["bottom8"], classes_u["random8"]]); R8 = unit(R8 - (R8 * U24).sum(1, keepdim=True) * U24); partners = ct * U24 + st * R8
img_u = images(U24, s_inj); img_p = images(partners, s_inj); img_r = images(R8, s_inj); cos_out = (unit(img_u) * unit(img_p)).sum(1); rho = img_u.norm(dim=1) / img_r.norm(dim=1); pred = ct * rho / torch.sqrt(ct ** 2 * rho ** 2 + st ** 2)
t2 = {nm: dict(measured=cos_out[8 * i: 8 * i + 8].median().item(), predicted=pred[8 * i: 8 * i + 8].median().item(), rho=rho[8 * i: 8 * i + 8].median().item()) for i, nm in enumerate(("top8", "bottom8", "random8"))}
log(f"{tag} (2) image cosine of pairs with input cosine 0.75: " + " ".join(f"{nm} measured {v['measured']:.2f} predicted {v['predicted']:.2f} (gain ratio {v['rho']:.2f})" for nm, v in t2.items()))
# (3) additivity vs total norm
torch.manual_seed(2); W16 = unit(torch.randn(16, D, device=DEV)); single = images(W16, s_inj / 4); t3 = {}
for m in (1, 2, 4, 8, 16):
    v = W16[:m].sum(0); lin = single[:m].sum(0)
    for regime, amp in (("constant_total_norm", s_inj / v.norm()), ("constant_per_write_norm", s_inj)):
        a = torch.randint(0, 1, (len(pool),), device=DEV); inj = torch.zeros(NT, D, device=DEV); inj[pool] = amp * v[None]; S2 = run(inject=inj, inject_block=b + 1); act = (S2[L] - S0[L])[pool].mean(0) / amp; t3[f"{regime}_m{m}"] = dict(total_norm=(amp * v.norm()).item() / s_inj.item(), additivity_error=((act - lin).norm() / act.norm()).item(), cos=(unit(act[None]) @ unit(lin[None]).T).item())
log(f"{tag} (3) additivity error of m superposed random writes (linear prediction from single images at 1/4 amplitude): constant total norm " + "/".join(f"{t3[f'constant_total_norm_m{m}']['additivity_error']:.2f}" for m in (1, 2, 4, 8, 16)) + " | constant per-write norm (total norm 1, 1.4, 2, 2.8, 4x) " + "/".join(f"{t3[f'constant_per_write_norm_m{m}']['additivity_error']:.2f}" for m in (1, 2, 4, 8, 16)))
record(f"e285_theory_{tag}", dict(model=tag, b=b, L=L, gain_by_direction=t1, pair_cosines=t2, additivity=t3, rms_sv=(S ** 2).mean().sqrt().item()), "(1) gain top/mid/bottom/random " + "/".join(f"{t1[nm]['gain']:.2f}" for nm in ("top8", "mid8", "bottom8", "random8")) + f" (rms sv {(S ** 2).mean().sqrt():.2f}) | (2) pair cos measured/predicted top {t2['top8']['measured']:.2f}/{t2['top8']['predicted']:.2f} bottom {t2['bottom8']['measured']:.2f}/{t2['bottom8']['predicted']:.2f} random {t2['random8']['measured']:.2f}/{t2['random8']['predicted']:.2f} | (3) additivity error const-norm " + "/".join(f"{t3[f'constant_total_norm_m{m}']['additivity_error']:.2f}" for m in (1, 2, 4, 8, 16)) + " vs growing-norm " + "/".join(f"{t3[f'constant_per_write_norm_m{m}']['additivity_error']:.2f}" for m in (1, 2, 4, 8, 16)))
