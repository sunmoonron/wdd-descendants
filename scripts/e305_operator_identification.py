"""e305: what operator generates the causal funnel? The data-free transport operators T(L->L+k), k = 1, 2, 4, from
random injections; their top-16 input and output singular spaces, the same for the update T - I, the slowest and
fastest eigenmodes (largest and smallest |lambda|, real and imaginary parts), and the read-out G estimated forward
from random injections at block L+1 (ridge to the top-256 footprint PCs), with its top-16 input singular space.
Energy overlaps of the function core and the identity core with each, against chance 16/D; the gain of core
directions under T relative to random directions; and the overlap of the read-out space with the transport spaces."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from func_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; NB = c.NB; b = 2; D = c.D; d = 16; ks = [k for k in (1, 2, 4) if L + k < NB]
led, tn, tc = dominant(model, arch, c, ids_seq, b); run = make_runner_logits(model, arch, c, ids_seq, [L] + [L + k for k in ks], NT, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep)
S0i = run(positions=idx); S1i = run(tn, positions=idx); dl = S0i["lg"] - S1i["lg"]; dl = dl - dl.mean(1, keepdim=True); F = (S0i[L] - S1i[L])[idx]; Fc = F - F.mean(0, keepdim=True); Ud, Sd, Vd = torch.linalg.svd(dl, full_matrices=False); B = Vd[:256].T; Zs = dl @ B; Zs = Zs - Zs.mean(0, keepdim=True)
S_fn = torch.linalg.svd(Fc.T @ Zs, full_matrices=False)[0][:, :d]; cents = torch.stack([Fc[lab_i == k].mean(0) for k in range(K)]); w = torch.bincount(lab_i, minlength=K).float(); Sb = (cents * w[:, None]).T @ cents / w.sum(); S_id = torch.linalg.eigh(Sb)[1].flip(1)[:, :d]; S_pca = torch.linalg.svd(Fc, full_matrices=False)[2][:d].T
pool = torch.nonzero(typ)[:, 0]; s_inj = tc.abs().median(); inside = lambda A, Bm: ((Bm.T @ A) ** 2).sum().item() / A.shape[1]
def fit_T(blk, lv, seed):
    torch.manual_seed(seed); Vr = unit(torch.randn(1024, D, device=DEV)); X = []; Y = []
    for p in range(2):
        a = torch.randint(0, 1024, (len(pool),), device=DEV); inj = torch.zeros(NT, D, device=DEV); inj[pool] = s_inj * Vr[a]; S2 = run(inject=inj, inject_block=blk); X.append(Vr[a]); Y.append((S2[lv] - S0[lv])[pool] / s_inj)
    X, Y = torch.cat(X), torch.cat(Y); G = X.T @ X; return torch.linalg.solve(G + 1e-2 * G.diagonal().mean() * torch.eye(D, device=DEV), X.T @ Y)
def eigspace(T, slow=True):
    ev, V = torch.linalg.eig(T); order = ev.abs().argsort(descending=slow)[:8]; W = torch.cat([V[:, order].real, V[:, order].imag], 1); return torch.linalg.qr(W)[0][:, :d]
out = {}
for k in ks:
    T = fit_T(L + 1, L + k, 10 + k); Ut, St, Vh = torch.linalg.svd(T); Dt = T - torch.eye(D, device=DEV); Uu, Su, Vu = torch.linalg.svd(Dt); spaces = {"T_input_top": Ut[:, :d], "T_output_top": Vh[:d].T, "update_input_top": Uu[:, :d], "update_output_top": Vu[:d].T, "slow_modes": eigspace(T, True), "fast_modes": eigspace(T, False), "T_input_bottom": Ut[:, -d:]}
    torch.manual_seed(3); Rn = unit(torch.randn(64, D, device=DEV)); gain_core = (S_fn.T @ T).norm(dim=1).median().item(); gain_id = (S_id.T @ T).norm(dim=1).median().item(); gain_rand = (Rn @ T).norm(dim=1).median().item()
    out[k] = dict(function_core={nm: inside(S_fn, S) for nm, S in spaces.items()}, identity_core={nm: inside(S_id, S) for nm, S in spaces.items()}, pca_core={nm: inside(S_pca, S) for nm, S in spaces.items()}, gain_function_core=gain_core, gain_identity_core=gain_id, gain_random=gain_rand, sv_top16_share=(St[:d] ** 2).sum().item() / (St ** 2).sum().item())
torch.manual_seed(7); Vr = unit(torch.randn(1024, D, device=DEV)); X = []; Y = []; base = run(positions=pool); lg0 = base["lg"]
for p in range(2):
    a = torch.randint(0, 1024, (len(pool),), device=DEV); inj = torch.zeros(NT, D, device=DEV); inj[pool] = s_inj * Vr[a]; r = run(positions=pool, inject=inj, inject_block=L + 1); dlr = r["lg"] - lg0; dlr = dlr - dlr.mean(1, keepdim=True); X.append(Vr[a]); Y.append((dlr @ B) / s_inj)
X, Y = torch.cat(X), torch.cat(Y); Gx = X.T @ X; Wg = torch.linalg.solve(Gx + 1e-2 * Gx.diagonal().mean() * torch.eye(D, device=DEV), X.T @ Y); S_G = torch.linalg.svd(Wg, full_matrices=False)[0][:, :d]
readout = dict(function_core_in_G=inside(S_fn, S_G), identity_core_in_G=inside(S_id, S_G), pca_core_in_G=inside(S_pca, S_G), G_in_T_input_top={k: None for k in ks}, chance=d / D)
for k in ks:
    T = fit_T(L + 1, L + k, 10 + k); Ut = torch.linalg.svd(T)[0]; readout["G_in_T_input_top"][k] = inside(S_G, Ut[:, :d])
log(f"{tag} (K {K}, D {D}, chance {d / D:.3f}): " + " | ".join(f"k={k}: function core in T input/output top {out[k]['function_core']['T_input_top']:.2f}/{out[k]['function_core']['T_output_top']:.2f}, update input/output {out[k]['function_core']['update_input_top']:.2f}/{out[k]['function_core']['update_output_top']:.2f}, slow/fast modes {out[k]['function_core']['slow_modes']:.2f}/{out[k]['function_core']['fast_modes']:.2f}, T input bottom {out[k]['function_core']['T_input_bottom']:.3f}; identity core in T input top {out[k]['identity_core']['T_input_top']:.2f}, slow {out[k]['identity_core']['slow_modes']:.2f}; gain of core vs random {out[k]['gain_function_core']:.2f}/{out[k]['gain_random']:.2f}; top-16 singular share {out[k]['sv_top16_share']:.2f}" for k in ks) + f" | read-out G: function core in G {readout['function_core_in_G']:.2f}, identity core in G {readout['identity_core_in_G']:.2f}, PCA core in G {readout['pca_core_in_G']:.2f}; G in T input top " + "/".join(f"{readout['G_in_T_input_top'][k]:.2f}" for k in ks))
record(f"e305_operator_{tag}", dict(model=tag, b=b, L=L, K=K, d=d, per_k={str(k): v for k, v in out.items()}, readout={kk: (vv if not isinstance(vv, dict) else {str(a): bb for a, bb in vv.items()}) for kk, vv in readout.items()}), " | ".join(f"k{k}: fn in T-in {out[k]['function_core']['T_input_top']:.2f} T-out {out[k]['function_core']['T_output_top']:.2f} upd-in {out[k]['function_core']['update_input_top']:.2f} slow {out[k]['function_core']['slow_modes']:.2f} fast {out[k]['function_core']['fast_modes']:.2f}; gain core/random {out[k]['gain_function_core']:.2f}/{out[k]['gain_random']:.2f}" for k in ks) + f" | fn in G {readout['function_core_in_G']:.2f} id in G {readout['identity_core_in_G']:.2f} pca in G {readout['pca_core_in_G']:.2f} (chance {d / D:.3f})")
