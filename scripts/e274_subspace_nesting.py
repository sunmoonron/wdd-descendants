"""e274: are the identity, function and behaviour subspaces of the descendant nested or rotated? At L, from the
natural footprints of the K candidates: the identity subspace = top-d eigenvectors of the between-neuron scatter;
the function subspace = top-d left singular vectors of the cross-covariance between centred descendants and the
(PCA-reduced) logit footprints; the behaviour direction = the ridge direction predicting the removal KL. For d = 8
and 32: the fraction of each subspace's energy inside each other (chance d/D), and the same against a random subspace.
Nested: function inside identity ~1 and behaviour inside function ~1; rotated: near chance."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from func_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 16; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; b = 2; D = c.D
led, tn, tc = dominant(model, arch, c, ids_seq, b); run = make_runner_logits(model, arch, c, ids_seq, [L], NT, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep)
S0 = run(positions=idx); S1 = run(tn, positions=idx); F = (S0[L] - S1[L])[idx]; dl = S0["lg"] - S1["lg"]; lp0, lp1 = torch.log_softmax(S0["lg"], -1), torch.log_softmax(S1["lg"], -1); kl = (lp0.exp() * (lp0 - lp1)).sum(1); dl = dl - dl.mean(1, keepdim=True)
Fc = F - F.mean(0, keepdim=True); cents = torch.stack([Fc[lab_i == k].mean(0) for k in range(K)]); w = torch.bincount(lab_i, minlength=K).float(); Sb = (cents * w[:, None]).T @ cents / w.sum(); ev, V = torch.linalg.eigh(Sb); U_id_full = V.flip(1)
G = dl @ dl.T; evg, Vg = torch.linalg.eigh(G); Z = Vg.flip(1)[:, :256] * evg.flip(0)[:256].clamp_min(0).sqrt()[None]; Z = Z - Z.mean(0, keepdim=True); C = Fc.T @ Z; U_fn_full = torch.linalg.svd(C, full_matrices=False)[0]
klc = kl - kl.mean(); ridge = torch.linalg.solve(Fc.T @ Fc + 1e-2 * (Fc.T @ Fc).diagonal().mean() * torch.eye(D, device=DEV), Fc.T @ klc); u_beh = unit(ridge[None])[0]
Fs = F.T @ F; evs, Vs = torch.linalg.eigh(Fs); U_st_full = Vs.flip(1)
def inside(Ua, Ub): return ((Ub.T @ Ua) ** 2).sum().item() / Ua.shape[1]
torch.manual_seed(0); out = {}
for d in (8, 32):
    U_id, U_fn, U_st = U_id_full[:, :d], U_fn_full[:, :d], U_st_full[:, :d]; U_rand = torch.linalg.qr(torch.randn(D, d, device=DEV))[0]
    out[d] = dict(chance=d / D, fn_in_id=inside(U_fn, U_id), id_in_fn=inside(U_id, U_fn), beh_in_fn=inside(u_beh[:, None], U_fn), beh_in_id=inside(u_beh[:, None], U_id), fn_in_state=inside(U_fn, U_st), id_in_state=inside(U_id, U_st), fn_in_random=inside(U_fn, U_rand), beh_in_random=inside(u_beh[:, None], U_rand))
    log(f"{tag} d={d} (chance {d / D:.3f}): function inside identity {out[d]['fn_in_id']:.2f}, identity inside function {out[d]['id_in_fn']:.2f}, behaviour inside function {out[d]['beh_in_fn']:.2f}, behaviour inside identity {out[d]['beh_in_id']:.2f} | function inside state-PCA {out[d]['fn_in_state']:.2f}, identity inside state-PCA {out[d]['id_in_state']:.2f} | random-subspace reference: function {out[d]['fn_in_random']:.3f}, behaviour {out[d]['beh_in_random']:.3f}")
record(f"e274_nesting_{tag}", dict(model=tag, b=b, L=L, K=K, per_d={str(k): v for k, v in out.items()}), " | ".join(f"d={d}: fn in id {v['fn_in_id']:.2f}, id in fn {v['id_in_fn']:.2f}, beh in fn {v['beh_in_fn']:.2f}, beh in id {v['beh_in_id']:.2f}, fn in state {v['fn_in_state']:.2f} (chance {v['chance']:.3f})" for d, v in out.items()))
