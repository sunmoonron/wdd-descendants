"""e313: the mathematical connection between WDD and the quotient, tested. WDD writes the state as a ledger of
coefficients over the model's own atoms, x = sum_i c_i(t) w_i; the quotient is linear on the linear range, so the
functional coordinate of a joint perturbation must be the ledger-weighted sum of the atoms' coordinates:
P^T d_S(t) ~ sum_{i in S} c_i(t) z_i, with z_i = P^T u_i the core coordinate of atom i's context-free transplant
image. Test A: random subsets S of 4 to 8 candidate atoms ablated jointly at all tokens; at tokens where S carries
weight, the joint footprint's core coordinate against the ledger prediction (cosine, R2), against a random-z control,
and the same in full dimension with the transplant images. Test B: the quotient's algebra on physically unrelated
atom pairs, z(w_i + w_j) against z_i + z_j from injections at foreign tokens."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from func_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; b = 2; D = c.D; lab = c.d["lab"]; A = c.d["A"]; R = A[(lab["type"] == T_MLP) & (lab["block"] == b)].float().to(DEV)
led, tn, tc = dominant(model, arch, c, ids_seq, b); run = make_runner_logits(model, arch, c, ids_seq, [L], NT, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep)
S0i = run(positions=idx); S1i = run(tn, positions=idx); dl = S0i["lg"] - S1i["lg"]; dl = dl - dl.mean(1, keepdim=True); F = (S0i[L] - S1i[L])[idx]; Fc = F - F.mean(0, keepdim=True); G = dl @ dl.T; evg, Vg = torch.linalg.eigh(G); Zt = Vg.flip(1)[:, :256] * evg.flip(0)[:256].clamp_min(0).sqrt()[None]; Zt = Zt - Zt.mean(0, keepdim=True); P = torch.linalg.svd(Fc.T @ Zt, full_matrices=False)[0][:, :16]
foreign = torch.nonzero(typ & ~torch.isin(tn, keep))[:, 0]; s_inj = tc.abs().median(); torch.manual_seed(0); a = torch.randint(0, K, (len(foreign),), device=DEV); inj = torch.zeros(NT, D, device=DEV); inj[foreign] = s_inj * unit(R[keep])[a]; S2 = run(inject=inj, inject_block=b + 1); U = torch.stack([(S2[L] - S0[L])[foreign][a == k].mean(0) for k in range(K)]) / s_inj; Zat = U @ P
def ablate_set(S):
    neur = keep[S]
    def pre(m, inp):
        x = inp[0].clone(); flat = x.reshape(-1, c.DFF); flat[:, neur] = 0; return (flat.reshape(x.shape),)
    h = arch.mlp_lin(b).register_forward_pre_hook(pre); r = run(); h.remove(); return r[L]
resA = []; resA_full = []; resA_rand = []; ntok = 0
for trial in range(8):
    S = torch.randperm(K, device=DEV)[:torch.randint(4, 9, (1,)).item()]; SL = ablate_set(S); dS = S0[L] - SL; coef = led[:, keep[S]]; weight = coef.abs().sum(1); rows = torch.nonzero(typ & (weight >= weight[typ].quantile(0.9)))[:, 0]
    pred_core = coef[rows] @ Zat[S]; act_core = dS[rows] @ P; pred_full = coef[rows] @ U[S]; rand_core = coef[rows] @ Zat[torch.randperm(K, device=DEV)[:len(S)]]
    resA.append(((unit(act_core) * unit(pred_core)).sum(1))); resA_full.append(((unit(dS[rows]) * unit(pred_full)).sum(1))); resA_rand.append(((unit(act_core) * unit(rand_core)).sum(1))); ntok += len(rows)
cosA, cosAf, cosAr = torch.cat(resA), torch.cat(resA_full), torch.cat(resA_rand)
npair = min(12, K // 2); pairs = [(int(i), int(j)) for i, j in torch.randperm(K, device=DEV)[:2 * npair].reshape(npair, 2).tolist()]; sub = foreign[::2][:1024]; cosB = []; cosBf = []
for i, j in pairs:
    inj = torch.zeros(NT, D, device=DEV); inj[sub] = s_inj * (unit(R[keep[i]]) + unit(R[keep[j]]))[None]; r = run(inject=inj, inject_block=b + 1); img = (r[L] - S0[L])[sub].mean(0) / s_inj; cosB.append((unit((img @ P)[None]) @ unit((Zat[i] + Zat[j])[None]).T).item()); cosBf.append((unit(img[None]) @ unit((U[i] + U[j])[None]).T).item())
res = dict(K=K, tokens_tested=ntok, ledger_core_cos_median=cosA.median().item(), ledger_core_cos_mean=cosA.mean().item(), ledger_core_random_control=cosAr.median().item(), ledger_full_cos_median=cosAf.median().item(), pair_core_cos_median=float(torch.tensor(cosB).median()), pair_full_cos_median=float(torch.tensor(cosBf).median()), pair_write_cos_median=float(torch.tensor([(unit(R[keep[i]][None]) @ unit(R[keep[j]][None]).T).item() for i, j in pairs]).abs().median()))
log(f"{tag} (K {K}): (A) ledger-weighted sum of atom cores vs the joint ablation's core coordinate over {ntok} tokens: median cos {res['ledger_core_cos_median']:.2f} (mean {res['ledger_core_cos_mean']:.2f}; random-atom control {res['ledger_core_random_control']:.2f}); in full dimension with transplant images {res['ledger_full_cos_median']:.2f} | (B) z(w_i + w_j) vs z_i + z_j over {npair} physically unrelated atom pairs (|write cos| median {res['pair_write_cos_median']:.2f}): core {res['pair_core_cos_median']:.2f}, full dimension {res['pair_full_cos_median']:.2f}")
record(f"e313_ledger_{tag}", dict(model=tag, b=b, L=L, **res), f"ledger->core cos {res['ledger_core_cos_median']:.2f} (random {res['ledger_core_random_control']:.2f}; full-D {res['ledger_full_cos_median']:.2f}) | pair additivity core {res['pair_core_cos_median']:.2f} full {res['pair_full_cos_median']:.2f}")
