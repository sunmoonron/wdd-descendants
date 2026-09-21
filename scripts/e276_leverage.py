"""e276: are natural descendants used by the downstream network, or only induced by it? Unit directions are injected
at the block-(L+1) input of foreign typical tokens at the natural footprint norm: the natural descendant centroids
of the block-2 candidates, the descendant centroids of random vectors (random directions injected at block 3 and
read at L), the raw write vectors at level L (the transported-dictionary assumption), random unit vectors, and
covariance-matched random directions (sampled from the state covariance at L). Leverage = median centred logit
response, median KL, and the state response three blocks later per unit injection. If downstream components are
tuned to the descendants of the model's own writes, the natural centroids out-lever the random-vector descendants
and both random nulls; if descendant leverage is generic to transport, natural and random-vector descendants tie."""
import sys, os, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from func_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; b = 2; D = c.D; NB = c.NB; L2 = min(L + 3, NB - 1); lab = c.d["lab"]; A = c.d["A"]; R = A[(lab["type"] == T_MLP) & (lab["block"] == b)].float().to(DEV)
led, tn, tc = dominant(model, arch, c, ids_seq, b); run = make_runner_logits(model, arch, c, ids_seq, [L, L2], NT, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep)
S1 = run(tn); F = (S0[L] - S1[L])[idx]; Cd = centroids(F, lab_i, K); fnorm = F.norm(dim=1).median()
torch.manual_seed(0); pool = torch.nonzero(typ)[:, 0]; s_inj = tc.abs().median(); NR = 16; Vr = unit(torch.randn(NR, D, device=DEV)); a = torch.randint(0, NR, (len(pool),), device=DEV); inj = torch.zeros(NT, D, device=DEV); inj[pool] = s_inj * Vr[a]; S2 = run(inject=inj, inject_block=b + 1); Crd = centroids((S2[L] - S0[L])[pool], a, NR)
foreign = torch.nonzero(typ & ~torch.isin(tn, keep))[:, 0][:1536]; NF = len(foreign); base = run(positions=foreign); lg0 = base["lg"]; lp0 = torch.log_softmax(lg0, -1)
Sc = S0[L][typ]; Sc = Sc - Sc.mean(0, keepdim=True); z = torch.randn(NF, Sc.shape[0], device=DEV) / math.sqrt(Sc.shape[0]); cov_dirs = unit(z @ Sc)
kinds = {"natural_descendant": Cd[torch.randint(0, K, (NF,), device=DEV)], "random_vector_descendant": Crd[torch.randint(0, NR, (NF,), device=DEV)], "write_vector": unit(R[keep])[torch.randint(0, K, (NF,), device=DEV)], "random": unit(torch.randn(NF, D, device=DEV)), "covariance_matched_random": cov_dirs}
out = {}
for nm, dirs in kinds.items():
    inj = torch.zeros(NT, D, device=DEV); inj[foreign] = fnorm * dirs; r = run(positions=foreign, inject=inj, inject_block=L + 1); dl = r["lg"] - lg0; dl = dl - dl.mean(1, keepdim=True); lp1 = torch.log_softmax(r["lg"], -1); kl = (lp0.exp() * (lp0 - lp1)).sum(1)
    out[nm] = dict(logit_response=dl.norm(dim=1).median().item(), kl=kl.median().item(), state_response=((r[L2] - S0[L2])[foreign].norm(dim=1) / fnorm).median().item())
    log(f"{tag} {nm}: logit response {out[nm]['logit_response']:.3f}, KL {out[nm]['kl']:.4f}, state response at block {L2} per unit injection {out[nm]['state_response']:.2f}")
nat = out["natural_descendant"]; ratios = {f"natural_over_{k}": {m: nat[m] / max(out[k][m], 1e-9) for m in ("logit_response", "kl", "state_response")} for k in ("random_vector_descendant", "write_vector", "random", "covariance_matched_random")}
record(f"e276_leverage_{tag}", dict(model=tag, b=b, L=L, L2=L2, K=K, injection_norm=fnorm.item(), per_kind=out, ratios=ratios), "logit response: " + " ".join(f"{k} {v['logit_response']:.3f}" for k, v in out.items()) + " | KL: " + " ".join(f"{k} {v['kl']:.4f}" for k, v in out.items()) + " | natural/random-vector-descendant logit {:.2f} KL {:.2f}; natural/covariance-matched logit {:.2f} KL {:.2f}; write/random logit {:.2f}".format(ratios["natural_over_random_vector_descendant"]["logit_response"], ratios["natural_over_random_vector_descendant"]["kl"], ratios["natural_over_covariance_matched_random"]["logit_response"], ratios["natural_over_covariance_matched_random"]["kl"], out["write_vector"]["logit_response"] / max(out["random"]["logit_response"], 1e-9)))
