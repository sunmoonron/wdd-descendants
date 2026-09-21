"""e295: what do the leftover dimensions do? The per-token descendant at L is split into its part in the top-32
principal directions of the cloud (z, the causal rung) and the complement (q, the leftover 500 to 2000 directions).
Unit q-vectors, z-vectors, full descendants and random directions are injected at the natural footprint norm at
every other foreign token. Measured on the injected tokens: logit response norm, KL, the response two blocks later
split into the z-subspace and the q-subspace of that level, and the persistence of the injected q (cosine of the
later response with the injected vector); measured on the untouched tokens: the cross-token response energy
relative to the injected tokens (routing / token interaction)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from func_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; NB = c.NB; b = 2; D = c.D; lf = L + 2 if L + 2 < NB - 1 else NB - 2
led, tn, tc = dominant(model, arch, c, ids_seq, b); run = make_runner_logits(model, arch, c, ids_seq, [L, lf], NT, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep)
S1 = run(tn); F = (S0[L] - S1[L])[idx]; Ffut = (S0[lf] - S1[lf])[idx]; fnorm = F.norm(dim=1).median(); Fc = F - F.mean(0, keepdim=True); U32 = torch.linalg.svd(Fc, full_matrices=False)[2][:32]; Uf32 = torch.linalg.svd(Ffut - Ffut.mean(0, keepdim=True), full_matrices=False)[2][:32]
zpart = (F @ U32.T) @ U32; qpart = F - zpart; energy_q = ((qpart ** 2).sum() / (F ** 2).sum()).item()
foreign_all = torch.nonzero(typ & ~torch.isin(tn, keep))[:, 0]; inj_tok = foreign_all[::2][:1024]; untouched = foreign_all[1::2][:1024]; NI = len(inj_tok); base_i = run(positions=inj_tok); lg0 = base_i["lg"]; lp0 = torch.log_softmax(lg0, -1); torch.manual_seed(0); pick = torch.randint(0, len(idx), (NI,), device=DEV)
kinds = {"q_leftover": unit(qpart[pick]), "z_causal": unit(zpart[pick]), "full_descendant": unit(F[pick]), "random": unit(torch.randn(NI, D, device=DEV))}; out = {}
for nm, dirs in kinds.items():
    inj = torch.zeros(NT, D, device=DEV); inj[inj_tok] = fnorm * dirs; r = run(positions=inj_tok, inject=inj, inject_block=L + 1); dlr = r["lg"] - lg0; dlr = dlr - dlr.mean(1, keepdim=True); lp1 = torch.log_softmax(r["lg"], -1); klv = (lp0.exp() * (lp0 - lp1)).sum(1); Rf = (r[lf] - S0[lf]); Ri = Rf[inj_tok]; Ru = Rf[untouched]
    out[nm] = dict(logit_response=dlr.norm(dim=1).median().item(), kl=klv.median().item(), later_norm_per_injection=(Ri.norm(dim=1) / fnorm).median().item(), later_in_z=(((Ri @ Uf32.T) ** 2).sum(1) / (Ri ** 2).sum(1).clamp_min(1e-9)).median().item(), persistence=((unit(Ri) * dirs).sum(1)).median().item(), cross_token=(Ru.norm(dim=1).median() / Ri.norm(dim=1).median()).item())
log(f"{tag} (K {K}; q holds {energy_q:.2f} of descendant energy): " + " | ".join(f"{nm}: logit response {v['logit_response']:.2f}, KL {v['kl']:.4f}, later norm/injection {v['later_norm_per_injection']:.2f}, later response in the z-subspace {v['later_in_z']:.2f}, persistence {v['persistence']:.2f}, cross-token {v['cross_token']:.2f}" for nm, v in out.items()))
record(f"e295_remainder_{tag}", dict(model=tag, b=b, L=L, future=lf, K=K, q_energy=energy_q, per_kind=out), f"q energy {energy_q:.2f} | " + " | ".join(f"{nm}: logit {v['logit_response']:.2f} KL {v['kl']:.4f} later-in-z {v['later_in_z']:.2f} persist {v['persistence']:.2f} cross {v['cross_token']:.2f}" for nm, v in out.items()))
