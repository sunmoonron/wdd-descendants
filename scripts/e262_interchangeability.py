"""e262: are neurons with similar descendants interchangeable under intervention? Pairs (A, B) of block-2 candidates:
the 5 pairs with the highest descendant similarity among pairs of low write similarity ('similar') and the 5 pairs
with the lowest descendant similarity ('control'). At B's natural tokens: clean; B removed; B removed and A's vector
injected with B's coefficient at the block-3 input. Repair = 1 - ||dL(remove+substitute)|| / ||dL(remove)|| on the
same-position logits, and the same in the state at L; also cos(dL(substitute), dL(remove)). Interchangeable = high
repair for similar pairs and low for control pairs."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from func_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; b = 2; D = c.D; lab = c.d["lab"]; A = c.d["A"]; R = A[(lab["type"] == T_MLP) & (lab["block"] == b)].float().to(DEV)
led, tn, tc = dominant(model, arch, c, ids_seq, b); run = make_runner_logits(model, arch, c, ids_seq, [L], NT, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep)
S0i = run(positions=idx); S1i = run(tn, positions=idx); F = S0i[L] - S1i[L]; Cd = unit(centroids(F[idx], lab_i, K)); Rw = unit(R[keep]); Gd, Gw = Cd @ Cd.T, Rw @ Rw.T; iu = torch.triu_indices(K, K, 1, device=DEV); gw, gd = Gw[iu[0], iu[1]], Gd[iu[0], iu[1]]; low_w = gw < gw.median(); cand = torch.nonzero(low_w)[:, 0]
sim_pairs = cand[gd[cand].argsort(descending=True)[:5]]; ctl_pairs = cand[gd[cand].argsort()[:5]]; out = {}
for nm, pairs in (("similar", sim_pairs), ("control", ctl_pairs)):
    partner = torch.full((K,), -1, device=DEV)
    for pidx in pairs.tolist(): a_, b_ = int(iu[0][pidx]), int(iu[1][pidx]); partner[b_] = a_; partner[a_] = b_             # both directions
    toks = idx[partner[lab_i] >= 0]; lb = lab_i[partner[lab_i] >= 0]; la = partner[lb]; S_cl = run(positions=toks); S_rm = run(tn, positions=toks)
    inj = torch.zeros(NT, D, device=DEV); inj[toks] = tc[toks][:, None] * R[keep[la]]; S_sub = run(tn, positions=toks, inject=inj, inject_block=b + 1)
    dl_rm = S_rm["lg"] - S_cl["lg"]; dl_sub = S_sub["lg"] - S_cl["lg"]; ds_rm = (S_rm[L] - S_cl[L])[toks]; ds_sub = (S_sub[L] - S_cl[L])[toks]
    out[nm] = dict(n_tokens=int(len(toks)), mean_desc_sim=gd[pairs].mean().item(), mean_write_sim=gw[pairs].mean().item(), repair_logits=(1 - dl_sub.norm(dim=1) / dl_rm.norm(dim=1).clamp_min(1e-6)).median().item(), repair_state=(1 - ds_sub.norm(dim=1) / ds_rm.norm(dim=1).clamp_min(1e-6)).median().item(), cos_sub_rm_logits=((dl_sub * dl_rm).sum(1) / (dl_sub.norm(dim=1) * dl_rm.norm(dim=1)).clamp_min(1e-9)).median().item())
    log(f"{tag} {nm} pairs (descendant sim {out[nm]['mean_desc_sim']:.2f}, write sim {out[nm]['mean_write_sim']:.2f}, {len(toks)} tokens): substituting the partner's write repairs the removal effect by {out[nm]['repair_logits']:.2f} in the logits and {out[nm]['repair_state']:.2f} in the state at L (cos of residual effects {out[nm]['cos_sub_rm_logits']:.2f})")
record(f"e262_interchange_{tag}", dict(model=tag, b=b, L=L, K=K, **out), f"similar-descendant pairs: repair logits {out['similar']['repair_logits']:.2f}, state {out['similar']['repair_state']:.2f} (desc sim {out['similar']['mean_desc_sim']:.2f}) | control pairs: repair logits {out['control']['repair_logits']:.2f}, state {out['control']['repair_state']:.2f} (desc sim {out['control']['mean_desc_sim']:.2f})")
