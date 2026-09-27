"""e526: what direction, computable at the origin from the network's own geometry, does a row's thousand-step motion
follow, and how long does the motion remember its direction? e525 showed the motion is not the row's mean loss
gradient at the time. The read-side account: a row's update is a sum, over the positions where its neuron fires, of
residual-space vectors, so the accumulated motion may follow what the neuron sees when it fires rather than any
instantaneous gradient. Candidate targets, all activation-weighted means over the positions where the neuron fires
(on 32 x 512 evaluation tokens at the origin checkpoint, sinks removed): the centred residual state at the row's own
block with the row's own write removed; the centred block-12 state with the own write removed; the co-active writes
(the block's MLP output minus the own write and the bias); and, per row, the neuron's read direction (the
up-projection row), the state at the row's best position (e523's target) and the row's own direction. Per row: the
cosine of the actual one-, two- and four-thousand-step updates (e524's records) with each target, and the share of the
thousand-step update's energy explained by the least-squares fit on the five non-radial targets together (five random
directions explain 0.005). Then the memory of the motion: the cosine between the unit thousand-step increment at the
origin and the increments k thousand steps later, k = 1 to 6, per row. Entrants within three thousand steps against
S-matched non-entrants, all non-words and the words; Pythia-410m, blocks 12 and 6, origins 4000, 8000, 12000.
Pre-registered (honest guesses), block 12:
- the thousand-step increment follows the co-active writes' mean at a median cosine above 0.1 for the entrants (0.4);
- the entrants' increment follows the own-block active-state mean more than the matched rows' (AUC 0.6 or above) (0.5);
- the read direction is not a target (median |cosine| below 0.05) (0.6);
- the five targets explain 20% or more of the entrants' thousand-step increment against 10% or less of the typical
  non-word's (0.4);
- the increment's direction has memory: the entrants' lag-k cosine stays above 0.1 out to k = 4 (0.4).
Arguments: name step (the origin checkpoint, thousands)."""
import sys, os, json as _json, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; t0 = int(sys.argv[2]) * 1000; CDIR = f"/workspace/wdd/cache/e524_{name}"; BL = [12, 6]; LB = 12; NWORD = 256; LAST = 16000
steps = list(range(t0, LAST + 1, 1000)); ck = {n: torch.load(f"{CDIR}/step{n}.pt") for n in steps}
model, tok, fam = load_model(name, revision=f"step{t0}"); arch = Arch(model, fam); D, DFF = arch.D, arch.DFF; R = (LB + 1) * DFF
for p in model.parameters(): p.requires_grad_(False)
ids = eval_ids(name).to(DEV); B_, T_ = ids.shape
def auc(pos, neg):
    if pos.numel() < 5 or neg.numel() < 5: return None
    x = torch.cat([pos, neg]); r = x.argsort().argsort().double() + 1; npos, nneg = pos.numel(), neg.numel(); return float((r[:npos].sum() - npos * (npos + 1) / 2) / (npos * nneg))
# one pass over the evaluation text: activations, block outputs and MLP outputs of blocks 0-12, positions 1:
cap = dict(act={bb: [] for bb in range(LB + 1)}, out={bb: [] for bb in range(LB + 1)}, mlp={bb: [] for bb in range(LB + 1)})
def h_act(bb):
    def f(m, a): cap["act"][bb].append(a[0][:, 1:].detach().float().clamp_min(0).reshape(-1, DFF).half()); return None
    return f
def h_out(bb):
    def f(m, i, o):
        cap["out"][bb].append(out_of(o)[:, 1:].detach().float().reshape(-1, D))
        if bb == LB: raise Stop
    return f
def h_mlp(bb):
    def f(m, i, o): cap["mlp"][bb].append(out_of(o)[:, 1:].detach().float().reshape(-1, D))
    return f
hs = [arch.mlp_lin(bb).register_forward_pre_hook(h_act(bb)) for bb in range(LB + 1)] + [arch.layers[bb].register_forward_hook(h_out(bb)) for bb in range(LB + 1)] + [arch.layers[bb].mlp.register_forward_hook(h_mlp(bb)) for bb in range(LB + 1)]
try:
    for s in range(0, B_, 4):
        try:
            with torch.no_grad(): model(ids[s:s + 4])
        except Stop: pass
finally: [h.remove() for h in hs]
A = {bb: torch.cat(cap["act"][bb]) for bb in range(LB + 1)}; X = {bb: torch.cat(cap["out"][bb]) for bb in range(LB + 1)}; MLP = {bb: torch.cat(cap["mlp"][bb]) for bb in range(LB + 1)}; del cap
keep = ~sinkmask(X[LB]); N = int(keep.sum()); log(f"{name} step{t0}: {N} of {X[LB].shape[0]} positions kept")
Xc = {bb: (X[bb][keep] - X[bb][keep].mean(0)) for bb in range(LB + 1)}
W0 = torch.cat([arch.wdir(bb) for bb in range(LB + 1)]); Rd = torch.cat([unitr(arch.rdir(bb)) for bb in range(LB + 1)])
T1 = torch.zeros(R, D, device=DEV); T2 = torch.zeros(R, D, device=DEV); T4 = torch.zeros(R, D, device=DEV); rate = torch.zeros(R, device=DEV)
for bb in range(LB + 1):
    a = A[bb][keep].float(); s = a.sum(0).clamp_min(1e-6); s2 = a.pow(2).sum(0); sl = slice(bb * DFF, (bb + 1) * DFF); w = W0[sl]; own = (s2 / s)[:, None] * w
    T1[sl] = (a.T @ Xc[bb]) / s[:, None] - own; T2[sl] = (a.T @ Xc[LB]) / s[:, None] - own
    bias = arch.mlp_bias(bb); mo = MLP[bb][keep] - (bias[None] if bias is not None else 0); T4[sl] = (a.T @ mo) / s[:, None] - own; rate[sl] = (a > 0).float().mean(0)
    del a
del A, X, MLP, Xc; torch.cuda.empty_cache()
Wv = {n: (ck[n]["rows"].float() * ck[n]["norms"][:, None]).to(DEV) for n in steps}
res = dict(model=name, origin=t0, n_positions=N, by_block={})
TG = {"own_block_state": T1, "block12_state": T2, "read_direction": Rd, "coactive_writes": T4}
for b in BL:
    Rb = (b + 1) * DFF; Bk = ck[t0]["blocks"][b]; U0 = Bk["U"].float().to(DEV); a0 = Bk["a"].to(DEV); S0 = Bk["S"].float()
    tg = {k: unitr(v[:Rb]) for k, v in TG.items()}; tg["best_state"] = U0[a0]; tg["own_direction"] = unitr(Wv[t0][:Rb])
    words = {}
    for n in steps:
        u = ck[n]["blocks"][b]["usage"]; wd = torch.zeros(Rb, dtype=torch.bool); wd[torch.nonzero(u > 0)[:, 0][u[u > 0].argsort(descending=True)[:NWORD]]] = True; words[n] = wd
    nonw0 = ~words[t0]; ent = torch.zeros(Rb, dtype=torch.bool); stay = nonw0.clone()
    for k in (1, 2, 3):
        n1, n2 = t0 + 1000 * k, t0 + 1000 * (k + 1)
        if n2 in words: e = stay & ~words[n1 - 1000] & words[n1] & words[n2]; ent |= e
        stay &= ~words[n1]
    never = nonw0.clone()
    for k in (1, 2, 3, 4):
        if t0 + 1000 * k in words: never &= ~words[t0 + 1000 * k]
    pool = torch.nonzero(never)[:, 0]; taken = torch.zeros(Rb, dtype=torch.bool); matched = []
    for i in torch.nonzero(ent)[:, 0].tolist():
        c = pool[~taken[pool]]; j = c[(S0[c] - S0[i]).abs().argmin()]; taken[j] = True; matched.append(int(j))
    mm = torch.zeros(Rb, dtype=torch.bool); mm[torch.tensor(matched, dtype=torch.long)] = True; groups = {"entrants": ent, "matched": mm, "non_words": nonw0, "words": words[t0]}
    Q = {"S": S0, "firing_rate": rate[:Rb].cpu()}
    for K in (1, 2, 4):
        if t0 + 1000 * K not in Wv: continue
        dw = Wv[t0 + 1000 * K][:Rb] - Wv[t0][:Rb]; du = unitr(dw)
        for k, v in tg.items(): Q[f"cos_{k}_{K}"] = (du * v).sum(1).cpu()
        # least-squares share of the increment's energy explained by the five non-radial targets, and by five random directions
        Mx = torch.stack([tg[k] for k in ("own_block_state", "block12_state", "read_direction", "coactive_writes", "best_state")], 2)                     # [Rb, D, 5]
        Gm = Mx.transpose(1, 2) @ Mx + 1e-6 * torch.eye(5, device=DEV); rhs = Mx.transpose(1, 2) @ du[:, :, None]; coef = torch.linalg.solve(Gm, rhs); Q[f"r2_targets_{K}"] = ((Mx @ coef)[:, :, 0]).pow(2).sum(1).cpu()
        gen = torch.Generator(device=DEV).manual_seed(K); Rr = unitr(torch.randn(Rb, D, 5, device=DEV, generator=gen).transpose(1, 2)).transpose(1, 2)
        Gm = Rr.transpose(1, 2) @ Rr + 1e-6 * torch.eye(5, device=DEV); rhs = Rr.transpose(1, 2) @ du[:, :, None]; coef = torch.linalg.solve(Gm, rhs); Q[f"r2_random_{K}"] = ((Rr @ coef)[:, :, 0]).pow(2).sum(1).cpu(); del Mx, Rr, Gm, rhs, coef
    # memory of the increment's direction: lag-k cosine between unit thousand-step increments
    incs = [unitr(Wv[steps[i + 1]][:Rb] - Wv[steps[i]][:Rb]).half() for i in range(len(steps) - 1)]
    for k in range(1, 7):
        if len(incs) > k:
            c = torch.stack([(incs[i].float() * incs[i + k].float()).sum(1) for i in range(len(incs) - k)]).mean(0); Q[f"lag_{k}"] = c.cpu()
    del incs
    out = dict(n_entrants=int(ent.sum()), n_non_words=int(nonw0.sum()), medians={g: {q: float(v[m].nanmedian()) for q, v in Q.items()} for g, m in groups.items()}, positive_fraction={g: {q: float((Q[q][m] > 0).float().mean()) for q in Q if q.startswith("cos_") or q.startswith("lag_")} for g, m in groups.items()},
               auc_entrants_vs_matched={q: auc(Q[q][ent].nan_to_num(0), Q[q][mm].nan_to_num(0)) for q in Q if q != "S"})
    res["by_block"][b] = out; e_, m_, a_, w_ = (out["medians"][g] for g in ("entrants", "matched", "non_words", "words")); A_ = out["auc_entrants_vs_matched"]; pf = out["positive_fraction"]; fm = lambda x: "n/a" if x is None else f"{x:.2f}"
    log(f"{name} step{t0} block {b} ({out['n_entrants']} entrants within three thousand steps): cosine of the thousand-step increment with each target, entrants/matched/non-words/words (AUC entrants vs matched): " + "; ".join(f"{k} {e_[f'cos_{k}_1']:.3f}/{m_[f'cos_{k}_1']:.3f}/{a_[f'cos_{k}_1']:.3f}/{w_[f'cos_{k}_1']:.3f} ({fm(A_[f'cos_{k}_1'])})" for k in tg) + f"; positive fraction (non-words) " + ", ".join(f"{k} {pf['non_words'][f'cos_{k}_1']:.2f}" for k in tg) + f"; energy explained by the five targets {e_['r2_targets_1']:.3f}/{m_['r2_targets_1']:.3f}/{a_['r2_targets_1']:.3f}/{w_['r2_targets_1']:.3f} (random five {a_['r2_random_1']:.3f}; AUC {fm(A_['r2_targets_1'])})")
    ks = [K for K in (2, 4) if f"cos_own_block_state_{K}" in e_]
    log(f"{name} step{t0} block {b}: over two and four thousand steps, entrants: " + " | ".join(f"K={K}: " + ", ".join(f"{k} {e_[f'cos_{k}_{K}']:.3f}" for k in tg) + f", explained {e_[f'r2_targets_{K}']:.3f} (non-words {a_[f'r2_targets_{K}']:.3f})" for K in ks) + "; memory of the increment's direction, lag-k cosine entrants/matched/non-words/words: " + ", ".join(f"k={k} {e_[f'lag_{k}']:.3f}/{m_[f'lag_{k}']:.3f}/{a_[f'lag_{k}']:.3f}/{w_[f'lag_{k}']:.3f}" for k in range(1, 7) if f"lag_{k}" in e_) + f"; firing rate {e_['firing_rate']:.3f}/{m_['firing_rate']:.3f}/{a_['firing_rate']:.3f}")
    del U0, a0; torch.cuda.empty_cache()
B = res["by_block"][12]; e_, a_ = B["medians"]["entrants"], B["medians"]["non_words"]; g_ = lambda x: -9 if x is None else x
res["checks"] = dict(coactive_over_0_1=e_["cos_coactive_writes_1"] > 0.1, own_block_state_auc_over_0_6=g_(B["auc_entrants_vs_matched"]["cos_own_block_state_1"]) >= 0.6, read_direction_not_target=abs(a_["cos_read_direction_1"]) < 0.05 and abs(e_["cos_read_direction_1"]) < 0.05, targets_explain_20_vs_10=e_["r2_targets_1"] >= 0.2 and a_["r2_targets_1"] <= 0.1, memory_to_lag_4=all(g_(e_.get(f"lag_{k}")) > 0.1 for k in range(1, 5)))
fm = lambda x: "n/a" if x is None else f"{x:.2f}"
summ = f"{name} step{t0}: " + " | ".join(f"block {b}: thousand-step increment's cosine with own-block state / block-12 state / read direction / co-active writes / best state / own direction, entrants {o['medians']['entrants']['cos_own_block_state_1']:.3f}/{o['medians']['entrants']['cos_block12_state_1']:.3f}/{o['medians']['entrants']['cos_read_direction_1']:.3f}/{o['medians']['entrants']['cos_coactive_writes_1']:.3f}/{o['medians']['entrants']['cos_best_state_1']:.3f}/{o['medians']['entrants']['cos_own_direction_1']:.3f}, non-words {o['medians']['non_words']['cos_own_block_state_1']:.3f}/{o['medians']['non_words']['cos_block12_state_1']:.3f}/{o['medians']['non_words']['cos_read_direction_1']:.3f}/{o['medians']['non_words']['cos_coactive_writes_1']:.3f}/{o['medians']['non_words']['cos_best_state_1']:.3f}/{o['medians']['non_words']['cos_own_direction_1']:.3f}; explained by the five targets {o['medians']['entrants']['r2_targets_1']:.3f} (non-words {o['medians']['non_words']['r2_targets_1']:.3f}, random {o['medians']['non_words']['r2_random_1']:.3f}); lag-k memory entrants " + "/".join(f"{o['medians']['entrants'][f'lag_{k}']:.2f}" for k in range(1, 7) if f'lag_{k}' in o['medians']['entrants']) + " (non-words " + "/".join(f"{o['medians']['non_words'][f'lag_{k}']:.2f}" for k in range(1, 7) if f'lag_{k}' in o['medians']['non_words']) + ")" for b, o in res["by_block"].items()) + f" | checks {_json.dumps(res['checks'])}"
log(summ); record(f"e526_input_side_{name}_step{t0}", res, summ)
