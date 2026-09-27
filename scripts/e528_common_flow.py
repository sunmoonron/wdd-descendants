"""e528: is the rows' motion a common flow? Every experiment so far asks what steers one row. But every row moves
(by a similar angle, and growing), the states are sums of the rows, and a common linear map applied to the whole row
set would give each row a persistent increment, larger for rows with more of their direction in the map's amplified
subspace, without any single row's gradient showing it. This run fits, for each thousand-step interval between the
sixteen Pythia checkpoints, the best linear map M with dW = W M + E over all 53,248 rows of blocks 0-12 (the flow),
against the same fit on rows paired with a permutation of the increments (the spurious share of a 1024 x 1024 map).
It reports the flow's share of the increments' energy, per group; its structure (symmetric against antisymmetric
part, the isotropic growth rate, and the symmetric part's quadratic form on the cloud's top-32 principal directions
against random directions); its persistence across intervals; the flow's share of each row's own increment; whether
the memory of the motion lives in the flow part or in the residual; whether the same map explains the states' change
(applied without fitting to the block-12 states on the measurement text); and the decisive number, how much of the
entrants' rise of S the flow alone reproduces when applied to the origin's rows against the origin's states.
Pre-registered (honest guesses), block 12:
- the flow explains 15% or more of the increments' energy (the permutation null about 2%) (0.5);
- the symmetric part amplifies the cloud's top-32 directions more than random directions, ratio 1.5 or above (0.5);
- the flow persists: the cosine of consecutive maps is 0.5 or above (0.5);
- entrants' increments have a larger flow share than matched rows' (AUC 0.6 or above) (0.4);
- the flow applied to the origin's rows reproduces 40% or more of the entrants' rise of S (0.35).
Arguments: name."""
import sys, os, json as _json, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; CDIR = f"/workspace/wdd/cache/e524_{name}"; BL = [12, 6]; LB = 12; NWORD = 256; NPC = 32
steps = list(range(1000, 16001, 1000)); ck = {n: torch.load(f"{CDIR}/step{n}.pt") for n in steps}; D = ck[1000]["rows"].shape[1]; DFF = 4096; R = (LB + 1) * DFF
idsB = eval_ids(name)[:8, :256].to(DEV); gen = torch.Generator(device=DEV).manual_seed(0)
def auc(pos, neg):
    if pos.numel() < 5 or neg.numel() < 5: return None
    x = torch.cat([pos, neg]); r = x.argsort().argsort().double() + 1; npos, nneg = pos.numel(), neg.numel(); return float((r[:npos].sum() - npos * (npos + 1) / 2) / (npos * nneg))
def strat_auc(q, s, y, nbins=10):
    edges = s.quantile(torch.linspace(0, 1, nbins + 1)); tot, wsum = 0.0, 0
    for i in range(nbins):
        m = (s >= edges[i]) & (s <= edges[i + 1]) if i == nbins - 1 else (s >= edges[i]) & (s < edges[i + 1])
        v = auc(q[m & y], q[m & ~y]); n = int((m & y).sum())
        if v is not None: tot += v * n; wsum += n
    return (tot / wsum) if wsum else None
def fit_map(W, dW, ridge=1e-4):
    G = W.T @ W; G = G + ridge * G.diagonal().mean() * torch.eye(D, device=DEV); return torch.linalg.solve(G, W.T @ dW)
r2 = lambda dW, F: 1 - float((dW - F).pow(2).sum() / dW.pow(2).sum().clamp_min(1e-12))
# raw centred block states on the measurement text at every checkpoint (for the states' change under the flow)
Xs = {}
for n in steps:
    model, tok, fam = load_model(name, revision=f"step{n}"); arch = Arch(model, fam)
    for p in model.parameters(): p.requires_grad_(False)
    X0 = block_states(model, arch, idsB, BL, chunk=4); Xs[n] = {b: X0[b].reshape(-1, D).cpu() for b in BL}; del model, X0; torch.cuda.empty_cache()
log(f"{name}: states recomputed at {len(steps)} checkpoints")
Wv = {n: (ck[n]["rows"].float() * ck[n]["norms"][:, None]) for n in steps}
words = {b: {} for b in BL}
for b in BL:
    Rb = (b + 1) * DFF
    for n in steps:
        u = ck[n]["blocks"][b]["usage"]; wd = torch.zeros(Rb, dtype=torch.bool); wd[torch.nonzero(u > 0)[:, 0][u[u > 0].argsort(descending=True)[:NWORD]]] = True; words[b][n] = wd
def groups_at(b, i):
    """entrants within three thousand steps (clean), S-matched non-entrants, non-words and words at steps[i]"""
    Rb = (b + 1) * DFF; n0 = steps[i]; S0 = ck[n0]["blocks"][b]["S"].float(); nonw0 = ~words[b][n0]; ent = torch.zeros(Rb, dtype=torch.bool); stay = nonw0.clone(); never = nonw0.clone()
    for k in (1, 2, 3):
        if i + k + 1 < len(steps): e = stay & ~words[b][steps[i + k - 1]] & words[b][steps[i + k]] & words[b][steps[i + k + 1]]; ent |= e
        if i + k < len(steps): stay &= ~words[b][steps[i + k]]; never &= ~words[b][steps[i + k]]
    if i + 4 < len(steps): never &= ~words[b][steps[i + 4]]
    pool = torch.nonzero(never)[:, 0]; taken = torch.zeros(Rb, dtype=torch.bool); matched = []
    for r in torch.nonzero(ent)[:, 0].tolist():
        c = pool[~taken[pool]]; j = c[(S0[c] - S0[r]).abs().argmin()]; taken[j] = True; matched.append(int(j))
    mm = torch.zeros(Rb, dtype=torch.bool); mm[torch.tensor(matched, dtype=torch.long)] = True
    return {"entrants": ent, "matched": mm, "non_words": nonw0, "words": words[b][n0]}, S0
res = dict(model=name, steps=steps, by_block={})
for b in BL:
    Rb = (b + 1) * DFF; per = {}; Ms = []; uE, udW = [], []; flow_share_rows = {}
    for i in range(len(steps) - 1):
        n0, n1 = steps[i], steps[i + 1]; W0 = Wv[n0][:Rb].to(DEV); W1 = Wv[n1][:Rb].to(DEV); dW = W1 - W0
        M = fit_map(W0, dW); F = W0 @ M; E = dW - F; Ms.append(M)
        perm = torch.randperm(Rb, device=DEV, generator=gen); Mn = fit_map(W0, dW[perm]); Fn = W0 @ Mn
        g, S0 = groups_at(b, i); gd = {k: v.to(DEV) for k, v in g.items()}
        Msym = (M + M.T) / 2; Manti = (M - M.T) / 2; V = ck[n0]["blocks"][b]["V"].to(DEV)[:, :NPC]
        q_pc = float(((V.T @ Msym @ V).diagonal()).mean()); Rr = unitr(torch.randn(256, D, device=DEV, generator=gen)); q_rand = float(((Rr @ Msym) * Rr).sum(1).mean()); ev = torch.linalg.eigvalsh(Msym)
        # the flow applied to the origin's rows against the origin's states: S reproduced
        U0 = ck[n0]["blocks"][b]["U"].float().to(DEV); keep0 = ck[n0]["blocks"][b]["keep"].to(DEV); L0 = ck[n0]["blocks"][b]["L"].to(DEV)
        def S_of_rows(Wr):
            P = U0 @ unitr(Wr).T; ratio = P.abs() / L0[:, None]; ratio[~keep0] = 0; return ratio.max(0).values
        S00 = S_of_rows(W0); S_flow = S_of_rows(W0 + F); S_row = S_of_rows(W1); dSf = (S_flow - S00).cpu(); dSr = (S_row - S00).cpu()
        ent, mm, nonw = g["entrants"], g["matched"], g["non_words"]; rise = dSr[ent] > 0.1
        # the states under the flow (applied, not fitted), block b states on the measurement text
        X0 = Xs[n0][b].to(DEV); X1 = Xs[n1][b].to(DEV); kp = ~sinkmask(X0) & ~sinkmask(X1); X0c = X0[kp] - X0[kp].mean(0); X1c = X1[kp] - X1[kp].mean(0); dX = X1c - X0c
        st_r2 = r2(dX, X0c @ M); st_r2_null = r2(dX, X0c @ Mn); st_r2_growth = r2(dX, X0c * (M.diagonal().mean()))
        share = (F.pow(2).sum(1) / dW.pow(2).sum(1).clamp_min(1e-12)).cpu(); cosF = ((dW * F).sum(1) / (dW.norm(dim=1) * F.norm(dim=1)).clamp_min(1e-12)).cpu()
        a0 = ck[n0]["blocks"][b]["a"].to(DEV); tgt = U0[a0]; cos_dw_best = ((unitr(dW) * tgt).sum(1)).cpu(); cos_E_best = ((unitr(E) * tgt).sum(1)).cpu(); cos_F_best = ((unitr(F) * tgt).sum(1)).cpu()
        per[n0] = dict(r2_flow=r2(dW, F), r2_null=r2(dW, Fn), r2_by_group={k: r2(dW[v], F[v]) for k, v in gd.items()}, sym_share=float(Msym.pow(2).sum() / M.pow(2).sum()), anti_share=float(Manti.pow(2).sum() / M.pow(2).sum()), growth_rate=float(M.diagonal().mean()),
                       quad_form_top32=q_pc, quad_form_random=q_rand, eig_max=float(ev[-1]), eig_min=float(ev[0]), eig_median=float(ev.median()), states_r2_flow=st_r2, states_r2_null=st_r2_null, states_r2_growth_only=st_r2_growth,
                       flow_share_median={k: float(share[v].median()) for k, v in g.items()}, cos_flow_median={k: float(cosF[v].median()) for k, v in g.items()}, auc_flow_share_entrants_vs_matched=auc(share[ent], share[mm]),
                       n_entrants=int(ent.sum()), entrants_dS_row=float(dSr[ent].median()), entrants_dS_flow=float(dSf[ent].median()), matched_dS_row=float(dSr[mm].median()), matched_dS_flow=float(dSf[mm].median()),
                       flow_share_of_rise=float((dSf[ent][rise] / dSr[ent][rise]).median()) if int(rise.sum()) >= 5 else None, auc_entry_dS_flow=auc(dSf[ent], dSf[nonw & ~ent]), auc_entry_dS_row=auc(dSr[ent], dSr[nonw & ~ent]), auc_entry_dS_flow_within_S=strat_auc(dSf[nonw], S0[nonw], ent[nonw]),
                       cos_best={"all": {k: float(v[ent].median()) for k, v in (("dW", cos_dw_best), ("flow", cos_F_best), ("residual", cos_E_best))}, "typical": {k: float(v[nonw].median()) for k, v in (("dW", cos_dw_best), ("flow", cos_F_best), ("residual", cos_E_best))}})
        uE.append(unitr(E).half()); udW.append(unitr(dW).half()); flow_share_rows[n0] = share
        o = per[n0]; log(f"{name} block {b} {n0}->{n1}: flow explains {o['r2_flow']:.3f} of the increments' energy (null {o['r2_null']:.3f}; entrants {o['r2_by_group']['entrants']:.3f}, matched {o['r2_by_group']['matched']:.3f}, words {o['r2_by_group']['words']:.3f}); symmetric share {o['sym_share']:.2f}, growth rate {o['growth_rate']:.3f}, quadratic form on the top-32 {o['quad_form_top32']:.3f} vs random {o['quad_form_random']:.3f} (eigenvalues {o['eig_min']:.3f}..{o['eig_median']:.3f}..{o['eig_max']:.3f}); states' change explained by the rows' flow {o['states_r2_flow']:.3f} (null {o['states_r2_null']:.3f}, growth only {o['states_r2_growth_only']:.3f}); per-row flow share entrants/matched/non-words/words {o['flow_share_median']['entrants']:.3f}/{o['flow_share_median']['matched']:.3f}/{o['flow_share_median']['non_words']:.3f}/{o['flow_share_median']['words']:.3f} (AUC {o['auc_flow_share_entrants_vs_matched'] if o['auc_flow_share_entrants_vs_matched'] is None else round(o['auc_flow_share_entrants_vs_matched'], 2)}); entrants' rise of S: rows moved {o['entrants_dS_row']:+.3f}, flow only {o['entrants_dS_flow']:+.3f} (share {o['flow_share_of_rise'] if o['flow_share_of_rise'] is None else round(o['flow_share_of_rise'], 2)}); matched {o['matched_dS_row']:+.3f}/{o['matched_dS_flow']:+.3f}; entry AUC of the flow's change of S {o['auc_entry_dS_flow'] if o['auc_entry_dS_flow'] is None else round(o['auc_entry_dS_flow'], 2)} (within S {o['auc_entry_dS_flow_within_S'] if o['auc_entry_dS_flow_within_S'] is None else round(o['auc_entry_dS_flow_within_S'], 2)}; rows' actual {o['auc_entry_dS_row'] if o['auc_entry_dS_row'] is None else round(o['auc_entry_dS_row'], 2)}); cosine with the best state, entrants: increment {o['cos_best']['all']['dW']:.3f}, flow part {o['cos_best']['all']['flow']:.3f}, residual {o['cos_best']['all']['residual']:.3f}")
        del W0, W1, dW, F, E, Fn, U0, X0, X1; torch.cuda.empty_cache()
    # persistence of the flow, and where the memory lives
    pers = {k: float(torch.tensor([float((Ms[i] * Ms[i + k]).sum() / (Ms[i].norm() * Ms[i + k].norm())) for i in range(len(Ms) - k)]).mean()) for k in range(1, 7)}
    pers_sym = {k: float(torch.tensor([float((((Ms[i] + Ms[i].T) * (Ms[i + k] + Ms[i + k].T)).sum()) / ((Ms[i] + Ms[i].T).norm() * (Ms[i + k] + Ms[i + k].T).norm())) for i in range(len(Ms) - k)]).mean()) for k in range(1, 7)}
    g4, _ = groups_at(b, steps.index(4000)); i4 = steps.index(4000); mem = {}
    for lab, seq in (("increment", udW), ("residual", uE)):
        mem[lab] = {}
        for k in range(1, 7):
            c = torch.stack([(seq[i].float() * seq[i + k].float()).sum(1) for i in range(i4, len(seq) - k)]).mean(0).cpu(); mem[lab][k] = {gname: float(c[m].median()) for gname, m in g4.items()}
    res["by_block"][b] = dict(per_interval=per, flow_persistence=pers, flow_persistence_symmetric=pers_sym, memory_from_4000=mem)
    log(f"{name} block {b}: persistence of the flow, cosine of maps k intervals apart: " + ", ".join(f"k={k} {v:.2f}" for k, v in pers.items()) + " (symmetric part " + ", ".join(f"{v:.2f}" for v in pers_sym.values()) + "); memory from step 4000, lag-k cosine entrants/matched/non-words/words, of the increments: " + ", ".join(f"k={k} {v['entrants']:.2f}/{v['matched']:.2f}/{v['non_words']:.2f}/{v['words']:.2f}" for k, v in mem["increment"].items()) + "; of the residuals after the flow: " + ", ".join(f"k={k} {v['entrants']:.2f}/{v['matched']:.2f}/{v['non_words']:.2f}/{v['words']:.2f}" for k, v in mem["residual"].items()))
    del Ms, uE, udW; torch.cuda.empty_cache()
P12 = res["by_block"][12]["per_interval"]; mid = [P12[n] for n in (4000, 8000, 12000)]; g_ = lambda x: -9 if x is None else x
res["checks"] = dict(flow_over_15=all(o["r2_flow"] >= 0.15 for o in mid), amplifies_top32=all(o["quad_form_top32"] >= 1.5 * o["quad_form_random"] if o["quad_form_random"] > 0 else o["quad_form_top32"] > o["quad_form_random"] for o in mid), persists_over_0_5=res["by_block"][12]["flow_persistence"][1] >= 0.5, entrants_larger_share=all(g_(o["auc_flow_share_entrants_vs_matched"]) >= 0.6 for o in mid), flow_reproduces_40=all(g_(o["flow_share_of_rise"]) >= 0.4 for o in mid))
fm = lambda x: "n/a" if x is None else f"{x:.2f}"
summ = f"{name}: " + " | ".join(f"block {b}: flow explains " + "/".join(f"{o['per_interval'][n]['r2_flow']:.2f}" for n in (2000, 4000, 8000, 12000)) + " of the increments at 2000/4000/8000/12000 (null " + "/".join(f"{o['per_interval'][n]['r2_null']:.2f}" for n in (2000, 4000, 8000, 12000)) + "), symmetric share " + "/".join(f"{o['per_interval'][n]['sym_share']:.2f}" for n in (4000, 8000, 12000)) + ", growth " + "/".join(f"{o['per_interval'][n]['growth_rate']:.3f}" for n in (4000, 8000, 12000)) + ", top-32 vs random quadratic form " + "/".join(f"{o['per_interval'][n]['quad_form_top32']:.3f}:{o['per_interval'][n]['quad_form_random']:.3f}" for n in (4000, 8000, 12000)) + f"; persistence k=1 {o['flow_persistence'][1]:.2f}, k=4 {o['flow_persistence'][4]:.2f}; states explained " + "/".join(f"{o['per_interval'][n]['states_r2_flow']:.2f}" for n in (4000, 8000, 12000)) + "; per-row flow share entrants/non-words " + "/".join(f"{o['per_interval'][n]['flow_share_median']['entrants']:.2f}:{o['per_interval'][n]['flow_share_median']['non_words']:.2f}" for n in (4000, 8000, 12000)) + "; flow's share of the entrants' rise " + "/".join(fm(o['per_interval'][n]['flow_share_of_rise']) for n in (4000, 8000, 12000)) + ", entry AUC of the flow's change " + "/".join(fm(o['per_interval'][n]['auc_entry_dS_flow']) for n in (4000, 8000, 12000)) + "; memory of increments vs residuals at lag 2, entrants " + f"{o['memory_from_4000']['increment'][2]['entrants']:.2f} vs {o['memory_from_4000']['residual'][2]['entrants']:.2f}" for b, o in res["by_block"].items()) + f" | checks {_json.dumps(res['checks'])}"
log(summ); record(f"e528_common_flow_{name}", res, summ)
