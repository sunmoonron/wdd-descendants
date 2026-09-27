"""e523: where do the entrants' rows move? e522 found that the rise of a future word's alignment from one checkpoint
to the next is mostly the row moving (its direction at the next checkpoint fits the origin's states better than its
old direction did), not the states moving onto the row, and that every row moves by a similar angle whether it
enters or not. This run asks what distinguishes the entrants' motion. Two readings: the row rotates toward the state
at the particular position where its projection is largest (a local pull), or it rotates into the principal subspace
of the state cloud, where projections are large at many positions (subspace capture). Per non-word row at the origin:
the fraction of its direction, of its next direction and of its change (the unit vector from the origin row to the
next checkpoint's row) inside the top-k principal subspace of the centred states (k = 8, 16, 32, 64, 128; k/D and the
cloud's own variance fraction as references); the cosine of its change with the centred state at its best position,
raw and after removing the top-32 principal components, against the same at a random position; whether the best
position persists under the new row; the rise of the projection at the old best position against the rise of the
maximum. Three predictors of entry within deciles of S: the row's fraction in the top-32 subspace, its mean squared
projection over positions (how much of the cloud it sees) and its count of positions over the floor. Entrants
against S-matched non-entrants, all non-words and the current words.
Setup: Pythia-410m, blocks 12 and 6; origin checkpoint as argument; 8 x 256 measurement tokens (e519's); the next
checkpoint's word set from e519's records.
Pre-registered (honest guesses), block 12:
- the entrants' row change lies more in the top-32 principal subspace than the matched rows' (AUC 0.6 or above) (0.5);
- the entrants' change points toward the state at their best position: median cosine above 0.1 and at least 0.05
  above the matched rows' (0.5);
- that specificity survives the removal of the top-32 components (0.4);
- the best position persists for at least 60% of the entrants (0.5);
- the mean squared projection predicts entry within deciles of S at 0.6 or above (0.4).
Arguments: name revision (the origin checkpoint)."""
import sys, os, json as _json, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; rev = sys.argv[2]; STEPS = ["step0", "step64", "step256", "step512", "step1000", "step2000", "step3000", "step4000", "step8000", "step16000", "step32000", "step64000", "main"]
assert rev in STEPS[:-1], rev; NEXTS = STEPS[STEPS.index(rev) + 1]; CDIR = f"/workspace/wdd/cache/e519_{name}"; NWORD = 256; BL = [12, 6]; KS = [8, 16, 32, 64, 128]; KRES = 32
ref = {s: torch.load(f"{CDIR}/{s}.pt") for s in (rev, NEXTS)}
idsB = eval_ids(name)[:8, :256].to(DEV)
model, tok, fam = load_model(name, revision=rev); arch = Arch(model, fam); D, DFF = arch.D, arch.DFF
for p in model.parameters(): p.requires_grad_(False)
def gabs(m, T=14.0, n=28001):
    t = torch.linspace(0, T, n, dtype=torch.float64); F = torch.special.erf(t / math.sqrt(2)); return float(torch.trapezoid(1 - F.pow(m), t))
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
def floor_of(arch_, b, U):
    """the per-position floor for the block's dictionary, and the row index map"""
    A, lab = build_dictionary(arch_, blocks=list(range(b + 1))); Au = unitr(A); Ar = unitr(rotate(A, seed=7)); del A; m = Au.shape[0]
    typ = lab["type"].to(DEV); blk = lab["block"].to(DEV); idx = lab["index"].to(DEV); gidx = torch.full((b + 1, DFF), -1, device=DEV, dtype=torch.long)
    for bb in range(b + 1): mm = (typ == T_MLP) & (blk == bb); gidx[bb, idx[mm]] = torch.nonzero(mm)[:, 0]
    CA = Au.T @ Au / m; CR = Ar.T @ Ar / m; gm = gabs(m); N = U.shape[0]
    s2 = ((U @ CA) * U).sum(1); s2r = ((U @ CR) * U).sum(1); Mr = torch.cat([(U[s:s + 128] @ Ar.T).abs().max(1).values for s in range(0, N, 128)])
    r_cal = float(Mr.mean() / (s2r.clamp_min(1e-12).sqrt() * gm).mean()); L = s2.clamp_min(1e-12).sqrt() * gm * r_cal; rows = gidx.reshape(-1); del Au, Ar; return L, rows
rows_of = lambda arch_, b: torch.cat([arch_.wdir(bb) for bb in range(b + 1)])
X0 = block_states(model, arch, idsB, BL, chunk=4); Wv0 = {b: rows_of(arch, b) for b in BL}
model1, _, _ = load_model(name, revision=NEXTS); arch1 = Arch(model1, fam)
for p in model1.parameters(): p.requires_grad_(False)
Wv1 = {b: rows_of(arch1, b) for b in BL}; del model1; torch.cuda.empty_cache()
res = dict(model=name, origin=rev, horizon=NEXTS, ks=KS, by_block={}); gen = torch.Generator(device=DEV).manual_seed(0)
for b in BL:
    X = X0[b].reshape(-1, D); keep = ~sinkmask(X); Xk = X[keep]; N = Xk.shape[0]; mu = Xk.mean(0); Xc = Xk - mu; U = unitr(Xc)
    L, rows = floor_of(arch, b, U); R = rows.numel()
    ev, V = torch.linalg.eigh(Xc.T @ Xc / N); V = V.flip(1); ev = ev.flip(0); varfrac = {k: float(ev[:k].sum() / ev.sum()) for k in KS}
    w0 = unitr(Wv0[b]); w1 = unitr(Wv1[b]); dw = Wv1[b] - Wv0[b]; dh = unitr(dw); rel = dw.norm(dim=1) / Wv0[b].norm(dim=1).clamp_min(1e-9)
    P0 = U @ w0.T; ratio0 = P0.abs() / L[:, None]; S0, a0 = ratio0.max(0); cnt = (ratio0 > 1).sum(0).float(); CU = U.T @ U / N; m2 = ((w0 @ CU) * w0).sum(1) * D
    P1 = U @ w1.T; ratio1 = P1.abs() / L[:, None]; S01, a01 = ratio1.max(0); persist = (a01 == a0).float(); local_rise = ratio1.gather(0, a0[None])[0] - S0; max_rise = S01 - S0
    del ratio0, ratio1
    fk = lambda M: {k: (M @ V[:, :k]).pow(2).sum(1) for k in KS}; f0 = fk(w0); f1 = fk(w1); g = fk(dh)
    xs = U[a0]; cos_best = (dh * xs).sum(1); rp = torch.randint(0, N, (R,), device=DEV, generator=gen); cos_rand = (dh * U[rp]).sum(1)
    Vr = V[:, :KRES]; resid = lambda M: M - (M @ Vr) @ Vr.T; dr = unitr(resid(dh)); cos_best_res = (dr * unitr(resid(xs))).sum(1); cos_rand_res = (dr * unitr(resid(U[rp]))).sum(1)
    del xs
    # groups
    u0, u1 = ref[rev][b]["usage"], ref[NEXTS][b]["usage"]
    Wset0 = set(torch.nonzero(u0 > 0)[:, 0][u0[u0 > 0].argsort(descending=True)[:NWORD]].tolist()); Wset1 = set(torch.nonzero(u1 > 0)[:, 0][u1[u1 > 0].argsort(descending=True)[:NWORD]].tolist())
    in0 = torch.zeros(R, dtype=torch.bool, device=DEV); in0[list(Wset0)] = True; in1 = torch.zeros(R, dtype=torch.bool, device=DEV); in1[list(Wset1)] = True
    ent = ~in0 & in1; nonw = ~in0; pool = torch.nonzero(~in0 & ~in1)[:, 0]; taken = torch.zeros(R, dtype=torch.bool, device=DEV); matched = []
    for i in torch.nonzero(ent)[:, 0].tolist():
        c = pool[~taken[pool]]; j = c[(S0[c] - S0[i]).abs().argmin()]; taken[j] = True; matched.append(int(j))
    mm = torch.zeros(R, dtype=torch.bool, device=DEV); mm[torch.tensor(matched, device=DEV)] = True
    groups = {"entrants": ent, "matched": mm, "non_words": nonw, "words": in0}
    med = lambda q, m: float(q[m].median())
    out = dict(n_entrants=int(ent.sum()), n_non_words=int(nonw.sum()), n_words=int(in0.sum()), cloud_variance_fraction=varfrac, isotropic_fraction={k: k / D for k in KS}, groups={}, auc_entrants_vs_matched={}, auc_entrants_vs_non_words={}, within_S_deciles={})
    for gname, gm in groups.items():
        out["groups"][gname] = dict(S=med(S0, gm), relative_row_change=med(rel, gm), row_cos=med((w0 * w1).sum(1), gm), fraction_in_top_k=dict(row={k: med(f0[k], gm) for k in KS}, next_row={k: med(f1[k], gm) for k in KS}, change={k: med(g[k], gm) for k in KS}),
                                    cos_change_with_best_state=med(cos_best, gm), cos_change_with_random_state=med(cos_rand, gm), cos_change_with_best_state_residual=med(cos_best_res, gm), cos_change_with_random_state_residual=med(cos_rand_res, gm), positive_cos_best=float((cos_best[gm] > 0).float().mean()),
                                    best_position_persists=float(persist[gm].mean()), local_rise=med(local_rise, gm), max_rise=med(max_rise, gm), count_over_floor=med(cnt, gm), mean_sq_projection=med(m2, gm))
    for k, q in (("change_in_top_32", g[32]), ("change_in_top_128", g[128]), ("cos_change_with_best_state", cos_best), ("cos_change_with_best_state_residual", cos_best_res), ("row_in_top_32", f0[32]), ("mean_sq_projection", m2), ("count_over_floor", cnt), ("best_position_persists", persist)):
        out["auc_entrants_vs_matched"][k] = auc(q[ent], q[mm]); out["auc_entrants_vs_non_words"][k] = auc(q[ent], q[nonw & ~ent])
    for k, q in (("row_in_top_32", f0[32]), ("row_in_top_128", f0[128]), ("mean_sq_projection", m2), ("count_over_floor", cnt)):
        out["within_S_deciles"][k] = strat_auc(q[nonw].cpu(), S0[nonw].cpu(), ent[nonw].cpu())
    res["by_block"][b] = out; e_, m_, a_, w_ = (out["groups"][k] for k in ("entrants", "matched", "non_words", "words")); fm = lambda x: "n/a" if x is None else f"{x:.2f}"
    log(f"{name} {rev} block {b} ({out['n_entrants']} entrants; cloud variance in the top 8/32/128 components {varfrac[8]:.2f}/{varfrac[32]:.2f}/{varfrac[128]:.2f}): fraction of the row's change in the top-32 subspace, entrants/matched/non-words/words {e_['fraction_in_top_k']['change'][32]:.3f}/{m_['fraction_in_top_k']['change'][32]:.3f}/{a_['fraction_in_top_k']['change'][32]:.3f}/{w_['fraction_in_top_k']['change'][32]:.3f} (isotropic {32 / D:.3f}; top-128 {e_['fraction_in_top_k']['change'][128]:.3f}/{m_['fraction_in_top_k']['change'][128]:.3f}/{a_['fraction_in_top_k']['change'][128]:.3f}, isotropic {128 / D:.3f}); the row itself in the top 32, origin -> next: entrants {e_['fraction_in_top_k']['row'][32]:.3f} -> {e_['fraction_in_top_k']['next_row'][32]:.3f}, matched {m_['fraction_in_top_k']['row'][32]:.3f} -> {m_['fraction_in_top_k']['next_row'][32]:.3f}, non-words {a_['fraction_in_top_k']['row'][32]:.3f} -> {a_['fraction_in_top_k']['next_row'][32]:.3f}, words {w_['fraction_in_top_k']['row'][32]:.3f} -> {w_['fraction_in_top_k']['next_row'][32]:.3f}; AUC entrants vs matched: change in top 32 {fm(out['auc_entrants_vs_matched']['change_in_top_32'])}, row in top 32 {fm(out['auc_entrants_vs_matched']['row_in_top_32'])}")
    log(f"{name} {rev} block {b}: cosine of the row's change with the state at its best position, entrants/matched/non-words/words {e_['cos_change_with_best_state']:.3f}/{m_['cos_change_with_best_state']:.3f}/{a_['cos_change_with_best_state']:.3f}/{w_['cos_change_with_best_state']:.3f} (random position {e_['cos_change_with_random_state']:.3f}/{m_['cos_change_with_random_state']:.3f}; positive for {e_['positive_cos_best']:.2f}/{m_['positive_cos_best']:.2f} of entrants/matched); after removing the top 32 components {e_['cos_change_with_best_state_residual']:.3f}/{m_['cos_change_with_best_state_residual']:.3f}/{a_['cos_change_with_best_state_residual']:.3f} (random {e_['cos_change_with_random_state_residual']:.3f}); AUC entrants vs matched {fm(out['auc_entrants_vs_matched']['cos_change_with_best_state'])}, residual {fm(out['auc_entrants_vs_matched']['cos_change_with_best_state_residual'])}; the best position persists for {e_['best_position_persists']:.2f}/{m_['best_position_persists']:.2f}/{a_['best_position_persists']:.2f} of entrants/matched/non-words; rise at the old best position vs of the maximum, entrants {e_['local_rise']:.2f}/{e_['max_rise']:.2f}, matched {m_['local_rise']:.2f}/{m_['max_rise']:.2f}; relative row change {e_['relative_row_change']:.2f}/{m_['relative_row_change']:.2f}/{a_['relative_row_change']:.2f}/{w_['relative_row_change']:.2f}")
    log(f"{name} {rev} block {b}: entry within deciles of S, AUC: row in top 32 {fm(out['within_S_deciles']['row_in_top_32'])}, in top 128 {fm(out['within_S_deciles']['row_in_top_128'])}, mean squared projection {fm(out['within_S_deciles']['mean_sq_projection'])}, count over the floor {fm(out['within_S_deciles']['count_over_floor'])}; medians entrants/matched: mean squared projection {e_['mean_sq_projection']:.2f}/{m_['mean_sq_projection']:.2f}, count over the floor {e_['count_over_floor']:.0f}/{m_['count_over_floor']:.0f}, row in top 32 {e_['fraction_in_top_k']['row'][32]:.3f}/{m_['fraction_in_top_k']['row'][32]:.3f}")
    del P0, P1, U, Xc; torch.cuda.empty_cache()
B = res["by_block"][12]; e_, m_ = B["groups"]["entrants"], B["groups"]["matched"]; g_ = lambda x: -1 if x is None else x
res["checks"] = dict(change_in_subspace_auc_over_0_6=g_(B["auc_entrants_vs_matched"]["change_in_top_32"]) >= 0.6, change_toward_best_state=e_["cos_change_with_best_state"] > 0.1 and e_["cos_change_with_best_state"] - m_["cos_change_with_best_state"] >= 0.05, specificity_survives_residual=e_["cos_change_with_best_state_residual"] > m_["cos_change_with_best_state_residual"] and e_["cos_change_with_best_state_residual"] > 0.05, best_position_persists_60=e_["best_position_persists"] >= 0.6, mean_sq_projection_within_S_over_0_6=g_(B["within_S_deciles"]["mean_sq_projection"]) >= 0.6)
fm = lambda x: "n/a" if x is None else f"{x:.2f}"
summ = f"{name} {rev} to {NEXTS}: " + " | ".join(f"block {b}: row change in the top-32 subspace entrants/matched/non-words {o['groups']['entrants']['fraction_in_top_k']['change'][32]:.3f}/{o['groups']['matched']['fraction_in_top_k']['change'][32]:.3f}/{o['groups']['non_words']['fraction_in_top_k']['change'][32]:.3f} (isotropic {32 / D:.3f}; AUC {fm(o['auc_entrants_vs_matched']['change_in_top_32'])}); cosine of the change with the best state {o['groups']['entrants']['cos_change_with_best_state']:.3f}/{o['groups']['matched']['cos_change_with_best_state']:.3f}/{o['groups']['non_words']['cos_change_with_best_state']:.3f} (random {o['groups']['entrants']['cos_change_with_random_state']:.3f}; residual {o['groups']['entrants']['cos_change_with_best_state_residual']:.3f}/{o['groups']['matched']['cos_change_with_best_state_residual']:.3f}); best position persists {o['groups']['entrants']['best_position_persists']:.2f}/{o['groups']['matched']['best_position_persists']:.2f}; within S deciles: row in top 32 {fm(o['within_S_deciles']['row_in_top_32'])}, mean sq projection {fm(o['within_S_deciles']['mean_sq_projection'])}, count over floor {fm(o['within_S_deciles']['count_over_floor'])}" for b, o in res["by_block"].items()) + f" | checks {_json.dumps(res['checks'])}"
log(summ); record(f"e523_row_motion_{name}_{rev}", res, summ)
