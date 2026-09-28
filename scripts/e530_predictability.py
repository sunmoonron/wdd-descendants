"""e530: how much of a row's eight-thousand-step displacement is predictable from anything at the origin? Sessions 79-81
tried the mechanisms one by one (the row's gradient, its whitened form, the co-active writes, the read direction,
flat directions, a common flow) and found each to carry a few percent or nothing. Before another mechanism is tried,
the honest step is to bound the whole: fit predictors of the actual displacement w(t+8000) - w(t) on half the rows
from every origin-time quantity in hand, and score them on the other half. Predictors, from the poorest to the
richest: the row's own direction with one fitted scale (growth); the common flow, a single linear map on the row
(e528); one fitted scale per origin-time target direction (the row itself, the context at the row's block and at
block 12 with the own write removed, the co-active writes, the read direction, the state at the best position, each
scaled by the row's norm); a full linear map on each target (six maps); a common direction (the mean displacement of
the training rows). Then the hindsight bound: the full linear model on the same context features computed at the
destination checkpoint (the own write removed, so the row's own displacement is not fed back). Direction and size
are scored apart: the energy explained and the median cosine on held-out rows, for the whole displacement and for
its part orthogonal to the row (the non-radial motion), by group; and the size of the displacement regressed on
scalar features (row norm, firing rate, mean activation, S, count over the floor, block) on held-out rows.
Pre-registered (honest guesses), origins 4000 and 8000, blocks 0-12 rows:
- growth alone explains 8-12% of the displacement's energy on held-out rows (0.5);
- the full linear model on origin-time targets explains under 20% (0.6), and under 10% of the non-radial part (0.6);
- the hindsight model explains at least twice what the origin model does (0.5);
- entrants are no more predictable than matched rows (energy explained within 0.05) (0.6);
- the size of the displacement is predictable from the scalar features at 0.3 or above (0.5).
Arguments: name step (the origin checkpoint, thousands)."""
import sys, os, json as _json, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; t0 = int(sys.argv[2]) * 1000; t1 = t0 + 8000; CDIR = f"/workspace/wdd/cache/e524_{name}"; LB = 12; B = 12; NWORD = 256
ck = {n: torch.load(f"{CDIR}/step{n}.pt") for n in (t0, t1)}; D = ck[t0]["rows"].shape[1]; DFF = 4096; R = (LB + 1) * DFF
ids = eval_ids(name).to(DEV); B_, T_ = ids.shape
def auc(pos, neg):
    if pos.numel() < 5 or neg.numel() < 5: return None
    x = torch.cat([pos, neg]); r = x.argsort().argsort().double() + 1; npos, nneg = pos.numel(), neg.numel(); return float((r[:npos].sum() - npos * (npos + 1) / 2) / (npos * nneg))
def features(step):
    """origin-time (or destination-time) context features for every row of blocks 0-12, from one pass over the evaluation text"""
    model, tok, fam = load_model(name, revision=f"step{step}"); arch = Arch(model, fam)
    for p in model.parameters(): p.requires_grad_(False)
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
    keep = ~sinkmask(X[LB]); Xc = {bb: (X[bb][keep] - X[bb][keep].mean(0)) for bb in range(LB + 1)}
    W = torch.cat([arch.wdir(bb) for bb in range(LB + 1)]); Rd = torch.cat([unitr(arch.rdir(bb)) for bb in range(LB + 1)])
    T1 = torch.zeros(R, D, device=DEV); T2 = torch.zeros(R, D, device=DEV); T4 = torch.zeros(R, D, device=DEV); rate = torch.zeros(R, device=DEV); amean = torch.zeros(R, device=DEV)
    for bb in range(LB + 1):
        a = A[bb][keep].float(); s = a.sum(0).clamp_min(1e-6); s2 = a.pow(2).sum(0); sl = slice(bb * DFF, (bb + 1) * DFF); w = W[sl]; own = (s2 / s)[:, None] * w
        T1[sl] = (a.T @ Xc[bb]) / s[:, None] - own; T2[sl] = (a.T @ Xc[LB]) / s[:, None] - own
        bias = arch.mlp_bias(bb); mo = MLP[bb][keep] - (bias[None] if bias is not None else 0); T4[sl] = (a.T @ mo) / s[:, None] - own; rate[sl] = (a > 0).float().mean(0); amean[sl] = a.mean(0); del a
    del model, A, X, MLP, Xc; torch.cuda.empty_cache()
    return dict(T1=unitr(T1), T2=unitr(T2), T4=unitr(T4), Rd=Rd, rate=rate, amean=amean, W=W)
F0 = features(t0); F1 = features(t1); log(f"{name} step{t0}: features at the origin and the destination computed")
W0 = (ck[t0]["rows"].float() * ck[t0]["norms"][:, None]).to(DEV); W1 = (ck[t1]["rows"].float() * ck[t1]["norms"][:, None]).to(DEV); Y = W1 - W0; wn = W0.norm(dim=1); wu = unitr(W0)
Bk0, Bk1 = ck[t0]["blocks"][B], ck[t1]["blocks"][B]; U0 = Bk0["U"].float().to(DEV)[Bk0["a"].to(DEV)]; U1 = Bk1["U"].float().to(DEV)[Bk1["a"].to(DEV)]; S0 = Bk0["S"].float().to(DEV); cnt0 = Bk0["cnt"].float().to(DEV)
words = {}
for n, Bk in ((t0, Bk0), (t1, Bk1)):
    u = Bk["usage"]; wd = torch.zeros(R, dtype=torch.bool); wd[torch.nonzero(u > 0)[:, 0][u[u > 0].argsort(descending=True)[:NWORD]]] = True; words[n] = wd.to(DEV)
ent = ~words[t0] & words[t1]; nonw = ~words[t0]; S0c = S0.cpu(); pool = torch.nonzero((nonw & ~words[t1]).cpu())[:, 0]; taken = torch.zeros(R, dtype=torch.bool); matched = []
for i in torch.nonzero(ent.cpu())[:, 0].tolist():
    c = pool[~taken[pool]]; j = c[(S0c[c] - S0c[i]).abs().argmin()]; taken[j] = True; matched.append(int(j))
mm = torch.zeros(R, dtype=torch.bool, device=DEV); mm[torch.tensor(matched, device=DEV)] = True; groups = {"entrants": ent, "matched": mm, "non_words": nonw, "words": words[t0]}
gen = torch.Generator(device=DEV).manual_seed(0); perm = torch.randperm(R, device=DEV, generator=gen); train = torch.zeros(R, dtype=torch.bool, device=DEV); train[perm[:R // 2]] = True; test = ~train
Yp = Y - (Y * wu).sum(1, keepdim=True) * wu                                                                                # the non-radial part of the displacement
def ridge_fit(Xf, Yt, lam=None):
    """ridge regression; with lam None the ridge is chosen on a validation fifth of the given rows over a grid, then refitted on all of them"""
    if lam is None:
        n = Xf.shape[0]; g_ = torch.Generator(device=DEV).manual_seed(1); pv = torch.randperm(n, device=DEV, generator=g_); va = pv[:n // 5]; tr = pv[n // 5:]; Gt = Xf[tr].T @ Xf[tr]; Bt = Xf[tr].T @ Yt[tr]; best = None
        for l in (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0):
            Mt = torch.linalg.solve(Gt + l * Gt.diagonal().mean() * torch.eye(Gt.shape[0], device=DEV), Bt); e = float((Yt[va] - Xf[va] @ Mt).pow(2).sum())
            if best is None or e < best[0]: best = (e, l)
        lam = best[1]; RIDGE_CHOSEN.append(lam)
    G = Xf.T @ Xf; G = G + lam * G.diagonal().mean() * torch.eye(G.shape[0], device=DEV); return torch.linalg.solve(G, Xf.T @ Yt)
RIDGE_CHOSEN = []
def score(pred, target, mask):
    e = float(1 - (target[mask] - pred[mask]).pow(2).sum() / target[mask].pow(2).sum()); c = float(((pred[mask] * target[mask]).sum(1) / (pred[mask].norm(dim=1) * target[mask].norm(dim=1)).clamp_min(1e-12)).median()); return dict(energy=e, cos=c)
def evaluate(pred):
    out = {}; pp = pred - (pred * wu).sum(1, keepdim=True) * wu
    for gname, gm in (("held_out", test), ("entrants", ent & test), ("matched", mm & test), ("non_words", nonw & test), ("words", words[t0] & test)):
        if int(gm.sum()) >= 5: out[gname] = dict(whole=score(pred, Y, gm), non_radial=score(pp, Yp, gm))
    return out
models = {}
# growth: one fitted scale on the row
c = float((Y[train] * W0[train]).sum() / W0[train].pow(2).sum()); models["growth"] = evaluate(c * W0)
# common direction: the mean displacement of the training rows, scaled by the row's norm
md = Y[train].mean(0); md = md / md.norm(); cd = float((Y[train] @ md).sum() / wn[train].sum()); models["common_direction"] = evaluate(cd * wn[:, None] * md[None])
# common flow: one linear map on the row
M = ridge_fit(W0[train], Y[train]); models["common_flow"] = evaluate(W0 @ M)
# origin targets: the row itself, contexts at the block and at block 12, co-active writes, read direction, best state
TG = {"row": wu, "context_block": F0["T1"], "context_12": F0["T2"], "coactive": F0["T4"], "read": F0["Rd"], "best_state": U0}
Xs = torch.stack([TG[k] * wn[:, None] for k in TG], 2)                                                                     # [R, D, 6]
# one scale per target: solve the 6 x 6 normal equations over all training rows and coordinates
A6 = torch.einsum("rdk,rdl->kl", Xs[train], Xs[train]); b6 = torch.einsum("rdk,rd->k", Xs[train], Y[train]); beta = torch.linalg.solve(A6 + 1e-6 * A6.diagonal().mean() * torch.eye(6, device=DEV), b6); models["target_scales"] = evaluate(Xs @ beta); models["target_scales"]["beta"] = {k: float(beta[i]) for i, k in enumerate(TG)}
# a full linear map on each target (six maps, fitted together)
Xf = torch.cat([TG[k] * wn[:, None] for k in TG], 1); Mf = ridge_fit(Xf[train], Y[train]); models["target_maps"] = evaluate(Xf @ Mf); del Xf
# hindsight: the same maps on the destination's context features (own write removed; the row's own displacement not fed back)
TGh = {"context_block": F1["T1"], "context_12": F1["T2"], "coactive": F1["T4"], "read": F1["Rd"], "best_state_dest": U1, "row": wu}
Xh = torch.cat([TGh[k] * wn[:, None] for k in TGh], 1); Mh = ridge_fit(Xh[train], Y[train]); models["hindsight_maps"] = evaluate(Xh @ Mh); del Xh
Xh2 = torch.cat([TGh[k] * wn[:, None] for k in ("context_block", "context_12", "coactive", "best_state_dest")], 1); Mh2 = ridge_fit(Xh2[train], Y[train]); models["hindsight_maps_no_row"] = evaluate(Xh2 @ Mh2); del Xh2
Xo2 = torch.cat([TG[k] * wn[:, None] for k in ("context_block", "context_12", "coactive", "best_state")], 1); Mo2 = ridge_fit(Xo2[train], Y[train]); models["origin_maps_no_row"] = evaluate(Xo2 @ Mo2); del Xo2
# the matched row's displacement as a predictor of the entrant's (shared displacement among rows of like size)
ents = torch.nonzero(ent)[:, 0]; mt = torch.tensor(matched, device=DEV); cm = ((Y[ents] * Y[mt]).sum(1) / (Y[ents].norm(dim=1) * Y[mt].norm(dim=1)).clamp_min(1e-12)); rnd = Y[torch.randperm(R, device=DEV, generator=gen)[:ents.numel()]]; cr = ((Y[ents] * rnd).sum(1) / (Y[ents].norm(dim=1) * rnd.norm(dim=1)).clamp_min(1e-12))
models["matched_row_displacement"] = dict(cos_entrant_vs_matched=float(cm.median()), cos_entrant_vs_random_row=float(cr.median()))
# the size of the displacement from scalar features
blk = torch.arange(R, device=DEV) // DFF; onehot = torch.nn.functional.one_hot(blk, LB + 1).float()
Z = torch.cat([wn[:, None], F0["rate"][:, None], F0["amean"][:, None], S0[:, None], cnt0[:, None], torch.log(wn.clamp_min(1e-6))[:, None], onehot, torch.ones(R, 1, device=DEV)], 1); yn = Y.norm(dim=1)
bz = ridge_fit(Z[train], yn[train, None], lam=1e-6)[:, 0]; pz = Z @ bz; size_r2 = float(1 - (yn[test] - pz[test]).pow(2).sum() / (yn[test] - yn[train].mean()).pow(2).sum())
Zr = torch.cat([F0["rate"][:, None], F0["amean"][:, None], S0[:, None], cnt0[:, None], onehot, torch.ones(R, 1, device=DEV)], 1); yr = yn / wn.clamp_min(1e-6); br = ridge_fit(Zr[train], yr[train, None], lam=1e-6)[:, 0]; pr = Zr @ br; rel_r2 = float(1 - (yr[test] - pr[test]).pow(2).sum() / (yr[test] - yr[train].mean()).pow(2).sum())
Zn = Z[:, [0, -1]]; size_norm_only = float(1 - (yn[test] - (Zn @ ridge_fit(Zn[train], yn[train, None], lam=1e-6))[test, 0]).pow(2).sum() / (yn[test] - yn[train].mean()).pow(2).sum())
res = dict(model=name, origin=t0, destination=t1, n_rows=R, n_train=int(train.sum()), n_entrants=int(ent.sum()), n_matched=len(matched), ridge_chosen=RIDGE_CHOSEN, models=models, size=dict(r2_scalar_features=size_r2, r2_norm_only=size_norm_only, r2_relative_size=rel_r2), radial_share_of_displacement=float(((Y * wu).sum(1).pow(2) / Y.pow(2).sum(1).clamp_min(1e-12)).median()), median_relative_displacement=float((yn / wn).median()))
g_ = lambda m, gname="held_out", part="whole", k="energy": models[m].get(gname, {}).get(part, {}).get(k)
res["checks"] = dict(growth_8_to_12=0.08 <= g_("growth") <= 0.12, origin_maps_under_20=g_("target_maps") < 0.2, origin_maps_non_radial_under_10=g_("target_maps", part="non_radial") < 0.1, hindsight_at_least_double=g_("hindsight_maps") >= 2 * max(g_("target_maps"), 1e-6), entrants_not_more_predictable=abs(g_("target_maps", "entrants") - g_("target_maps", "matched")) < 0.05, size_predictable_0_3=size_r2 >= 0.3)
fm = lambda x: "n/a" if x is None else f"{x:.3f}"
for m in ("growth", "common_direction", "common_flow", "target_scales", "target_maps", "origin_maps_no_row", "hindsight_maps", "hindsight_maps_no_row"):
    o = models[m]; log(f"{name} step{t0}->{t1} {m}: held-out energy {o['held_out']['whole']['energy']:.3f} (cosine {o['held_out']['whole']['cos']:.3f}); non-radial {o['held_out']['non_radial']['energy']:.3f} ({o['held_out']['non_radial']['cos']:.3f}); entrants {fm(o.get('entrants', {}).get('whole', {}).get('energy'))} / matched {fm(o.get('matched', {}).get('whole', {}).get('energy'))} / words {fm(o.get('words', {}).get('whole', {}).get('energy'))}" + (f"; scales {_json.dumps({k: round(v, 3) for k, v in o['beta'].items()})}" if "beta" in o else ""))
log(f"{name} step{t0}->{t1}: ridge chosen for the map models {RIDGE_CHOSEN}; radial share of the displacement {res['radial_share_of_displacement']:.3f}, median relative displacement {res['median_relative_displacement']:.2f}; size from scalar features R2 {size_r2:.3f} (norm alone {size_norm_only:.3f}; relative size {rel_r2:.3f}); entrant's displacement vs matched row's cosine {models['matched_row_displacement']['cos_entrant_vs_matched']:.3f} (vs a random row {models['matched_row_displacement']['cos_entrant_vs_random_row']:.3f})")
summ = f"{name} step{t0}->{t1} ({res['n_entrants']} entrants over the window): held-out energy explained, whole / non-radial: growth {g_('growth'):.3f}/{g_('growth', part='non_radial'):.3f}, common direction {g_('common_direction'):.3f}, common flow {g_('common_flow'):.3f}/{g_('common_flow', part='non_radial'):.3f}, six target scales {g_('target_scales'):.3f}, six target maps {g_('target_maps'):.3f}/{g_('target_maps', part='non_radial'):.3f} (entrants {fm(g_('target_maps', 'entrants'))}, matched {fm(g_('target_maps', 'matched'))}), hindsight maps {g_('hindsight_maps'):.3f}/{g_('hindsight_maps', part='non_radial'):.3f} (without the row {g_('hindsight_maps_no_row'):.3f}; origin without the row {g_('origin_maps_no_row'):.3f}); median cosine origin maps {g_('target_maps', k='cos'):.3f}, hindsight {g_('hindsight_maps', k='cos'):.3f}; size R2 {size_r2:.3f} (norm alone {size_norm_only:.3f}); radial share {res['radial_share_of_displacement']:.3f} | checks {_json.dumps(res['checks'])}"
log(summ); record(f"e530_predictability_{name}_step{t0}_v2", res, summ)
