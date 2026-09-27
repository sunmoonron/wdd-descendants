"""e522: does the loss's gradient explain which native rows the state comes to align with? e521 found that a future
word's alignment is made locally by the mass of small writes at its position, by no one in particular. The question
moves to training: why does optimisation make that mass point along particular native rows? The direct test is the
gradient. At an origin checkpoint the mean loss on 24 x 512 held-out tokens (the gradient text) is differentiated
with respect to every parameter, and the model is moved one small step along the negative gradient (the gradient
flow; the step sized so that the first-order loss decrease is 0.01 nats) and, separately, one step along the sign of
the gradient (the Adam-like direction, sized the same way). The alignment of every non-word row, S (its largest
centred projection over the extreme-value floor on 8 x 256 measurement tokens, e516's criterion), is recomputed after
each step, with the rows and states both moved, the states alone moved (rows held at the origin) and the rows alone
moved (states held): the one-step prediction of the change of S and its split into state motion and row motion.
The same split is applied to the actual change from the origin to the next checkpoint: S with the next checkpoint's
states and rows, with its states and the origin's rows, and with the origin's states and its rows. Two local pulls
are recorded as well, at each row's best position: the state pull, the negative gradient of the measurement text's
loss with respect to the state there, projected on the row (signed along the projection), and the row pull, the
negative gradient of the loss with respect to the row's own parameters projected on the state's component orthogonal
to the row (the first-order rotation of the row toward the state); with the row's growth pull and gradient norm.
Predictors are scored for entry at the next checkpoint (AUC over all non-words; within deciles of S, the
conditional test the relayed take asked for; against S-matched non-entrants) and for the actual change of S
(Spearman, over all non-words and within deciles of S). Linearity is checked with a doubled step.
Setup: Pythia-410m, blocks 12 and 6; origin checkpoint as argument; the next checkpoint's word set from e519's records.
Pre-registered (honest guesses), block 12:
- the gradient-flow prediction of the change of S predicts entry over all non-words at AUC 0.7 or above (0.35);
- within deciles of S it separates entrants from non-entrants at 0.6 or above (0.3);
- its Spearman with the actual change of S over all non-words is 0.2 or above (0.5);
- the actual rise of the entrants' S is mostly state motion (state share above 0.5) (0.6);
- the local state pull is positive for at least 60% of the entrants (0.5).
Arguments: name revision (the origin checkpoint)."""
import sys, os, json as _json, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; rev = sys.argv[2]; STEPS = ["step0", "step64", "step256", "step512", "step1000", "step2000", "step3000", "step4000", "step8000", "step16000", "step32000", "step64000", "main"]
assert rev in STEPS[:-1], rev; NEXTS = STEPS[STEPS.index(rev) + 1]; CDIR = f"/workspace/wdd/cache/e519_{name}"; NWORD = 256; BL = [12, 6]; LB = max(BL); DLOSS = 0.01
ref = {s: torch.load(f"{CDIR}/{s}.pt") for s in (rev, NEXTS)}
ids_all = eval_ids(name); idsB = ids_all[:8, :256].to(DEV); idsA = ids_all[8:32, :512].to(DEV)
model, tok, fam = load_model(name, revision=rev); arch = Arch(model, fam); D, DFF = arch.D, arch.DFF; layers = arch.layers
for p in model.parameters(): p.requires_grad_(True)
def gabs(m, T=14.0, n=28001):
    t = torch.linspace(0, T, n, dtype=torch.float64); F = torch.special.erf(t / math.sqrt(2)); return float(torch.trapezoid(1 - F.pow(m), t))
def auc(pos, neg):
    if pos.numel() < 5 or neg.numel() < 5: return None
    x = torch.cat([pos, neg]); r = x.argsort().argsort().double() + 1; npos, nneg = pos.numel(), neg.numel(); return float((r[:npos].sum() - npos * (npos + 1) / 2) / (npos * nneg))
def spearman(a, b):
    if a.numel() < 5: return None
    ra = a.argsort().argsort().double(); rb = b.argsort().argsort().double(); ra = ra - ra.mean(); rb = rb - rb.mean(); return float((ra * rb).sum() / (ra.norm() * rb.norm()).clamp_min(1e-12))
def strat(fn, q, s, y, nbins=10):
    """fn applied within deciles of s (auc: y boolean; spearman: y real), weighted by the positives (auc) or the count (spearman)"""
    edges = s.quantile(torch.linspace(0, 1, nbins + 1)); tot, wsum = 0.0, 0
    for i in range(nbins):
        m = (s >= edges[i]) & (s <= edges[i + 1]) if i == nbins - 1 else (s >= edges[i]) & (s < edges[i + 1])
        if fn == "auc": v = auc(q[m & y], q[m & ~y]); n = int((m & y).sum())
        else: v = spearman(q[m], y[m]); n = int(m.sum())
        if v is not None: tot += v * n; wsum += n
    return (tot / wsum) if wsum else None
def mean_loss(ids, chunk=4, backward=False):
    tot = 0.0; ntok = ids.shape[0] * (ids.shape[1] - 1)
    for s in range(0, ids.shape[0], chunk):
        c = ids[s:s + chunk]; wgt = c.shape[0] * (c.shape[1] - 1) / ntok
        if backward:
            with torch.enable_grad(): l = model(c, labels=c, use_cache=False).loss * wgt; l.backward(); tot += float(l)
        else:
            with torch.no_grad(): tot += float(model(c, labels=c, use_cache=False).loss) * wgt
    return tot
def state_grads(ids):
    """{block: [B, T-1, D]} gradient of the mean loss on ids with respect to the block's output at positions 1:"""
    cap = {}
    def hk(bb):
        def f(m, i, o): cap[bb] = o[0] if isinstance(o, tuple) else o
        return f
    hs = [layers[bb].register_forward_hook(hk(bb)) for bb in BL]
    try:
        with torch.enable_grad(): l = model(ids, labels=ids, use_cache=False).loss; g = torch.autograd.grad(l, [cap[bb] for bb in BL])
    finally: [h.remove() for h in hs]
    return {bb: g[i][:, 1:].detach().float() for i, bb in enumerate(BL)}
def dictionary(arch_, b):
    A, lab = build_dictionary(arch_, blocks=list(range(b + 1))); Au = unitr(A); Ar = unitr(rotate(A, seed=7)); del A
    typ = lab["type"].to(DEV); blk = lab["block"].to(DEV); idx = lab["index"].to(DEV); gidx = torch.full((b + 1, DFF), -1, device=DEV, dtype=torch.long)
    for bb in range(b + 1): mm = (typ == T_MLP) & (blk == bb); gidx[bb, idx[mm]] = torch.nonzero(mm)[:, 0]
    m = Au.shape[0]; d = dict(Am=Au[gidx.reshape(-1)].clone(), CA=Au.T @ Au / m, CR=Ar.T @ Ar / m, Ar=Ar, gm=gabs(m)); del Au; return d
def S_of(X, d):
    """max over the floor [R], its position [R], the signed projection there (unit-state units) [R], the centred states, for kept states X [N, D]"""
    mu = X.mean(0); Xc = X - mu; U = unitr(Xc); N = U.shape[0]
    s2 = ((U @ d["CA"]) * U).sum(1); s2r = ((U @ d["CR"]) * U).sum(1); Mr = torch.cat([(U[s:s + 128] @ d["Ar"].T).abs().max(1).values for s in range(0, N, 128)])
    r_cal = float(Mr.mean() / (s2r.clamp_min(1e-12).sqrt() * d["gm"]).mean()); L = s2.clamp_min(1e-12).sqrt() * d["gm"] * r_cal
    P = U @ d["Am"].T; v, a = (P.abs() / L[:, None]).max(0); return v, a, P.gather(0, a[None])[0], Xc
kept = lambda X: X.reshape(-1, D)[~sinkmask(X.reshape(-1, D))]
# the origin: states, dictionaries, S, local gradients
X0 = block_states(model, arch, idsB, BL, chunk=4); d0 = {b: dictionary(arch, b) for b in BL}; base = {b: S_of(kept(X0[b]), d0[b]) for b in BL}
wn0 = {b: torch.cat([arch.wdir(bb).norm(dim=-1) for bb in range(b + 1)]) for b in BL}; W0 = {b: d0[b]["Am"] for b in BL}
gB = state_grads(idsB); lossB0 = mean_loss(idsB)
for p in model.parameters(): p.grad = None
lossA0 = mean_loss(idsA, backward=True)
gn2 = sum(float(p.grad.pow(2).sum()) for p in model.parameters()); gn1 = sum(float(p.grad.abs().sum()) for p in model.parameters()); eta = {"flow": DLOSS / gn2, "sign": DLOSS / gn1}
Gw = {bb: (arch.mlp_lin(bb).weight.grad.float() if fam == "gpt2" else arch.mlp_lin(bb).weight.grad.float().T).clone() for bb in range(LB + 1)}
theta0 = [p.detach().clone() for p in model.parameters()]; tn = math.sqrt(sum(float(t.pow(2).sum()) for t in theta0))
local = {}
for b in BL:
    v0, a0, p0, Xc0 = base[b]; w = W0[b]; sgn = torch.sign(p0); sgn[sgn == 0] = 1
    gsel = gB[b].reshape(-1, D)[~sinkmask(X0[b].reshape(-1, D))][a0]; g_state = -(gsel * w).sum(1) * sgn; g_state_cos = g_state / gsel.norm(dim=1).clamp_min(1e-12)
    xsel = Xc0[a0]; xperp = xsel - w * (xsel * w).sum(1, keepdim=True); G = torch.cat([Gw[bb] for bb in range(b + 1)]); gr = -(G * xperp).sum(1)
    local[b] = dict(g_state=g_state.cpu(), g_state_cos=g_state_cos.cpu(), g_row=(gr / wn0[b] * sgn).cpu(), g_row_cos=(gr / (G.norm(dim=1) * xperp.norm(dim=1)).clamp_min(1e-12) * sgn).cpu(), row_growth=(-(G * w).sum(1)).cpu(), row_grad_norm=(G.norm(dim=1) / wn0[b]).cpu())
    del gsel, xsel, xperp, G
# one step along the gradient flow and along the sign of the gradient, single and doubled
step = {}
for kind in ("flow", "sign"):
    for scale in (1, 2):
        with torch.no_grad():
            for p in model.parameters(): p.sub_((eta[kind] * scale) * (p.grad if kind == "flow" else p.grad.sign()))
            dn = math.sqrt(sum(float((p - t).pow(2).sum()) for p, t in zip(model.parameters(), theta0)))
        Xs = block_states(model, arch, idsB, BL, chunk=4); rec = dict(lossA=mean_loss(idsA), lossB=mean_loss(idsB), rel_param_change=dn / tn, by_block={})
        for b in BL:
            ds = dictionary(arch, b); Xks = kept(Xs[b]); wn = torch.cat([arch.wdir(bb).norm(dim=-1) for bb in range(b + 1)])
            rec["by_block"][b] = dict(full=S_of(Xks, ds)[0].cpu(), state=S_of(Xks, d0[b])[0].cpu(), row=S_of(kept(X0[b]), ds)[0].cpu(), row_rel_change=float((((ds["Am"] * wn[:, None]) - (W0[b] * wn0[b][:, None])).norm(dim=1) / wn0[b]).median()), state_rel_change=float(((Xks - kept(X0[b])).norm(dim=1) / kept(X0[b]).norm(dim=1)).median()) if Xks.shape == kept(X0[b]).shape else None)
            del ds
        step[(kind, scale)] = rec
        with torch.no_grad():
            for p, t in zip(model.parameters(), theta0): p.copy_(t)
        log(f"{name} {rev}: {kind} step x{scale}: loss on the gradient text {lossA0:.4f} -> {rec['lossA']:.4f}, on the measurement text {lossB0:.4f} -> {rec['lossB']:.4f}; relative parameter change {rec['rel_param_change']:.2e}; block 12 row change {rec['by_block'][12]['row_rel_change']:.2e}, state change {rec['by_block'][12]['state_rel_change']}")
for p in model.parameters(): p.grad = None; p.requires_grad_(False)
del theta0, Gw, gB; torch.cuda.empty_cache()
# the next checkpoint: the actual change and its split
model1, _, _ = load_model(name, revision=NEXTS); arch1 = Arch(model1, fam)
for p in model1.parameters(): p.requires_grad_(False)
X1 = block_states(model1, arch1, idsB, BL, chunk=4); nxt = {}
for b in BL:
    d1 = dictionary(arch1, b); Xk1 = kept(X1[b]); wn1 = torch.cat([arch1.wdir(bb).norm(dim=-1) for bb in range(b + 1)])
    nxt[b] = dict(full=S_of(Xk1, d1)[0].cpu(), state=S_of(Xk1, d0[b])[0].cpu(), row=S_of(kept(X0[b]), d1)[0].cpu(), row_cos=(d1["Am"] * W0[b]).sum(1).cpu(), row_norm_ratio=(wn1 / wn0[b]).cpu()); del d1
del model1, X1; torch.cuda.empty_cache()
# scoring
res = dict(model=name, origin=rev, horizon=NEXTS, loss=dict(gradient_text=lossA0, measurement_text=lossB0), eta=eta, steps={f"{k}_x{s}": dict(lossA=r["lossA"], lossB=r["lossB"], rel_param_change=r["rel_param_change"], row_rel_change={b: r["by_block"][b]["row_rel_change"] for b in BL}, state_rel_change={b: r["by_block"][b]["state_rel_change"] for b in BL}) for (k, s), r in step.items()}, by_block={})
for b in BL:
    u0, u1 = ref[rev][b]["usage"], ref[NEXTS][b]["usage"]; Sc = ref[rev][b]["max_over_floor"]["real"].float(); R = u0.numel()
    Wset0 = set(torch.nonzero(u0 > 0)[:, 0][u0[u0 > 0].argsort(descending=True)[:NWORD]].tolist()); Wset1 = set(torch.nonzero(u1 > 0)[:, 0][u1[u1 > 0].argsort(descending=True)[:NWORD]].tolist())
    in0 = torch.zeros(R, dtype=torch.bool); in0[list(Wset0)] = True; in1 = torch.zeros(R, dtype=torch.bool); in1[list(Wset1)] = True; cand = torch.nonzero(~in0)[:, 0]; ent = in1[cand]
    S0 = base[b][0].cpu()[cand]; Q = dict(S=S0)
    for kind in ("flow", "sign"):
        for part in ("full", "state", "row"): Q[f"{kind}_{part}"] = step[(kind, 1)]["by_block"][b][part][cand] - S0
    for k in ("g_state", "g_state_cos", "g_row", "g_row_cos", "row_growth", "row_grad_norm"): Q[k] = local[b][k][cand]
    dS = nxt[b]["full"][cand] - S0; dS_state = nxt[b]["state"][cand] - S0; dS_row = nxt[b]["row"][cand] - S0
    pool = torch.nonzero(~ent)[:, 0]; taken = torch.zeros(cand.numel(), dtype=torch.bool); matched = []
    for i in torch.nonzero(ent)[:, 0].tolist():
        c = pool[~taken[pool]]; j = c[(S0[c] - S0[i]).abs().argmin()]; taken[j] = True; matched.append(int(j))
    matched = torch.tensor(matched, dtype=torch.long); ent_idx = torch.nonzero(ent)[:, 0]
    pred = {}
    for k, q in Q.items():
        q = q.nan_to_num(0)
        pred[k] = dict(auc_all=auc(q[ent], q[~ent]), auc_within_S=None if k == "S" else strat("auc", q, S0, ent), auc_matched=auc(q[ent_idx], q[matched]), spearman_dS=spearman(q, dS), spearman_dS_within_S=None if k == "S" else strat("sp", q, S0, dS), spearman_dS_state=spearman(q, dS_state), spearman_dS_row=spearman(q, dS_row), positive_fraction_entrants=float((q[ent] > 0).float().mean()), median_entrants=float(q[ent].median()), median_matched=float(q[matched].median()), median_all=float(q.median()))
    def split(full, st, rw, m):
        rise = full[m]; ok = rise > 0.2
        return dict(n=int(m.sum()), n_rising=int(ok.sum()), median_S0=float(S0[m].median()), median_dS=float(rise.median()), state_share=float((st[m][ok] / rise[ok]).median()) if int(ok.sum()) >= 5 else None, row_share=float((rw[m][ok] / rise[ok]).median()) if int(ok.sum()) >= 5 else None, sum_of_shares=float(((st[m][ok] + rw[m][ok]) / rise[ok]).median()) if int(ok.sum()) >= 5 else None)
    mm = torch.zeros(cand.numel(), dtype=torch.bool); mm[matched] = True
    who = {g: split(dS, dS_state, dS_row, m) for g, m in (("entrants", ent), ("matched", mm), ("all", torch.ones_like(ent)))}
    ff, fs, fr = Q["flow_full"], Q["flow_state"], Q["flow_row"]; flow_split = {}
    for g, m in (("entrants", ent), ("matched", mm), ("all", torch.ones_like(ent))):
        ok = m & (ff.abs() > 1e-6); flow_split[g] = dict(median_dS_flow=float(ff[m].median()), state_share=float((fs[ok] / ff[ok]).median()) if int(ok.sum()) >= 5 else None, row_share=float((fr[ok] / ff[ok]).median()) if int(ok.sum()) >= 5 else None)
    f2 = step[("flow", 2)]["by_block"][b]["full"][cand] - S0; s2 = step[("sign", 2)]["by_block"][b]["full"][cand] - S0
    lin = dict(flow_spearman_x1_x2=spearman(ff, f2), flow_median_ratio_x2_over_x1=float((f2[ff.abs() > 1e-6] / ff[ff.abs() > 1e-6]).median()), sign_spearman_x1_x2=spearman(Q["sign_full"], s2), flow_vs_sign_spearman=spearman(ff, Q["sign_full"]))
    res["by_block"][b] = dict(n_candidates=int(cand.numel()), n_entrants=int(ent.sum()), S_check_spearman_with_e519=spearman(S0, Sc[cand]), predictors=pred, who_moved=who, flow_split=flow_split, linearity=lin,
                              rows=dict(entrants_row_cos=float(nxt[b]["row_cos"][cand][ent].median()), matched_row_cos=float(nxt[b]["row_cos"][cand][matched].median()), all_row_cos=float(nxt[b]["row_cos"][cand].median()), entrants_row_norm_ratio=float(nxt[b]["row_norm_ratio"][cand][ent].median()), all_row_norm_ratio=float(nxt[b]["row_norm_ratio"][cand].median())))
    fm = lambda x: "n/a" if x is None else f"{x:.2f}"; P_ = pred
    log(f"{name} {rev} block {b} ({int(ent.sum())} entrants of {cand.numel()} non-words; S agrees with e519 at Spearman {fm(res['by_block'][b]['S_check_spearman_with_e519'])}): entry AUC over all non-words / within deciles of S / vs matched: S {fm(P_['S']['auc_all'])}; gradient-flow dS {fm(P_['flow_full']['auc_all'])}/{fm(P_['flow_full']['auc_within_S'])}/{fm(P_['flow_full']['auc_matched'])} (state part {fm(P_['flow_state']['auc_all'])}/{fm(P_['flow_state']['auc_within_S'])}, row part {fm(P_['flow_row']['auc_all'])}/{fm(P_['flow_row']['auc_within_S'])}); sign-step dS {fm(P_['sign_full']['auc_all'])}/{fm(P_['sign_full']['auc_within_S'])}/{fm(P_['sign_full']['auc_matched'])}; state pull {fm(P_['g_state']['auc_all'])}/{fm(P_['g_state']['auc_within_S'])} (cos {fm(P_['g_state_cos']['auc_all'])}/{fm(P_['g_state_cos']['auc_within_S'])}); row pull {fm(P_['g_row']['auc_all'])}/{fm(P_['g_row']['auc_within_S'])} (cos {fm(P_['g_row_cos']['auc_all'])}/{fm(P_['g_row_cos']['auc_within_S'])}); row growth {fm(P_['row_growth']['auc_all'])}/{fm(P_['row_growth']['auc_within_S'])}; row gradient norm {fm(P_['row_grad_norm']['auc_all'])}/{fm(P_['row_grad_norm']['auc_within_S'])}")
    log(f"{name} {rev} block {b}: Spearman with the actual change of S, all / within deciles of S: S {fm(P_['S']['spearman_dS'])}; flow dS {fm(P_['flow_full']['spearman_dS'])}/{fm(P_['flow_full']['spearman_dS_within_S'])} (state part with the actual state part {fm(P_['flow_state']['spearman_dS_state'])}, row part with the actual row part {fm(P_['flow_row']['spearman_dS_row'])}); sign dS {fm(P_['sign_full']['spearman_dS'])}/{fm(P_['sign_full']['spearman_dS_within_S'])}; state pull {fm(P_['g_state']['spearman_dS'])}/{fm(P_['g_state']['spearman_dS_within_S'])}; row pull {fm(P_['g_row']['spearman_dS'])}/{fm(P_['g_row']['spearman_dS_within_S'])}; linearity: flow x1 vs x2 Spearman {fm(lin['flow_spearman_x1_x2'])}, ratio {fm(lin['flow_median_ratio_x2_over_x1'])}; flow vs sign {fm(lin['flow_vs_sign_spearman'])}")
    w_ = who["entrants"]; wm = who["matched"]; fe = flow_split["entrants"]
    log(f"{name} {rev} block {b}: who moved, entrants: S {w_['median_S0']:.2f} -> +{w_['median_dS']:.2f}; state share {fm(w_['state_share'])}, row share {fm(w_['row_share'])}, sum {fm(w_['sum_of_shares'])} (matched: +{wm['median_dS']:.2f}, {fm(wm['state_share'])}/{fm(wm['row_share'])}); the flow step's split, entrants: state {fm(fe['state_share'])}, row {fm(fe['row_share'])}; row cosine origin to next, entrants/matched/all {res['by_block'][b]['rows']['entrants_row_cos']:.3f}/{res['by_block'][b]['rows']['matched_row_cos']:.3f}/{res['by_block'][b]['rows']['all_row_cos']:.3f}; positive fractions among entrants: flow dS {P_['flow_full']['positive_fraction_entrants']:.2f}, state pull {P_['g_state']['positive_fraction_entrants']:.2f}, row pull {P_['g_row']['positive_fraction_entrants']:.2f}")
B12 = res["by_block"][12]; Pf = B12["predictors"]["flow_full"]; g_ = lambda x: -1 if x is None else x
res["checks"] = dict(flow_entry_auc_over_0_7=g_(Pf["auc_all"]) >= 0.7, flow_within_S_over_0_6=g_(Pf["auc_within_S"]) >= 0.6, flow_spearman_actual_over_0_2=g_(Pf["spearman_dS"]) >= 0.2, state_motion_over_half=g_(B12["who_moved"]["entrants"]["state_share"]) > 0.5, state_pull_positive_60=B12["predictors"]["g_state_cos"]["positive_fraction_entrants"] >= 0.6)
fm = lambda x: "n/a" if x is None else f"{x:.2f}"
summ = f"{name} {rev} to {NEXTS}: " + " | ".join(f"block {b}: entry AUC all/within S/matched: flow dS {fm(o['predictors']['flow_full']['auc_all'])}/{fm(o['predictors']['flow_full']['auc_within_S'])}/{fm(o['predictors']['flow_full']['auc_matched'])}, sign dS {fm(o['predictors']['sign_full']['auc_all'])}/{fm(o['predictors']['sign_full']['auc_within_S'])}/{fm(o['predictors']['sign_full']['auc_matched'])}, state pull {fm(o['predictors']['g_state']['auc_all'])}/{fm(o['predictors']['g_state']['auc_within_S'])}, row pull {fm(o['predictors']['g_row']['auc_all'])}/{fm(o['predictors']['g_row']['auc_within_S'])} (S alone {fm(o['predictors']['S']['auc_all'])}); Spearman with actual dS: flow {fm(o['predictors']['flow_full']['spearman_dS'])} (within S {fm(o['predictors']['flow_full']['spearman_dS_within_S'])}), sign {fm(o['predictors']['sign_full']['spearman_dS'])}; entrants' rise: state share {fm(o['who_moved']['entrants']['state_share'])}, row share {fm(o['who_moved']['entrants']['row_share'])}" for b, o in res["by_block"].items()) + f" | checks {_json.dumps(res['checks'])}"
log(summ); record(f"e522_gradient_{name}_{rev}", res, summ)
