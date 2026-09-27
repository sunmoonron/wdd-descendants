"""e522b: is e522's null real or is the gradient estimate too noisy to see? e522 found that a one-step move along the
loss gradient (estimated on 12k held-out tokens) predicts neither which rows enter the vocabulary nor how their
alignment changes, and that the rise of an entrant's alignment is mostly the row rotating toward the states, not the
states moving. A null from a noisy estimate is not a finding. This run estimates the gradient on 65k tokens of the
training distribution (the Pile, NeelNanda/pile-10k; wikitext if unavailable) in two halves, so that the reliability
of every gradient-derived quantity can be read from its split-half agreement across rows, and asks two direct
questions of the actual training interval: whether the rows' accumulated change from the origin to the next
checkpoint follows the origin's gradient (the cosine of w_next - w_origin with the negative row gradient, per row),
and whether the states' actual change at the entrants' best positions follows the one-step flow's predicted change
(the cosine of the two changes, per position; random positions as the reference). The e522 predictions are then
repeated with the larger estimate: the one-step change of S for entry over all non-words, within deciles of S and
against S-matched non-entrants, and against the actual change of S and its row part.
Setup: Pythia-410m, blocks 12 and 6; origin checkpoint as argument; 8 x 256 measurement tokens (e519's);
gradient text 2 x 64 x 512 tokens; the next checkpoint's word set from e519's records.
Pre-registered (honest guesses), block 12:
- the split-half Spearman of the one-step change of S across rows is 0.5 or above (0.4);
- the split-half Spearman of the row pull across rows is 0.5 or above (0.4);
- the rows' accumulated change follows the origin's gradient: positive cosine for at least 70% of rows (0.5);
- the actual state change at entrants' best positions has a positive cosine with the flow's predicted change for
  at least 60% of them (0.4);
- with 65k tokens, the one-step change of S separates entrants within deciles of S at 0.55 or above (0.3).
Arguments: name revision (the origin checkpoint)."""
import sys, os, json as _json, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; rev = sys.argv[2]; STEPS = ["step0", "step64", "step256", "step512", "step1000", "step2000", "step3000", "step4000", "step8000", "step16000", "step32000", "step64000", "main"]
assert rev in STEPS[:-1], rev; NEXTS = STEPS[STEPS.index(rev) + 1]; CDIR = f"/workspace/wdd/cache/e519_{name}"; NWORD = 256; BL = [12, 6]; LB = max(BL); DLOSS = 0.01; NSEQ = 64; SEQ = 512
ref = {s: torch.load(f"{CDIR}/{s}.pt") for s in (rev, NEXTS)}
idsB = eval_ids(name)[:8, :256].to(DEV)
model, tok, fam = load_model(name, revision=rev); arch = Arch(model, fam); D, DFF = arch.D, arch.DFF; layers = arch.layers
def gradient_text():
    from datasets import load_dataset
    need = 2 * NSEQ * SEQ + 1; chunks, n, src = [], 0, "pile-10k"
    try:
        ds = load_dataset("NeelNanda/pile-10k", split="train")
        for t in ds["text"]:
            if not t.strip(): continue
            ii = tok(t).input_ids; chunks.append(torch.tensor(ii)); n += len(ii)
            if n >= need: break
    except Exception as e:
        log(f"pile-10k unavailable ({type(e).__name__}); using wikitext test"); src = "wikitext"; chunks = [corpus_ids(tok, "wikitext", "test")]
    ids = torch.cat(chunks)[:need]; assert ids.numel() >= need, ids.numel()
    return ids[:2 * NSEQ * SEQ].view(2 * NSEQ, SEQ).to(DEV), src
idsG, src = gradient_text(); H = {1: idsG[:NSEQ], 2: idsG[NSEQ:]}
def gabs(m, T=14.0, n=28001):
    t = torch.linspace(0, T, n, dtype=torch.float64); F = torch.special.erf(t / math.sqrt(2)); return float(torch.trapezoid(1 - F.pow(m), t))
def auc(pos, neg):
    if pos.numel() < 5 or neg.numel() < 5: return None
    x = torch.cat([pos, neg]); r = x.argsort().argsort().double() + 1; npos, nneg = pos.numel(), neg.numel(); return float((r[:npos].sum() - npos * (npos + 1) / 2) / (npos * nneg))
def spearman(a, b):
    if a.numel() < 5: return None
    ra = a.argsort().argsort().double(); rb = b.argsort().argsort().double(); ra = ra - ra.mean(); rb = rb - rb.mean(); return float((ra * rb).sum() / (ra.norm() * rb.norm()).clamp_min(1e-12))
def strat(fn, q, s, y, nbins=10):
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
def dictionary(arch_, b):
    A, lab = build_dictionary(arch_, blocks=list(range(b + 1))); Au = unitr(A); Ar = unitr(rotate(A, seed=7)); del A
    typ = lab["type"].to(DEV); blk = lab["block"].to(DEV); idx = lab["index"].to(DEV); gidx = torch.full((b + 1, DFF), -1, device=DEV, dtype=torch.long)
    for bb in range(b + 1): mm = (typ == T_MLP) & (blk == bb); gidx[bb, idx[mm]] = torch.nonzero(mm)[:, 0]
    m = Au.shape[0]; d = dict(Am=Au[gidx.reshape(-1)].clone(), CA=Au.T @ Au / m, CR=Ar.T @ Ar / m, Ar=Ar, gm=gabs(m)); del Au; return d
def S_of(X, d):
    mu = X.mean(0); Xc = X - mu; U = unitr(Xc); N = U.shape[0]
    s2 = ((U @ d["CA"]) * U).sum(1); s2r = ((U @ d["CR"]) * U).sum(1); Mr = torch.cat([(U[s:s + 128] @ d["Ar"].T).abs().max(1).values for s in range(0, N, 128)])
    r_cal = float(Mr.mean() / (s2r.clamp_min(1e-12).sqrt() * d["gm"]).mean()); L = s2.clamp_min(1e-12).sqrt() * d["gm"] * r_cal
    P = U @ d["Am"].T; v, a = (P.abs() / L[:, None]).max(0); return v, a, P.gather(0, a[None])[0], Xc
flat = lambda X: X.reshape(-1, D)
rows_of = lambda arch_, b: torch.cat([arch_.wdir(bb) for bb in range(b + 1)])
# the origin
for p in model.parameters(): p.requires_grad_(False)
X0 = block_states(model, arch, idsB, BL, chunk=4); keep0 = {b: ~sinkmask(flat(X0[b])) for b in BL}; d0 = {b: dictionary(arch, b) for b in BL}; base = {b: S_of(flat(X0[b])[keep0[b]], d0[b]) for b in BL}
W0 = {b: d0[b]["Am"] for b in BL}; Wvec0 = {b: rows_of(arch, b) for b in BL}; lossB0 = mean_loss(idsB)
# gradients on the two halves
for p in model.parameters(): p.requires_grad_(True)
G = {}; Gw = {}; lossG = {}
for h in (1, 2):
    for p in model.parameters(): p.grad = None
    lossG[h] = mean_loss(H[h], backward=True); G[h] = [p.grad.detach().clone() for p in model.parameters()]
    Gw[h] = {b: torch.cat([(arch.mlp_lin(bb).weight.grad.float() if fam == "gpt2" else arch.mlp_lin(bb).weight.grad.float().T) for bb in range(b + 1)]).clone() for b in BL}
for p in model.parameters(): p.grad = None
G["all"] = [(a + b_) / 2 for a, b_ in zip(G[1], G[2])]; Gw["all"] = {b: (Gw[1][b] + Gw[2][b]) / 2 for b in BL}
theta0 = [p.detach().clone() for p in model.parameters()]
log(f"{name} {rev}: gradient text {src}, {2 * NSEQ * SEQ} tokens; loss on the halves {lossG[1]:.4f}/{lossG[2]:.4f}, on the measurement text {lossB0:.4f}; cosine of the halves' row gradients (block 12 rows, median) {float(((Gw[1][12] * Gw[2][12]).sum(1) / (Gw[1][12].norm(dim=1) * Gw[2][12].norm(dim=1)).clamp_min(1e-12)).median()):.3f}")
# local row pulls per half, at each row's best position
local = {}
for b in BL:
    v0, a0, p0, Xc0 = base[b]; w = W0[b]; sgn = torch.sign(p0); sgn[sgn == 0] = 1; xsel = Xc0[a0]; xperp = xsel - w * (xsel * w).sum(1, keepdim=True); wn = Wvec0[b].norm(dim=1)
    local[b] = {h: (-(Gw[h][b] * xperp).sum(1) / wn * sgn).cpu() for h in (1, 2, "all")}; del xsel, xperp
# one flow step per gradient estimate
step = {}
for h in (1, 2, "all"):
    gn2 = sum(float(g.pow(2).sum()) for g in G[h]); eta = DLOSS / gn2
    with torch.no_grad():
        for p, g in zip(model.parameters(), G[h]): p.sub_(eta * g)
    Xs = block_states(model, arch, idsB, BL, chunk=4); rec = dict(lossB=mean_loss(idsB), by_block={})
    for b in BL:
        ds = dictionary(arch, b); Xks = flat(Xs[b])[keep0[b]]
        rec["by_block"][b] = dict(dS=(S_of(Xks, ds)[0] - base[b][0]).cpu(), dS_row=(S_of(flat(X0[b])[keep0[b]], ds)[0] - base[b][0]).cpu(), dX=(flat(Xs[b]) - flat(X0[b])) if b == 12 else None); del ds
    step[h] = rec
    with torch.no_grad():
        for p, t in zip(model.parameters(), theta0): p.copy_(t)
    log(f"{name} {rev}: flow step from half {h}: loss on the measurement text {lossB0:.4f} -> {rec['lossB']:.4f}")
for p in model.parameters(): p.requires_grad_(False)
del G, theta0; torch.cuda.empty_cache()
# the next checkpoint
model1, _, _ = load_model(name, revision=NEXTS); arch1 = Arch(model1, fam)
for p in model1.parameters(): p.requires_grad_(False)
X1 = block_states(model1, arch1, idsB, BL, chunk=4); nxt = {}
for b in BL:
    d1 = dictionary(arch1, b); Xk1 = flat(X1[b])[keep0[b]]
    nxt[b] = dict(dS=(S_of(Xk1, d1)[0] - base[b][0]).cpu(), dS_row=(S_of(flat(X0[b])[keep0[b]], d1)[0] - base[b][0]).cpu(), dW=rows_of(arch1, b) - Wvec0[b], dX=(flat(X1[b]) - flat(X0[b])) if b == 12 else None); del d1
del model1; torch.cuda.empty_cache()
# scoring
res = dict(model=name, origin=rev, horizon=NEXTS, gradient_text=src, n_gradient_tokens=2 * NSEQ * SEQ, by_block={})
gen = torch.Generator().manual_seed(0)
for b in BL:
    u0, u1 = ref[rev][b]["usage"], ref[NEXTS][b]["usage"]; R = u0.numel()
    Wset0 = set(torch.nonzero(u0 > 0)[:, 0][u0[u0 > 0].argsort(descending=True)[:NWORD]].tolist()); Wset1 = set(torch.nonzero(u1 > 0)[:, 0][u1[u1 > 0].argsort(descending=True)[:NWORD]].tolist())
    in0 = torch.zeros(R, dtype=torch.bool); in0[list(Wset0)] = True; in1 = torch.zeros(R, dtype=torch.bool); in1[list(Wset1)] = True; cand = torch.nonzero(~in0)[:, 0]; ent = in1[cand]; ent_idx = torch.nonzero(ent)[:, 0]
    S0 = base[b][0].cpu()[cand]; dS = nxt[b]["dS"][cand]; dS_row = nxt[b]["dS_row"][cand]
    pool = torch.nonzero(~ent)[:, 0]; taken = torch.zeros(cand.numel(), dtype=torch.bool); matched = []
    for i in ent_idx.tolist():
        c = pool[~taken[pool]]; j = c[(S0[c] - S0[i]).abs().argmin()]; taken[j] = True; matched.append(int(j))
    matched = torch.tensor(matched, dtype=torch.long)
    Q = {"flow_dS": {h: step[h]["by_block"][b]["dS"][cand] for h in (1, 2, "all")}, "row_pull": {h: local[b][h][cand] for h in (1, 2, "all")}}
    # the rows' accumulated change against the origin's gradient
    dW = nxt[b]["dW"][cand.to(DEV)]; cosg = {h: ((dW * -Gw[h][b][cand.to(DEV)]).sum(1) / (dW.norm(dim=1) * Gw[h][b][cand.to(DEV)].norm(dim=1)).clamp_min(1e-12)).cpu() for h in (1, 2, "all")}
    cos12 = ((Gw[1][b][cand.to(DEV)] * Gw[2][b][cand.to(DEV)]).sum(1) / (Gw[1][b][cand.to(DEV)].norm(dim=1) * Gw[2][b][cand.to(DEV)].norm(dim=1)).clamp_min(1e-12)).cpu()
    out = dict(n_candidates=int(cand.numel()), n_entrants=int(ent.sum()), reliability={}, predictors={}, row_change={}, state_change=None)
    for k in ("flow_dS", "row_pull"):
        q1, q2, qa = Q[k][1], Q[k][2], Q[k]["all"]
        out["reliability"][k] = dict(split_half_spearman_all=spearman(q1, q2), split_half_spearman_entrants=spearman(q1[ent], q2[ent]), split_half_spearman_top_decile_S=spearman(q1[S0 >= S0.quantile(0.9)], q2[S0 >= S0.quantile(0.9)]))
        out["predictors"][k] = dict(auc_all=auc(qa[ent], qa[~ent]), auc_within_S=strat("auc", qa, S0, ent), auc_matched=auc(qa[ent_idx], qa[matched]), spearman_dS=spearman(qa, dS), spearman_dS_within_S=strat("sp", qa, S0, dS), spearman_dS_row=spearman(qa, dS_row), positive_fraction_entrants=float((qa[ent] > 0).float().mean()), positive_fraction_all=float((qa > 0).float().mean()))
    out["row_change"] = dict(cos_with_negative_gradient={str(h): dict(median_all=float(cosg[h].median()), positive_fraction_all=float((cosg[h] > 0).float().mean()), median_entrants=float(cosg[h][ent].median()), positive_fraction_entrants=float((cosg[h][ent] > 0).float().mean()), median_matched=float(cosg[h][matched].median())) for h in (1, 2, "all")},
                             halves_gradient_cos=dict(median_all=float(cos12.median()), median_entrants=float(cos12[ent].median()), positive_fraction_all=float((cos12 > 0).float().mean())), cos_gradient_halves_agreement=spearman(cosg[1], cosg[2]), median_relative_row_change=float((dW.norm(dim=1) / Wvec0[b][cand.to(DEV)].norm(dim=1)).median()))
    if b == 12:
        a0 = base[b][1]; kidx = torch.nonzero(keep0[b])[:, 0]; pos_ent = kidx[a0[cand[ent_idx].to(DEV)]]; pos_rand = kidx[torch.randperm(kidx.numel(), generator=gen)[:2000].to(DEV)]
        def cosr(P):
            dxa = nxt[b]["dX"][P]; out_ = {}
            for h in (1, 2, "all"):
                dxf = step[h]["by_block"][b]["dX"][P]; c = (dxa * dxf).sum(1) / (dxa.norm(dim=1) * dxf.norm(dim=1)).clamp_min(1e-12); out_[str(h)] = dict(median=float(c.median()), positive_fraction=float((c > 0).float().mean()))
            c12 = (step[1]["by_block"][b]["dX"][P] * step[2]["by_block"][b]["dX"][P]).sum(1) / (step[1]["by_block"][b]["dX"][P].norm(dim=1) * step[2]["by_block"][b]["dX"][P].norm(dim=1)).clamp_min(1e-12)
            out_["halves"] = dict(median=float(c12.median()), positive_fraction=float((c12 > 0).float().mean())); out_["n"] = int(P.numel()); return out_
        out["state_change"] = dict(entrants_best_positions=cosr(pos_ent), random_positions=cosr(pos_rand), median_relative_actual_change=float((nxt[b]["dX"][pos_rand].norm(dim=1) / flat(X0[b])[pos_rand].norm(dim=1)).median()))
    res["by_block"][b] = out; fm = lambda x: "n/a" if x is None else f"{x:.2f}"; rl = out["reliability"]; pr = out["predictors"]; rc = out["row_change"]
    log(f"{name} {rev} block {b} ({int(ent.sum())} entrants of {cand.numel()} non-words): split-half Spearman across rows, all/entrants/top decile of S: flow dS {fm(rl['flow_dS']['split_half_spearman_all'])}/{fm(rl['flow_dS']['split_half_spearman_entrants'])}/{fm(rl['flow_dS']['split_half_spearman_top_decile_S'])}, row pull {fm(rl['row_pull']['split_half_spearman_all'])}/{fm(rl['row_pull']['split_half_spearman_entrants'])}/{fm(rl['row_pull']['split_half_spearman_top_decile_S'])}; halves' row-gradient cosine (median) {rc['halves_gradient_cos']['median_all']:.3f}; with 65k tokens: entry AUC all/within S/matched: flow dS {fm(pr['flow_dS']['auc_all'])}/{fm(pr['flow_dS']['auc_within_S'])}/{fm(pr['flow_dS']['auc_matched'])}, row pull {fm(pr['row_pull']['auc_all'])}/{fm(pr['row_pull']['auc_within_S'])}/{fm(pr['row_pull']['auc_matched'])}; Spearman with the actual dS: flow {fm(pr['flow_dS']['spearman_dS'])} (within S {fm(pr['flow_dS']['spearman_dS_within_S'])}), row pull with the actual row part {fm(pr['row_pull']['spearman_dS_row'])}")
    ca = rc["cos_with_negative_gradient"]["all"]
    log(f"{name} {rev} block {b}: the rows' accumulated change against the origin's negative gradient: cosine median {ca['median_all']:.3f}, positive for {ca['positive_fraction_all']:.2f} of rows (entrants {ca['median_entrants']:.3f}, {ca['positive_fraction_entrants']:.2f}; matched {ca['median_matched']:.3f}); halves agree at Spearman {fm(rc['cos_gradient_halves_agreement'])}; median relative row change {rc['median_relative_row_change']:.2f}" + (f"; the states' actual change against the flow's predicted change, cosine median / positive fraction: entrants' best positions {out['state_change']['entrants_best_positions']['all']['median']:.3f}/{out['state_change']['entrants_best_positions']['all']['positive_fraction']:.2f}, random positions {out['state_change']['random_positions']['all']['median']:.3f}/{out['state_change']['random_positions']['all']['positive_fraction']:.2f}; the two halves' predicted changes agree at cosine {out['state_change']['random_positions']['halves']['median']:.3f}; median relative actual state change {out['state_change']['median_relative_actual_change']:.2f}" if out["state_change"] else ""))
B12 = res["by_block"][12]; g_ = lambda x: -1 if x is None else x
res["checks"] = dict(flow_dS_split_half_over_0_5=g_(B12["reliability"]["flow_dS"]["split_half_spearman_all"]) >= 0.5, row_pull_split_half_over_0_5=g_(B12["reliability"]["row_pull"]["split_half_spearman_all"]) >= 0.5, row_change_follows_gradient_70=B12["row_change"]["cos_with_negative_gradient"]["all"]["positive_fraction_all"] >= 0.7, state_change_follows_flow_60=B12["state_change"]["entrants_best_positions"]["all"]["positive_fraction"] >= 0.6, flow_within_S_over_0_55=g_(B12["predictors"]["flow_dS"]["auc_within_S"]) >= 0.55)
fm = lambda x: "n/a" if x is None else f"{x:.2f}"
summ = f"{name} {rev} to {NEXTS} ({src}, {2 * NSEQ * SEQ} gradient tokens): " + " | ".join(f"block {b}: split-half reliability flow dS {fm(o['reliability']['flow_dS']['split_half_spearman_all'])}, row pull {fm(o['reliability']['row_pull']['split_half_spearman_all'])}; entry AUC all/within S/matched: flow dS {fm(o['predictors']['flow_dS']['auc_all'])}/{fm(o['predictors']['flow_dS']['auc_within_S'])}/{fm(o['predictors']['flow_dS']['auc_matched'])}, row pull {fm(o['predictors']['row_pull']['auc_all'])}/{fm(o['predictors']['row_pull']['auc_within_S'])}/{fm(o['predictors']['row_pull']['auc_matched'])}; Spearman with actual dS: flow {fm(o['predictors']['flow_dS']['spearman_dS'])}; rows' change vs negative gradient: cosine {o['row_change']['cos_with_negative_gradient']['all']['median_all']:.3f}, positive {o['row_change']['cos_with_negative_gradient']['all']['positive_fraction_all']:.2f}" + (f"; states' change vs flow at entrants' positions: cosine {o['state_change']['entrants_best_positions']['all']['median']:.3f}, positive {o['state_change']['entrants_best_positions']['all']['positive_fraction']:.2f} (random {o['state_change']['random_positions']['all']['median']:.3f})" if o["state_change"] else "") for b, o in res["by_block"].items()) + f" | checks {_json.dumps(res['checks'])}"
log(summ); record(f"e522b_reliability_{name}_{rev}", res, summ)
