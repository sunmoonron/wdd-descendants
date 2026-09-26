"""e507: does the centred chord predict the whole projection profile, not only the maximum? e506 found the centred
chord (the position's 64 largest MLP writes plus its embeddings, less their common part over positions) predicts
GPT-2's native maximum within 1-13% and names the winning atom in a third to a half of states. The stronger question
is whether it predicts the state's projection onto every atom: the ranks of all atoms, the identity of the top few,
the size of the largest projections, and where the error comes from. A state is exactly its writes, so for every atom
    observed = chord + crowd + rest,
with the crowd the centred sum of the remaining MLP writes and the rest the centred non-MLP part (attention, biases);
the decomposition is exact, and the gain of each term along the chord's prediction says how much of the chord the
crowd takes away and attention adds back.
Per typical state: the observed profile o = <u, a> over all atoms (u the centred unit state); the prediction
p = <chord centred, a> / |x - mean|; the crowd c and the rest r likewise (o = p + c + r exactly). Reported per block:
Spearman of |o| with |p| over all atoms (median across states); the recall of the observed top 1 / 5 / 10 / 50 atoms
by the predicted top k (mean across states); Spearman of |o| with |p| among the state's 100 largest observed
projections (does the chord order the true words?) and among the 100 largest predicted (medians); on the 200 largest
predicted (a set chosen by the predictor, so the regression is unbiased), the gain of o on p (least squares through
the origin; 1 is calibrated), the gains of c and of r on p (the crowd's contraction and attention's addition along
the chord), the R^2 of p for o, and the variance shares of c and r in the residual o - p with the cross term; and
the same for the sum of all writes centred, whose residual is the rest alone.
v2: recalls as means (a median of a 0/1 recall is 0 or 1), the tail chosen by one variable at a time (the union of
two top sets induces a negative correlation by construction), the regression set chosen by the predictor.
Setup: a model at a checkpoint (arguments), blocks NB/4, NB/2, 3NB/4; 8 x 256 evaluation tokens, typical positions.
Pre-registered (honest guesses):
- in GPT-2 the observed top 10 is recalled by the predicted top 10 at over 0.5 and the tail Spearman is above 0.6
  at the middle block (0.5);
- the crowd's gain on the chord is negative and the rest's positive at every block of every model, and their sum is
  within 0.15 of zero at GPT-2's middle block (0.5);
- over Pythia's training the chord's tail Spearman rises from step 512 to the end (0.6; read across runs).
Arguments: name [revision] [v2]."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; rev = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] not in ("none", "main") else None
model, tok, fam = load_model(name, revision=rev); arch = Arch(model, fam); NB = arch.NB; D = arch.D; DFF = arch.DFF; NW = 64; KS = (1, 5, 10, 50); TAIL = 100; REG = 200
blocks = sorted({NB // 4, NB // 2, (3 * NB) // 4}); LB = max(blocks); ids = eval_ids(name)[:8, :256].to(DEV); B_, T_ = ids.shape
acts = {b: [] for b in range(LB + 1)}
def mk(b):
    def pre(m, a): acts[b].append(a[0].detach().float()[:, 1:].reshape(-1, DFF)); return None
    return pre
hs = [arch.mlp_lin(b).register_forward_pre_hook(mk(b)) for b in range(LB + 1)]
try: S_ = block_states(model, arch, ids, blocks, chunk=4)
finally: [h.remove() for h in hs]
A_all = {b: torch.cat(acts[b]) for b in range(LB + 1)}; del acts; rown = {b: arch.wdir(b).float().norm(dim=-1) for b in range(LB + 1)}
tokens = ids[:, 1:].reshape(-1); positions = torch.arange(1, T_, device=DEV).repeat(B_); has_pos = len(arch.emb) > 1
def rowcorr(X, Y):
    """Pearson correlation per row of two [n, m] tensors"""
    Xc_, Yc_ = X - X.mean(1, keepdim=True), Y - Y.mean(1, keepdim=True); return (Xc_ * Yc_).sum(1) / (Xc_.norm(dim=1) * Yc_.norm(dim=1)).clamp_min(1e-9)
def rowspear(X, Y): return rowcorr(X.argsort(1).argsort(1).float(), Y.argsort(1).argsort(1).float())
res = dict(model=name, revision=rev, blocks=blocks, by_block={})
for b in blocks:
    X = S_[b].reshape(-1, D); keep = ~sinkmask(X); Xk = X[keep]; Xc = Xk - Xk.mean(0); U = unitr(Xc); xn = Xc.norm(dim=-1); N = U.shape[0]
    A, lab = build_dictionary(arch, blocks=list(range(b + 1))); anorm = A.float().norm(dim=-1); Au = unitr(A); del A; m = Au.shape[0]
    typ = lab["type"].to(DEV); blk = lab["block"].to(DEV); idx = lab["index"].to(DEV)
    gidx = torch.full((b + 1, DFF), -1, device=DEV, dtype=torch.long)
    for bb in range(b + 1): mm = (typ == T_MLP) & (blk == bb); gidx[bb, idx[mm]] = torch.nonzero(mm)[:, 0]
    Cled = torch.cat([A_all[bb][keep] * rown[bb][None] for bb in range(b + 1)], 1); top = Cled.abs().topk(NW, dim=1); writers = gidx.reshape(-1)[top.indices]; wcoef = Cled.gather(1, top.indices)
    mt = typ == T_TOK; tok_lut = torch.full((int(idx[mt].max()) + 1,), -1, device=DEV, dtype=torch.long); tok_lut[idx[mt]] = torch.nonzero(mt)[:, 0]; tatom = tok_lut[tokens[keep]]
    writers = torch.cat([writers, tatom[:, None]], 1); wcoef = torch.cat([wcoef, anorm[tatom][:, None]], 1)
    if has_pos:
        mp = typ == T_POS; pos_lut = torch.full((int(idx[mp].max()) + 1,), -1, device=DEV, dtype=torch.long); pos_lut[idx[mp]] = torch.nonzero(mp)[:, 0]; patom = pos_lut[positions[keep]]
        writers = torch.cat([writers, patom[:, None]], 1); wcoef = torch.cat([wcoef, anorm[patom][:, None]], 1)
    assert bool((writers >= 0).all())
    W64r = torch.einsum("nk,nkd->nd", wcoef, Au[writers]); Wallr = sum(A_all[bb][keep] @ arch.wdir(bb).float() for bb in range(b + 1)) + arch.emb[0].detach().float()[tokens[keep]] + (arch.emb[1].detach().float()[positions[keep]] if has_pos else 0)
    Pch = (W64r - W64r.mean(0)) / xn[:, None]; Pall = (Wallr - Wallr.mean(0)) / xn[:, None]; Ccr = Pall - Pch                  # chord, all writes, crowd: centred, in units of the state norm (rest = u - Pall)
    acc = {k: [] for k in ("sp_all_chord", "sp_all_allw", "sp_obs100_chord", "sp_pred100_chord", "sp_obs100_allw", "sp_pred100_allw", "gain_o", "gain_crowd", "gain_rest", "r2_chord", "gain_o_allw", "r2_allw", "share_crowd", "share_rest", "share_cross")}
    for k in KS: acc[f"recall{k}_chord"] = []; acc[f"recall{k}_allw"] = []
    for s in range(0, N, 128):
        O = U[s:s + 128] @ Au.T; P = Pch[s:s + 128] @ Au.T; P2 = Pall[s:s + 128] @ Au.T; C = Ccr[s:s + 128] @ Au.T; Rr = O - P2; Oa, Pa, P2a = O.abs(), P.abs(), P2.abs()
        acc["sp_all_chord"].append(rowspear(Oa, Pa)); acc["sp_all_allw"].append(rowspear(Oa, P2a))
        to = {k: Oa.topk(k, 1).indices for k in KS}
        for k in KS:
            acc[f"recall{k}_chord"].append((to[k][:, :, None] == Pa.topk(k, 1).indices[:, None, :]).any(2).float().mean(1)); acc[f"recall{k}_allw"].append((to[k][:, :, None] == P2a.topk(k, 1).indices[:, None, :]).any(2).float().mean(1))
        obs100 = Oa.topk(TAIL, 1).indices; pred100 = Pa.topk(TAIL, 1).indices; pred100b = P2a.topk(TAIL, 1).indices; reg = Pa.topk(REG, 1).indices; regb = P2a.topk(REG, 1).indices
        acc["sp_obs100_chord"].append(rowspear(Oa.gather(1, obs100), Pa.gather(1, obs100))); acc["sp_pred100_chord"].append(rowspear(Oa.gather(1, pred100), Pa.gather(1, pred100)))
        acc["sp_obs100_allw"].append(rowspear(Oa.gather(1, obs100), P2a.gather(1, obs100))); acc["sp_pred100_allw"].append(rowspear(Oa.gather(1, pred100b), P2a.gather(1, pred100b)))
        o, p, c, r = O.gather(1, reg), P.gather(1, reg), C.gather(1, reg), Rr.gather(1, reg); pp = (p * p).sum(1).clamp_min(1e-12)
        acc["gain_o"].append((o * p).sum(1) / pp); acc["gain_crowd"].append((c * p).sum(1) / pp); acc["gain_rest"].append((r * p).sum(1) / pp)
        acc["r2_chord"].append(1 - ((o - p) ** 2).sum(1) / ((o - o.mean(1, keepdim=True)) ** 2).sum(1).clamp_min(1e-12)); e = o - p; ve = e.var(1).clamp_min(1e-12)
        acc["share_crowd"].append(c.var(1) / ve); acc["share_rest"].append(r.var(1) / ve); acc["share_cross"].append(2 * ((c - c.mean(1, keepdim=True)) * (r - r.mean(1, keepdim=True))).mean(1) / ve)
        o2, p2 = O.gather(1, regb), P2.gather(1, regb); acc["gain_o_allw"].append((o2 * p2).sum(1) / (p2 * p2).sum(1).clamp_min(1e-12)); acc["r2_allw"].append(1 - ((o2 - p2) ** 2).sum(1) / ((o2 - o2.mean(1, keepdim=True)) ** 2).sum(1).clamp_min(1e-12))
    out = {k: float(torch.cat(vs).mean() if k.startswith("recall") else torch.cat(vs).median()) for k, vs in acc.items()}
    out.update(n_states=N, n_atoms=m); res["by_block"][b] = out; o = out
    log(f"{name}{' ' + rev if rev else ''} block {b} ({N} states, {m} atoms): chord centred: Spearman over all atoms {o['sp_all_chord']:.2f}, among the observed top 100 {o['sp_obs100_chord']:.2f}, among the predicted top 100 {o['sp_pred100_chord']:.2f}; recall of the observed top 1/5/10/50 {o['recall1_chord']:.2f}/{o['recall5_chord']:.2f}/{o['recall10_chord']:.2f}/{o['recall50_chord']:.2f}; on the predicted top 200: gain {o['gain_o']:.2f} (crowd {o['gain_crowd']:+.2f}, rest {o['gain_rest']:+.2f}), R2 {o['r2_chord']:.2f}, residual variance shares crowd {o['share_crowd']:.2f} rest {o['share_rest']:.2f} cross {o['share_cross']:+.2f} | all writes centred: Spearman all {o['sp_all_allw']:.2f}, observed top 100 {o['sp_obs100_allw']:.2f}, predicted top 100 {o['sp_pred100_allw']:.2f}; recall 1/5/10/50 {o['recall1_allw']:.2f}/{o['recall5_allw']:.2f}/{o['recall10_allw']:.2f}/{o['recall50_allw']:.2f}; gain {o['gain_o_allw']:.2f}, R2 {o['r2_allw']:.2f}")
    del Au, Cled, W64r, Wallr, Pch, Pall, Ccr; torch.cuda.empty_cache()
mid = res["by_block"][NB // 2]; Bk = res["by_block"]
res["checks"] = dict(recall10_over_half_and_tail_spearman_over_0_6=mid["recall10_chord"] > 0.5 and mid["sp_obs100_chord"] > 0.6, crowd_negative_rest_positive=all(o["gain_crowd"] < 0 and o["gain_rest"] > 0 for o in Bk.values()), gains_cancel_at_mid=abs(mid["gain_crowd"] + mid["gain_rest"]) < 0.15)
summ = (f"{name}{' ' + rev if rev else ''}: by block " + " | ".join(f"{b}: chord centred Spearman all/observed top 100/predicted top 100 {o['sp_all_chord']:.2f}/{o['sp_obs100_chord']:.2f}/{o['sp_pred100_chord']:.2f}, recall of the observed top 1/5/10/50 {o['recall1_chord']:.2f}/{o['recall5_chord']:.2f}/{o['recall10_chord']:.2f}/{o['recall50_chord']:.2f}, gain {o['gain_o']:.2f} (crowd {o['gain_crowd']:+.2f}, rest {o['gain_rest']:+.2f}), R2 {o['r2_chord']:.2f}, residual shares crowd/rest/cross {o['share_crowd']:.2f}/{o['share_rest']:.2f}/{o['share_cross']:+.2f}; all writes centred Spearman all/observed/predicted {o['sp_all_allw']:.2f}/{o['sp_obs100_allw']:.2f}/{o['sp_pred100_allw']:.2f}, recall 1/5/10/50 {o['recall1_allw']:.2f}/{o['recall5_allw']:.2f}/{o['recall10_allw']:.2f}/{o['recall50_allw']:.2f}, gain {o['gain_o_allw']:.2f}, R2 {o['r2_allw']:.2f}" for b, o in Bk.items())
        + f" | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e507_profile_{name}{'_' + rev if rev else ''}", res, summ)
