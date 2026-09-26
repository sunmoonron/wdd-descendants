"""e506: does the chord predict the native maximum? e505 found that the winning projection at a position is carried
by the position's few largest writes acting through the dictionary's Gram matrix: the own coefficient is 0.3-0.5 of
it, the other top-64 writes' cross terms 0.6-0.7, and the top non-writer atom's projection is 0.8 cross terms from
the same writes. So the corrected model is: the native maximum is the maximum over atoms of the projection of the
position's largest writes (the chord), and the extreme-value floor is for the atoms the chord does not touch. This
run makes the prediction from the ledger alone, state by state, and asks whether it reproduces the observed native
maximum in level, in rank across states, in the identity of the winning atom, and, across Pythia's checkpoints, in
the provenance factor (the observed maximum over the rotated dictionary's).
Per typical state x: W64 = the sum of the 64 largest MLP writes (|activation x row norm|) plus the embeddings, in
the atoms' coordinates; the chord prediction M_c = max over all atoms of |<W64, a>| / |x - mean|; the calibrated
Gumbel floor L of e503 (from the second moment of all atoms along the state, calibrated on the rotated dictionary);
the combined prediction max(M_c, L). Observed: the native maximum M, its atom, the rotated maximum.
Reported per block: mean ratios M / M_c and M / max(M_c, L); Spearman across states of M with M_c; the share of
states where the chord's argmax atom is the observed argmax atom, and where it is in the observed top 4 (by
projection); the provenance factor observed (mean M over mean rotated maximum) and predicted (mean max(M_c, L) over
the same); the same for the chord alone. Also the chord's own maximum as a share of the state norm (its amplitude)
and the number of atoms above the floor under the chord and observed.
Setup: a model at a checkpoint (arguments), blocks NB/4, NB/2, 3NB/4; 8 x 256 evaluation tokens, typical positions.
Pre-registered (honest guesses):
- at the end of training the observed maximum is within 20% of the combined prediction at every block, and the
  Spearman across states is above 0.7 at the middle block (0.5);
- the chord's argmax atom is the observed argmax for over half of the states at the middle block (0.5);
- across Pythia's checkpoints the predicted provenance factor tracks the observed (to be read across runs) (0.6).
v2: beside the raw chord, the chord centred over positions (the states are centred, so the common direction the chord
carries at every position is removed the same way) and the full write-sum (every MLP row plus the embeddings) centred.
Pre-registered addendum: the centred chord brings the level within 20% (0.5).
Arguments: name [revision] [v2]."""
import sys, os, json as _json, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; rev = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] not in ("none", "main") else None
model, tok, fam = load_model(name, revision=rev); arch = Arch(model, fam); NB = arch.NB; D = arch.D; DFF = arch.DFF; NW = 64
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
def gabs(m, T=14.0, n=28001):
    t = torch.linspace(0, T, n, dtype=torch.float64); F = torch.special.erf(t / math.sqrt(2)); return float(torch.trapezoid(1 - F.pow(m), t))
def spear(x, y):
    rx, ry = x.argsort().argsort().double(), y.argsort().argsort().double(); rx, ry = rx - rx.mean(), ry - ry.mean(); return float((rx * ry).sum() / (rx.norm() * ry.norm()).clamp_min(1e-9))
res = dict(model=name, revision=rev, blocks=blocks, by_block={})
for b in blocks:
    X = S_[b].reshape(-1, D); keep = ~sinkmask(X); Xk = X[keep]; Xc = Xk - Xk.mean(0); U = unitr(Xc); xn = Xc.norm(dim=-1); N = U.shape[0]
    A, lab = build_dictionary(arch, blocks=list(range(b + 1))); anorm = A.float().norm(dim=-1); Au = unitr(A); Ar = unitr(rotate(A, seed=7)); del A; m = Au.shape[0]
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
    W64r = torch.einsum("nk,nkd->nd", wcoef, Au[writers]); W64 = W64r / xn[:, None]                                # the chord, in units of the state norm
    Wallr = sum(A_all[bb][keep] @ arch.wdir(bb).float() for bb in range(b + 1)) + arch.emb[0].detach().float()[tokens[keep]] + (arch.emb[1].detach().float()[positions[keep]] if has_pos else 0)
    preds = {"chord": W64, "chord_centred": (W64r - W64r.mean(0)) / xn[:, None], "all_writes_centred": (Wallr - Wallr.mean(0)) / xn[:, None]}   # v2: centred over positions, as the states are
    CA = Au.T @ Au / m; CR = Ar.T @ Ar / m; s2_all = ((U @ CA) * U).sum(1); s2_rot = ((U @ CR) * U).sum(1)
    M, arg, Mr, n_above_obs = [], [], [], []; Mc = {k: [] for k in preds}; argc = {k: [] for k in preds}; top4hit = {k: [] for k in preds}; n_above_chord = {k: [] for k in preds}
    Lr = s2_rot.clamp_min(1e-12).sqrt() * gabs(m); L0 = s2_all.clamp_min(1e-12).sqrt() * gabs(m)
    for s in range(0, N, 128):
        C = (U[s:s + 128] @ Au.T).abs(); Mr.append((U[s:s + 128] @ Ar.T).abs().max(1).values); M.append(C.max(1).values); arg.append(C.argmax(1)); n_above_obs.append((C > L0[s:s + 128, None]).sum(1)); t4 = C.topk(4, dim=1).indices
        for k, W in preds.items():
            P = (W[s:s + 128] @ Au.T).abs(); Mc[k].append(P.max(1).values); argc[k].append(P.argmax(1)); top4hit[k].append((t4 == P.argmax(1)[:, None]).any(1)); n_above_chord[k].append((P > L0[s:s + 128, None]).sum(1))
    M, arg, Mr, n_above_obs = map(torch.cat, (M, arg, Mr, n_above_obs)); r_cal = float(Mr.mean() / Lr.mean()); L = L0 * r_cal
    out = dict(n_states=N, n_atoms=m, calibration_rotated=r_cal, factor_observed=float(M.mean() / Mr.mean()), median_observed_max=float(M.median()), median_floor=float(L.median()), atoms_above_floor_observed=float(n_above_obs.float().median()), predictors={})
    for k in preds:
        Mk, ak, t4k, nk = torch.cat(Mc[k]), torch.cat(argc[k]), torch.cat(top4hit[k]), torch.cat(n_above_chord[k]); Mcomb = torch.maximum(Mk, L)
        out["predictors"][k] = dict(ratio_observed_over_chord=float(M.mean() / Mk.mean()), ratio_observed_over_combined=float(M.mean() / Mcomb.mean()), median_ratio_observed_over_combined=float((M / Mcomb).median()), spearman_observed_chord=spear(M, Mk), spearman_observed_combined=spear(M, Mcomb),
                                    chord_argmax_is_observed_argmax=float((ak == arg).float().mean()), chord_argmax_in_observed_top4=float(t4k.float().mean()), factor_predicted=float(Mcomb.mean() / Mr.mean()), factor_chord_alone=float(Mk.mean() / Mr.mean()), chord_amplitude=float(preds[k].norm(dim=-1).median()),
                                    median_chord_max=float(Mk.median()), atoms_above_floor_chord=float(nk.float().median()), share_chord_above_floor=float((Mk > L).float().mean()))
    res["by_block"][b] = out; o = out
    log(f"{name}{' ' + rev if rev else ''} block {b} ({N} states, {m} atoms): observed max {o['median_observed_max']:.3f}, floor {o['median_floor']:.3f}, atoms above the floor {o['atoms_above_floor_observed']:.0f}, provenance factor {o['factor_observed']:.2f} | " + " | ".join(f"{k}: max {p['median_chord_max']:.3f}, observed over chord {p['ratio_observed_over_chord']:.2f}, over combined {p['ratio_observed_over_combined']:.2f} (median {p['median_ratio_observed_over_combined']:.2f}), Spearman {p['spearman_observed_chord']:.2f}, argmax is the observed atom {p['chord_argmax_is_observed_argmax']:.2f} (top 4 {p['chord_argmax_in_observed_top4']:.2f}), factor predicted {p['factor_predicted']:.2f}, amplitude {p['chord_amplitude']:.2f}, atoms above the floor {p['atoms_above_floor_chord']:.0f}" for k, p in o["predictors"].items()))
    del Au, Ar, Cled, W64, W64r, Wallr, preds, CA, CR; torch.cuda.empty_cache()
mid = res["by_block"][NB // 2]; Bk = res["by_block"]; pc = lambda o, k: o["predictors"][k]
res["checks"] = dict(raw_chord_level_within_20pct=all(0.8 <= pc(o, "chord")["ratio_observed_over_combined"] <= 1.2 for o in Bk.values()) and pc(mid, "chord")["spearman_observed_combined"] > 0.7,
                     centred_chord_level_within_20pct=all(0.8 <= pc(o, "chord_centred")["ratio_observed_over_combined"] <= 1.2 for o in Bk.values()), centred_chord_argmax_over_half=pc(mid, "chord_centred")["chord_argmax_is_observed_argmax"] > 0.5)
summ = (f"{name}{' ' + rev if rev else ''}: by block " + " | ".join(f"{b}: provenance factor observed {o['factor_observed']:.2f}, atoms above the floor {o['atoms_above_floor_observed']:.0f}; " + "; ".join(f"{k}: observed max over prediction {p['ratio_observed_over_combined']:.2f}, Spearman {p['spearman_observed_combined']:.2f}, argmax is the observed atom {p['chord_argmax_is_observed_argmax']:.2f} (top 4 {p['chord_argmax_in_observed_top4']:.2f}), factor predicted {p['factor_predicted']:.2f}, atoms above the floor {p['atoms_above_floor_chord']:.0f}" for k, p in o["predictors"].items()) for b, o in Bk.items())
        + f" | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e506_chord_{name}{'_' + rev if rev else ''}", res, summ)
