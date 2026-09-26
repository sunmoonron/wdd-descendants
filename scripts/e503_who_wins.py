"""e503: does the extreme-value model predict which atom wins? THEORY 3m says an atom is a word for a state when its
projection stands above the Gumbel level of the competing atoms; a writer clears it through its own coefficient (its
prominence), a non-writer through the second-order alignment of the state cloud with the atoms. Both halves make
numbers, and this run compares them with the observed maxima, state by state, with nothing fitted.
Per typical state x (u its centred unit direction), the position's writers taken as in e493/e498 (the 64 largest MLP
writes of blocks 0-b by |activation x row norm|, plus the token's embedding):
- observed: the writers' maximum |<u, a>|, the non-writers' maximum, which of the two wins (the argmax is a writer),
  and the rotated dictionary's maximum;
- the predicted non-writer level sigma_n(x) g(m_n): sigma_n^2 = u^T C_n u is the second moment of the non-writing
  atoms along u, g(m) the exact mean of the maximum of m independent |N(0,1)| (a numerical integral), m_n their
  number. The same law on the rotated dictionary gives a calibration ratio r (observed over predicted) that absorbs
  the atom cloud's non-Gaussian tails on a provenance-free dictionary; the calibrated level is r times the analytic one;
- the predicted writer level: the prominence of the largest write, max_j |c_j| |w_j| / |x| (own coefficient only, no
  cross terms);
- the predicted winner: prominence above the calibrated non-writer level. A hybrid uses the observed writer maximum
  against the predicted level, to see which half fails when the joint prediction does.
Reported per block: observed and predicted writer-win rates, the accuracy and the AUC of (prominence - level) for the
observed winner; the non-writer maximum against its second-order level (mean ratio, Spearman across states) and the
non-writer factor over the rotated maximum, observed and predicted from second order alone; the writer maximum
against the prominence; the whole maximum against the predicted maximum; the provenance factor observed and
predicted; the share of observed writer wins below the level (won through cross terms) and of observed non-writer
wins above it.
Setup: a model at a checkpoint (arguments), blocks NB/4, NB/2, 3NB/4; 8 x 256 evaluation tokens, typical positions.
Pre-registered (honest guesses):
- the non-writers' maximum is within 15% of its calibrated second-order level at the middle block (0.4);
- the writers' maximum is within a factor 1.3 of the prominence at the middle block (0.5);
- the predicted writer-win rate is within 0.1 of the observed at every block, and the AUC of (prominence - level) for
  the observed winner is above 0.8 at the middle block (0.5).
Arguments: name [revision]."""
import sys, os, json as _json, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; rev = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] not in ("none", "main") else None
model, tok, fam = load_model(name, revision=rev); arch = Arch(model, fam); NB = arch.NB; D = arch.D; DFF = arch.DFF; NW = 64
blocks = sorted({NB // 4, NB // 2, (3 * NB) // 4}); LB = max(blocks); ids = eval_ids(name)[:8, :256].to(DEV)
acts = {b: [] for b in range(LB + 1)}
def mk(b):
    def pre(m, a): acts[b].append(a[0].detach().float()[:, 1:].reshape(-1, DFF)); return None
    return pre
hs = [arch.mlp_lin(b).register_forward_pre_hook(mk(b)) for b in range(LB + 1)]
try: S_ = block_states(model, arch, ids, blocks, chunk=4)
finally: [h.remove() for h in hs]
A_all = {b: torch.cat(acts[b]) for b in range(LB + 1)}; del acts; rown = {b: arch.wdir(b).float().norm(dim=-1) for b in range(LB + 1)}; tokens = ids[:, 1:].reshape(-1)
def gabs(m, T=14.0, n=28001):
    """the mean of the maximum of m independent |N(0,1)|: the integral over t of 1 - (2 Phi(t) - 1)^m"""
    t = torch.linspace(0, T, n, dtype=torch.float64); F = torch.special.erf(t / math.sqrt(2)); return float(torch.trapezoid(1 - F.pow(m), t))
def auc1(score, pos):
    r = score.argsort().argsort().double() + 1; npos = pos.double().sum(); nneg = pos.numel() - npos; return float(((r * pos.double()).sum() - npos * (npos + 1) / 2) / (npos * nneg).clamp_min(1))
def spear(x, y):
    rx, ry = x.argsort().argsort().double(), y.argsort().argsort().double(); rx, ry = rx - rx.mean(), ry - ry.mean(); return float((rx * ry).sum() / (rx.norm() * ry.norm()).clamp_min(1e-9))
res = dict(model=name, revision=rev, blocks=blocks, n_writers_mlp=NW, by_block={})
for b in blocks:
    X = S_[b].reshape(-1, D); keep = ~sinkmask(X); Xk = X[keep]; Xc = Xk - Xk.mean(0); U = unitr(Xc); xn = Xc.norm(dim=-1); N = U.shape[0]
    A, lab = build_dictionary(arch, blocks=list(range(b + 1))); anorm = A.float().norm(dim=-1); Au = unitr(A); Ar = unitr(rotate(A, seed=7)); del A; m = Au.shape[0]
    typ = lab["type"].to(DEV); blk = lab["block"].to(DEV); idx = lab["index"].to(DEV)
    gidx = torch.full((b + 1, DFF), -1, device=DEV, dtype=torch.long)
    for bb in range(b + 1): mm = (typ == T_MLP) & (blk == bb); gidx[bb, idx[mm]] = torch.nonzero(mm)[:, 0]
    Cled = torch.cat([A_all[bb][keep] * rown[bb][None] for bb in range(b + 1)], 1); top = Cled.abs().topk(NW, dim=1); writers = gidx.reshape(-1)[top.indices]; prom = top.values / xn[:, None]
    mt = typ == T_TOK; tok_lut = torch.full((int(idx[mt].max()) + 1,), -1, device=DEV, dtype=torch.long); tok_lut[idx[mt]] = torch.nonzero(mt)[:, 0]; tatom = tok_lut[tokens[keep]]
    writers = torch.cat([writers, tatom[:, None]], 1); prom = torch.cat([prom, (anorm[tatom] / xn)[:, None]], 1); assert bool((writers >= 0).all()); pi_max = prom.max(1).values; nw = writers.shape[1]
    CA = Au.T @ Au / m; CR = Ar.T @ Ar / m; s2_all = ((U @ CA) * U).sum(1); s2_rot = ((U @ CR) * U).sum(1)
    Mw, Mn, Mr, ssw = [], [], [], []
    for s in range(0, N, 128):
        C = (U[s:s + 128] @ Au.T).abs(); wm = torch.zeros_like(C, dtype=torch.bool); wm.scatter_(1, writers[s:s + 128], True); Cw_ = torch.where(wm, C, torch.zeros_like(C))
        Mw.append(Cw_.max(1).values); Mn.append(torch.where(wm, torch.zeros_like(C), C).max(1).values); ssw.append(Cw_.pow(2).sum(1)); Mr.append((U[s:s + 128] @ Ar.T).abs().max(1).values)
    Mw, Mn, Mr, ssw = map(torch.cat, (Mw, Mn, Mr, ssw)); Mall = torch.maximum(Mw, Mn); win = Mw > Mn
    mn_ = m - nw; sig_n = ((m * s2_all - ssw) / mn_).clamp_min(1e-12).sqrt(); sig_r = s2_rot.clamp_min(1e-12).sqrt()
    Ln = sig_n * gabs(mn_); Lr = sig_r * gabs(m); r_cal = float(Mr.mean() / Lr.mean()); Ln_cal = Ln * r_cal
    pred = pi_max > Ln; pred_cal = pi_max > Ln_cal; hyb = Mw > Ln_cal; Mpred = torch.maximum(pi_max, Ln_cal)
    out = dict(n_states=N, n_atoms=m, n_writers=nw, calibration_rotated=r_cal, gumbel_mean=gabs(mn_), win_rate_observed=float(win.float().mean()), win_rate_predicted=float(pred.float().mean()), win_rate_predicted_calibrated=float(pred_cal.float().mean()),
               accuracy_calibrated=float((pred_cal == win).float().mean()), accuracy_hybrid=float((hyb == win).float().mean()), win_rate_hybrid=float(hyb.float().mean()), auc_prominence_minus_level=auc1(pi_max - Ln_cal, win),
               writer_wins_below_level=float((win & ~pred_cal).float().sum() / win.float().sum().clamp_min(1)), nonwriter_wins_above_level=float((~win & pred_cal).float().sum() / (~win).float().sum().clamp_min(1)),
               nonwriters=dict(ratio_to_level=float(Mn.mean() / Ln.mean()), ratio_to_calibrated_level=float(Mn.mean() / Ln_cal.mean()), spearman_with_level=spear(Mn, Ln), factor_observed=float(Mn.mean() / Mr.mean()), factor_second_order=float(Ln.mean() / Lr.mean()), median_sigma_ratio=float((sig_n / sig_r).median())),
               writers=dict(ratio_to_prominence=float(Mw.mean() / pi_max.mean()), spearman_with_prominence=spear(Mw, pi_max), median_prominence=float(pi_max.median()), median_level=float(Ln_cal.median()), median_max=float(Mw.median())),
               all=dict(ratio_to_predicted=float(Mall.mean() / Mpred.mean()), spearman_with_predicted=spear(Mall, Mpred), factor_observed=float(Mall.mean() / Mr.mean()), factor_predicted=float(Mpred.mean() / Mr.mean())))
    res["by_block"][b] = out; o = out
    log(f"{name}{' ' + rev if rev else ''} block {b} ({N} states, {m} atoms): writer wins observed {o['win_rate_observed']:.2f}, predicted {o['win_rate_predicted_calibrated']:.2f} (uncalibrated {o['win_rate_predicted']:.2f}; hybrid {o['win_rate_hybrid']:.2f}), accuracy {o['accuracy_calibrated']:.2f} (hybrid {o['accuracy_hybrid']:.2f}), AUC {o['auc_prominence_minus_level']:.2f}; writer wins below the level {o['writer_wins_below_level']:.2f}, non-writer wins above it {o['nonwriter_wins_above_level']:.2f} | non-writers: max over level {o['nonwriters']['ratio_to_level']:.2f} (calibrated {o['nonwriters']['ratio_to_calibrated_level']:.2f}), Spearman {o['nonwriters']['spearman_with_level']:.2f}, factor observed {o['nonwriters']['factor_observed']:.2f} vs second order {o['nonwriters']['factor_second_order']:.2f} | writers: max over prominence {o['writers']['ratio_to_prominence']:.2f}, Spearman {o['writers']['spearman_with_prominence']:.2f}, median prominence {o['writers']['median_prominence']:.3f} vs level {o['writers']['median_level']:.3f} | all: max over predicted {o['all']['ratio_to_predicted']:.2f}, Spearman {o['all']['spearman_with_predicted']:.2f}, factor observed {o['all']['factor_observed']:.2f} vs predicted {o['all']['factor_predicted']:.2f} | rotated calibration {r_cal:.3f}")
    del Au, Ar, Cled, CA, CR; torch.cuda.empty_cache()
mid = res["by_block"][NB // 2]; Bk = res["by_block"]
res["checks"] = dict(nonwriter_level_second_order=0.87 <= mid["nonwriters"]["ratio_to_calibrated_level"] <= 1.15, writer_level_is_prominence=0.77 <= mid["writers"]["ratio_to_prominence"] <= 1.3,
                     win_rate_predicted=all(abs(o["win_rate_predicted_calibrated"] - o["win_rate_observed"]) < 0.1 for o in Bk.values()) and mid["auc_prominence_minus_level"] > 0.8)
summ = (f"{name}{' ' + rev if rev else ''}: by block " + " | ".join(f"{b}: writer wins observed/predicted {o['win_rate_observed']:.2f}/{o['win_rate_predicted_calibrated']:.2f} (accuracy {o['accuracy_calibrated']:.2f}, AUC {o['auc_prominence_minus_level']:.2f}), non-writer max over second-order level {o['nonwriters']['ratio_to_calibrated_level']:.2f} (factor observed {o['nonwriters']['factor_observed']:.2f} vs second order {o['nonwriters']['factor_second_order']:.2f}), writer max over prominence {o['writers']['ratio_to_prominence']:.2f}, provenance factor observed/predicted {o['all']['factor_observed']:.2f}/{o['all']['factor_predicted']:.2f}, rotated calibration {o['calibration_rotated']:.2f}" for b, o in Bk.items())
        + f" | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e503_whowins_{name}{'_' + rev if rev else ''}", res, summ)
