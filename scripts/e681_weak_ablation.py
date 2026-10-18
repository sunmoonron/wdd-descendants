"""e681 (session 155): do weak activations matter, and to whom? The essay says almost all weak activations can be removed
for almost no cost, except in dials (early vision, the last layer's token suppression) and the cost is near-ties; Gao
expects substantial degradation. Per neuron, activations below theta times the neuron's own maximum (the third largest
magnitude over a calibration corpus) are zeroed, in every block; theta 0.02 to 0.5; the mean of the weak activations as
the replacement control (e678's lesson); block subsets (first half, second half, last block); and neuron classes by the
share of a neuron's activation energy in its top 1% of positions (high share = detector, low = dial), tertiles within a
block, a random third as the control. Measured on wikitext-2 test: next-token KL to the unedited model, loss change,
top-1 flips and the clean margin of the flipped predictions against the rest. Arguments: --model M [--smoke].
Pre-registered in e681_prereg.json."""
import argparse
from c11 import *
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--smoke", action="store_true"); A_ = ap.parse_args(); SMOKE = A_.smoke
name = A_.model; TAG = f"e681_{name}" + ("_smoke" if SMOKE else ""); M = load(name); NB, D, DFF = M["NB"], M["D"], M["DFF"]; T = 256
NWC = 16 if SMOKE else 128; NWE = 8 if SMOKE else 48; THETAS = [0.02, 0.05, 0.1, 0.2, 0.5]; L_ = lambda s: log(TAG, s)
cal_ids = windows_from(wikitext_ids(M, "train"), NWC, T).to(DEV); ev_ids = windows_from(wikitext_ids(M, "test"), NWE, T).to(DEV)
# ---------------- calibration: per neuron, the maximum (third largest |a|), the energy, the top-1% energy, the mean of the weak activations
NPOS = NWC * (T - 1); K1 = max(1, NPOS // 100); TOP = {b: None for b in range(NB)}; E = torch.zeros(NB, DFF, device=DEV); MAX3 = torch.zeros(NB, DFF, device=DEV); TOPABS = {b: None for b in range(NB)}; ABS = {b: [] for b in range(NB)}
QS = [0.25, 0.1, 0.05, 0.01]   # keep the strongest quarter, tenth, twentieth, hundredth of each neuron's |activation| (by count), zero the rest
with Capture(M, list(range(NB)), ["act"]) as C:
    for s0 in range(0, NWC, 8):
        run_base(M, cal_ids[s0:s0 + 8])
        for b in range(NB):
            a = C.cap[("act", b)][:, 1:].float().reshape(-1, DFF); a2 = a * a; E[b] += a2.sum(0); ABS[b].append(a.abs().half().cpu())
            TOP[b] = a2 if TOP[b] is None else torch.cat([TOP[b], a2]); TOP[b] = TOP[b].topk(min(K1, TOP[b].shape[0]), dim=0).values
            TOPABS[b] = a.abs() if TOPABS[b] is None else torch.cat([TOPABS[b], a.abs()]); TOPABS[b] = TOPABS[b].topk(3, dim=0).values
for b in range(NB): MAX3[b] = TOPABS[b][-1]
E1 = torch.stack([TOP[b].sum(0) for b in range(NB)]) / E.clamp_min(1e-12)   # share of the energy in the top 1% of positions
TAUQ = {q: torch.zeros(NB, DFF, device=DEV) for q in QS}
for b in range(NB):
    X_ = torch.cat(ABS[b]); ABS[b] = None
    for c0 in range(0, DFF, 1024):
        xs = X_[:, c0:c0 + 1024].to(DEV).float()
        for q in QS: TAUQ[q][b, c0:c0 + 1024] = xs.quantile(1 - q, dim=0)
    del X_
THR = {("theta", th): th * MAX3 for th in THETAS}; THR.update({("keep", q): TAUQ[q] for q in QS})   # per-neuron thresholds: below = weak
SUMW = {k: torch.zeros(NB, DFF, device=DEV) for k in (("theta", 0.1), ("keep", 0.25))}; CNTW = {k: torch.zeros(NB, DFF, device=DEV) for k in SUMW}
with Capture(M, list(range(NB)), ["act"]) as C:   # second pass: the mean of the weak activations (for the replacement control)
    for s0 in range(0, NWC, 8):
        run_base(M, cal_ids[s0:s0 + 8])
        for b in range(NB):
            a = C.cap[("act", b)][:, 1:].float().reshape(-1, DFF)
            for k in SUMW: wk = a.abs() < THR[k][b][None]; SUMW[k][b] += (a * wk).sum(0); CNTW[k][b] += wk.sum(0)
MEANW = {k: SUMW[k] / CNTW[k].clamp_min(1) for k in SUMW}; WEAKSHARE = {str(k): float((CNTW[k] / NPOS).mean()) for k in SUMW}
alive = MAX3 > 1e-6; tert = torch.zeros(NB, DFF, dtype=torch.long, device=DEV)   # 0 dial (low top-1% share), 1 middle, 2 detector
for b in range(NB):
    v = E1[b][alive[b]]; q1, q2 = float(v.quantile(1 / 3)), float(v.quantile(2 / 3)); tert[b] = torch.where(E1[b] < q1, 0, torch.where(E1[b] < q2, 1, 2))
rnd3 = torch.rand(NB, DFF, device=DEV, generator=torch.Generator(device=DEV).manual_seed(681)) < 1 / 3
L_(f"calibrated on {NPOS} positions: top-1% energy share median {float(E1[alive].median()):.3f} (dial tertile below {float(E1[alive].quantile(1 / 3)):.3f}); weak activations: below 0.1 max {WEAKSHARE[str(('theta', 0.1))]:.3f} of positions, outside the strongest quarter {WEAKSHARE[str(('keep', 0.25))]:.3f}; "
   f"0.1 max sits at the {float((TAUQ[0.25] < 0.1 * MAX3).float().mean()):.2f} share of neurons above their 75th percentile")
# ---------------- conditions
COND = {}; STAT = collections.defaultdict(float)
def hook_for(b):
    def h(mod, a):
        if not COND: return None
        x = a[0]; thr = COND["thr"][b]; inc = COND["include"][b]
        if not bool(inc.any()): return None
        wk = (x.abs() < thr.to(x.dtype)[None, None]) & inc[None, None]; STAT["removed"] += float((x.float() ** 2)[wk].sum()); STAT["all"] += float((x.float() ** 2).sum()); STAT["npos"] += float(wk.float().sum()); STAT["nall"] += float(wk.numel())
        rep = COND["rep"][b].to(x.dtype)[None, None].expand_as(x) if COND["rep"] is not None else torch.zeros_like(x)
        return (torch.where(wk, rep, x),)
    return h
HS = [write_mod(M, b).register_forward_pre_hook(hook_for(b)) for b in range(NB)]
def evaluate():
    STAT.clear(); LPs = []
    for s0 in range(0, NWE, 8): o = M["base"](input_ids=ev_ids[s0:s0 + 8], use_cache=False); LPs.append(logprobs_at(M, o.last_hidden_state[:, :-1]))   # fp32, see e680
    return torch.cat(LPs), ((STAT["removed"] / max(STAT["all"], 1e-12), STAT["npos"] / max(STAT["nall"], 1e-12)) if STAT else (0.0, 0.0))
LP0, _ = evaluate(); TGT = ev_ids[:, 1:]; PRED0 = LP0.argmax(-1); LOSS0 = float(-LP0.gather(-1, TGT[..., None]).float().mean()); MARG0 = torch.zeros(NWE, T - 1, device=DEV)
for w in range(NWE): t2 = LP0[w].float().exp().topk(2, dim=-1).values; MARG0[w] = t2[:, 0] - t2[:, 1]
def compare(LP1):
    """per token: KL(clean || edited), the flips of the top-1 prediction, the edited loss; one window at a time to bound memory"""
    ks, fl = [], []
    for w in range(NWE): ks.append(kl(LP0[w].float(), LP1[w].float())); fl.append(LP1[w].argmax(-1) != PRED0[w])
    return torch.stack(ks), torch.stack(fl), float(-LP1.gather(-1, TGT[..., None]).float().mean())
ALL = torch.ones(NB, DFF, dtype=torch.bool, device=DEV); half = NB // 2
SETS = {"all": ALL, "first_half": torch.cat([ALL[:half], ~ALL[half:]]), "second_half": torch.cat([~ALL[:half], ALL[half:]]), "last_block": torch.cat([~ALL[:-1], ALL[-1:]]), "dial": tert == 0, "middle": tert == 1, "detector": tert == 2, "random_third": rnd3}
RUNS = [("theta", th, "all", False) for th in THETAS] + [("keep", q, "all", False) for q in QS] + [("theta", 0.1, "all", True), ("keep", 0.25, "all", True)] + [("keep", 0.25, s, False) for s in ("first_half", "second_half", "last_block", "dial", "middle", "detector", "random_third")] + [("theta", 0.1, s, False) for s in ("last_block", "dial", "detector")]
res = dict(model=M["hf"], n_blocks=NB, DFF=DFF, calib_positions=NPOS, eval_positions=int(TGT.numel()), base_loss=LOSS0, top1pct_energy_share_median=float(E1[alive].median()), weak_position_share=WEAKSHARE, conditions={})
for kind, v, sname, use_mean in RUNS:
    COND.update(thr=THR[(kind, v)], include=SETS[sname], rep=(MEANW[(kind, v)] if use_mean else None)); LP1, (remf, posf) = evaluate(); k, flip, loss1 = compare(LP1); del LP1
    mf, mn = MARG0[flip], MARG0[~flip]; key = f"{kind}_{v}_{sname}" + ("_mean" if use_mean else "")
    res["conditions"][key] = dict(kind=kind, value=v, set=sname, mean_replaced=use_mean, n_neurons=int(SETS[sname].sum()), removed_energy_fraction=remf, removed_position_fraction=posf, kl_mean=float(k.mean()), kl_median=float(k.median()), loss_delta=loss1 - LOSS0, flip_rate=float(flip.float().mean()),
                                  margin_flipped_median=float(mf.median()) if mf.numel() else None, margin_unflipped_median=float(mn.median()), flipped_share_near_tie=float((mf < 0.1).float().mean()) if mf.numel() else None)
    r = res["conditions"][key]; L_(f"{key}: removed energy {remf:.4f} (positions {posf:.3f}), KL mean {r['kl_mean']:.4f}, loss {r['loss_delta']:+.4f}, flips {r['flip_rate']:.4f} (flipped margin median {r['margin_flipped_median']}, unflipped {r['margin_unflipped_median']:.3f})")
COND.clear(); [h.remove() for h in HS]
c = res["conditions"]; per = lambda key: c[key]["kl_mean"] / max(c[key]["n_neurons"], 1)
V = dict(W1_keep_quarter_kl_below_005=bool(c["keep_0.25_all"]["kl_mean"] < 0.05), W1b_keep_quarter_above_02=bool(c["keep_0.25_all"]["kl_mean"] > 0.2), W1c_theta01_kl_below_005=bool(c["theta_0.1_all"]["kl_mean"] < 0.05),
         W2_dial_cost_2x_detector=bool(c["keep_0.25_dial"]["kl_mean"] >= 2 * c["keep_0.25_detector"]["kl_mean"]), W3_flipped_margin_half=bool(c["keep_0.25_all"]["margin_flipped_median"] is not None and c["keep_0.25_all"]["margin_flipped_median"] < 0.5 * c["keep_0.25_all"]["margin_unflipped_median"]),
         W4_last_block_per_neuron_2x=bool(per("keep_0.25_last_block") >= 2 * per("keep_0.25_all")), W5_mean_cheaper_than_zero=bool(c["keep_0.25_all_mean"]["kl_mean"] < c["keep_0.25_all"]["kl_mean"]))
res["verdicts"] = V
summ = (f"[{name}] weak-activation ablation on {TGT.numel()} tokens: KL (loss change) keeping the strongest " + ", ".join(f"{q}: {c[f'keep_{q}_all']['kl_mean']:.3f} ({c[f'keep_{q}_all']['loss_delta']:+.3f}, energy removed {c[f'keep_{q}_all']['removed_energy_fraction']:.3f})" for q in QS) +
        "; below theta of max " + ", ".join(f"{th}: {c[f'theta_{th}_all']['kl_mean']:.3f}" for th in THETAS) + f"; mean replacement (quarter / 0.1 max) {c['keep_0.25_all_mean']['kl_mean']:.3f} / {c['theta_0.1_all_mean']['kl_mean']:.3f}; "
        f"by set, strongest quarter kept: first half {c['keep_0.25_first_half']['kl_mean']:.3f}, second half {c['keep_0.25_second_half']['kl_mean']:.3f}, last block {c['keep_0.25_last_block']['kl_mean']:.3f}, dial {c['keep_0.25_dial']['kl_mean']:.3f}, middle {c['keep_0.25_middle']['kl_mean']:.3f}, detector {c['keep_0.25_detector']['kl_mean']:.3f}, random third {c['keep_0.25_random_third']['kl_mean']:.3f}; "
        f"flips {c['keep_0.25_all']['flip_rate']:.3f}, flipped-margin median {c['keep_0.25_all']['margin_flipped_median']} vs unflipped {c['keep_0.25_all']['margin_unflipped_median']:.3f}; verdicts {V}")
save(TAG, res, summ)
