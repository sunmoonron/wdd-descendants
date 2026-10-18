"""e685b (session 155, the control e685 asked for): the readers' ledger with the crowd's mean added back. e685 drove a neuron by its K largest inputs and found the simulation worse than ablating the neuron at mid depth; the crowd it leaves out carries a net offset. Here each K-input simulation also comes in a second form: the K largest inputs plus the corpus mean of what they leave out (the crowd replaced by its mean, a per-neuron constant), which is the error-node convention of attribution graphs. Fewer neurons, three K values.
Original docstring: e685 (session 155): the readers' ledger. Nanda expects a pruned neuron attribution graph to carry significant
error; the essay expects variable resolution with almost no error where it matters; Gao wants the test in downstream
KL. Per neuron at a block, the exact split of its pre-activation at every position into the writes of every upstream
MLP neuron, every attention output and the embedding (e676's split through the normalisation): the effective number
of contributors at its strong firings (top 1%) and at weak ones (10-30% of the maximum); the share of its input carried
by upstream writes that clear the floor at that position (readable by projection competition) against the crowd below
it; and the mechanistic simulation score: the neuron driven by its K largest inputs only (K = 1, 4, 16, 64, with the
constants), swapped in at every position, the downstream KL at its strong positions and over all positions, relative
to the KL of replacing the neuron by its mean. Arguments: --model M --block B [--smoke]. Pre-registered in e685_prereg.json."""
import argparse
from c11 import *
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--block", type=int, required=True); ap.add_argument("--smoke", action="store_true"); A_ = ap.parse_args(); SMOKE = A_.smoke
name = A_.model; B = A_.block; TAG = f"e685b_{name}_b{B}" + ("_smoke" if SMOKE else ""); M = load(name); NB, D, DFF = M["NB"], M["D"], M["DFF"]; T = 256; L_ = lambda s: log(TAG, s)
NWIN = 8 if SMOKE else 64; NNEU = 6 if SMOKE else 40; KS = [16, 64, 256, 10 ** 7]; NATT = n_att_feeding(M, B); NSTRONG = 160; NWEAK = 160   # the last K is every input: the simulation must then reproduce the neuron (a check of the split)
ids = windows_from(wikitext_ids(M, "train"), NWIN, T).to(DEV); N = NWIN * (T - 1); idx_of = lambda w, p: w * (T - 1) + (p - 1)
UP, AT, EM, XM, AB, LP = [], [], [], [], [], []; G_, U_ = [], []
with Capture(M, list(range(B + 1)), ["act", "att", "xin", "emb", "pre"]) as C:
    for s0 in range(0, NWIN, 8):
        o = M["base"](input_ids=ids[s0:s0 + 8], use_cache=False); LP.append(logprobs_at(M, o.last_hidden_state[:, :-1]))   # fp32: half-precision log-probs put a 1e-3-nat noise floor under the KL, the size of one neuron's effect
        UP.append(torch.cat([C.cap[("act", b)][:, 1:].half() for b in range(B)], -1).reshape(-1, B * DFF) if B else torch.zeros(8 * (T - 1), 0, device=DEV, dtype=torch.half)); AT.append(torch.stack([C.cap[("att", b)][:, 1:].half() for b in range(NATT)], 2).reshape(-1, NATT, D))
        EM.append(C.cap[("emb", 0)][:, 1:].half().reshape(-1, D)); XM.append(C.cap[("xin", B)][:, 1:].float().reshape(-1, D)); AB.append(C.cap[("act", B)][:, 1:].half().reshape(-1, DFF))
        if M["fam"] == "gated": G_.append(C.cap[("pre_g", B)][:, 1:].half().reshape(-1, DFF)); U_.append(C.cap[("pre_u", B)][:, 1:].half().reshape(-1, DFF))
UP, AT, EM, XM, AB, LP = torch.cat(UP), torch.cat(AT), torch.cat(EM), torch.cat(XM), torch.cat(AB), torch.cat(LP); LP = LP.reshape(N, -1); TGT = ids[:, 1:].reshape(-1)
Rd = read_w(M, B); RB = read_bias(M, B); RT = {k: read_tilde(M, B, Rd[k]) for k in Rd}; RC = {k: read_const(M, B, Rd[k], RB[k]) for k in Rd}; WUP = torch.cat([write_w(M, b).T for b in range(B)]) if B else torch.zeros(0, D, device=DEV)
BIAS = sum((write_bias(M, b) for b in range(B)), torch.zeros(D, device=DEV)); P_ = ln_params(M, B); SIG = (XM.var(-1, unbiased=False) + P_["eps"]).sqrt() if P_["centre"] else (XM.pow(2).mean(-1) + P_["eps"]).sqrt()
A_rows, blk_, neu_, nr_ = rows_dictionary(M, max(B - 1, 0)); U = unitr(XM - XM.mean(0)); cal = floor_calibration(U[~sinkmask(XM)], A_rows); LF = floor_of(U, cal)
ab = AB.float(); mx = ab.max(0).values; a100 = ab.topk(100, dim=0).values[-1]; floor_act = 1.0 if M["fam"] != "gated" else float(mx.quantile(0.1)); cand = torch.nonzero((mx > floor_act) & (a100 >= 0.25 * mx))[:, 0].tolist(); NEU = sorted(random.Random(685 + B).sample(cand, min(NNEU, len(cand))))
L_(f"block {B} of {NB}: {N} positions, {B * DFF} upstream MLP units + {NATT} attention outputs + embedding; {len(cand)} candidates, {len(NEU)} neurons sampled; r_cal {cal['r_cal']:.3f}")
def contrib(k, P, key):
    """[len(P), units]: the exact split at positions P of read key's pre-activation of neuron k, and the constant"""
    r = RT[key][k]; V = WUP @ r; s = SIG[P]
    C = torch.cat([UP[P].float() * V[None] / s[:, None], (AT[P].float() @ r) / s[:, None], ((EM[P].float() @ r) / s)[:, None]], 1); return C, (BIAS @ r) / s + RC[key][k]
def pre_true(k, P, key): xn, _ = ln_apply(M, B, XM[P]); return xn @ Rd[key][k] + RB[key][k]
def sim_all(k, K):
    """the neuron's activation at every position from its K largest inputs (per read key) plus the constants; and the same with the crowd it leaves out replaced by its corpus mean"""
    out = torch.zeros(N, device=DEV); outm = torch.zeros(N, device=DEV); PRE = {key: torch.zeros(N, device=DEV) for key in Rd}; CROWD = {key: torch.zeros(N, device=DEV) for key in Rd}
    for s in range(0, N, 2048):
        P = torch.arange(s, min(s + 2048, N), device=DEV)
        for key in Rd:
            C, c0 = contrib(k, P, key); top = C.abs().topk(min(K, C.shape[1]), dim=1).indices; PRE[key][P] = C.gather(1, top).sum(1) + c0; CROWD[key][P] = C.sum(1) - C.gather(1, top).sum(1)
    for key in Rd: PRE[key + "_m"] = PRE[key] + CROWD[key].mean()
    act = lambda pre, sfx: M["act"](pre["h" + sfx]) if M["fam"] != "gated" else torch.nn.functional.silu(pre["g" + sfx]) * pre["u" + sfx]
    return act(PRE, ""), act(PRE, "_m")
def kl_with(k, repl, strong):
    """downstream KL at the strong positions and over all positions when neuron k's activation is replaced by repl [N] at every position.
    Activation row i is position i+1 of its window; log-prob row j is position j (predicting token j+1), so a firing at row i is scored at log-prob row i+1 (dropped at a window's last position)."""
    rows_ = torch.arange(N, device=DEV).reshape(NWIN, T - 1); tot_s, tot_a = [], []; strong = strong[(strong % (T - 1)) != (T - 2)] + 1
    for s0 in range(0, NWIN, 8):
        x = ids[s0:s0 + 8]; rr = rows_[s0:s0 + 8]
        def hook(m, a):
            h = a[0].clone(); h[:, 1:, k] = repl[rr].to(h.dtype); return (h,)
        hd = write_mod(M, B).register_forward_pre_hook(hook)
        try: o = M["base"](input_ids=x, use_cache=False)
        finally: hd.remove()
        lp = logprobs_at(M, o.last_hidden_state[:, :-1]).reshape(-1, LP.shape[-1]); k_ = kl(LP[rr.reshape(-1)].float(), lp); tot_a.append(k_); tot_s.append(k_[torch.isin(rr.reshape(-1), strong)])
    return float(torch.cat(tot_s).mean()), float(torch.cat(tot_a).mean())
rows = {}; err_max = 0.0
for k in NEU:
    a = ab[:, k]; strong = a.topk(NSTRONG).indices; wk = torch.nonzero((a > 0.1 * mx[k]) & (a < 0.3 * mx[k]))[:, 0]; weak = wk[torch.randperm(wk.numel(), device=DEV)[:NWEAK]] if wk.numel() else strong[-NWEAK:]
    key0 = "h" if M["fam"] != "gated" else "g"; Cs, c0s = contrib(k, strong, key0); Cw, c0w = contrib(k, weak, key0); err_max = max(err_max, float(((Cs.sum(1) + c0s) - pre_true(k, strong, key0)).abs().max() / pre_true(k, strong, key0).abs().max().clamp_min(1e-9)))
    neff = lambda C: float((C.abs().sum(1) ** 2 / (C ** 2).sum(1).clamp_min(1e-12)).median()); neff_mlp = lambda C: float((C[:, : B * DFF].abs().sum(1) ** 2 / (C[:, : B * DFF] ** 2).sum(1).clamp_min(1e-12)).median()) if B else None
    over = ((U[strong] @ A_rows.T).abs() / LF[strong][:, None]) > 1 if B else torch.zeros(len(strong), 0, dtype=torch.bool, device=DEV); Cm = Cs[:, : B * DFF]
    over_share = float(((Cm * over).sum(1) / Cm.sum(1).abs().clamp_min(1e-9) * torch.sign(Cm.sum(1))).median()) if B else None; over_energy = float((((Cm * over) ** 2).sum(1) / (Cm ** 2).sum(1).clamp_min(1e-12)).median()) if B else None; over_count = float(over.sum(1).float().median()) if B else None
    top16_share = float((Cs.abs().topk(16, dim=1).values.sum(1) / Cs.abs().sum(1).clamp_min(1e-9)).median()); top16_share_w = float((Cw.abs().topk(16, dim=1).values.sum(1) / Cw.abs().sum(1).clamp_min(1e-9)).median())
    mean_act = torch.full((N,), float(a.mean()), device=DEV); kl_abl = kl_with(k, mean_act, strong); sims = {}
    simsm = {}
    for K in KS: a_plain, a_mean = sim_all(k, K); sims[str(K)] = kl_with(k, a_plain, strong); simsm[str(K)] = kl_with(k, a_mean, strong)
    rows[k] = dict(neff_strong=neff(Cs), neff_weak=neff(Cw), neff_mlp_strong=neff_mlp(Cs), neff_mlp_weak=neff_mlp(Cw), top16_share_strong=top16_share, top16_share_weak=top16_share_w, over_floor_share=over_share, over_floor_energy=over_energy, over_floor_count=over_count,
                   kl_ablate_strong=kl_abl[0], kl_ablate_all=kl_abl[1], sim={K: dict(strong=v[0], all=v[1], ratio_strong=v[0] / max(kl_abl[0], 1e-12), ratio_all=v[1] / max(kl_abl[1], 1e-12)) for K, v in sims.items()}, simm={K: dict(strong=v[0], all=v[1], ratio_strong=v[0] / max(kl_abl[0], 1e-12), ratio_all=v[1] / max(kl_abl[1], 1e-12)) for K, v in simsm.items()}, max_act=float(mx[k]))
    if len(rows) % 10 == 0: L_(f"{len(rows)} neurons; last: KL ratio at K=16 strong {rows[k]['sim']['16']['ratio_strong']:.2f}, with the crowd's mean {rows[k]['simm']['16']['ratio_strong']:.2f}")
R = list(rows.values()); rk = lambda K, key: med([r["sim"][str(K)][key] for r in R]); rkm = lambda K, key: med([r["simm"][str(K)][key] for r in R])
S_ = dict(n=len(R), split_max_rel_error=err_max, neff_strong=med([r["neff_strong"] for r in R]), neff_weak=med([r["neff_weak"] for r in R]), neff_ratio=med([r["neff_strong"] / max(r["neff_weak"], 1e-9) for r in R]), top16_share_strong=med([r["top16_share_strong"] for r in R]), top16_share_weak=med([r["top16_share_weak"] for r in R]),
          over_floor_share=med([r["over_floor_share"] for r in R]), over_floor_energy=med([r["over_floor_energy"] for r in R]), over_floor_count=med([r["over_floor_count"] for r in R]), kl_ablate_strong=med([r["kl_ablate_strong"] for r in R]), kl_ablate_all=med([r["kl_ablate_all"] for r in R]),
          sim_ratio_strong={str(K): rk(K, "ratio_strong") for K in KS}, sim_ratio_all={str(K): rk(K, "ratio_all") for K in KS}, simm_ratio_strong={str(K): rkm(K, "ratio_strong") for K in KS}, simm_ratio_all={str(K): rkm(K, "ratio_all") for K in KS}, share_ratio16_strong_below_025=mean([float(r["sim"]["16"]["ratio_strong"] < 0.25) for r in R]), share_ratio16_strong_above_05=mean([float(r["sim"]["16"]["ratio_strong"] > 0.5) for r in R]))
kstar = lambda key: next((K for K in KS if S_[key][str(K)] < 0.5), None)   # the fewest inputs at which the simulation beats half of the ablation's damage
S_["k_half_strong"] = kstar("sim_ratio_strong"); S_["k_half_all"] = kstar("sim_ratio_all")
V = dict(L0_all_inputs_reproduce=bool(S_["sim_ratio_all"][str(KS[-1])] < 0.05), L4_crowd_mean_halves_ratio16=bool(S_["simm_ratio_strong"]["16"] < 0.5 * S_["sim_ratio_strong"]["16"]), L5_crowd_mean_ratio16_below_025=bool(S_["simm_ratio_strong"]["16"] < 0.25), L6_crowd_mean_ratio64_below_025=bool(S_["simm_ratio_strong"]["64"] < 0.25))
res = dict(model=M["hf"], block=B, n_blocks=NB, positions=N, upstream_units=B * DFF + NATT + 1, ks=KS, summary=S_, verdicts=V, neurons={str(k): v for k, v in rows.items()})
summ = (f"[{name} block {B}] {len(R)} neurons, split error {err_max:.1e}: effective contributors at strong firings {S_['neff_strong']:.0f} vs weak {S_['neff_weak']:.0f} (ratio {S_['neff_ratio']:.2f}); 16 largest inputs carry {S_['top16_share_strong']:.2f} of the strong input (weak {S_['top16_share_weak']:.2f}); over-floor upstream writes: {S_['over_floor_count']} per position, {S_['over_floor_energy']} of the MLP input energy; "
        f"KL of the K-input simulation over the mean-ablation KL at strong positions, K {KS[:-1]} and all: " + " ".join(f"{S_['sim_ratio_strong'][str(K)]:.2f}" for K in KS) + " (all positions " + " ".join(f"{S_['sim_ratio_all'][str(K)]:.2f}" for K in KS) + "); with the crowd's mean added back: " + " ".join(f"{S_['simm_ratio_strong'][str(K)]:.2f}" for K in KS) + " (all positions " + " ".join(f"{S_['simm_ratio_all'][str(K)]:.2f}" for K in KS) + f"); verdicts {V}")
save(TAG, res, summ)
