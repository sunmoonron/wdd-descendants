"""e697 (session 157): the brake, causally. e695 read the crowd as a suppressor: at a neuron's strong firings its sixteen
largest inputs carry the net drive, and off the class the crowd's net sum opposes them. Here that reading is tested by
intervention on e692's model-blocks and neurons: (1) where the sixteen-input simulation's damage comes from, the
simulation swapped in only at the neuron's top-5% positions against only at the rest; (2) signed wires, the sixteen
strongest excitatory inputs with the sixteen strongest inhibitory ones, against sixteen and thirty-two by magnitude;
(3) the weak-write bus, the crowd's write projected on the upstream blocks' weak-write principal directions (sixteen per
block, from held-out text, e693's subspace), alone with the sixteen wires and with the signed wires. Scored as e685:
downstream KL at the strong firings and over all positions relative to mean-ablating the neuron. Arguments: --model M
--block B [--smoke]. Pre-registered in e696_e701_prereg.json."""
import argparse
from c11 import *
F_ = torch.nn.functional
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--block", type=int, required=True); ap.add_argument("--smoke", action="store_true"); A_ = ap.parse_args(); SMOKE = A_.smoke
name = A_.model; B = A_.block; TAG = f"e697_{name}_b{B}" + ("_smoke" if SMOKE else ""); M = load(name); NB, D, DFF = M["NB"], M["D"], M["DFF"]; T = 256; L_ = lambda s: log(TAG, s)
NWIN = 8 if SMOKE else int(os.environ.get("WDD_NWIN", 64)); NWH = 16 if SMOKE else int(os.environ.get("WDD_NWH", 96)); CH_ = int(os.environ.get("WDD_CHUNK", 2048)); NNEU = 4 if SMOKE else 40; NATT = n_att_feeding(M, B); NSTRONG = 160; Q = 0.25; RB_ = 16
ids_all = wikitext_ids(M, "train"); half = len(ids_all) // 2
ids = windows_from(ids_all[:half], NWIN, T).to(DEV); ids_h = windows_from(ids_all[half:], NWH, T).to(DEV); N = NWIN * (T - 1); NH = NWH * (T - 1); NMLP = B * DFF; NU = NMLP + NATT + 1
def capture(ids_, nwin, need_lp):
    UP, AT, EM, XM, AB, LP = [], [], [], [], [], []
    with Capture(M, list(range(B + 1)), ["act", "att", "xin", "emb"]) as C:
        for s0 in range(0, nwin, 8):
            o = M["base"](input_ids=ids_[s0:s0 + 8], use_cache=False)
            if need_lp: LP.append(logprobs_at(M, o.last_hidden_state[:, :-1]))
            UP.append(torch.cat([C.cap[("act", b)][:, 1:].half() for b in range(B)], -1).reshape(-1, NMLP) if B else torch.zeros(8 * (T - 1), 0, device=DEV, dtype=torch.half)); AT.append(torch.stack([C.cap[("att", b)][:, 1:].half() for b in range(NATT)], 2).reshape(-1, NATT, D))
            EM.append(C.cap[("emb", 0)][:, 1:].half().reshape(-1, D)); XM.append(C.cap[("xin", B)][:, 1:].float().reshape(-1, D)); AB.append(C.cap[("act", B)][:, 1:].half().reshape(-1, DFF))
    S = dict(UP=torch.cat(UP), AT=torch.cat(AT), EM=torch.cat(EM), XM=torch.cat(XM), AB=torch.cat(AB)); S["LP"] = torch.cat(LP).reshape(nwin * (T - 1), -1) if need_lp else None; return S
E = capture(ids, NWIN, True); H = capture(ids_h, NWH, False); L_(f"block {B} of {NB}: {N} evaluation and {NH} held-out positions, {NU} upstream units")
Rd = read_w(M, B); RB = read_bias(M, B); RT = {k: read_tilde(M, B, Rd[k]) for k in Rd}; RC = {k: read_const(M, B, Rd[k], RB[k]) for k in Rd}; WUP = torch.cat([write_w(M, b).T for b in range(B)]) if B else torch.zeros(0, D, device=DEV)
BIAS = sum((write_bias(M, b) for b in range(B)), torch.zeros(D, device=DEV)); P_ = ln_params(M, B)
sig = lambda XM: (XM.var(-1, unbiased=False) + P_["eps"]).sqrt() if P_["centre"] else (XM.pow(2).mean(-1) + P_["eps"]).sqrt(); E["SIG"] = sig(E["XM"]); H["SIG"] = sig(H["XM"]); key0 = "h" if M["fam"] != "gated" else "g"
# ---------------- the weak-write bus: each upstream block's weak write (all but the strongest quarter of each unit's activations) on held-out text, its 16 principal directions; concatenated and orthonormalised
PB = []
for b in range(B):
    cols = slice(b * DFF, (b + 1) * DFF); tau = torch.cat([H["UP"][:, b * DFF + c0:b * DFF + min(c0 + 512, DFF)].float().abs().quantile(1 - Q, dim=0) for c0 in range(0, DFF, 512)]); W = write_w(M, b); cov = torch.zeros(D, D, device=DEV)   # chunks clipped to the block (Qwen widths are not multiples of 512)
    for s in range(0, NH, 2048):
        a = H["UP"][s:s + 2048, cols].float(); dx = (a * (a.abs() < tau[None])) @ W.T; cov += dx.T @ dx
    ev, evec = torch.linalg.eigh(cov); PB.append(evec[:, -RB_:])
PBUS, _ = torch.linalg.qr(torch.cat(PB, 1)) if B else (torch.zeros(D, 0, device=DEV), None); L_(f"bus: {PBUS.shape[1]} directions from {B} upstream blocks")
def contrib(S, k, P, key):
    r = RT[key][k]; V = WUP @ r; s = S["SIG"][P]
    C = torch.cat([S["UP"][P].float() * V[None] / s[:, None], (S["AT"][P].float() @ r) / s[:, None], ((S["EM"][P].float() @ r) / s)[:, None]], 1); return C, (BIAS @ r) / s + RC[key][k]
WIRES = ["abs16", "abs32", "pos16", "pos16neg16"]
def ledger(k, key):
    """fp16 contributions [N, NU], the constant, the total, and for each wire set its indices and sum"""
    C16 = torch.empty(N, NU, device=DEV, dtype=torch.half); c0 = torch.empty(N, device=DEV); tot = torch.empty(N, device=DEV); IDX = {w: [] for w in WIRES}; TS = {w: torch.empty(N, device=DEV) for w in WIRES}
    for s in range(0, N, CH_):
        P = torch.arange(s, min(s + CH_, N), device=DEV); c, cc = contrib(E, k, P, key); C16[P] = c.half(); c0[P] = cc; tot[P] = c.sum(1)
        sets = {"abs16": c.abs().topk(16, dim=1).indices, "abs32": c.abs().topk(32, dim=1).indices, "pos16": c.topk(16, dim=1).indices}; sets["pos16neg16"] = torch.cat([sets["pos16"], (-c).topk(16, dim=1).indices], 1)
        for w in WIRES: IDX[w].append(sets[w]); TS[w][P] = c.gather(1, sets[w]).sum(1)
    return C16, c0, tot, {w: torch.cat(v) for w, v in IDX.items()}, TS
def bus_summary(idx, r_):
    """the crowd's write (the stream minus the wires' writes, scaled as the read is) on the bus, read by the neuron"""
    out = torch.empty(N, device=DEV); pr = PBUS @ (PBUS.T @ r_); K = idx.shape[1]
    for s in range(0, N, CH_):
        P = torch.arange(s, min(s + CH_, N), device=DEV); top = idx[P]; isM = (top < NMLP).float(); isA = (top >= NMLP) & (top < NMLP + NATT); isE = (top >= NMLP + NATT).any(1).float()
        ti = top.clamp(max=max(NMLP - 1, 0)); a = E["UP"][P].gather(1, ti).float() * isM; xm = torch.einsum("nk,nkd->nd", a, WUP[ti]) if NMLP else torch.zeros(len(P), D, device=DEV)
        ai_ = (top - NMLP).clamp(0, NATT - 1); xa = (E["AT"][P].float().gather(1, ai_[:, :, None].expand(-1, -1, D)) * isA[:, :, None]).sum(1)
        xc = (E["XM"][P] - BIAS[None] - xm - xa - E["EM"][P].float() * isE[:, None]) / E["SIG"][P][:, None]; out[P] = xc @ pr
    return out
ab = E["AB"].float(); mx = ab.max(0).values; a100 = ab.topk(100, dim=0).values[-1]; floor_act = 1.0 if M["fam"] != "gated" else float(mx.quantile(0.1)); cand = torch.nonzero((mx > floor_act) & (a100 >= 0.25 * mx))[:, 0].tolist(); NEU = sorted(random.Random(685 + B).sample(cand, min(NNEU, len(cand))))
def act_of(PRE): return M["act"](PRE["h"]) if M["fam"] != "gated" else F_.silu(PRE["g"]) * PRE["u"]
def kl_with(k, repl, strong):
    rows_ = torch.arange(N, device=DEV).reshape(NWIN, T - 1); tot_s, tot_a = [], []; strong = strong[(strong % (T - 1)) != (T - 2)] + 1
    for s0 in range(0, NWIN, 8):
        x = ids[s0:s0 + 8]; rr = rows_[s0:s0 + 8]
        def hook(m, a):
            h = a[0].clone(); h[:, 1:, k] = repl[rr].to(h.dtype); return (h,)
        hd = write_mod(M, B).register_forward_pre_hook(hook)
        try: o = M["base"](input_ids=x, use_cache=False)
        finally: hd.remove()
        lp = logprobs_at(M, o.last_hidden_state[:, :-1]).reshape(-1, E["LP"].shape[-1]); k_ = kl(E["LP"][rr.reshape(-1)].float(), lp); tot_a.append(k_); tot_s.append(k_[torch.isin(rr.reshape(-1), strong)])
    return float(torch.cat(tot_s).mean()), float(torch.cat(tot_a).mean())
CONDS = ["abs16", "abs16_on", "abs16_off", "abs32", "pos16", "pos16neg16", "abs16_bus", "pos16neg16_bus", "all"]
rows = {}
for k in NEU:
    a = ab[:, k]; strong = a.topk(NSTRONG).indices; onmask = torch.zeros(N, dtype=torch.bool, device=DEV); onmask[a.topk(max(1, N // 20)).indices] = True
    mean_act = torch.full((N,), float(a.mean()), device=DEV); kl_abl = kl_with(k, mean_act, strong); LG = {key: ledger(k, key) for key in Rd}; PRE = {}
    for key, (C16, c0, tot, IDX, TS) in LG.items():
        bus16 = bus_summary(IDX["abs16"], RT[key][k]); bus32 = bus_summary(IDX["pos16neg16"], RT[key][k])
        PRE[key] = {"abs16": TS["abs16"] + c0, "abs32": TS["abs32"] + c0, "pos16": TS["pos16"] + c0, "pos16neg16": TS["pos16neg16"] + c0, "abs16_bus": TS["abs16"] + c0 + bus16, "pos16neg16_bus": TS["pos16neg16"] + c0 + bus32, "all": tot + c0}
    sims = {}
    for cnd in CONDS:
        base = cnd.split("_")[0] if cnd in ("abs16_on", "abs16_off") else cnd; act = act_of({key: PRE[key][base] for key in Rd})
        if cnd == "abs16_on": act = torch.where(onmask, act, a)
        if cnd == "abs16_off": act = torch.where(onmask, a, act)
        v = kl_with(k, act, strong); sims[cnd] = dict(strong=v[0], all=v[1], ratio_strong=v[0] / max(kl_abl[0], 1e-12), ratio_all=v[1] / max(kl_abl[1], 1e-12))
    C16, c0, tot, IDX, TS = LG[key0]; brake = float(((tot - TS["abs16"])[~onmask] * torch.sign(TS["abs16"][~onmask]) / TS["abs16"][~onmask].abs().clamp_min(1e-9)).median()); brake_on = float(((tot - TS["abs16"])[onmask] * torch.sign(TS["abs16"][onmask]) / TS["abs16"][onmask].abs().clamp_min(1e-9)).median())
    neg_share = float(((TS["pos16neg16"] - TS["pos16"])[~onmask] / (tot - TS["pos16"])[~onmask].clamp(max=-1e-9)).median()) if bool((~onmask).any()) else None   # how much of the off-class net opposition the sixteen inhibitory wires carry
    rows[k] = dict(kl_ablate_strong=kl_abl[0], kl_ablate_all=kl_abl[1], sim=sims, off_share_all=sims["abs16_off"]["all"] / max(sims["abs16"]["all"], 1e-12), on_share_all=sims["abs16_on"]["all"] / max(sims["abs16"]["all"], 1e-12), off_share_strong=sims["abs16_off"]["strong"] / max(sims["abs16"]["strong"], 1e-12), crowd_over_wires_off=brake, crowd_over_wires_on=brake_on, neg16_share_of_opposition=neg_share)
    if len(rows) % 5 == 0: L_(f"{len(rows)} neurons; last: abs16 ratio strong/all {sims['abs16']['ratio_strong']:.2f}/{sims['abs16']['ratio_all']:.2f}, off-only {sims['abs16_off']['ratio_all']:.2f} on-only {sims['abs16_on']['ratio_all']:.2f}, signed {sims['pos16neg16']['ratio_strong']:.2f}, bus {sims['abs16_bus']['ratio_strong']:.2f}, signed+bus {sims['pos16neg16_bus']['ratio_strong']:.2f}")
R = list(rows.values()); mr = lambda c, key: med([r["sim"][c][key] for r in R])
S_ = dict(n=len(R), bus_dims=int(PBUS.shape[1]), kl_ablate_strong=med([r["kl_ablate_strong"] for r in R]), kl_ablate_all=med([r["kl_ablate_all"] for r in R]), ratio_strong={c: mr(c, "ratio_strong") for c in CONDS}, ratio_all={c: mr(c, "ratio_all") for c in CONDS},
          off_share_all=med([r["off_share_all"] for r in R]), on_share_all=med([r["on_share_all"] for r in R]), off_share_strong=med([r["off_share_strong"] for r in R]), crowd_over_wires_off=med([r["crowd_over_wires_off"] for r in R]), crowd_over_wires_on=med([r["crowd_over_wires_on"] for r in R]), neg16_share_of_opposition=med([r["neg16_share_of_opposition"] for r in R]))
rs, ra = S_["ratio_strong"], S_["ratio_all"]
V = dict(B0_all_inputs_reproduce=bool(ra["all"] < 0.05), B1_off_class_carries_damage=bool(S_["off_share_all"] >= 0.7), B1b_on_class_alone_small=bool(S_["on_share_all"] <= 0.3), B2_signed_halves_abs16=bool(rs["pos16neg16"] < 0.5 * rs["abs16"]), B2b_signed_beats_abs32=bool(rs["pos16neg16"] < rs["abs32"] - 0.1),
         B3_bus_halves_abs16=bool(rs["abs16_bus"] < 0.5 * rs["abs16"]), B4_signed_bus_below_025=bool(rs["pos16neg16_bus"] < 0.25), B5_neg16_carries_half_of_opposition=bool(S_["neg16_share_of_opposition"] is not None and S_["neg16_share_of_opposition"] >= 0.5), B6_bus_halves_all_positions=bool(ra["abs16_bus"] < 0.5 * ra["abs16"]))
res = dict(model=M["hf"], block=B, n_blocks=NB, positions=N, heldout_positions=NH, upstream_units=NU, conditions=CONDS, summary=S_, verdicts=V, neurons={str(k): v for k, v in rows.items()})
summ = (f"[{name} block {B}] {len(R)} neurons: KL ratio of the simulation over the ablation's, at the strong firings / over all positions: " + ", ".join(f"{c} {rs[c]:.2f}/{ra[c]:.2f}" for c in CONDS) + f"; the share of the sixteen-input damage (all positions) from the off-class positions {S_['off_share_all']:.2f}, from the top-5% positions {S_['on_share_all']:.2f}; "
        f"the crowd's net sum over the sixteen wires (signed) off the class {S_['crowd_over_wires_off']:.2f}, on it {S_['crowd_over_wires_on']:.2f}; the sixteen inhibitory wires' share of the off-class opposition {S_['neg16_share_of_opposition']}; bus {S_['bus_dims']} directions; verdicts {V}")
save(TAG, res, summ)
