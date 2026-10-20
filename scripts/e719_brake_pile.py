"""e719 (session 163): the brake on the Pile. e697 and e716 measured the brake on wikitext; here e716's table (the off-class share
at class cuts of 1, 5 and 20 percent, the excitatory-only runaway and the signed repair at k of 8, 16 and 32 wires) is taken on
windows of the Pile, with the same neurons as the wikitext run when its result file is given. Arguments: --model M --block B
[--neurons-from RESULT.json] [--smoke]. Pre-registered in e717_e722_prereg.json."""

import argparse
from c11 import *
F_ = torch.nn.functional
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--block", type=int, required=True); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--neurons-from", default=None); A_ = ap.parse_args(); SMOKE = A_.smoke
name = A_.model; B = A_.block; TAG = f"e719_{name}_b{B}" + ("_smoke" if SMOKE else ""); M = load(name); NB, D, DFF = M["NB"], M["D"], M["DFF"]; T = 256; L_ = lambda s: log(TAG, s)
NWIN = 8 if SMOKE else int(os.environ.get("WDD_NWIN", 64)); NWH = 16 if SMOKE else int(os.environ.get("WDD_NWH", 96)); CH_ = int(os.environ.get("WDD_CHUNK", 2048)); NNEU = 4 if SMOKE else 40; NATT = n_att_feeding(M, B); NSTRONG = 160; Q = 0.25; RB_ = 16
ids_all = pile_ids(M, ndocs=3000); half = len(ids_all) // 2
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
E = capture(ids, NWIN, True); L_(f"block {B} of {NB}: {N} evaluation positions, {NU} upstream units")
Rd = read_w(M, B); RB = read_bias(M, B); RT = {k: read_tilde(M, B, Rd[k]) for k in Rd}; RC = {k: read_const(M, B, Rd[k], RB[k]) for k in Rd}; WUP = torch.cat([write_w(M, b).T for b in range(B)]) if B else torch.zeros(0, D, device=DEV)
BIAS = sum((write_bias(M, b) for b in range(B)), torch.zeros(D, device=DEV)); P_ = ln_params(M, B)
sig = lambda XM: (XM.var(-1, unbiased=False) + P_["eps"]).sqrt() if P_["centre"] else (XM.pow(2).mean(-1) + P_["eps"]).sqrt(); E["SIG"] = sig(E["XM"]); key0 = "h" if M["fam"] != "gated" else "g"
def contrib(S, k, P, key):
    r = RT[key][k]; V = WUP @ r; s = S["SIG"][P]
    C = torch.cat([S["UP"][P].float() * V[None] / s[:, None], (S["AT"][P].float() @ r) / s[:, None], ((S["EM"][P].float() @ r) / s)[:, None]], 1); return C, (BIAS @ r) / s + RC[key][k]
KS = (8, 16, 32); CUTS = (0.01, 0.05, 0.20); WIRES = [f"abs{k}" for k in KS] + [f"pos{k}" for k in KS] + [f"pos{k}neg{k}" for k in KS]
def ledger(k, key):
    """the constant, the total, and for each wire set its sum (indices are not kept: no bus here)"""
    c0 = torch.empty(N, device=DEV); tot = torch.empty(N, device=DEV); TS = {w: torch.empty(N, device=DEV) for w in WIRES}
    for s in range(0, N, CH_):
        P = torch.arange(s, min(s + CH_, N), device=DEV); c, cc = contrib(E, k, P, key); c0[P] = cc; tot[P] = c.sum(1)
        for kk in KS:
            TS[f"abs{kk}"][P] = c.gather(1, c.abs().topk(kk, dim=1).indices).sum(1); pos = c.topk(kk, dim=1).indices; neg = (-c).topk(kk, dim=1).indices
            TS[f"pos{kk}"][P] = c.gather(1, pos).sum(1); TS[f"pos{kk}neg{kk}"][P] = TS[f"pos{kk}"][P] + c.gather(1, neg).sum(1)
    return c0, tot, TS
ab = E["AB"].float(); mx = ab.max(0).values; a100 = ab.topk(100, dim=0).values[-1]; floor_act = 1.0 if M["fam"] != "gated" else float(mx.quantile(0.1)); cand = torch.nonzero((mx > floor_act) & (a100 >= 0.25 * mx))[:, 0].tolist(); NEU = sorted(random.Random(685 + B).sample(cand, min(NNEU, len(cand))))
REF = json.load(open(A_.neurons_from)) if A_.neurons_from else None
if REF is not None: NEU = sorted(int(x) for x in REF["neurons"])[:NNEU]; L_(f"{len(NEU)} neurons taken from {os.path.basename(A_.neurons_from)}")
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
CONDS = WIRES + [f"abs16_on{int(c * 100)}" for c in CUTS] + [f"abs16_off{int(c * 100)}" for c in CUTS] + ["all"]
rows = {}
for k in NEU:
    a = ab[:, k]; strong = a.topk(NSTRONG).indices; masks = {}
    for c in CUTS: m_ = torch.zeros(N, dtype=torch.bool, device=DEV); m_[a.topk(max(1, int(N * c))).indices] = True; masks[int(c * 100)] = m_
    mean_act = torch.full((N,), float(a.mean()), device=DEV); kl_abl = kl_with(k, mean_act, strong); LG = {key: ledger(k, key) for key in Rd}; PRE = {}
    for key, (c0, tot, TS) in LG.items(): PRE[key] = {w: TS[w] + c0 for w in WIRES}; PRE[key]["all"] = tot + c0
    sims = {}
    for cnd in CONDS:
        base = "abs16" if cnd.startswith("abs16_") else cnd; act = act_of({key: PRE[key][base] for key in Rd})
        if cnd.startswith("abs16_on"): act = torch.where(masks[int(cnd[8:])], act, a)
        if cnd.startswith("abs16_off"): act = torch.where(masks[int(cnd[9:])], a, act)
        v = kl_with(k, act, strong); sims[cnd] = dict(strong=v[0], all=v[1], ratio_strong=v[0] / max(kl_abl[0], 1e-12), ratio_all=v[1] / max(kl_abl[1], 1e-12))
    rows[k] = dict(kl_ablate_strong=kl_abl[0], kl_ablate_all=kl_abl[1], sim=sims, off_share={int(c * 100): sims[f"abs16_off{int(c * 100)}"]["all"] / max(sims["abs16"]["all"], 1e-12) for c in CUTS})
    if len(rows) % 5 == 0: L_(f"{len(rows)} neurons; last: " + ", ".join(f"{w} {sims[w]['ratio_strong']:.2f}/{sims[w]['ratio_all']:.2f}" for w in WIRES) + "; off share " + ", ".join(f"{int(c * 100)}% {rows[k]['off_share'][int(c * 100)]:.2f}" for c in CUTS))
R = list(rows.values()); mr = lambda c, key: med([r["sim"][c][key] for r in R])
S_ = dict(n=len(R), kl_ablate_strong=med([r["kl_ablate_strong"] for r in R]), kl_ablate_all=med([r["kl_ablate_all"] for r in R]), ratio_strong={c: mr(c, "ratio_strong") for c in CONDS}, ratio_all={c: mr(c, "ratio_all") for c in CONDS}, off_share={int(c * 100): med([r["off_share"][int(c * 100)] for r in R]) for c in CUTS})
rs, ra, osh = S_["ratio_strong"], S_["ratio_all"], S_["off_share"]
REFO = None
if REF is not None:
    ss = REF.get("summary_stats") if isinstance(REF.get("summary_stats"), dict) else None
    if ss and isinstance(ss.get("off_share"), dict): REFO = ss["off_share"].get("5", ss["off_share"].get(5))
    elif ss and "off_share_all" in ss: REFO = ss["off_share_all"]   # e697's name for the same share at the five percent class
V = dict(U1_off_share_half_at_5=bool(osh[5] >= 0.5), U2_excitatory_runaway_16=bool(ra["pos16"] > 3), U3_signed_below_abs_16=bool(rs["pos16neg16"] < rs["abs16"]), U4_off_share_within_015_of_wikitext=(bool(abs(osh[5] - REFO) < 0.15) if REFO is not None else None), R1_off_share_half_at_every_cut=bool(all(osh[int(c * 100)] >= 0.5 for c in CUTS)), R2_excitatory_runaway_every_k=bool(all(ra[f"pos{k}"] > 3 for k in KS)), R3_signed_below_abs_every_k=bool(all(rs[f"pos{k}neg{k}"] < rs[f"abs{k}"] for k in KS)), R4_off_share_falls_with_cut=bool(osh[1] > osh[5] > osh[20]), R5_abs32_leaves_half=bool(rs["abs32"] >= 0.5))
res = dict(corpus="pile-10k", wikitext_off_share_5=REFO, model=M["hf"], block=B, n_blocks=NB, positions=N, upstream_units=NU, conditions=CONDS, summary=S_, verdicts=V, neurons={str(k): v for k, v in rows.items()})
summ = (f"[{name} block {B}, the Pile] {len(R)} neurons: KL ratio at the firings / over all positions: " + ", ".join(f"{c} {rs[c]:.2f}/{ra[c]:.2f}" for c in CONDS) + "; off-class share of the sixteen-input damage at class cuts 1, 5, 20 percent: " + ", ".join(f"{osh[int(c * 100)]:.2f}" for c in CUTS) + f"; verdicts {V}")
save(TAG, res, summ)
