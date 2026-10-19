"""e706 (session 159): the strong field. e703 found the weak-write field specific for a neuron's readers and the attention
units no help, with the best training-free graph leaving a quarter to a half of a pruned neuron's error. The crowd outside
the sixteen wires also holds the strong writes of units that are not wires; here each upstream block's strong write (the
kept quarter of its activations) gets its own sixteen principal directions, and the reader graph is scored with the weak
field, the strong field, both, a random subspace of both's dimension, and both with the attention units as exact wires.
Arguments: --model M --block B [--smoke]. Pre-registered in e705_e706_prereg.json; the neurons are e685b's sample."""
import argparse
from c11 import *
F_ = torch.nn.functional
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--block", type=int, required=True); ap.add_argument("--smoke", action="store_true"); A_ = ap.parse_args(); SMOKE = A_.smoke
name = A_.model; B = A_.block; TAG = f"e706_{name}_b{B}" + ("_smoke" if SMOKE else ""); M = load(name); NB, D, DFF = M["NB"], M["D"], M["DFF"]; T = 256; L_ = lambda s: log(TAG, s)
NWIN = 8 if SMOKE else 64; NWH = 16 if SMOKE else 96; NNEU = 4 if SMOKE else 40; NATT = n_att_feeding(M, B); NSTRONG = 160; Q = 0.25
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
# ---------------- the fields: each upstream block's weak write (all but the strongest quarter) and strong write (the strongest quarter) on held-out text, sixteen directions each
PBW, PBS = [], []
for b in range(B):
    cols = slice(b * DFF, (b + 1) * DFF); tau = torch.cat([H["UP"][:, b * DFF + c0:b * DFF + min(c0 + 512, DFF)].float().abs().quantile(1 - Q, dim=0) for c0 in range(0, DFF, 512)]); W = write_w(M, b); cw = torch.zeros(D, D, device=DEV); cs = torch.zeros(D, D, device=DEV)
    for s_ in range(0, NH, 2048):
        a = H["UP"][s_:s_ + 2048, cols].float(); wk = a.abs() < tau[None]; dw = (a * wk) @ W.T; ds = (a * ~wk) @ W.T; cw += dw.T @ dw; cs += ds.T @ ds
    PBW.append(torch.linalg.eigh(cw)[1][:, -16:]); PBS.append(torch.linalg.eigh(cs)[1][:, -16:])
FIELDW = torch.linalg.qr(torch.cat(PBW, 1))[0]; FIELDS = torch.linalg.qr(torch.cat(PBS, 1))[0]; BOTH = torch.linalg.qr(torch.cat(PBW + PBS, 1))[0]; KB = BOTH.shape[1]
gen = torch.Generator(device=DEV).manual_seed(706 + B); RANDB = torch.linalg.qr(torch.randn(D, KB, device=DEV, generator=gen))[0]; OVWS = float((FIELDW.T @ FIELDS).pow(2).sum() / FIELDW.shape[1])
L_(f"fields: weak {FIELDW.shape[1]}, strong {FIELDS.shape[1]}, both {KB} directions from {B} upstream blocks; weak-strong overlap {OVWS:.2f}")
# ---------------- (B) the readers
def contrib(S, k, P, key):
    r = RT[key][k]; V = WUP @ r; s = S["SIG"][P]
    C = torch.cat([S["UP"][P].float() * V[None] / s[:, None], (S["AT"][P].float() @ r) / s[:, None], ((S["EM"][P].float() @ r) / s)[:, None]], 1); return C, (BIAS @ r) / s + RC[key][k]
ATTIDX = torch.arange(NMLP, NU, device=DEV)   # the attention units and the embedding
def ledger(k, key):
    c0 = torch.empty(N, device=DEV); tot = torch.empty(N, device=DEV); I16 = torch.empty(N, 16, device=DEV, dtype=torch.long); TS16 = torch.empty(N, device=DEV); TSA = torch.empty(N, device=DEV); IA = torch.empty(N, 16 + ATTIDX.numel(), device=DEV, dtype=torch.long)
    for s in range(0, N, 2048):
        Pp = torch.arange(s, min(s + 2048, N), device=DEV); c, cc = contrib(E, k, Pp, key); c0[Pp] = cc; tot[Pp] = c.sum(1); top = c.abs().topk(16, dim=1).indices; I16[Pp] = top; TS16[Pp] = c.gather(1, top).sum(1)
        cm = c.clone(); cm[:, NMLP:] = 0; topm = cm.abs().topk(16, dim=1).indices; ia = torch.cat([topm, ATTIDX[None].expand(len(Pp), -1)], 1); IA[Pp] = ia; TSA[Pp] = c.gather(1, ia).sum(1)   # sixteen MLP wires plus every attention unit and the embedding, exact
    return c0, tot, I16, TS16, IA, TSA
def bus_summary(idx, r_, Pm):
    out = torch.empty(N, device=DEV); pr = Pm @ (Pm.T @ r_)
    for s in range(0, N, 2048):
        Pp = torch.arange(s, min(s + 2048, N), device=DEV); top = idx[Pp]; isM = (top < NMLP).float(); isA = (top >= NMLP) & (top < NMLP + NATT); isE = (top >= NMLP + NATT).any(1).float()
        ti = top.clamp(max=max(NMLP - 1, 0)); a = E["UP"][Pp].gather(1, ti).float() * isM; xm = torch.einsum("nk,nkd->nd", a, WUP[ti]) if NMLP else torch.zeros(len(Pp), D, device=DEV)
        ai_ = (top - NMLP).clamp(0, NATT - 1); xa = (E["AT"][Pp].float().gather(1, ai_[:, :, None].expand(-1, -1, D)) * isA[:, :, None]).sum(1)
        xc = (E["XM"][Pp] - BIAS[None] - xm - xa - E["EM"][Pp].float() * isE[:, None]) / E["SIG"][Pp][:, None]; out[Pp] = xc @ pr
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
CONDS = ["abs16", "abs16_weak", "abs16_strong", "abs16_both", "abs16_random", "abs16_att_both", "all"]
rows = {}
for k in NEU:
    a = ab[:, k]; strong = a.topk(NSTRONG).indices; mean_act = torch.full((N,), float(a.mean()), device=DEV); kl_abl = kl_with(k, mean_act, strong); PRE = {}
    for key in Rd:
        c0, tot, I16, TS16, IA, TSA = ledger(k, key); r_ = RT[key][k]
        PRE[key] = {"abs16": TS16 + c0, "abs16_weak": TS16 + c0 + bus_summary(I16, r_, FIELDW), "abs16_strong": TS16 + c0 + bus_summary(I16, r_, FIELDS), "abs16_both": TS16 + c0 + bus_summary(I16, r_, BOTH), "abs16_random": TS16 + c0 + bus_summary(I16, r_, RANDB), "abs16_att_both": TSA + c0 + bus_summary(IA, r_, BOTH), "all": tot + c0}
    sims = {}
    for cnd in CONDS: v = kl_with(k, act_of({key: PRE[key][cnd] for key in Rd}), strong); sims[cnd] = dict(strong=v[0], all=v[1], ratio_strong=v[0] / max(kl_abl[0], 1e-12), ratio_all=v[1] / max(kl_abl[1], 1e-12))
    rows[k] = dict(kl_ablate_strong=kl_abl[0], kl_ablate_all=kl_abl[1], sim=sims)
    if len(rows) % 5 == 0: L_(f"{len(rows)} neurons; last ratios at the firings: " + ", ".join(f"{c} {sims[c]['ratio_strong']:.2f}" for c in CONDS))
R = list(rows.values()); mr = lambda c, key: med([r["sim"][c][key] for r in R])
S_ = dict(n=len(R), field_dims={"weak": int(FIELDW.shape[1]), "strong": int(FIELDS.shape[1]), "both": KB}, weak_strong_overlap=OVWS, kl_ablate_strong=med([r["kl_ablate_strong"] for r in R]), kl_ablate_all=med([r["kl_ablate_all"] for r in R]), ratio_strong={c: mr(c, "ratio_strong") for c in CONDS}, ratio_all={c: mr(c, "ratio_all") for c in CONDS})
rs, ra = S_["ratio_strong"], S_["ratio_all"]
V = dict(H0_all_inputs_reproduce=bool(ra["all"] < 0.05), H1_strong_alone_halves_abs16=bool(rs["abs16_strong"] < 0.5 * rs["abs16"]), H2_both_beats_weak_by_01=bool(rs["abs16_both"] < rs["abs16_weak"] - 0.1), H3_both_att_below_025=bool(rs["abs16_att_both"] < 0.25), H4_both_below_05=bool(rs["abs16_both"] < 0.5), H5_both_beats_random_by_02=bool(rs["abs16_both"] < rs["abs16_random"] - 0.2), H6_both_halves_all_positions=bool(ra["abs16_both"] < 0.5 * ra["abs16"]))
res = dict(model=M["hf"], block=B, n_blocks=NB, positions=N, heldout_positions=NH, upstream_units=NU, conditions=CONDS, summary=S_, verdicts=V, neurons={str(k): v for k, v in rows.items()})
summ = (f"[{name} block {B}] {len(R)} neurons, KL ratio at the firings / over all positions: " + ", ".join(f"{c} {rs[c]:.2f}/{ra[c]:.2f}" for c in CONDS) + f"; the weak and strong fields overlap by {OVWS:.2f}; verdicts {V}")
save(TAG, res, summ)
