"""e703 (session 158): rows plus field. WDD reads a position's state as a few native write rows over the extreme-value
floor plus a crowd below it; e693 found the crowd's weak part living in sixteen directions per block and e697 that those
directions carry the brake a pruned neuron graph lacks. Here the dictionary gets the field as a second kind of atom and
two things are measured at a block's input: (A) the state: the centred residual state decomposed by OMP over the upstream
write rows (sixteen atoms, WDD's reading) and then by projection on the field (the upstream blocks' weak-write principal
directions, sixteen or sixty-four per block, from held-out text), against a random subspace and the stream's own principal
directions of the same dimension, scored by the energy left and by the rows still over the floor in the residual; (B) the
readers: e697's sixteen-wire simulation of a neuron plus the field (sixteen or sixty-four per block), plus the attention
units and the embedding as exact wires, against the random subspace, scored by downstream KL relative to ablation.
Arguments: --model M --block B [--smoke]. Pre-registered in e701_e704_prereg.json; the neurons are e685b's sample."""
import argparse
from c11 import *
F_ = torch.nn.functional
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--block", type=int, required=True); ap.add_argument("--smoke", action="store_true"); A_ = ap.parse_args(); SMOKE = A_.smoke
name = A_.model; B = A_.block; TAG = f"e703_{name}_b{B}" + ("_smoke" if SMOKE else ""); M = load(name); NB, D, DFF = M["NB"], M["D"], M["DFF"]; T = 256; L_ = lambda s: log(TAG, s)
NWIN = 8 if SMOKE else 64; NWH = 16 if SMOKE else 96; NNEU = 4 if SMOKE else 40; NPOS = 400 if SMOKE else 2000; NATT = n_att_feeding(M, B); NSTRONG = 160; Q = 0.25; KOMP = 16
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
# ---------------- the field: each upstream block's weak-write principal directions on held-out text; controls of the same dimension
PB16, PBW = [], []
for b in range(B):
    cols = slice(b * DFF, (b + 1) * DFF); tau = torch.cat([H["UP"][:, b * DFF + c0:b * DFF + min(c0 + 512, DFF)].float().abs().quantile(1 - Q, dim=0) for c0 in range(0, DFF, 512)]); W = write_w(M, b); cov = torch.zeros(D, D, device=DEV)
    for s in range(0, NH, 2048):
        a = H["UP"][s:s + 2048, cols].float(); dx = (a * (a.abs() < tau[None])) @ W.T; cov += dx.T @ dx
    ev, evec = torch.linalg.eigh(cov / NH); PB16.append(evec[:, -16:]); PBW.append(evec[:, -64:] * ev[-64:].clamp_min(0).sqrt()[None])   # sixteen per block for the bus; sixty-four per block weighted by their standard deviations for the union
KU = min(128, D // 4); FIELD16 = torch.linalg.qr(torch.cat(PB16, 1))[0] if B else torch.zeros(D, 0, device=DEV); FIELD128 = torch.linalg.svd(torch.cat(PBW, 1), full_matrices=False)[0][:, :KU].contiguous() if B else torch.zeros(D, 0, device=DEV); K16 = FIELD16.shape[1]
gen = torch.Generator(device=DEV).manual_seed(703 + B); RAND16 = torch.linalg.qr(torch.randn(D, K16, device=DEV, generator=gen))[0]; RAND128 = torch.linalg.qr(torch.randn(D, KU, device=DEV, generator=gen))[0]; Xs = H["XM"] / H["SIG"][:, None]; ev_s, evec_s = torch.linalg.eigh(Xs.T @ Xs / NH); SPCA = evec_s[:, -KU:].contiguous(); del Xs
SUBS = {"field16": FIELD16, "random16": RAND16, "field128": FIELD128, "random128": RAND128, "stream128": SPCA}; L_(f"field: {K16} directions (16 per upstream block) and the union's top {KU}; controls of {K16} and {KU} random directions and the stream's {KU}")
# ---------------- (A) the state: rows over the floor, OMP over the rows, then the field
A_rows, blk_, neu_, nr_ = rows_dictionary(M, max(B - 1, 0)); XC = E["XM"] - E["XM"].mean(0); U = unitr(XC); cal = floor_calibration(U[~sinkmask(E["XM"])], A_rows); LF = floor_of(U, cal); A_rot = unitr(rotate(A_rows, seed=5))
P = torch.randperm(N, device=DEV, generator=gen)[:NPOS]; XP = XC[P]; en = lambda X: (X ** 2).sum(1)
def over(Xr, A):
    """rows over the floor in the (unit) residual rows, counted against the dictionary A"""
    Ur = unitr(Xr); lf = floor_of(Ur, cal); return torch.cat([(((Ur[s:s + 512] @ A.T).abs() / lf[s:s + 512, None]) > 1).sum(1) for s in range(0, Ur.shape[0], 512)]).float()
sel, cof = omp(XP, A_rows, k=KOMP); XR = XP - torch.einsum("nk,nkd->nd", cof, A_rows[sel]); over0 = over(XP, A_rows); over_rows = over(XR, A_rows); over_rows_null = over(XR, A_rot)
ST = dict(over_floor_state=float(over0.median()), energy_left_rows=float((en(XR) / en(XP)).median()), over_floor_after_rows=float(over_rows.median()), over_floor_after_rows_null=float(over_rows_null.median()))
for nm, Pm in SUBS.items():
    XF = XR - (XR @ Pm) @ Pm.T; ST[f"energy_left_rows_{nm}"] = float((en(XF) / en(XP)).median()); ST[f"over_floor_after_rows_{nm}"] = float(over(XF, A_rows).median()); ST[f"over_floor_after_rows_{nm}_null"] = float(over(XF, A_rot).median())
    XO = XP - (XP @ Pm) @ Pm.T; ST[f"energy_left_{nm}_only"] = float((en(XO) / en(XP)).median())
L_(f"state: rows over the floor {ST['over_floor_state']:.1f}; energy left after 16 OMP rows {ST['energy_left_rows']:.2f}, then the field16 {ST['energy_left_rows_field16']:.2f} (random {ST['energy_left_rows_random16']:.2f}), the field128 {ST['energy_left_rows_field128']:.2f} (random {ST['energy_left_rows_random128']:.2f}, stream {ST['energy_left_rows_stream128']:.2f}); rows over the floor left {ST['over_floor_after_rows']:.1f} -> field128 {ST['over_floor_after_rows_field128']:.1f} (null {ST['over_floor_after_rows_field128_null']:.1f})")
del A_rows, A_rot, U
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
CONDS = ["abs16", "abs16_field16", "abs16_field128", "abs16_att", "abs16_att_field128", "abs16_random16", "abs16_random128", "abs16_stream128", "all"]
rows = {}
for k in NEU:
    a = ab[:, k]; strong = a.topk(NSTRONG).indices; mean_act = torch.full((N,), float(a.mean()), device=DEV); kl_abl = kl_with(k, mean_act, strong); PRE = {}
    for key in Rd:
        c0, tot, I16, TS16, IA, TSA = ledger(k, key); r_ = RT[key][k]
        PRE[key] = {"abs16": TS16 + c0, "abs16_field16": TS16 + c0 + bus_summary(I16, r_, FIELD16), "abs16_field128": TS16 + c0 + bus_summary(I16, r_, FIELD128), "abs16_att": TSA + c0, "abs16_att_field128": TSA + c0 + bus_summary(IA, r_, FIELD128), "abs16_random16": TS16 + c0 + bus_summary(I16, r_, RAND16), "abs16_random128": TS16 + c0 + bus_summary(I16, r_, RAND128), "abs16_stream128": TS16 + c0 + bus_summary(I16, r_, SPCA), "all": tot + c0}
    sims = {}
    for cnd in CONDS: v = kl_with(k, act_of({key: PRE[key][cnd] for key in Rd}), strong); sims[cnd] = dict(strong=v[0], all=v[1], ratio_strong=v[0] / max(kl_abl[0], 1e-12), ratio_all=v[1] / max(kl_abl[1], 1e-12))
    rows[k] = dict(kl_ablate_strong=kl_abl[0], kl_ablate_all=kl_abl[1], sim=sims)
    if len(rows) % 5 == 0: L_(f"{len(rows)} neurons; last ratios at the firings: " + ", ".join(f"{c} {sims[c]['ratio_strong']:.2f}" for c in CONDS))
R = list(rows.values()); mr = lambda c, key: med([r["sim"][c][key] for r in R])
S_ = dict(n=len(R), field_dims={"16": K16, "128": KU}, state=ST, kl_ablate_strong=med([r["kl_ablate_strong"] for r in R]), kl_ablate_all=med([r["kl_ablate_all"] for r in R]), ratio_strong={c: mr(c, "ratio_strong") for c in CONDS}, ratio_all={c: mr(c, "ratio_all") for c in CONDS})
rs, ra = S_["ratio_strong"], S_["ratio_all"]
V = dict(S0_all_inputs_reproduce=bool(ra["all"] < 0.05), S1_field128_halves_energy_left_by_rows=bool(ST["energy_left_rows_field128"] < 0.5 * ST["energy_left_rows"]), S2_field128_beats_random_by_01=bool(ST["energy_left_rows_field128"] < ST["energy_left_rows_random128"] - 0.1), S2b_field128_beats_stream=bool(ST["energy_left_rows_field128"] < ST["energy_left_rows_stream128"] - 0.05),
         S3_no_rows_over_floor_left=bool(ST["over_floor_after_rows_field128"] <= ST["over_floor_after_rows_field128_null"] + 1), R1_att_field128_below_025=bool(rs["abs16_att_field128"] < 0.25), R2_att_field128_below_05=bool(rs["abs16_att_field128"] < 0.5), R3_field128_beats_field16_by_01=bool(rs["abs16_field128"] < rs["abs16_field16"] - 0.1),
         R4_att_wires_halve_abs16=bool(rs["abs16_att"] < 0.5 * rs["abs16"]), R5_field128_beats_random_by_02=bool(rs["abs16_field128"] < rs["abs16_random128"] - 0.2), R5b_field16_beats_random16_by_01=bool(rs["abs16_field16"] < rs["abs16_random16"] - 0.1), R6_att_field128_halves_all_positions=bool(ra["abs16_att_field128"] < 0.5 * ra["abs16"]))
res = dict(model=M["hf"], block=B, n_blocks=NB, positions=N, heldout_positions=NH, upstream_units=NU, omp_atoms=KOMP, state_positions=NPOS, conditions=CONDS, summary=S_, verdicts=V, neurons={str(k): v for k, v in rows.items()})
summ = (f"[{name} block {B}] the state at {NPOS} positions: {ST['over_floor_state']:.1f} rows over the floor; after 16 OMP rows {ST['energy_left_rows']:.2f} of the energy is left, after the field16 too ({K16} directions) {ST['energy_left_rows_field16']:.2f} (random {ST['energy_left_rows_random16']:.2f}), after the field128 ({KU}) {ST['energy_left_rows_field128']:.2f} (random {ST['energy_left_rows_random128']:.2f}, the stream's {ST['energy_left_rows_stream128']:.2f}); "
        f"rows over the floor left in the residual after rows {ST['over_floor_after_rows']:.1f} (rotated null {ST['over_floor_after_rows_null']:.1f}), after rows and the field128 {ST['over_floor_after_rows_field128']:.1f} (null {ST['over_floor_after_rows_field128_null']:.1f}); the readers, {len(R)} neurons, KL ratio at the firings / over all positions: " + ", ".join(f"{c} {rs[c]:.2f}/{ra[c]:.2f}" for c in CONDS) + f"; verdicts {V}")
save(TAG, res, summ)
