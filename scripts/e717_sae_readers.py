"""e717 (session 163): the SAE on the readers' turf. e715 scored the fields against a public SAE on a position's state, where the
fields add nothing; e706 scored the fields where they were found, on a neuron's input, where they halve a pruned neuron's error.
Here the SAE is scored there too. At GPT-2 small a block-B neuron is driven by the sixteen-input simulation with the two fields
(e706), and by the state entering the block as jbloom's residual SAE reconstructs it (its full active set; its top-k latents by
activation, the encoder's own choice; its top-k latents by their contribution to this neuron, the ledger on the SAE's atoms), with
the block's own attention exact and the position's true normalisation scale held fixed; and by the sixteen wires plus the SAE's
error, which asks whether the error carries the brake. Scored by the downstream KL at the firings and over all positions relative
to mean ablation, and by the off-class share of the damage at the five percent class. Arguments: --block B [--smoke].
Pre-registered in e717_e722_prereg.json; the neurons are e685b's sample."""
import argparse
from c11 import *
F_ = torch.nn.functional
ap = argparse.ArgumentParser(); ap.add_argument("--block", type=int, required=True); ap.add_argument("--smoke", action="store_true"); A_ = ap.parse_args(); SMOKE = A_.smoke
name = "gpt2"; B = A_.block; TAG = f"e717_{name}_b{B}" + ("_smoke" if SMOKE else ""); M = load(name); NB, D, DFF = M["NB"], M["D"], M["DFF"]; T = 256; L_ = lambda s: log(TAG, s)
NWIN = 8 if SMOKE else int(os.environ.get("WDD_NWIN", 64)); NWH = 16 if SMOKE else int(os.environ.get("WDD_NWH", 96)); CH_ = int(os.environ.get("WDD_CHUNK", 2048)); NNEU = 4 if SMOKE else 40; NATT = n_att_feeding(M, B); NSTRONG = 160; Q = 0.25; CUT = 0.05
assert B >= 1 and not M["parallel"], "the block's own attention must feed its MLP and the SAE sits at the block's input"
ids_all = wikitext_ids(M, "train"); half = len(ids_all) // 2
ids = windows_from(ids_all[:half], NWIN, T).to(DEV); ids_h = windows_from(ids_all[half:], NWH, T).to(DEV); N = NWIN * (T - 1); NH = NWH * (T - 1); NMLP = B * DFF; NU = NMLP + NATT + 1
def capture(ids_, nwin, need_lp):
    UP, AT, EM, XM, XP, AB, LP = [], [], [], [], [], [], []
    with Capture(M, list(range(B + 1)), ["act", "att", "xin", "emb", "x"]) as C:
        for s0 in range(0, nwin, 8):
            o = M["base"](input_ids=ids_[s0:s0 + 8], use_cache=False)
            if need_lp: LP.append(logprobs_at(M, o.last_hidden_state[:, :-1]))
            UP.append(torch.cat([C.cap[("act", b)][:, 1:].half() for b in range(B)], -1).reshape(-1, NMLP)); AT.append(torch.stack([C.cap[("att", b)][:, 1:].half() for b in range(NATT)], 2).reshape(-1, NATT, D))
            EM.append(C.cap[("emb", 0)][:, 1:].half().reshape(-1, D)); XM.append(C.cap[("xin", B)][:, 1:].float().reshape(-1, D)); XP.append(C.cap[("x", B - 1)][:, 1:].float().reshape(-1, D)); AB.append(C.cap[("act", B)][:, 1:].half().reshape(-1, DFF))
    S = dict(UP=torch.cat(UP), AT=torch.cat(AT), EM=torch.cat(EM), XM=torch.cat(XM), XP=torch.cat(XP), AB=torch.cat(AB)); S["LP"] = torch.cat(LP).reshape(nwin * (T - 1), -1) if need_lp else None; return S
E = capture(ids, NWIN, True); H = capture(ids_h, NWH, False); L_(f"block {B} of {NB}: {N} evaluation and {NH} held-out positions, {NU} upstream units")
Rd = read_w(M, B); RB = read_bias(M, B); RT = {k: read_tilde(M, B, Rd[k]) for k in Rd}; RC = {k: read_const(M, B, Rd[k], RB[k]) for k in Rd}; WUP = torch.cat([write_w(M, b).T for b in range(B)])
BIAS = sum((write_bias(M, b) for b in range(B)), torch.zeros(D, device=DEV)); P_ = ln_params(M, B)
sig = lambda XM: (XM.var(-1, unbiased=False) + P_["eps"]).sqrt() if P_["centre"] else (XM.pow(2).mean(-1) + P_["eps"]).sqrt(); E["SIG"] = sig(E["XM"]); H["SIG"] = sig(H["XM"])
# ---------------- the fields (e706): each upstream block's weak write and strong write on held-out text, sixteen directions each
PBW, PBS = [], []
for b in range(B):
    cols = slice(b * DFF, (b + 1) * DFF); tau = torch.cat([H["UP"][:, b * DFF + c0:b * DFF + min(c0 + 512, DFF)].float().abs().quantile(1 - Q, dim=0) for c0 in range(0, DFF, 512)]); W = write_w(M, b); cw = torch.zeros(D, D, device=DEV); cs = torch.zeros(D, D, device=DEV)
    for s_ in range(0, NH, 2048):
        a = H["UP"][s_:s_ + 2048, cols].float(); wk = a.abs() < tau[None]; dw = (a * wk) @ W.T; ds = (a * ~wk) @ W.T; cw += dw.T @ dw; cs += ds.T @ ds
    PBW.append(torch.linalg.eigh(cw)[1][:, -16:]); PBS.append(torch.linalg.eigh(cs)[1][:, -16:])
BOTH = torch.linalg.qr(torch.cat(PBW + PBS, 1))[0]; KB = BOTH.shape[1]; gen = torch.Generator(device=DEV).manual_seed(706 + B); RANDB = torch.linalg.qr(torch.randn(D, KB, device=DEV, generator=gen))[0]
del H
# ---------------- the SAE at the block's input (jbloom's resid_pre SAE of block B, on the stream as TransformerLens centres it)
from huggingface_hub import snapshot_download; from safetensors.torch import load_file
hp = f"blocks.{B}.hook_resid_pre"; pth = snapshot_download("jbloom/GPT2-Small-SAEs-Reformatted", allow_patterns=[f"{hp}/*"]); Wt = load_file(f"{pth}/{hp}/sae_weights.safetensors"); cfg_ = json.load(open(f"{pth}/{hp}/cfg.json"))
We, be, Wd, bd = Wt["W_enc"].to(DEV).float(), Wt["b_enc"].to(DEV).float(), Wt["W_dec"].to(DEV).float(), Wt["b_dec"].to(DEV).float(); ADB = bool(cfg_.get("apply_b_dec_to_input", True)); NL = Wd.shape[0]
XT = E["XP"] - E["XP"].mean(-1, keepdim=True); ATTB = E["XM"] - E["XP"]   # the state entering block B (centred) and the block's own attention output
HL = torch.cat([torch.relu(((XT[s:s + CH_] - bd) if ADB else XT[s:s + CH_]) @ We + be) for s in range(0, N, CH_)]); NACT = (HL > 0).sum(-1).float(); WDN = Wd.norm(dim=1)
ERR = {"full": XT - (HL @ Wd + bd)}
for k in (16, 32):
    top = (HL * WDN[None]).topk(k, dim=1).indices; Hk = torch.zeros_like(HL).scatter_(1, top, HL.gather(1, top)); ERR[f"top{k}"] = XT - (Hk @ Wd + bd); del Hk
ESH = {k: float(v.pow(2).sum() / XT.pow(2).sum()) for k, v in ERR.items()}; L_(f"SAE {hp}: {NL} latents, {med(NACT):.0f} active per position; error energy share full {ESH['full']:.3f}, top16 {ESH['top16']:.3f}, top32 {ESH['top32']:.3f}")
# ---------------- the readers
def contrib(S, k, P, key):
    r = RT[key][k]; V = WUP @ r; s = S["SIG"][P]
    C = torch.cat([S["UP"][P].float() * V[None] / s[:, None], (S["AT"][P].float() @ r) / s[:, None], ((S["EM"][P].float() @ r) / s)[:, None]], 1); return C, (BIAS @ r) / s + RC[key][k]
def ledger(k, key):
    c0 = torch.empty(N, device=DEV); tot = torch.empty(N, device=DEV); I16 = torch.empty(N, 16, device=DEV, dtype=torch.long); TS16 = torch.empty(N, device=DEV)
    for s in range(0, N, CH_):
        Pp = torch.arange(s, min(s + CH_, N), device=DEV); c, cc = contrib(E, k, Pp, key); c0[Pp] = cc; tot[Pp] = c.sum(1); top = c.abs().topk(16, dim=1).indices; I16[Pp] = top; TS16[Pp] = c.gather(1, top).sum(1)
    return c0, tot, I16, TS16
def bus_summary(idx, r_, Pm):
    out = torch.empty(N, device=DEV); pr = Pm @ (Pm.T @ r_)
    for s in range(0, N, CH_):
        Pp = torch.arange(s, min(s + CH_, N), device=DEV); top = idx[Pp]; isM = (top < NMLP).float(); isA = (top >= NMLP) & (top < NMLP + NATT); isE = (top >= NMLP + NATT).any(1).float()
        ti = top.clamp(max=NMLP - 1); a = E["UP"][Pp].gather(1, ti).float() * isM; xm = torch.einsum("nk,nkd->nd", a, WUP[ti])
        ai_ = (top - NMLP).clamp(0, NATT - 1); xa = (E["AT"][Pp].float().gather(1, ai_[:, :, None].expand(-1, -1, D)) * isA[:, :, None]).sum(1)
        xc = (E["XM"][Pp] - BIAS[None] - xm - xa - E["EM"][Pp].float() * isE[:, None]) / E["SIG"][Pp][:, None]; out[Pp] = xc @ pr
    return out
def sae_ledger(r_):
    """the SAE's atoms as this neuron's inputs: the total of the latents' contributions and the sum of the k largest by magnitude"""
    V = Wd @ r_; ctot = torch.empty(N, device=DEV); CT = {k: torch.empty(N, device=DEV) for k in (16, 32)}
    for s in range(0, N, CH_):
        Pp = torch.arange(s, min(s + CH_, N), device=DEV); c = HL[Pp] * V[None] / E["SIG"][Pp][:, None]; ctot[Pp] = c.sum(1)
        for k in CT: CT[k][Pp] = c.gather(1, c.abs().topk(k, dim=1).indices).sum(1)
    return ctot, CT
ab = E["AB"].float(); mx = ab.max(0).values; a100 = ab.topk(100, dim=0).values[-1]; cand = torch.nonzero((mx > 1.0) & (a100 >= 0.25 * mx))[:, 0].tolist(); NEU = sorted(random.Random(685 + B).sample(cand, min(NNEU, len(cand))))
def act_of(PRE): return M["act"](PRE["h"])
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
BASES = ["abs16", "abs16_both", "abs16_random", "abs16_saeerr", "sae_full", "sae_top16", "sae_top32", "sae_led16", "sae_led32", "all"]; SPLIT = ["abs16", "sae_full", "sae_top16", "sae_led16"]
CONDS = BASES + [f"{b_}_on5" for b_ in SPLIT] + [f"{b_}_off5" for b_ in SPLIT]
rows = {}
for k in NEU:
    a = ab[:, k]; strong = a.topk(NSTRONG).indices; mask = torch.zeros(N, dtype=torch.bool, device=DEV); mask[a.topk(max(1, int(N * CUT))).indices] = True
    mean_act = torch.full((N,), float(a.mean()), device=DEV); kl_abl = kl_with(k, mean_act, strong); PRE = {}
    for key in Rd:
        c0, tot, I16, TS16 = ledger(k, key); r_ = RT[key][k]; full = tot + c0; e_ = {nm: (ERR[nm] @ r_) / E["SIG"] for nm in ERR}; ctot, CT = sae_ledger(r_)
        PRE[key] = {"abs16": TS16 + c0, "abs16_both": TS16 + c0 + bus_summary(I16, r_, BOTH), "abs16_random": TS16 + c0 + bus_summary(I16, r_, RANDB), "abs16_saeerr": TS16 + c0 + e_["full"],
                    "sae_full": full - e_["full"], "sae_top16": full - e_["top16"], "sae_top32": full - e_["top32"], "sae_led16": full - e_["full"] - ctot + CT[16], "sae_led32": full - e_["full"] - ctot + CT[32], "all": full}
    sims = {}
    for cnd in CONDS:
        base = cnd.split("_on5")[0].split("_off5")[0]; act = act_of({key: PRE[key][base] for key in Rd})
        if cnd.endswith("_on5"): act = torch.where(mask, act, a)
        if cnd.endswith("_off5"): act = torch.where(mask, a, act)
        v = kl_with(k, act, strong); sims[cnd] = dict(strong=v[0], all=v[1], ratio_strong=v[0] / max(kl_abl[0], 1e-12), ratio_all=v[1] / max(kl_abl[1], 1e-12))
    rows[k] = dict(kl_ablate_strong=kl_abl[0], kl_ablate_all=kl_abl[1], sim=sims, off_share={b_: sims[f"{b_}_off5"]["all"] / max(sims[b_]["all"], 1e-12) for b_ in SPLIT})
    if len(rows) % 5 == 0 or SMOKE: L_(f"{len(rows)} neurons; last ratios at the firings / over all positions: " + ", ".join(f"{c} {sims[c]['ratio_strong']:.2f}/{sims[c]['ratio_all']:.2f}" for c in BASES) + "; off share " + ", ".join(f"{b_} {rows[k]['off_share'][b_]:.2f}" for b_ in SPLIT))
R = list(rows.values()); mr = lambda c, key: med([r["sim"][c][key] for r in R])
S_ = dict(n=len(R), sae=hp, sae_latents=NL, sae_active_latents=med(NACT), sae_error_energy_share=ESH, field_dims=KB, kl_ablate_strong=med([r["kl_ablate_strong"] for r in R]), kl_ablate_all=med([r["kl_ablate_all"] for r in R]), ratio_strong={c: mr(c, "ratio_strong") for c in CONDS}, ratio_all={c: mr(c, "ratio_all") for c in CONDS}, off_share={b_: med([r["off_share"][b_] for r in R]) for b_ in SPLIT})
rs, ra, osh = S_["ratio_strong"], S_["ratio_all"], S_["off_share"]
V = dict(S0_all_inputs_reproduce=bool(ra["all"] < 0.05), S1_saefull_below_both_at_firings=bool(rs["sae_full"] < rs["abs16_both"]), S2_saefull_all_below_05=bool(ra["sae_full"] < 0.5), S3_saefull_offshare_below_abs16=bool(osh["sae_full"] < osh["abs16"]), S4_saetop16_runaway=bool(ra["sae_top16"] > 1),
         S5_led16_beats_top16_at_firings=bool(rs["sae_led16"] < rs["sae_top16"]), S6_led16_below_abs16_at_firings=bool(rs["sae_led16"] < rs["abs16"]), S7_saeerr_helps_wires_by_01=bool(rs["abs16_saeerr"] < rs["abs16"] - 0.1), S8_saefull_all_below_both_all=bool(ra["sae_full"] < ra["abs16_both"]))
res = dict(model=M["hf"], block=B, n_blocks=NB, positions=N, heldout_positions=NH, upstream_units=NU, conditions=CONDS, summary=S_, verdicts=V, neurons={str(k): v for k, v in rows.items()})
summ = (f"[{name} block {B}] {len(R)} neurons, SAE {hp} with {med(NACT):.0f} active latents and error energy share {ESH['full']:.3f}: KL ratio at the firings / over all positions: " + ", ".join(f"{c} {rs[c]:.2f}/{ra[c]:.2f}" for c in BASES) + "; off-class share of the damage at the five percent class: " + ", ".join(f"{b_} {osh[b_]:.2f}" for b_ in SPLIT) + f"; verdicts {V}")
save(TAG, res, summ)
