"""e723 (session 164): the matched-budget comparison on the readers' turf. e717 scored a public SAE's reconstruction as a neuron's
input beside the sixteen wires plus the two fields, but the fields used their full dimension (thirty-two directions per upstream
block) and the SAE its own active set. Here the budget is matched and the selection is held out: for each neuron, k directions of
the fields, k decoder directions of the SAE and k random directions, chosen on held-out text by their mean absolute contribution
to the neuron, carry the crowd's projection beside the sixteen wires, for k of 16, 64 and the fields' full dimension; the SAE's
own coefficients on its k latents largest by contribution (e717's rule) stand beside the projection; and the SAE's full
reconstruction is scored alone and with its error projected on the fields, on the SAE's own directions and on random directions
(does the field carry what the SAE misses?). Scored by the downstream KL at the firings and over all positions relative to mean
ablation, by the off-class share of the damage at the five percent class, and by the brake fidelity, the signed share and the
correlation of the crowd's true net sum that each input reproduces off the class. Arguments: --block B --sae {jbloom,oai32k,oai128k}
[--smoke]. Pre-registered in e723_e724_prereg.json; the neurons are e685b's sample."""
import argparse, urllib.request
from c11 import *
F_ = torch.nn.functional
ap = argparse.ArgumentParser(); ap.add_argument("--block", type=int, required=True); ap.add_argument("--sae", default="jbloom"); ap.add_argument("--smoke", action="store_true"); A_ = ap.parse_args(); SMOKE = A_.smoke
name = "gpt2"; B = A_.block; SAE = A_.sae; TAG = f"e723_{name}_b{B}_{SAE}" + ("_smoke" if SMOKE else ""); M = load(name); NB, D, DFF = M["NB"], M["D"], M["DFF"]; T = 256; L_ = lambda s: log(TAG, s)
NWIN = 8 if SMOKE else int(os.environ.get("WDD_NWIN", 64)); NWH = 16 if SMOKE else int(os.environ.get("WDD_NWH", 96)); CH_ = int(os.environ.get("WDD_CHUNK", 2048)); NNEU = 4 if SMOKE else 40; NATT = n_att_feeding(M, B); NSTRONG = 160; Q = 0.25; CUT = 0.05
assert B >= 1 and not M["parallel"]
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
# ---------------- the fields (e706) on held-out text, and a random orthonormal set of their dimension
PBW, PBS = [], []
for b in range(B):
    cols = slice(b * DFF, (b + 1) * DFF); tau = torch.cat([H["UP"][:, b * DFF + c0:b * DFF + min(c0 + 512, DFF)].float().abs().quantile(1 - Q, dim=0) for c0 in range(0, DFF, 512)]); W = write_w(M, b); cw = torch.zeros(D, D, device=DEV); cs = torch.zeros(D, D, device=DEV)
    for s_ in range(0, NH, 2048):
        a = H["UP"][s_:s_ + 2048, cols].float(); wk = a.abs() < tau[None]; dw = (a * wk) @ W.T; ds = (a * ~wk) @ W.T; cw += dw.T @ dw; cs += ds.T @ ds
    PBW.append(torch.linalg.eigh(cw)[1][:, -16:]); PBS.append(torch.linalg.eigh(cs)[1][:, -16:])
BOTH = torch.linalg.qr(torch.cat(PBW + PBS, 1))[0]; KB = BOTH.shape[1]; gen = torch.Generator(device=DEV).manual_seed(706 + B); RANDB = torch.linalg.qr(torch.randn(D, KB, device=DEV, generator=gen))[0]
KS = sorted(set([16, 64, KB])); BF = (H["XM"].abs() @ BOTH).sum(0) if False else torch.cat([(H["XM"][s:s + 2048] @ BOTH).abs().div(H["SIG"][s:s + 2048, None]).sum(0, keepdim=True) for s in range(0, NH, 2048)]).sum(0)   # held-out size of each field direction in the stream
# ---------------- the SAE at the block's input: jbloom (L1, TransformerLens-centred) or OpenAI TopK (normalised input, rescaled output)
from huggingface_hub import snapshot_download; from safetensors.torch import load_file
if SAE == "jbloom":
    hp = f"blocks.{B}.hook_resid_pre"; pth = snapshot_download("jbloom/GPT2-Small-SAEs-Reformatted", allow_patterns=[f"{hp}/*"]); Wt = load_file(f"{pth}/{hp}/sae_weights.safetensors"); cfg_ = json.load(open(f"{pth}/{hp}/cfg.json"))
    We, be, Wd, bd = Wt["W_enc"].to(DEV).float(), Wt["b_enc"].to(DEV).float(), Wt["W_dec"].to(DEV).float(), Wt["b_dec"].to(DEV).float(); ADB = bool(cfg_.get("apply_b_dec_to_input", True)); KTOP = None; KSTORE = 256
elif SAE in ("oai32k", "oai128k"):
    hp = f"resid_post_mlp_v5_{SAE[3:]}/autoencoders/{B - 1}.pt"; os.makedirs(CACHED, exist_ok=True); f_ = f"{CACHED}/oai_gpt2_{SAE[3:]}_{B - 1}.pt"
    if not os.path.exists(f_): urllib.request.urlretrieve("https://openaipublic.blob.core.windows.net/sparse-autoencoder/gpt2-small/" + hp, f_ + ".tmp"); os.replace(f_ + ".tmp", f_)
    sd_ = torch.load(f_, map_location="cpu", weights_only=False); assert sd_["activation"] == "TopK"
    We, be, Wd, bd = sd_["encoder.weight"].T.to(DEV).float(), sd_["latent_bias"].to(DEV).float(), sd_["decoder.weight"].T.to(DEV).float(), sd_["pre_bias"].to(DEV).float(); ADB = True; KTOP = int(sd_["activation_state_dict"]["k"]); KSTORE = KTOP
else: raise SystemExit(SAE)
NL = Wd.shape[0]
def sae_sparse(x):
    """top-KSTORE latents (indices, values) of the SAE on x, the input as the SAE sees it, and the per-position scale and shift of its output"""
    if KTOP is not None: mu = x.mean(-1, keepdim=True); xc = x - mu; sd = xc.std(-1, keepdim=True); xt = xc / (sd + 1e-5); SC, SH = sd[:, 0], mu
    else: xt = x - x.mean(-1, keepdim=True); SC, SH = torch.ones(x.shape[0], device=DEV), 0.0
    pre = ((xt - bd) if ADB else xt) @ We + be; top = pre.topk(KSTORE, dim=-1); return top.indices, torch.relu(top.values), xt, SC, SH
HI, HV, ERRF, SCE = [], [], [], []
for s in range(0, N, CH_):
    hi, hv, xt, sc, sh = sae_sparse(E["XP"][s:s + CH_]); rec = torch.einsum("nk,nkd->nd", hv, Wd[hi]) + bd; HI.append(hi); HV.append(hv); SCE.append(sc); ERRF.append(xt - rec if KTOP is None else (E["XP"][s:s + CH_] - (rec * sc[:, None] + sh)))
HI = torch.cat(HI); HV = torch.cat(HV); ERRF = torch.cat(ERRF); SCE = torch.cat(SCE); NACT = (HV > 0).sum(-1).float()
AH = torch.zeros(NL, device=DEV)   # held-out: the sum over positions of each latent's activation over the scale, the size of the latent's contribution to any reader
for s in range(0, NH, CH_):
    hi, hv, xt, sc, sh = sae_sparse(H["XP"][s:s + CH_]); AH.scatter_add_(0, hi.reshape(-1), (hv * (sc / H["SIG"][s:s + CH_])[:, None]).reshape(-1))
ESH = float(ERRF.pow(2).sum() / (E["XP"] - E["XP"].mean(-1, keepdim=True)).pow(2).sum()); L_(f"SAE {SAE} {hp}: {NL} latents, {med(NACT):.0f} active per position, error energy share {ESH:.3f}; fields {KB} directions; budgets {KS}")
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
def sae_contrib(r_):
    """each stored latent's contribution to this neuron at each position (the ledger on the SAE's atoms), its total, and the sums of the k largest"""
    V = Wd @ r_; c = HV * V[HI] * (SCE / E["SIG"])[:, None]; ctot = c.sum(1); CT = {k: c.gather(1, c.abs().topk(min(k, KSTORE), dim=1).indices).sum(1) for k in (16, 64)}; return ctot, CT
def err_on(Pm, r_): pr = Pm @ (Pm.T @ r_); return (ERRF @ pr) / E["SIG"]
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
BASES = ["abs16"] + [f"f{k}" for k in KS] + [f"s{k}" for k in KS] + [f"r{k}" for k in KS] + ["led16", "led64", "sae_full", "sae_full_f", "sae_full_s", "sae_full_r", "all"]; SPLIT = ["abs16", f"f{KB}", f"s{KB}", "sae_full", "sae_full_f"]
CONDS = BASES + [f"{b_}_on5" for b_ in SPLIT] + [f"{b_}_off5" for b_ in SPLIT]
rows = {}
for k in NEU:
    a = ab[:, k]; strong = a.topk(NSTRONG).indices; mask = torch.zeros(N, dtype=torch.bool, device=DEV); mask[a.topk(max(1, int(N * CUT))).indices] = True
    mean_act = torch.full((N,), float(a.mean()), device=DEV); kl_abl = kl_with(k, mean_act, strong); PRE = {}; BRK = {}
    for key in Rd:
        c0, tot, I16, TS16 = ledger(k, key); r_ = RT[key][k]; full = tot + c0; e_full = (ERRF @ r_) / E["SIG"]; ctot, CT = sae_contrib(r_)
        fs = (BOTH.T @ r_).abs() * BF; fo = fs.argsort(descending=True); ss = (Wd @ r_).abs() * AH; so = ss.argsort(descending=True)
        P = {"abs16": TS16 + c0, "led16": full - e_full - ctot + CT[16], "led64": full - e_full - ctot + CT[64], "sae_full": full - e_full, "all": full}
        for kk in KS:
            P[f"f{kk}"] = TS16 + c0 + bus_summary(I16, r_, BOTH[:, fo[:kk]]); P[f"r{kk}"] = TS16 + c0 + bus_summary(I16, r_, RANDB[:, :kk]); P[f"s{kk}"] = TS16 + c0 + bus_summary(I16, r_, torch.linalg.qr(Wd[so[:kk]].T)[0])
        P["sae_full_f"] = full - e_full + err_on(BOTH, r_); P["sae_full_r"] = full - e_full + err_on(RANDB, r_); P["sae_full_s"] = full - e_full + err_on(torch.linalg.qr(Wd[so[:KB]].T)[0], r_)
        PRE[key] = P
        if key == "h":   # brake fidelity: what each input adds beyond the wires against the crowd's true net sum, off and on the class
            crowd = tot - TS16; off = ~mask
            for nm in BASES:
                imp = P[nm] - TS16 - c0; v = {}
                for lab_, m_ in (("off", off), ("on", mask)):
                    ct, ci = crowd[m_], imp[m_]; v[f"share_{lab_}"] = float(ci.sum() / ct.sum()) if float(ct.sum().abs()) > 1e-9 else None; v[f"corr_{lab_}"] = float(torch.corrcoef(torch.stack([ct, ci]))[0, 1]) if ci.std() > 0 else None
                BRK[nm] = v
    sims = {}
    for cnd in CONDS:
        base = cnd.split("_on5")[0].split("_off5")[0]; act = act_of({key: PRE[key][base] for key in Rd})
        if cnd.endswith("_on5"): act = torch.where(mask, act, a)
        if cnd.endswith("_off5"): act = torch.where(mask, a, act)
        v = kl_with(k, act, strong); sims[cnd] = dict(strong=v[0], all=v[1], ratio_strong=v[0] / max(kl_abl[0], 1e-12), ratio_all=v[1] / max(kl_abl[1], 1e-12))
    rows[k] = dict(kl_ablate_strong=kl_abl[0], kl_ablate_all=kl_abl[1], sim=sims, brake=BRK, off_share={b_: sims[f"{b_}_off5"]["all"] / max(sims[b_]["all"], 1e-12) for b_ in SPLIT})
    if len(rows) % 5 == 0 or SMOKE: L_(f"{len(rows)} neurons; last ratios at the firings / all: " + ", ".join(f"{c} {sims[c]['ratio_strong']:.2f}/{sims[c]['ratio_all']:.2f}" for c in BASES) + "; brake share off: " + ", ".join(f"{c} {BRK[c]['share_off']:.2f}" for c in ("f16", "s16", "r16", f"f{KB}", f"s{KB}", "led16", "sae_full") if BRK[c]["share_off"] is not None))
R = list(rows.values()); mr = lambda c, key: med([r["sim"][c][key] for r in R]); mb = lambda c, key: med([r["brake"][c][key] for r in R])
S_ = dict(n=len(R), sae=SAE, hook=hp, sae_latents=NL, sae_active_latents=med(NACT), sae_error_energy_share=ESH, field_dims=KB, budgets=KS, kl_ablate_strong=med([r["kl_ablate_strong"] for r in R]), kl_ablate_all=med([r["kl_ablate_all"] for r in R]),
          ratio_strong={c: mr(c, "ratio_strong") for c in CONDS}, ratio_all={c: mr(c, "ratio_all") for c in CONDS}, off_share={b_: med([r["off_share"][b_] for r in R]) for b_ in SPLIT}, brake={c: {key: mb(c, key) for key in ("share_off", "corr_off", "share_on", "corr_on")} for c in BASES})
rs, ra, osh, bk = S_["ratio_strong"], S_["ratio_all"], S_["off_share"], S_["brake"]
V = dict(M0_all_inputs_reproduce=bool(ra["all"] < 0.05), M1_f16_beats_s16_at_firings=bool(rs["f16"] < rs["s16"]), M2_sKB_within_005_of_fKB_at_firings=bool(abs(rs[f"s{KB}"] - rs[f"f{KB}"]) < 0.05), M3_saefull_fields_gain_005=bool(rs["sae_full"] - rs["sae_full_f"] >= 0.05),
         M4_saefull_random_gain_under_002=bool(rs["sae_full"] - rs["sae_full_r"] < 0.02), M5_f16_brake_corr_beats_s16=bool((bk["f16"]["corr_off"] or 0) > (bk["s16"]["corr_off"] or 0)), M6_s16_beats_led16_at_firings=bool(rs["s16"] < rs["led16"]), M7_rKB_worse_than_fKB_by_01=bool(rs[f"r{KB}"] > rs[f"f{KB}"] + 0.1), M8_f64_within_005_of_fKB=bool(abs(rs["f64"] - rs[f"f{KB}"]) < 0.05))
res = dict(model=M["hf"], block=B, sae=SAE, hook=hp, n_blocks=NB, positions=N, heldout_positions=NH, upstream_units=NU, conditions=CONDS, summary=S_, verdicts=V, neurons={str(k): v for k, v in rows.items()})
summ = (f"[{name} block {B}, {SAE}] {len(R)} neurons, {KB} field directions, budgets {KS}: KL ratio at the firings / over all positions: " + ", ".join(f"{c} {rs[c]:.2f}/{ra[c]:.2f}" for c in BASES) + "; off-class share: " + ", ".join(f"{b_} {osh[b_]:.2f}" for b_ in SPLIT) + "; brake fidelity off the class (share / correlation): " + ", ".join(f"{c} {bk[c]['share_off']:.2f}/{bk[c]['corr_off']:.2f}" for c in BASES if bk[c]["share_off"] is not None and bk[c]["corr_off"] is not None) + f"; verdicts {V}")
save(TAG, res, summ)
