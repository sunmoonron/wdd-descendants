"""e724 (session 164): do the fields add anything conditional on an SAE reconstruction of a state? e715 and e718 scored fields
beside rows and SAE latents at a fixed budget; here the SAE's full reconstruction is taken as given and sixteen more atoms are
chosen by pursuit on what it leaves, from the field directions, from the native rows, from the SAE's own directions and from
random directions, scored as in e718 by the recovered mean-ablation KL and the eight-perturbation intervention fidelity, with
jbloom's L1 and OpenAI's TopK dictionaries on GPT-2 small. Arguments: --model gpt2 --block B --sae {jbloom,oai32k,oai128k}
[--smoke]. Pre-registered in e723_e724_prereg.json."""
import urllib.request
import argparse
from c11 import *
ap = argparse.ArgumentParser(); ap.add_argument("--model", default="gpt2"); ap.add_argument("--sae", default="oai32k"); ap.add_argument("--block", type=int, required=True); ap.add_argument("--smoke", action="store_true"); A_ = ap.parse_args(); SMOKE = A_.smoke
name = A_.model; B = A_.block; SAE = A_.sae; TAG = f"e724_{name}_b{B}_{SAE}" + ("_smoke" if SMOKE else ""); M = load(name); NB, D, DFF = M["NB"], M["D"], M["DFF"]; T = 256; L_ = lambda s: log(TAG, s)
NWIN = 4 if SMOKE else 32; NS = 128 if SMOKE else 2048; NPOS = 24 if SMOKE else 256; HOR = 64; NKL = 8 if SMOKE else 128; NPERT = 8; Q = 0.25; KS = (16, 32, 64)
ids = windows_from(wikitext_ids(M, "test"), NWIN, T).to(DEV)
# ---------------- the state after block B and the upstream activations
X, AB = [], {b: [] for b in range(B)}
with Capture(M, list(range(B + 1)), ["x", "act"]) as C:
    for s0 in range(0, NWIN, 8):
        run_base(M, ids[s0:s0 + 8]); X.append(C.cap[("x", B)].float())
        for b in range(B): AB[b].append(C.cap[("act", b)].reshape(-1, DFF).half())
X = torch.cat(X); XBAR = X.reshape(-1, D).mean(0); XC = X - XBAR
FW, FS = [], []
for b in range(B):
    a = torch.cat(AB[b]); W = write_w(M, b); tau = torch.cat([a[:, c0:c0 + 512].float().abs().quantile(1 - Q, dim=0) for c0 in range(0, DFF, 512)]); cw = torch.zeros(D, D, device=DEV); cs = torch.zeros(D, D, device=DEV)
    for s_ in range(0, a.shape[0], 2048):
        af = a[s_:s_ + 2048].float(); wk = af.abs() < tau[None]; dw = (af * wk) @ W.T; ds = (af * ~wk) @ W.T; cw += dw.T @ dw; cs += ds.T @ ds
    FW.append(torch.linalg.eigh(cw)[1][:, -16:].T); FS.append(torch.linalg.eigh(cs)[1][:, -16:].T)
del AB
RD, _, _, _ = rows_dictionary(M, B); FA = torch.cat(FW + FS) if B else torch.zeros(0, D, device=DEV); PX = torch.linalg.eigh(XC.reshape(-1, D).T @ XC.reshape(-1, D))[1].flip(-1)
# ---------------- the SAE at the state after block B: jbloom (resid_pre of block B + 1, as TransformerLens centres it), OpenAI TopK (resid_post_mlp of block B, uncentred), EleutherAI TopK (layers.B, uncentred)
from huggingface_hub import snapshot_download; from safetensors.torch import load_file
CENTRE = False; ADB = False; KTOP = None
if SAE == "jbloom":
    hp = f"blocks.{B + 1}.hook_resid_pre"; pth = snapshot_download("jbloom/GPT2-Small-SAEs-Reformatted", allow_patterns=[f"{hp}/*"]); Wt = load_file(f"{pth}/{hp}/sae_weights.safetensors"); cfg_ = json.load(open(f"{pth}/{hp}/cfg.json"))
    We, be, Wd, bd = Wt["W_enc"].to(DEV).float(), Wt["b_enc"].to(DEV).float(), Wt["W_dec"].to(DEV).float(), Wt["b_dec"].to(DEV).float(); ADB = bool(cfg_.get("apply_b_dec_to_input", True)); CENTRE = True
elif SAE in ("oai32k", "oai128k"):
    assert name == "gpt2"; hp = f"resid_post_mlp_v5_{SAE[3:]}/autoencoders/{B}.pt"; os.makedirs(CACHED, exist_ok=True); f_ = f"{CACHED}/oai_gpt2_{SAE[3:]}_{B}.pt"
    if not os.path.exists(f_): urllib.request.urlretrieve("https://openaipublic.blob.core.windows.net/sparse-autoencoder/gpt2-small/" + hp, f_ + ".tmp"); os.replace(f_ + ".tmp", f_)
    sd = torch.load(f_, map_location="cpu", weights_only=False); assert sd["activation"] == "TopK", sd["activation"]
    We, be, Wd, bd = sd["encoder.weight"].T.to(DEV).float(), sd["latent_bias"].to(DEV).float(), sd["decoder.weight"].T.to(DEV).float(), sd["pre_bias"].to(DEV).float(); ADB = True; KTOP = int(sd["activation_state_dict"]["k"])
elif SAE == "el32k":
    hp = f"layers.{B}"; pth = snapshot_download(f"EleutherAI/sae-{M['hf'].split('/')[-1]}-32k", allow_patterns=[f"{hp}/*"]); Wt = load_file(f"{pth}/{hp}/sae.safetensors"); cfg_ = json.load(open(f"{pth}/{hp}/cfg.json"))
    We, be, Wd, bd = Wt["encoder.weight"].T.to(DEV).float(), Wt["encoder.bias"].to(DEV).float(), Wt["W_dec"].to(DEV).float(), Wt["b_dec"].to(DEV).float(); ADB = True; KTOP = int(cfg_["k"])
else: raise SystemExit(f"unknown SAE {SAE}")
NL = Wd.shape[0]
def sae_latents(x):
    """latents, the input as the SAE sees it, and the per-position scale and shift that take the SAE's output back to the stream"""
    if SAE in ("oai32k", "oai128k"):   # OpenAI's TopK dictionaries normalise the input (a LayerNorm without gain, eps 1e-5) and rescale the output by the same std and mean (their from_state_dict sets normalize for TopK)
        mu = x.mean(-1, keepdim=True); xc = x - mu; sd = xc.std(-1, keepdim=True); xt = xc / (sd + 1e-5); SC, SH = sd, mu
    else: xt = x - x.mean(-1, keepdim=True) if CENTRE else x; SC, SH = 1.0, (x.mean(-1, keepdim=True) if CENTRE else 0.0)
    pre = ((xt - bd) if ADB else xt) @ We + be
    if KTOP is None: return torch.relu(pre), xt, SC, SH
    top = pre.topk(KTOP, dim=-1); return torch.zeros_like(pre).scatter_(-1, top.indices, torch.relu(top.values)), xt, SC, SH
SD = unitr(Wd); DICT = {"rows": RD, "both": torch.cat([RD, FA]), "saeomp": SD, "rowssae": torch.cat([RD, SD])}; NROWS = RD.shape[0]
L_(f"dictionaries: {NROWS} rows, {FA.shape[0]} field atoms, {NL} SAE directions ({SAE} {hp}, top-k {KTOP}, centred {CENTRE})")
# ---------------- positions and the sampled Fisher (e707)
rng = np.random.default_rng(715 + B); per = NPOS // NWIN; SW = torch.tensor(np.repeat(np.arange(NWIN), per), device=DEV); ST = torch.tensor(np.concatenate([rng.choice(np.arange(8, T - HOR - 1), per, replace=False) for _ in range(NWIN)]), device=DEV); NPOS = SW.numel()
F = torch.zeros(NPOS, D, D, device=DEV); CNT = torch.zeros(NPOS, device=DEV); LEAF = {}; gen = torch.Generator(device=DEV).manual_seed(715 + B)
def leaf_hook(m, a, o):
    x = o[0] if isinstance(o, tuple) else o; l = x.detach().requires_grad_(True); LEAF["x"] = l
    return ((l,) + tuple(o[1:])) if isinstance(o, tuple) else l
hd = M["layers"][B].register_forward_hook(leaf_hook)
for s0 in range(0, NWIN, 8):
    nb = min(8, NWIN - s0); inb = (SW >= s0) & (SW < s0 + nb); idx = inb.nonzero()[:, 0]; wl = SW[idx] - s0; tl = ST[idx]; bw = torch.arange(nb, device=DEV)
    with torch.enable_grad():
        o = M["base"](input_ids=ids[s0:s0 + nb], use_cache=False); lp = logprobs_at(M, o.last_hidden_state); Xl = LEAF["x"]; P = lp.detach().exp()
        for s in range(NS):
            sw = torch.randint(0, T, (nb,), generator=gen, device=DEV); y = torch.multinomial(P[bw, sw], 1, generator=gen)[:, 0]
            g, = torch.autograd.grad(lp[bw, sw, y].sum(), Xl, retain_graph=True); m = (sw[wl] >= tl) & (sw[wl] < tl + HOR)
            if m.any(): G = g[wl[m], tl[m]]; F[idx[m]] += torch.einsum("pi,pj->pij", G, G); CNT[idx[m]] += 1
    del o, lp, P, Xl; LEAF.clear(); torch.cuda.empty_cache()
hd.remove(); F /= CNT.clamp_min(1)[:, None, None]; L_(f"Fisher sampled at {NPOS} positions ({med(CNT):.0f} samples each)")
# ---------------- the reconstructions (of the centred content; the replacement state is XBAR + recon)
xs = XC[SW, ST]; x0 = X[SW, ST]; H, XT, SC, SH = sae_latents(x0); NACT = (H > 0).sum(-1).float(); ESH = float((XT - (H @ Wd + bd)).pow(2).sum() / XT.pow(2).sum()); L_(f"SAE: {med(NACT):.0f} active latents per position, reconstruction error energy share {ESH:.3f}")
REC = {}; CH = {}
for dk in ("rows", "both", "saeomp", "rowssae"):
    for k in KS: sel, cof = omp(xs, DICT[dk], k=k); REC[f"{dk}{k}"] = torch.einsum("nk,nkd->nd", cof, DICT[dk][sel]); CH[f"{dk}{k}"] = sel
full = (H @ Wd + bd) * SC + SH; REC["saefull"] = full - XBAR; ESHS = float((x0 - full).pow(2).sum() / x0.pow(2).sum()); L_(f"SAE full reconstruction on the stream: error energy share {ESHS:.3f}")
contrib = H * Wd.norm(dim=1)[None]
for k in KS:
    top = contrib.topk(k, dim=1).indices; Hk = torch.zeros_like(H).scatter_(1, top, H.gather(1, top)); REC[f"saetop{k}"] = (Hk @ Wd + bd) * SC + SH - XBAR
genr = torch.Generator(device=DEV).manual_seed(7150 + B)
for k in KS:
    R_ = torch.zeros(NPOS, D, device=DEV); A = DICT["both"]
    for p in range(NPOS): ri = torch.randperm(A.shape[0], generator=genr, device=DEV)[:k]; Ar = A[ri]; R_[p] = torch.linalg.lstsq(Ar.T, xs[p][:, None]).solution[:, 0] @ Ar
    REC[f"random{k}"] = R_; Pk = PX[:, :k]; REC[f"pcs{k}"] = xs @ Pk @ Pk.T
RS_ = xs - REC["saefull"]; genc = torch.Generator(device=DEV).manual_seed(724 + B); RANDA = torch.linalg.qr(torch.randn(D, max(FA.shape[0], 16), device=DEV, generator=genc))[0].T
for dk, A in (("f", FA), ("rows", RD), ("s", SD), ("r", RANDA)):
    sel, cof = omp(RS_, A, k=16); REC[f"saefull_{dk}16"] = REC["saefull"] + torch.einsum("nk,nkd->nd", cof, A[sel])
fe = lambda nm: ((xs - REC[nm]).pow(2).sum(-1) / xs.pow(2).sum(-1).clamp_min(1e-12)); ff = lambda nm: (torch.einsum("pd,pde,pe->p", xs - REC[nm], F, xs - REC[nm]) / torch.einsum("pd,pde,pe->p", xs, F, xs).clamp_min(1e-30))
RES_E = {nm: med(fe(nm)) for nm in REC}; RES_F = {nm: med(ff(nm)) for nm in REC}; FIELD_SHARE = {f"both{k}": float((CH[f"both{k}"] >= NROWS).float().mean()) for k in KS}; SAE_SHARE = {f"rowssae{k}": float((CH[f"rowssae{k}"] >= NROWS).float().mean()) for k in KS}
# ---------------- the finite scores: downstream KL of the replacement, and the intervention fidelity
def lp_window(w, hook=None):
    hd_ = M["layers"][B].register_forward_hook(hook) if hook else None
    try: o = M["base"](input_ids=ids[w:w + 1], use_cache=False)
    finally:
        if hd_: hd_.remove()
    return logprobs_at(M, o.last_hidden_state)[0]
def replace_hook(t, v):
    def h(m, a, o):
        x = o[0] if isinstance(o, tuple) else o; x = x.clone(); x[0, t] = v.to(x.dtype); return ((x,) + tuple(o[1:])) if isinstance(o, tuple) else x
    return h
METH = list(REC) + ["mean"]; KLC = {nm: [] for nm in METH}; ksel = torch.tensor(rng.choice(NPOS, min(NKL, NPOS), replace=False), device=DEV)
DELTA = torch.randn(NPERT, D, device=DEV, generator=gen); DELTA = DELTA / DELTA.norm(dim=1, keepdim=True); FID_METH = [f"rows16", f"both16", f"saetop16", f"saeomp16", f"rowssae16", "saefull", "pcs16", "saefull_f16", "saefull_rows16", "saefull_s16", "saefull_r16"]; EFF = {nm: [] for nm in ["true"] + FID_METH}
for p in ksel.tolist():
    w, t = int(SW[p]), int(ST[p]); lp0 = lp_window(w); sc = 0.5 * float(xs[p].norm())
    for nm in METH: v = XBAR if nm == "mean" else XBAR + REC[nm][p]; lp1 = lp_window(w, replace_hook(t, v)); KLC[nm].append(float(kl(lp0[t:], lp1[t:]).mean()))
    for m in range(NPERT):
        d = sc * DELTA[m]; lpd = lp_window(w, replace_hook(t, x0[p] + d)); EFF["true"].append(float(kl(lp0[t:], lpd[t:]).mean()))
        for nm in FID_METH: v = XBAR + REC[nm][p]; lpa = lp_window(w, replace_hook(t, v)); lpb = lp_window(w, replace_hook(t, v + d)); EFF[nm].append(float(kl(lpa[t:], lpb[t:]).mean()))
KLm = {nm: med(v) for nm, v in KLC.items()}; rec = {nm: med([1 - a / max(b, 1e-12) for a, b in zip(KLC[nm], KLC["mean"])]) for nm in REC}
def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); rx = x.argsort().argsort(); ry = y.argsort().argsort(); return float(np.corrcoef(rx, ry)[0, 1])
FID = {nm: spearman(EFF["true"], EFF[nm]) for nm in FID_METH}
L_("recovered: " + ", ".join(f"{nm} {rec[nm]:.2f}" for nm in REC)); L_("intervention fidelity: " + ", ".join(f"{nm} {FID[nm]:.2f}" for nm in FID_METH))
S_ = dict(n_positions=NPOS, n_scored=int(ksel.numel()), sae_active_latents=med(NACT), n_rows=NROWS, n_field_atoms=int(FA.shape[0]), n_sae_directions=NL, recovered=rec, kl=KLm, euclid_residual=RES_E, fisher_residual=RES_F, field_share=FIELD_SHARE, sae_share_in_rowssae=SAE_SHARE, fidelity=FID)
V = dict(P1_both16_beats_saetop16=bool(rec["both16"] > rec["saetop16"]), P2_both16_beats_rows16_by_005=bool(rec["both16"] >= rec["rows16"] + 0.05), P3_saefull_beats_both64=bool(rec["saefull"] > rec["both64"]), P4_saeomp16_beats_saetop16=bool(rec["saeomp16"] > rec["saetop16"]),
         P5_rowssae16_beats_both_by_003=bool(rec["rowssae16"] >= max(rec["rows16"], rec["saeomp16"]) + 0.03), P6_both16_fidelity_matches_saetop16=bool(FID["both16"] >= FID["saetop16"]), P7_rows16_beats_pcs16=bool(rec["rows16"] > rec["pcs16"]), P8_saetop16_beats_rows16=bool(rec["saetop16"] > rec["rows16"]))
D64 = ("rows64", "both64", "saeomp64", "rowssae64", "saetop64"); V2 = dict(T1_saeomp16_beats_rows16=bool(rec["saeomp16"] > rec["rows16"]), T2_saeomp16_beats_saetop16=bool(rec["saeomp16"] > rec["saetop16"]), T3_fields_add_under_003=bool(rec["both16"] < rec["rows16"] + 0.03), T4_rows_fidelity_beats_saetop16=bool(FID["rows16"] > FID["saetop16"]), T5_all_dictionaries_095_at_64=bool(min(rec[d] for d in D64) >= 0.95), T6_rows16_at_least_085=bool(rec["rows16"] >= 0.85)); V.update(V2); V.update(dict(N1_fields_add_under_002_given_sae=bool(rec["saefull_f16"] - rec["saefull"] < 0.02), N2_rows_add_002_given_sae=bool(rec["saefull_rows16"] - rec["saefull"] >= 0.02), N3_sae_own_atoms_add_at_least_fields=bool(rec["saefull_s16"] - rec["saefull"] >= rec["saefull_f16"] - rec["saefull"]), N4_fidelity_f16_at_least_saefull=bool(FID["saefull_f16"] >= FID["saefull"])))
S_["sae_error_energy_share"] = ESH; S_["sae_error_energy_share_stream"] = ESHS
res = dict(model=M["hf"], block=B, sae=hp, sae_family=SAE, sae_k=KTOP, sae_centred=CENTRE, summary=S_, verdicts=V, kl_lists=KLC, effects=EFF)
summ = (f"[{name} block {B}, {SAE}] recovered of the mean-ablation KL at {int(ksel.numel())} positions: " + ", ".join(f"{nm} {rec[nm]:.2f}" for nm in REC) + f"; the SAE has {med(NACT):.0f} active latents per position; Euclidean residual at 16: rows {RES_E['rows16']:.2f}, both {RES_E['both16']:.2f}, SAE top {RES_E['saetop16']:.2f}, SAE OMP {RES_E['saeomp16']:.2f}; Fisher residual at 16: rows {RES_F['rows16']:.2f}, both {RES_F['both16']:.2f}, SAE top {RES_F['saetop16']:.2f}; intervention fidelity: " + ", ".join(f"{nm} {FID[nm]:.2f}" for nm in FID_METH) + "; conditional on the SAE's full reconstruction, sixteen more atoms of fields / rows / SAE directions / random recover " + " / ".join(f"{rec[m]:.3f}" for m in ("saefull_f16", "saefull_rows16", "saefull_s16", "saefull_r16")) + f" against {rec['saefull']:.3f}, with fidelity " + " / ".join(f"{FID[m]:.2f}" for m in ("saefull_f16", "saefull_rows16", "saefull_s16", "saefull_r16")) + f" against {FID['saefull']:.2f}" + f"; verdicts {V}")
save(TAG, res, summ)
