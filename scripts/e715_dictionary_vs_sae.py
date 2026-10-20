"""e715 (session 162): the dictionary against the SAE. The review of the note asked the decisive question: do collective fields
improve the account of a position's state over native neuron atoms alone, SAE features alone, and combinations? At positions of
GPT-2 small the state after block B is reconstructed with k atoms by seven rules and the reconstruction replaces the state,
scored by the downstream KL relative to mean ablation: orthogonal matching pursuit over the native rows (rows), over rows plus
the weak and strong field directions of the upstream blocks (both), over a public SAE's decoder directions (saeomp), over rows
plus those directions (rowssae), the SAE encoder's own top-k latents (saetop), all of the SAE's active latents (saefull), and
controls (k random atoms, the stream's top k principal directions). Beside the KL: the Euclidean residual, the read-energy
residual under the sampled Fisher of e707, and an intervention fidelity, the rank agreement between the effect of eight fixed
perturbations on the true state and on the reconstruction. Arguments: --block B [--smoke]. Pre-registered in e715_e716_prereg.json."""
import argparse
from c11 import *
ap = argparse.ArgumentParser(); ap.add_argument("--block", type=int, required=True); ap.add_argument("--smoke", action="store_true"); A_ = ap.parse_args(); SMOKE = A_.smoke
name = "gpt2"; B = A_.block; TAG = f"e715_{name}_b{B}" + ("_smoke" if SMOKE else ""); M = load(name); NB, D, DFF = M["NB"], M["D"], M["DFF"]; T = 256; L_ = lambda s: log(TAG, s)
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
# ---------------- the SAE (jbloom's resid_pre SAE of block B + 1, on the stream as TransformerLens centres it)
from huggingface_hub import snapshot_download; from safetensors.torch import load_file
hp = f"blocks.{B + 1}.hook_resid_pre"; pth = snapshot_download("jbloom/GPT2-Small-SAEs-Reformatted", allow_patterns=[f"{hp}/*"]); Wt = load_file(f"{pth}/{hp}/sae_weights.safetensors"); cfg_ = json.load(open(f"{pth}/{hp}/cfg.json"))
We, be, Wd, bd = Wt["W_enc"].to(DEV).float(), Wt["b_enc"].to(DEV).float(), Wt["W_dec"].to(DEV).float(), Wt["b_dec"].to(DEV).float(); ADB = bool(cfg_.get("apply_b_dec_to_input", True)); NL = Wd.shape[0]
def sae_latents(x):
    xt = x - x.mean(-1, keepdim=True); return torch.relu(((xt - bd) if ADB else xt) @ We + be), xt
SD = unitr(Wd); DICT = {"rows": RD, "both": torch.cat([RD, FA]), "saeomp": SD, "rowssae": torch.cat([RD, SD])}; NROWS = RD.shape[0]
L_(f"dictionaries: {NROWS} rows, {FA.shape[0]} field atoms, {NL} SAE directions ({hp})")
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
xs = XC[SW, ST]; x0 = X[SW, ST]; H, XT = sae_latents(x0); NACT = (H > 0).sum(-1).float(); MU = x0.mean(-1, keepdim=True)
REC = {}; CH = {}
for dk in ("rows", "both", "saeomp", "rowssae"):
    for k in KS: sel, cof = omp(xs, DICT[dk], k=k); REC[f"{dk}{k}"] = torch.einsum("nk,nkd->nd", cof, DICT[dk][sel]); CH[f"{dk}{k}"] = sel
full = H @ Wd + bd + MU; REC["saefull"] = full - XBAR
contrib = H * Wd.norm(dim=1)[None]
for k in KS:
    top = contrib.topk(k, dim=1).indices; Hk = torch.zeros_like(H).scatter_(1, top, H.gather(1, top)); REC[f"saetop{k}"] = Hk @ Wd + bd + MU - XBAR
genr = torch.Generator(device=DEV).manual_seed(7150 + B)
for k in KS:
    R_ = torch.zeros(NPOS, D, device=DEV); A = DICT["both"]
    for p in range(NPOS): ri = torch.randperm(A.shape[0], generator=genr, device=DEV)[:k]; Ar = A[ri]; R_[p] = torch.linalg.lstsq(Ar.T, xs[p][:, None]).solution[:, 0] @ Ar
    REC[f"random{k}"] = R_; Pk = PX[:, :k]; REC[f"pcs{k}"] = xs @ Pk @ Pk.T
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
DELTA = torch.randn(NPERT, D, device=DEV, generator=gen); DELTA = DELTA / DELTA.norm(dim=1, keepdim=True); FID_METH = [f"rows16", f"both16", f"saetop16", f"saeomp16", f"rowssae16", "saefull", "pcs16"]; EFF = {nm: [] for nm in ["true"] + FID_METH}
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
res = dict(model=M["hf"], block=B, sae=hp, summary=S_, verdicts=V, kl_lists=KLC, effects=EFF)
summ = (f"[{name} block {B}] recovered of the mean-ablation KL at {int(ksel.numel())} positions: " + ", ".join(f"{nm} {rec[nm]:.2f}" for nm in REC) + f"; the SAE has {med(NACT):.0f} active latents per position; Euclidean residual at 16: rows {RES_E['rows16']:.2f}, both {RES_E['both16']:.2f}, SAE top {RES_E['saetop16']:.2f}, SAE OMP {RES_E['saeomp16']:.2f}; Fisher residual at 16: rows {RES_F['rows16']:.2f}, both {RES_F['both16']:.2f}, SAE top {RES_F['saetop16']:.2f}; intervention fidelity: " + ", ".join(f"{nm} {FID[nm]:.2f}" for nm in FID_METH) + f"; verdicts {V}")
save(TAG, res, summ)
