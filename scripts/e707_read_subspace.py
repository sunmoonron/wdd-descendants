"""e707 (session 160): the read subspace. Both camps score a decomposition by how much of the state it reconstructs; the
model scores a state by what its readers do with it. At a position the Fisher information of the model's own output
distribution over the next sixty-four tokens with respect to the state after block B, estimated from sampled tokens and
their log-probability gradients, gives every vector a read energy (its Fisher quadratic form) beside its Euclidean energy.
Measured per position for the parts of the state (the content, the block's strong and weak writes, the attention output,
the OMP rows part and its residual, and on GPT-2 small a public residual SAE's reconstruction and error) and for the weak
field, the strong field, the stream's principal directions and random directions: read energy per unit energy against
chance, each part's share of the content's read energy against its share of the Euclidean energy, the dimension carrying
nine tenths of the content's read energy, and the finite check: the downstream KL of removing each part from the state
against the Fisher prediction and against removing a random vector of the same norm. Arguments: --model M --block B
[--smoke]. Pre-registered in e707_e712_prereg.json (revised after the smokes, see the file)."""
import argparse
from c11 import *
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--block", type=int, required=True); ap.add_argument("--smoke", action="store_true"); A_ = ap.parse_args(); SMOKE = A_.smoke
name = A_.model; B = A_.block; TAG = f"e707_{name}_b{B}" + ("_smoke" if SMOKE else ""); M = load(name); NB, D, DFF = M["NB"], M["D"], M["DFF"]; T = 256; L_ = lambda s: log(TAG, s)
NWIN = 4 if SMOKE else 32; NS = int(os.environ.get("E707_NS", 128 if SMOKE else 4096)); NPOS = 32 if SMOKE else 512; HOR = 64; NKL = 8 if SMOKE else 128; Q = 0.25; MASS = 0.9
ids = windows_from(wikitext_ids(M, "test"), NWIN, T).to(DEV); W = write_w(M, B)
# ---------------- baseline capture: the state after block B, the block's activations and attention output
X, AB, AT = [], [], []
with Capture(M, [B], ["x", "act", "att"]) as C:
    for s0 in range(0, NWIN, 8):
        run_base(M, ids[s0:s0 + 8]); X.append(C.cap[("x", B)].float()); AB.append(C.cap[("act", B)].float()); AT.append(C.cap[("att", B)].float())
X = torch.cat(X); AB = torch.cat(AB); AT = torch.cat(AT); XBAR = X.reshape(-1, D).mean(0)
TAU = torch.cat([AB.reshape(-1, DFF)[:, c0:c0 + 512].abs().quantile(1 - Q, dim=0) for c0 in range(0, DFF, 512)]); WK = AB.abs() < TAU
WEAK = (AB * WK) @ W.T; STRONG = (AB * ~WK) @ W.T; XC = X - XBAR
def pcs(Z, k=16): Zf = Z.reshape(-1, D); return torch.linalg.eigh(Zf.T @ Zf)[1][:, -k:].contiguous()
FW, FS, PX = pcs(WEAK), pcs(STRONG), pcs(XC); PXK = pcs(XC, k=D); gen = torch.Generator(device=DEV).manual_seed(707 + B); RR = torch.linalg.qr(torch.randn(D, 16, device=DEV, generator=gen))[0]
# the rows part of the centred state: OMP with sixteen native rows of blocks 0..B
RD, _, _, _ = rows_dictionary(M, B)
# the selected positions
rng = np.random.default_rng(707 + B); per = NPOS // NWIN; SW = torch.tensor(np.repeat(np.arange(NWIN), per), device=DEV); ST = torch.tensor(np.concatenate([rng.choice(np.arange(8, T - HOR - 1), per, replace=False) for _ in range(NWIN)]), device=DEV); NPOS = SW.numel()
# ---------------- the Fisher information of the output over the next HOR tokens with respect to the state at the selected positions
F = torch.zeros(NPOS, D, D, device=DEV); CNT = torch.zeros(NPOS, device=DEV); LEAF = {}
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
    del o, lp, P, Xl; LEAF.clear(); torch.cuda.empty_cache(); L_(f"Fisher: windows {s0 + nb} of {NWIN}")
hd.remove(); F /= CNT.clamp_min(1)[:, None, None]
EV, U = torch.linalg.eigh(F); EV = EV.flip(-1).clamp_min(0); U = U.flip(-1); TR = EV.sum(-1).clamp_min(1e-30); cum = EV.cumsum(-1) / TR[:, None]; K90T = (cum < MASS).sum(-1) + 1
xs = XC[SW, ST]; MKL = EV * torch.einsum("pdk,pd->pk", U, xs).pow(2); cumk = MKL.sort(-1, descending=True).values.cumsum(-1) / MKL.sum(-1, keepdim=True).clamp_min(1e-30); K90 = (cumk < MASS).sum(-1) + 1; K50 = (cumk < 0.5).sum(-1) + 1
L_(f"the content's read energy: k50 {med(K50):.0f}, k90 {med(K90):.0f} directions of {D} (by Fisher trace k90 {med(K90T):.0f}); samples per position {med(CNT):.0f}")
# ---------------- read energy of the parts of the state and of the subspaces
sel, cof = omp(xs, RD, k=16); XROWS = torch.einsum("nk,nkd->nd", cof, RD[sel]); XRES = xs - XROWS; wn = WEAK[SW, ST].norm(dim=-1, keepdim=True); RV = torch.randn(NPOS, D, device=DEV, generator=gen); RV = RV / RV.norm(dim=-1, keepdim=True) * wn
VEC = {"state": xs, "strong": STRONG[SW, ST], "weak": WEAK[SW, ST], "att": AT[SW, ST], "rows": XROWS, "rows_residual": XRES, "random_weaknorm": RV}
SAE = None
if name == "gpt2":
    try:
        from huggingface_hub import snapshot_download; from safetensors.torch import load_file
        hp = f"blocks.{B + 1}.hook_resid_pre"; pth = snapshot_download("jbloom/GPT2-Small-SAEs-Reformatted", allow_patterns=[f"{hp}/*"]); Wt = load_file(f"{pth}/{hp}/sae_weights.safetensors"); cfg_ = json.load(open(f"{pth}/{hp}/cfg.json"))
        We, be, Wd, bd = Wt["W_enc"].to(DEV).float(), Wt["b_enc"].to(DEV).float(), Wt["W_dec"].to(DEV).float(), Wt["b_dec"].to(DEV).float(); adb = bool(cfg_.get("apply_b_dec_to_input", True))
        def SAE(x):
            xt = x - x.mean(-1, keepdim=True); h = torch.relu(((xt - bd) if adb else xt) @ We + be); return h @ Wd + bd, xt   # the stream as TransformerLens centres it
        xh, xt = SAE(X[SW, ST]); XTBAR = (X.reshape(-1, D) - X.reshape(-1, D).mean(-1, keepdim=True)).mean(0); VEC["sae_content"] = xt - XTBAR; VEC["sae_recon"] = xh - XTBAR; VEC["sae_error"] = xt - xh
        en = VEC["sae_error"].norm(dim=-1, keepdim=True); RV2 = torch.randn(NPOS, D, device=DEV, generator=gen); VEC["random_errnorm"] = RV2 / RV2.norm(dim=-1, keepdim=True) * en
        L_(f"SAE {hp}: {We.shape[1]} latents, error energy {float((xt - xh).pow(2).sum(-1).mean() / (xt - XTBAR).pow(2).sum(-1).mean()):.3f} of the centred state's")
    except Exception as ex: L_(f"SAE unavailable: {ex}"); SAE = None
rq = lambda V: torch.einsum("pd,pde,pe->p", V, F, V); RE = {k: rq(v) for k, v in VEC.items()}; EE = {k: v.pow(2).sum(-1) for k, v in VEC.items()}; RHO_REF = TR / D
RHO = {k: RE[k] / EE[k].clamp_min(1e-12) / RHO_REF for k in VEC}   # read energy per unit energy, against a random direction's
SHR = {k: RE[k] / RE["state"].clamp_min(1e-30) for k in VEC}; SHE = {k: EE[k] / EE["state"].clamp_min(1e-12) for k in VEC}
if SAE is not None: SHR["sae_error"] = RE["sae_error"] / RE["sae_content"].clamp_min(1e-30); SHE["sae_error"] = EE["sae_error"] / EE["sae_content"].clamp_min(1e-12); SHR["sae_recon"] = RE["sae_recon"] / RE["sae_content"].clamp_min(1e-30); SHE["sae_recon"] = EE["sae_recon"] / EE["sae_content"].clamp_min(1e-12)
SUBS = {"weak_field": FW, "strong_field": FS, "stream_pcs": PX, "random16": RR}; RHOS = {k: torch.einsum("dk,pde,ek->p", Q, F, Q) / 16 / RHO_REF for k, Q in SUBS.items()}
# ---------------- the finite check: remove each part from the state, downstream KL, against the Fisher prediction
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
PARTS = ["state", "strong", "weak", "att", "rows", "rows_residual", "random_weaknorm"] + (["sae_error", "random_errnorm"] if SAE is not None else []); KLC = {k: [] for k in PARTS}; PRED = {k: [] for k in PARTS}; ksel = torch.tensor(rng.choice(NPOS, min(NKL, NPOS), replace=False), device=DEV)
for p in ksel.tolist():
    w, t = int(SW[p]), int(ST[p]); lp0 = lp_window(w); x0 = X[w, t]
    for k in PARTS: lp1 = lp_window(w, replace_hook(t, x0 - VEC[k][p])); KLC[k].append(float(kl(lp0[t:], lp1[t:]).mean())); PRED[k].append(float(0.5 * RE[k][p]))
KLm = {k: med(v) for k, v in KLC.items()}; L_("KL of removing each part (mean per downstream position): " + ", ".join(f"{k} {v:.2e}" for k, v in KLm.items()))
def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); rx = x.argsort().argsort(); ry = y.argsort().argsort(); return float(np.corrcoef(rx, ry)[0, 1]) if len(x) > 2 else None
allk = sum([KLC[k] for k in PARTS], []); allp = sum([PRED[k] for k in PARTS], []); SP_ALL = spearman(allp, allk); SP_PART = spearman([KLm[k] for k in PARTS], [med(PRED[k]) for k in PARTS])
# ---------------- summary and verdicts
S_ = dict(n_positions=NPOS, samples_per_position=med(CNT), k50=med(K50), k90=med(K90), k90_trace=med(K90T), k90_over_D=med(K90.float() / D), rho={k: med(v) for k, v in RHO.items()}, share_read={k: med(v) for k, v in SHR.items()}, share_energy={k: med(v) for k, v in SHE.items()}, rho_subspace={k: med(v) for k, v in RHOS.items()},
          kl=KLm, kl_predicted={k: med(PRED[k]) for k in PARTS}, spearman_all=SP_ALL, spearman_parts=SP_PART, sae=bool(SAE is not None))
rho, shr, she, rs = S_["rho"], S_["share_read"], S_["share_energy"], S_["rho_subspace"]
V = dict(R1_read_dim_under_tenth=bool(S_["k90"] < 0.1 * D), R2_fisher_predicts_finite_kl=bool(SP_ALL is not None and SP_ALL >= 0.8), R3_weak_less_read_than_strong=bool(rho["weak"] < rho["strong"]), R3b_weak_costs_more_than_random=bool(KLm["weak"] > KLm["random_weaknorm"]),
         R4_weak_field_read_3x_chance=bool(rs["weak_field"] > 3), R4b_strong_field_read_3x_chance=bool(rs["strong_field"] > 3), R6_rows_residual_less_read=bool(rho["rows_residual"] < rho["rows"]))
if SAE is not None: V["R5_sae_error_read_share_under_half_energy_share"] = bool(shr["sae_error"] < 0.5 * she["sae_error"]); V["R5b_sae_error_costs_more_than_random"] = bool(KLm["sae_error"] > KLm["random_errnorm"])
res = dict(model=M["hf"], block=B, n_blocks=NB, D=D, horizon=HOR, mass=MASS, summary=S_, verdicts=V, per_position=dict(k90=K90.tolist(), rho={k: v.tolist() for k, v in RHO.items()}, rho_subspace={k: v.tolist() for k, v in RHOS.items()}), kl_lists=KLC, kl_predicted_lists=PRED)
summ = (f"[{name} block {B}] at {NPOS} positions the content's read energy sits in {S_['k90']:.0f} directions of {D} (k50 {S_['k50']:.0f}; by Fisher trace {S_['k90_trace']:.0f}); read energy per unit energy against a random direction: content {rho['state']:.2f}, strong write {rho['strong']:.2f}, weak write {rho['weak']:.2f}, attention {rho['att']:.2f}, rows {rho['rows']:.2f}, rows residual {rho['rows_residual']:.2f}"
        + (f", SAE recon {rho['sae_recon']:.2f}, SAE error {rho['sae_error']:.2f} (the error's share of the content's read energy {shr['sae_error']:.2f} against {she['sae_error']:.2f} of its energy)" if SAE is not None else "") + f"; subspaces: weak field {rs['weak_field']:.2f}, strong field {rs['strong_field']:.2f}, stream PCs {rs['stream_pcs']:.2f}, random {rs['random16']:.2f}; "
        f"KL of removing: " + ", ".join(f"{k} {KLm[k]:.2e}" for k in PARTS) + f"; Spearman of the Fisher prediction with the finite KL {S_['spearman_all']:.2f} over positions and parts; verdicts {V}")
save(TAG, res, summ)
