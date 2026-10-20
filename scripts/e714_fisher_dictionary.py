"""e714 (session 161): the Fisher-graded dictionary. e707 found that the output grades a state by a quadratic form, the Fisher
information of its own next tokens, and not by Euclidean energy. Here that grade becomes the dictionary's objective: at a
position, k atoms are chosen from the native rows of blocks 0 to B, with or without the weak and strong field directions of
the upstream blocks, by orthogonal matching pursuit in a blended metric, the sampled Fisher normalised to unit mean eigenvalue plus lambda times
the identity (lambda 1 primary, 0.3 secondary; the pure Fisher metric kept as a diagnostic that the smokes showed to
blow up in the directions the sampled Fisher cannot see), against the same pursuit in the Euclidean metric; each reconstruction replaces the position's state
and is scored by the downstream KL, against mean ablation, k random atoms and the stream's top k principal directions.
Arguments: --model M --block B [--smoke]. Pre-registered in e713_e714_prereg.json."""
import argparse
from c11 import *
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--block", type=int, required=True); ap.add_argument("--smoke", action="store_true"); A_ = ap.parse_args(); SMOKE = A_.smoke
name = A_.model; B = A_.block; TAG = f"e714_{name}_b{B}" + ("_smoke" if SMOKE else ""); M = load(name); NB, D, DFF = M["NB"], M["D"], M["DFF"]; T = 256; L_ = lambda s: log(TAG, s)
NWIN = 4 if SMOKE else 32; NS = int(os.environ.get("E714_NS", 128 if SMOKE else 2048)); NPOS = 32 if SMOKE else 256; HOR = 64; NKL = 8 if SMOKE else 128; Q = 0.25; KS = (16, 32)
ids = windows_from(wikitext_ids(M, "test"), NWIN, T).to(DEV)
# ---------------- baseline capture: the state after block B and every block's activations up to B
X, AB = [], {b: [] for b in range(B + 1)}
with Capture(M, list(range(B + 1)), ["x", "act"]) as C:
    for s0 in range(0, NWIN, 8):
        run_base(M, ids[s0:s0 + 8]); X.append(C.cap[("x", B)].float())
        for b in range(B + 1): AB[b].append(C.cap[("act", b)].reshape(-1, DFF).half())
X = torch.cat(X); XBAR = X.reshape(-1, D).mean(0); XC = X - XBAR
# the field atoms: sixteen weak-write and sixteen strong-write directions per upstream block
FW, FS = [], []
for b in range(B):
    a = torch.cat(AB[b]); W = write_w(M, b); tau = torch.cat([a[:, c0:c0 + 512].float().abs().quantile(1 - Q, dim=0) for c0 in range(0, DFF, 512)]); cw = torch.zeros(D, D, device=DEV); cs = torch.zeros(D, D, device=DEV)
    for s_ in range(0, a.shape[0], 2048):
        af = a[s_:s_ + 2048].float(); wk = af.abs() < tau[None]; dw = (af * wk) @ W.T; ds = (af * ~wk) @ W.T; cw += dw.T @ dw; cs += ds.T @ ds
    FW.append(torch.linalg.eigh(cw)[1][:, -16:].T); FS.append(torch.linalg.eigh(cs)[1][:, -16:].T)
del AB
RD, _, _, _ = rows_dictionary(M, B); FA = torch.cat(FW + FS) if B else torch.zeros(0, D, device=DEV); DICT = {"rows": RD, "both": torch.cat([RD, FA])}; NROWS = RD.shape[0]; PX = torch.linalg.eigh(XC.reshape(-1, D).T @ XC.reshape(-1, D))[1].flip(-1)
L_(f"dictionary: {NROWS} native rows of blocks 0..{B}, {FA.shape[0]} field atoms")
# ---------------- the Fisher at the selected positions (e707's estimator)
rng = np.random.default_rng(714 + B); per = NPOS // NWIN; SW = torch.tensor(np.repeat(np.arange(NWIN), per), device=DEV); ST = torch.tensor(np.concatenate([rng.choice(np.arange(8, T - HOR - 1), per, replace=False) for _ in range(NWIN)]), device=DEV); NPOS = SW.numel()
F = torch.zeros(NPOS, D, D, device=DEV); CNT = torch.zeros(NPOS, device=DEV); LEAF = {}; gen = torch.Generator(device=DEV).manual_seed(714 + B)
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
hd.remove(); F /= CNT.clamp_min(1)[:, None, None]; FN = F * (D / F.diagonal(dim1=1, dim2=2).sum(-1).clamp_min(1e-30))[:, None, None]   # the Fisher normalised to unit mean eigenvalue
EV, U = torch.linalg.eigh(FN); EV = EV.clamp_min(0); LSQ = {lam: torch.einsum("pk,pdk->pkd", (EV + lam).sqrt(), U) for lam in (0.0, 0.3, 1.0)}   # L = (Lambda + lambda)^1/2 U^T: v^T (FN + lambda I) v = |L v|^2
# ---------------- the pursuits
xs = XC[SW, ST]
def pursuit(p, A, k, metric):
    """k atoms of A for position p: OMP in the Euclidean metric or in the Fisher metric (whitened by L); returns the reconstruction in the original space and the chosen indices"""
    x = xs[p:p + 1]
    if metric == "E": sel, cof = omp(x, A, k=k); return torch.einsum("nk,nkd->nd", cof, A[sel])[0], sel[0]
    L = LSQ[metric][p]; Aw = A @ L.T; nrm = Aw.norm(dim=1).clamp_min(1e-12); Aw = Aw / nrm[:, None]; sel, cof = omp(x @ L.T, Aw, k=k); return torch.einsum("nk,nkd->nd", cof, (A / nrm[:, None])[sel])[0], sel[0]
METH = [("ompE_rows16", "rows", 16, "E"), ("ompE_both16", "both", 16, "E"), ("ompE_both32", "both", 32, "E"), ("ompF_both16", "both", 16, 0.0), ("ompM1_rows16", "rows", 16, 1.0), ("ompM1_both16", "both", 16, 1.0), ("ompM1_both32", "both", 32, 1.0), ("ompM03_both16", "both", 16, 0.3)]
REC = {m_[0]: torch.zeros(NPOS, D, device=DEV) for m_ in METH}; FIELD_SHARE = {m_[0]: torch.zeros(NPOS, device=DEV) for m_ in METH}
for p in range(NPOS):
    for nm, dk, k, met in METH: r, sel = pursuit(p, DICT[dk], k, met); REC[nm][p] = r; FIELD_SHARE[nm][p] = float((sel >= NROWS).float().mean())
    if (p + 1) % 64 == 0: L_(f"pursuits at {p + 1} of {NPOS} positions")
genr = torch.Generator(device=DEV).manual_seed(7140 + B)
for p in range(NPOS):   # controls: sixteen random atoms of the two-field dictionary, least squares; the stream's top sixteen principal directions
    A = DICT["both"]; ri = torch.randperm(A.shape[0], generator=genr, device=DEV)[:16]; Ar = A[ri]; c = torch.linalg.lstsq(Ar.T, xs[p][:, None]).solution[:, 0]; REC.setdefault("random16", torch.zeros(NPOS, D, device=DEV))[p] = c @ Ar
    Pk = PX[:, :16]; REC.setdefault("pcs16", torch.zeros(NPOS, D, device=DEV))[p] = Pk @ (Pk.T @ xs[p])
fe = lambda nm: ((xs - REC[nm]).pow(2).sum(-1) / xs.pow(2).sum(-1).clamp_min(1e-12)); ff = lambda nm: (torch.einsum("pd,pde,pe->p", xs - REC[nm], F, xs - REC[nm]) / torch.einsum("pd,pde,pe->p", xs, F, xs).clamp_min(1e-30))
RES_E = {nm: fe(nm) for nm in REC}; RES_F = {nm: ff(nm) for nm in REC}
# ---------------- the finite score: the position's state replaced by the mean plus the reconstruction, downstream KL
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
KLC = {nm: [] for nm in list(REC) + ["mean"]}; ksel = torch.tensor(rng.choice(NPOS, min(NKL, NPOS), replace=False), device=DEV)
for p in ksel.tolist():
    w, t = int(SW[p]), int(ST[p]); lp0 = lp_window(w)
    for nm in KLC: v = XBAR if nm == "mean" else XBAR + REC[nm][p]; lp1 = lp_window(w, replace_hook(t, v)); KLC[nm].append(float(kl(lp0[t:], lp1[t:]).mean()))
KLm = {nm: med(v) for nm, v in KLC.items()}; rec = {nm: med([1 - a / max(b, 1e-12) for a, b in zip(KLC[nm], KLC["mean"])]) for nm in REC}
L_("recovered of the mean-ablation KL: " + ", ".join(f"{nm} {rec[nm]:.2f}" for nm in REC))
S_ = dict(n_positions=NPOS, samples_per_position=med(CNT), n_rows=NROWS, n_field_atoms=int(FA.shape[0]), recovered=rec, kl=KLm, euclid_residual={nm: med(v) for nm, v in RES_E.items()}, fisher_residual={nm: med(v) for nm, v in RES_F.items()}, field_share={nm: med(v) for nm, v in FIELD_SHARE.items()})
V = dict(F1_blended_beats_euclid_by_01_at_16=bool(rec["ompM1_both16"] >= rec["ompE_both16"] + 0.1), F2_blended_fisher_residual_lower_by_02=bool(S_["fisher_residual"]["ompM1_both16"] <= S_["fisher_residual"]["ompE_both16"] - 0.2), F3_blended_euclid_residual_higher=bool(S_["euclid_residual"]["ompM1_both16"] > S_["euclid_residual"]["ompE_both16"]),
         F4_blended_prefers_fields_by_01=bool(S_["field_share"]["ompM1_both16"] >= S_["field_share"]["ompE_both16"] + 0.1), F5_fields_help_blended_by_005=bool(rec["ompM1_both16"] >= rec["ompM1_rows16"] + 0.05), F6_blended16_beats_pcs16=bool(rec["ompM1_both16"] > rec["pcs16"]), F7_blended16_matches_euclid32=bool(rec["ompM1_both16"] >= rec["ompE_both32"]), F8_pure_fisher_fails_finite=bool(rec["ompF_both16"] < rec["ompE_both16"]), F9_lambda03_beats_lambda1=bool(rec["ompM03_both16"] > rec["ompM1_both16"]))
res = dict(model=M["hf"], block=B, n_blocks=NB, D=D, horizon=HOR, summary=S_, verdicts=V, kl_lists=KLC)
summ = (f"[{name} block {B}] k atoms by pursuit, recovered of the mean-ablation KL (median over {len(ksel)} positions): " + ", ".join(f"{nm} {rec[nm]:.2f}" for nm in REC) + f"; Euclidean residual at 16 both: E {S_['euclid_residual']['ompE_both16']:.2f} blended {S_['euclid_residual']['ompM1_both16']:.2f} pure Fisher {S_['euclid_residual']['ompF_both16']:.2f}; Fisher residual: E {S_['fisher_residual']['ompE_both16']:.2f} blended {S_['fisher_residual']['ompM1_both16']:.2f}; field share of the chosen atoms: E {S_['field_share']['ompE_both16']:.2f} blended {S_['field_share']['ompM1_both16']:.2f}; verdicts {V}")
save(TAG, res, summ)
