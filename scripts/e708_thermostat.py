"""e708 (session 160): the thermostat. Both camps treat the crowd as a thing; here it is tested as a loop. Block B's
strong writes (its activations over their per-neuron top-quarter threshold) are scaled by alpha from a half to double,
and the next three blocks' responses are read against the injected vector: how much of it each block's weak writes,
strong writes and attention output undo along its own direction, the total displacement after three blocks relative to
the injection, and the output KL of the whole perturbation against the KL of its direct path (the same vector added to
the final residual with every later block frozen). Arguments: --model M --block B [--smoke]. Pre-registered in
e707_e712_prereg.json."""
import argparse
from c11 import *
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--block", type=int, required=True); ap.add_argument("--smoke", action="store_true"); A_ = ap.parse_args(); SMOKE = A_.smoke
name = A_.model; B = A_.block; TAG = f"e708_{name}_b{B}" + ("_smoke" if SMOKE else ""); M = load(name); NB, D, DFF = M["NB"], M["D"], M["DFF"]; T = 256; L_ = lambda s: log(TAG, s)
NWIN = 4 if SMOKE else 32; Q = 0.25; ALPHAS = [0.5, 0.75, 1.25, 1.5, 2.0]; KD = 3; BL = [b for b in range(B, B + KD + 1) if b < NB]; LAST = BL[-1]
ids = windows_from(wikitext_ids(M, "test"), NWIN, T).to(DEV); N = NWIN * T; WM = {b: write_w(M, b) for b in BL}; TAU = {}; SCALE = {"alpha": 1.0}
def scale_hook(m, a):
    if SCALE["alpha"] == 1.0: return None
    h = a[0]; return (torch.where(h.abs() >= TAU[B].to(h.dtype), h * SCALE["alpha"], h),)
def run():
    out = {("weak", b): [] for b in BL}; out.update({("strong", b): [] for b in BL}); out.update({("att", b): [] for b in BL}); out[("x", LAST)] = []; H, LP = [], []
    with Capture(M, BL, ["act", "att", "x"]) as C:
        for s0 in range(0, NWIN, 8):
            o = M["base"](input_ids=ids[s0:s0 + 8], use_cache=False); H.append(o.last_hidden_state.float().reshape(-1, D)); LP.append(logprobs_at(M, o.last_hidden_state).reshape(-1, M["V"]))
            for b in BL:
                a = C.cap[("act", b)].float().reshape(-1, DFF)
                if b not in TAU: TAU[b] = torch.cat([a[:, c0:c0 + 512].abs().quantile(1 - Q, dim=0) for c0 in range(0, DFF, 512)])   # thresholds from the first baseline batch
                wk = a.abs() < TAU[b]; out[("weak", b)].append(((a * wk) @ WM[b].T).half()); out[("strong", b)].append(((a * ~wk) @ WM[b].T).half()); out[("att", b)].append(C.cap[("att", b)].reshape(-1, D).half())
            out[("x", LAST)].append(C.cap[("x", LAST)].reshape(-1, D).half())
    R = {k: torch.cat(v) for k, v in out.items()}; R["H"] = torch.cat(H); R["LP"] = torch.cat(LP); return R
hd = write_mod(M, B).register_forward_pre_hook(scale_hook); BASE = run(); L_(f"baseline captured at blocks {BL}; {N} positions")
def proj(d, dp, keep): return (-(d.float() * dp).sum(-1) / dp.pow(2).sum(-1).clamp_min(1e-12))[keep]
RES = {}
for alpha in ALPHAS:
    SCALE["alpha"] = alpha; R = run(); dp = (alpha - 1) * BASE[("strong", B)].float(); nrm = dp.norm(dim=-1); keep = nrm > nrm.median(); r = {}
    dsum = {k: torch.zeros(N, D, device=DEV) for k in ("weak", "strong", "att")}
    for b in BL[1:]:
        for k in ("weak", "strong", "att"): d = R[(k, b)].float() - BASE[(k, b)].float(); dsum[k] += d; r[f"c_{k}_b{b}"] = med(proj(d, dp, keep))
    for k in ("weak", "strong", "att"): r[f"c_{k}_cum"] = med(proj(dsum[k], dp, keep))
    dx = R[("x", LAST)].float() - BASE[("x", LAST)].float(); r["c_total"] = med(proj(dx - dp, dp, keep)); r["gain"] = med((dx.norm(dim=-1) / nrm.clamp_min(1e-12))[keep])
    klc = lambda P, Q: torch.cat([kl(P[s_:s_ + 1024], Q[s_:s_ + 1024]) for s_ in range(0, P.shape[0], 1024)]); kl_total = float(klc(BASE["LP"], R["LP"]).mean()); kl_direct = float(torch.cat([kl(BASE["LP"][s_:s_ + 1024], logprobs_at(M, BASE["H"][s_:s_ + 1024] + dp[s_:s_ + 1024])) for s_ in range(0, N, 1024)]).mean()); r.update(kl_total=kl_total, kl_direct=kl_direct, kl_ratio=kl_total / max(kl_direct, 1e-12), inject_norm=float(nrm[keep].median()))
    RES[str(alpha)] = r; L_(f"alpha {alpha}: undone by the next blocks' weak writes {r['c_weak_cum']:+.2f}, strong writes {r['c_strong_cum']:+.2f}, attention {r['c_att_cum']:+.2f}; total undone after {KD} blocks {r['c_total']:+.2f}, gain {r['gain']:.2f}; KL total {kl_total:.4f} / direct {kl_direct:.4f} = {r['kl_ratio']:.2f}"); del R
SCALE["alpha"] = 1.0; hd.remove(); A15, A05 = RES["1.5"], RES["0.5"]
V = dict(T1_weak_undoes_fifth=bool(A15["c_weak_cum"] >= 0.2), T2_weak_over_strong=bool(A15["c_weak_cum"] > A15["c_strong_cum"]), T3_self_repair_under_07=bool(A15["kl_ratio"] < 0.7), T4_asymmetric=bool(A15["c_total"] > A05["c_total"] + 0.1), T5_attention_not_the_repair=bool(A15["c_att_cum"] < A15["c_weak_cum"]))
res = dict(model=M["hf"], block=B, blocks_read=BL, n_blocks=NB, positions=N, alphas=ALPHAS, results=RES, verdicts=V)
summ = (f"[{name} block {B}] block {B}'s strong writes scaled: " + "; ".join(f"alpha {a}: weak {RES[str(a)]['c_weak_cum']:+.2f}, strong {RES[str(a)]['c_strong_cum']:+.2f}, attention {RES[str(a)]['c_att_cum']:+.2f}, total {RES[str(a)]['c_total']:+.2f}, gain {RES[str(a)]['gain']:.2f}, KL ratio {RES[str(a)]['kl_ratio']:.2f}" for a in ALPHAS) + f"; verdicts {V}")
save(TAG, res, summ)
