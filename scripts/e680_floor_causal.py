"""e680 (session 155): is the extreme-value floor a causal boundary for the model's own readers? The floor (e483, e666)
is the level a write's projection must clear to be identified against the competition of every other atom; the paper
showed a probe still recovers writes below it, so the information survives, but nobody asked whether the downstream
blocks USE sub-floor writes. Cammarata's reading ('the network keeps unrelated noise down by mostly ignoring weak
activations') predicts that a native write injected below the floor costs no more than a random direction of the same
norm, and far more above it. At the middle block, at 1024 positions of wikitext-2 test, a vector of norm alpha times the
position's floor is added to the residual along: own (the position's largest MLP write's direction, its own sign),
foreign (a native row of an earlier block not prominent at the position, random sign), random (a Gaussian direction),
rotated (the foreign row rotated, the paper's control). Measured: the next-token KL at the position and over the next
eight, the next block's MLP activation change, and the next block's directional response along the injected
direction. Arguments: --model M [--smoke]. Pre-registered in e680_prereg.json."""
import argparse
from c11 import *
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--smoke", action="store_true"); A_ = ap.parse_args(); SMOKE = A_.smoke
name = A_.model; TAG = f"e680_{name}" + ("_smoke" if SMOKE else ""); M = load(name); NB, D, DFF = M["NB"], M["D"], M["DFF"]; B = NB // 2; T = 256
NWIN = 8 if SMOKE else 64; NPOS = 64 if SMOKE else 1024; ALPHAS = [0.25, 0.5, 1.0, 2.0, 4.0]; NEXT = 8; BS = 32 if M["V"] > 100000 else 64; CONDS = ["own", "foreign", "random", "rotated"]
L_ = lambda s: log(TAG, s)
ids = windows_from(wikitext_ids(M, "test"), NWIN, T).to(DEV)
# ---------------- clean pass: block-B states, the ledger of blocks <= B, the next block's activations and output, logprobs at every position
X, XN1, AN1, ACT = [], [], [], []; LP = []
with Capture(M, list(range(B + 2)), ["act", "x"]) as C:
    for s0 in range(0, NWIN, 8):
        o = M["base"](input_ids=ids[s0:s0 + 8], use_cache=False); X.append(C.cap[("x", B)].float()); XN1.append(C.cap[("x", B + 1)].float()); AN1.append(C.cap[("act", B + 1)].float())
        ACT.append(torch.cat([C.cap[("act", b)].float() for b in range(B + 1)], -1).half()); LP.append(logprobs_at(M, o.last_hidden_state))   # fp32: a half-precision log-prob puts a 1e-3-nat noise floor under every KL
X, XN1, AN1, ACT, LP = torch.cat(X), torch.cat(XN1), torch.cat(AN1), torch.cat(ACT), torch.cat(LP)   # [NWIN, T, .]
A, blk, neu, nr = rows_dictionary(M, B); m = A.shape[0]
Xf = X.reshape(-1, D); keep = ~sinkmask(Xf); MU = Xf[keep].mean(0); U = unitr(Xf - MU); cal = floor_calibration(U[keep], A); Lf = floor_of(U, cal).reshape(NWIN, T)
L_(f"block {B} of {NB}, {m} rows, floor calibration r_cal {cal['r_cal']:.3f}; floor as a share of the centred norm, median {float((Lf.reshape(-1)[keep]).median()):.3f}")
rng = np.random.default_rng(680); cand = [(w, p) for w in range(NWIN) for p in range(16, T - NEXT - 1) if keep[w * T + p]]; pick = [cand[i] for i in rng.choice(len(cand), NPOS, replace=False)]
R = rotate(A, seed=7); gauss = torch.Generator(device=DEV).manual_seed(680)
ITEMS = []   # (w, p, cond, alpha, direction unit, norm)
for (w, p) in pick:
    x = X[w, p]; u = U[w * T + p]; fl = float(Lf[w, p]) * float((x - MU).norm()); c = ACT[w, p].float() * nr   # ledger coefficients in write units
    j_own = int(c.abs().argmax()); d_own = A[j_own] * torch.sign(c[j_own]); pr = (u @ A.T).abs() / float(Lf[w, p])
    quiet = torch.nonzero((pr < 0.25) & (blk < B))[:, 0]; j_for = int(quiet[int(rng.integers(len(quiet)))]) if len(quiet) else int(rng.integers(m)); s = float(rng.choice([-1.0, 1.0]))
    g = torch.randn(D, device=DEV, generator=gauss); dirs = dict(own=d_own, foreign=s * A[j_for], random=g / g.norm(), rotated=s * unitr(R[j_for][None])[0])
    for cond in CONDS:
        for a in ALPHAS: ITEMS.append((w, p, cond, a, dirs[cond], a * fl))
L_(f"{len(ITEMS)} injections ({NPOS} positions x {len(CONDS)} conditions x {len(ALPHAS)} amplitudes)")
# ---------------- the injections, batched: each batch row is one window copy with one vector added at one position after block B
OUT = collections.defaultdict(list)
for s0 in range(0, len(ITEMS), BS):
    ch = ITEMS[s0:s0 + BS]; n = len(ch); wi = torch.tensor([c[0] for c in ch], device=DEV); pi = torch.tensor([c[1] for c in ch], device=DEV); V = torch.stack([c[4] * c[5] for c in ch]); rows = torch.arange(n, device=DEV)
    def inject(mod, a, o):
        h = (o[0] if isinstance(o, tuple) else o).clone(); h[rows, pi] += V.to(h.dtype); return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
    hd = M["layers"][B].register_forward_hook(inject)
    try:
        with Capture(M, [B + 1], ["act", "x"]) as C: o = M["base"](input_ids=ids[wi], use_cache=False)
    finally: hd.remove()
    span = pi[:, None] + torch.arange(NEXT + 1, device=DEV)[None]; h = o.last_hidden_state[rows[:, None], span]; lp = logprobs_at(M, h)   # [n, NEXT+1, V]
    lp0 = LP[wi[:, None], span].float(); k = kl(lp0, lp); a1 = C.cap[("act", B + 1)][rows, pi].float(); a0 = AN1[wi, pi]; x1 = C.cap[("x", B + 1)][rows, pi].float(); x0 = XN1[wi, pi]
    dn = torch.stack([c[4] for c in ch]); resp = (((x1 - x0) - V) * dn).sum(-1) / V.norm(dim=-1)
    for i, c in enumerate(ch): OUT[(c[2], c[3])].append(dict(kl_p=float(k[i, 0]), kl_next=float(k[i, 1:].mean()), dact=float((a1[i] - a0[i]).norm() / a0[i].norm().clamp_min(1e-9)), resp=float(resp[i]), w=c[0], p=c[1]))
    if (s0 // BS) % 40 == 0: L_(f"batch {s0 // BS + 1} of {(len(ITEMS) + BS - 1) // BS}")
# ---------------- summaries: medians, ratios to random at the same amplitude, paired shares
KEYS = ["kl_p", "kl_next", "dact", "resp"]; S = {}
for cond in CONDS:
    S[cond] = {}
    for a in ALPHAS:
        v = OUT[(cond, a)]; r = OUT[("random", a)]; S[cond][str(a)] = {k: med([d[k] for d in v]) for k in KEYS}
        S[cond][str(a)].update({k + "_mean": mean([d[k] for d in v]) for k in KEYS}); S[cond][str(a)]["ratio_kl_p"] = S[cond][str(a)]["kl_p"] / max(med([d["kl_p"] for d in r]), 1e-12)
        S[cond][str(a)]["share_kl_p_above_random"] = mean([float(x["kl_p"] > y["kl_p"]) for x, y in zip(v, r)]); S[cond][str(a)]["ratio_kl_next"] = S[cond][str(a)]["kl_next"] / max(med([d["kl_next"] for d in r]), 1e-12)
def first_depart(cond, thr=1.5):
    for a in ALPHAS:
        if S[cond][str(a)]["ratio_kl_p"] > thr: return a
    return None
inb = lambda x, lo=0.7, hi=1.5: x is not None and lo <= x <= hi
V = dict(C1_sub_floor_native_is_random=bool(inb(S["own"]["0.5"]["ratio_kl_p"]) and inb(S["foreign"]["0.5"]["ratio_kl_p"])), C2_own_above_floor_exceeds_random=bool(S["own"]["2.0"]["ratio_kl_p"] > 2),
         C3_foreign_above_floor_is_random=bool(inb(S["foreign"]["2.0"]["ratio_kl_p"])), C4_rotated_is_random=bool(all(inb(S["rotated"][str(a)]["ratio_kl_p"]) for a in ALPHAS)), depart=dict(own=first_depart("own"), foreign=first_depart("foreign"), rotated=first_depart("rotated")))
res = dict(model=M["hf"], block=B, n_blocks=NB, rows=m, r_cal=cal["r_cal"], positions=NPOS, alphas=ALPHAS, conditions=CONDS, summary_by_condition=S, verdicts=V, rows_per_item=[dict(cond=c[2], alpha=c[3]) for c in ITEMS[:0]])
summ = (f"[{name} block {B}] KL at the position, median, ratio to a random direction of the same norm (and the share of positions above it), by amplitude in floors {ALPHAS}: " +
        "; ".join(f"{cond} " + " ".join(f"{S[cond][str(a)]['ratio_kl_p']:.2f}({S[cond][str(a)]['share_kl_p_above_random']:.2f})" for a in ALPHAS) for cond in ("own", "foreign", "rotated")) +
        f"; random median KL by amplitude " + " ".join(f"{S['random'][str(a)]['kl_p']:.2e}" for a in ALPHAS) + f"; next-block activation change (own/foreign/random at 1 floor) {S['own']['1.0']['dact']:.3f}/{S['foreign']['1.0']['dact']:.3f}/{S['random']['1.0']['dact']:.3f}; directional response at 1 floor {S['own']['1.0']['resp']:+.3f}/{S['foreign']['1.0']['resp']:+.3f}/{S['random']['1.0']['resp']:+.3f}; verdicts {V}")
save(TAG, res, summ)
