"""e686 (session 155): is a neuron's negative pole its nearest neighbour? The essay reads neurons as gates: the negative
pole is a closely related distinction (a V against an O, a taxi against an ambulance), not an unrelated second job. Per
neuron at three depths: the positive-pole class (the top 0.1% of positions by pre-activation, by the SwiGLU product in
gated models) and the negative-pole class (the bottom 0.1%). Their mean normalised input states, with the neuron's own
read axis (or axes, gate and up) projected out, since the two classes are opposite along it by construction; the cosine
between them, against a null of the same statistic with another neuron's negative class (100 partners). Also the
overlap of the two classes' current tokens and next tokens against the same null. Near = above the null's 95th
percentile. Arguments: --model M [--smoke]. Pre-registered in e686_prereg.json."""
import argparse
from c11 import *
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--smoke", action="store_true"); A_ = ap.parse_args(); SMOKE = A_.smoke
name = A_.model; TAG = f"e686_{name}" + ("_smoke" if SMOKE else ""); M = load(name); NB, D, DFF = M["NB"], M["D"], M["DFF"]; T = 512
NWIN = 16 if SMOKE else 256; BLOCKS = sorted(set([max(1, NB // 4), NB // 2, (3 * NB) // 4])); NPART = 100; FRAC = 0.001; NNEU = 200 if SMOKE else 1500; L_ = lambda s: log(TAG, s)
ids = windows_from(pile_ids(M), NWIN, T, seed=686).to(DEV); N = NWIN * (T - 2)
PRE = {b: [] for b in BLOCKS}; XN = {b: [] for b in BLOCKS}
with Capture(M, BLOCKS, ["pre", "xin"], stop_after=max(BLOCKS)) as C:
    for s0 in range(0, NWIN, 8):
        run_base(M, ids[s0:s0 + 8])
        for b in BLOCKS:
            xin = C.cap[("xin", b)][:, 1:T - 1].float().reshape(-1, D); xn, _ = ln_apply(M, b, xin); XN[b].append(xn.half())
            if M["fam"] == "gated": g, u = C.cap[("pre_g", b)][:, 1:T - 1].float(), C.cap[("pre_u", b)][:, 1:T - 1].float(); PRE[b].append((torch.nn.functional.silu(g) * u).reshape(-1, DFF).half())
            else: PRE[b].append(C.cap[("pre_h", b)][:, 1:T - 1].float().reshape(-1, DFF).half())
PRE = {b: torch.cat(v) for b, v in PRE.items()}; XN = {b: torch.cat(v) for b, v in XN.items()}; CUR = ids[:, 1:T - 1].reshape(-1); NXT = ids[:, 2:T].reshape(-1); K = max(10, int(N * FRAC))
L_(f"{N} positions, blocks {BLOCKS} of {NB}, pole class size {K}")
def jacc(a, b): a, b = set(a.tolist()), set(b.tolist()); return len(a & b) / max(len(a | b), 1)
def pole_stats(Xb, mu, proj_Q, top, bot):
    """for one unit: the cosine of its two pole-class mean states with the given axes projected out, and the two classes' token overlaps"""
    mp = Xb[top].float().mean(0) - mu; mn = Xb[bot].float().mean(0) - mu; mp = mp - (mp @ proj_Q) @ proj_Q.T; mn = mn - (mn @ proj_Q) @ proj_Q.T
    return float(torch.nn.functional.cosine_similarity(mp, mn, dim=0)), jacc(CUR[top], CUR[bot]), jacc(NXT[top], NXT[bot])
res = dict(model=M["hf"], n_blocks=NB, positions=N, class_size=K, blocks={}); rng = np.random.default_rng(686); NNULL = 300
for b in BLOCKS:
    P = PRE[b].float(); Xb = XN[b]; Xf = Xb.float(); mu = Xf.mean(0); Rd = read_w(M, b); alive = torch.nonzero(P.abs().max(0).values > 1e-4)[:, 0].tolist(); neu = sorted(rng.choice(alive, min(NNEU, len(alive)), replace=False).tolist())
    # the null: the two tails of a direction that is not a neuron's read: rotated read rows (the same norm spectrum) and isotropic directions; the direction projected out as the neuron's read axes are
    keyr = "h" if "h" in Rd else "g"; Rrot = unitr(rotate(Rd[keyr], seed=686 + b)[torch.randperm(DFF, generator=torch.Generator().manual_seed(b))[:NNULL].to(DEV)]); Riso = unitr(torch.randn(NNULL, D, device=DEV, generator=torch.Generator(device=DEV).manual_seed(686 + b)))
    NUL = {}
    for nm, Rn in (("rotated", Rrot), ("isotropic", Riso)):
        pr = Xf @ Rn.T; tops = pr.topk(K, dim=0).indices; bots = (-pr).topk(K, dim=0).indices; vals = [pole_stats(Xb, mu, Rn[i][:, None], tops[:, i], bots[:, i]) for i in range(NNULL)]
        NUL[nm] = dict(cos=np.array([v[0] for v in vals]), jc=np.array([v[1] for v in vals]), jn=np.array([v[2] for v in vals])); del pr
    top = P[:, neu].topk(K, dim=0).indices; bot = (-P[:, neu]).topk(K, dim=0).indices; rows = []
    for i, j in enumerate(neu):
        Q = torch.linalg.qr(torch.stack([Rd[k][j] for k in Rd]).T).Q; own, jc, jn = pole_stats(Xb, mu, Q, top[:, i], bot[:, i]); nr = NUL["rotated"]; ni = NUL["isotropic"]
        rows.append(dict(neuron=j, cos=own, z_rot=float((own - nr["cos"].mean()) / max(nr["cos"].std(), 1e-9)), z_iso=float((own - ni["cos"].mean()) / max(ni["cos"].std(), 1e-9)), near=bool(own > np.quantile(nr["cos"], 0.95)), near_iso=bool(own > np.quantile(ni["cos"], 0.95)),
                         jac_cur=jc, jac_next=jn, cur_above_null=bool(jc > np.quantile(nr["jc"], 0.95)), next_above_null=bool(jn > np.quantile(nr["jn"], 0.95)), pos_token=M["tok"].decode([int(torch.mode(CUR[top[:, i]].cpu()).values)]), neg_token=M["tok"].decode([int(torch.mode(CUR[bot[:, i]].cpu()).values)])))
    sh = lambda key: mean([r[key] for r in rows]); S = dict(n=len(rows), near_share=sh("near"), near_share_iso=sh("near_iso"), median_cos=med([r["cos"] for r in rows]), null_rot_mean=float(NUL["rotated"]["cos"].mean()), null_rot_p95=float(np.quantile(NUL["rotated"]["cos"], 0.95)), null_iso_mean=float(NUL["isotropic"]["cos"].mean()),
                                                             median_z_rot=med([r["z_rot"] for r in rows]), median_z_iso=med([r["z_iso"] for r in rows]), cur_overlap_above_null=sh("cur_above_null"), next_overlap_above_null=sh("next_above_null"), median_jac_cur=med([r["jac_cur"] for r in rows]), median_jac_next=med([r["jac_next"] for r in rows]), null_jac_next_mean=float(NUL["rotated"]["jn"].mean()))
    res["blocks"][str(b)] = dict(summary=S, neurons=rows[:200]); L_(f"block {b}: pole-state cosine median {S['median_cos']:+.3f} vs the rotated-read tails' {S['null_rot_mean']:+.3f} (95th {S['null_rot_p95']:+.3f}; isotropic {S['null_iso_mean']:+.3f}); near {S['near_share']:.2f} of {S['n']} neurons (isotropic null {S['near_share_iso']:.2f}; z {S['median_z_rot']:.2f}/{S['median_z_iso']:.2f}); next-token overlap {S['median_jac_next']:.3f} vs null {S['null_jac_next_mean']:.3f}, above null {S['next_overlap_above_null']:.2f}")
    del P, Xb, Xf; torch.cuda.empty_cache()
sh = [res["blocks"][str(b)]["summary"]["near_share"] for b in BLOCKS]
V = dict(N1_near_above_half_at_middle=bool(res["blocks"][str(NB // 2)]["summary"]["near_share"] > 0.5), N2_rises_with_depth=bool(sh[-1] > sh[0]), N3_next_token_overlap_above_03=bool(res["blocks"][str(NB // 2)]["summary"]["next_overlap_above_null"] > 0.3))
res["verdicts"] = V
summ = f"[{name}] negative pole as nearest neighbour (null: the two tails of a rotated read row), blocks {BLOCKS}: near share " + ", ".join(f"{b}: {res['blocks'][str(b)]['summary']['near_share']:.2f} (median cos {res['blocks'][str(b)]['summary']['median_cos']:+.2f} vs null {res['blocks'][str(b)]['summary']['null_rot_mean']:+.2f}, z {res['blocks'][str(b)]['summary']['median_z_rot']:.2f}; next-token overlap above null {res['blocks'][str(b)]['summary']['next_overlap_above_null']:.2f})" for b in BLOCKS) + f"; verdicts {V}"
save(TAG, res, summ)
