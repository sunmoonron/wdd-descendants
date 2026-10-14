"""e656 (session 143; Gemini 4's RYS ideas 3a and 3b, after Ng's RYS, 2024, and Sun et al.'s Transformer Layers as
Painters, AAAI 2025): the repeat map of three small models and what predicts it. For every block [i, j) of 1, 2, 3, 4,
6 or 8 layers the model runs with that block traversed twice (no weight change), scored by the change in wikitext-2
loss and in accuracy on the probes of rys_common (few-shot single-digit addition, e560's capitals, weekday and month
offsets); the probe gain is judged on held-out items (the best block chosen on one random half of the items, its gain
read on the other half). Single-pass predictors of the map, each a Spearman across blocks with -dloss and with the
probe gain (raw, and averaged within block size): the centred cosine of x_i and x_j and their log norm ratio (plain
baselines), against the Jaccard of their 16-word OMP supports over the rows of blocks 0..j-1 and the reader
compatibility of x_j with block i's readers (the reader Gram of sd_common: x_j's readable energy over x_i's). Junctions
(idea 3a), on the four least and four most damaging blocks of two or more layers: the second traversal fed x_j (plain),
x_j norm-matched to x_i, x_i plus the band increment's 16-word native chord, the same from the rotated dictionary, or
x_j with the band's own first-pass MLP writes removed; the second traversal's increment is added to x_j. Arguments:
model [--smoke]."""
from rys_common import *
from rys_common import _hs, _out
name = sys.argv[1]; SMOKE = "--smoke" in sys.argv; t0 = time.time(); rng = np.random.default_rng(656)
M = load(name); N = M["NB"]; D = M["D"]; A, blk = dictionary(M); DFF = A.shape[0] // N
W = wikitext(M, "test", 6 if SMOKE else 24, 128); P = probes(M); KD = kinds(P); NI = len(P[0]); half = rng.permutation(NI) < NI // 2
base_loss = lm_loss(M, W); STOP = stopset(M, P); base_ok, base_okm = probe_acc_both(M, P, STOP)
print(f"[{name}] {N} blocks; base loss {base_loss:.4f}; {NI} probes, base accuracy {base_ok.mean():.3f} (" + ", ".join(f"{k} {base_ok[KD == k].mean():.2f} of {(KD == k).sum()}" for k in sorted(set(KD))) + f") ; with the function words masked {base_okm.mean():.3f} ({time.time() - t0:.0f}s)", flush=True)
def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float); ok = np.isfinite(a) & np.isfinite(b); a, b = a[ok], b[ok]
    if len(a) < 4 or a.std() == 0 or b.std() == 0: return float("nan")
    return float(np.corrcoef(a.argsort().argsort(), b.argsort().argsort())[0, 1])
SIZES = [1, 2, 4] if SMOKE else [1, 2, 3, 4, 6, 8]
configs = [(i, i + s) for s in SIZES for i in range(0, N - s + 1)]
if SMOKE: configs = configs[::6]
rows = []
for n_, (i, j) in enumerate(configs):
    with Repeat(M, rys_seq(N, i, j)): loss = lm_loss(M, W); ok, okm = probe_acc_both(M, P, STOP)
    rows.append(dict(i=i, j=j, s=j - i, dloss=loss - base_loss, gain=float(ok.mean() - base_ok.mean()), gainA=float(ok[half].mean() - base_ok[half].mean()), gainB=float(ok[~half].mean() - base_ok[~half].mean()),
                     by_kind={k: float(ok[KD == k].mean() - base_ok[KD == k].mean()) for k in sorted(set(KD))}, ok=ok.astype(int).tolist(),
                     gain_m=float(okm.mean() - base_okm.mean()), gainA_m=float(okm[half].mean() - base_okm[half].mean()), gainB_m=float(okm[~half].mean() - base_okm[~half].mean()), by_kind_m={k: float(okm[KD == k].mean() - base_okm[KD == k].mean()) for k in sorted(set(KD))}))
    if n_ % 20 == 0: print(f"[{name}] {n_ + 1}/{len(configs)} blocks ({time.time() - t0:.0f}s)", flush=True)
# held-out probe gain: the block chosen on half A, read on half B; per kind too
bestA = max(rows, key=lambda r: r["gainA"]); okB = np.array(bestA["ok"])[~half]
heldout = dict(block=[bestA["i"], bestA["j"]], gainA=bestA["gainA"], gainB=bestA["gainB"], mean_gainB_all_blocks=float(np.mean([r["gainB"] for r in rows])), se_halfB=float(np.sqrt(base_ok[~half].mean() * (1 - base_ok[~half].mean()) / (~half).sum())), by_kind={})
for k in sorted(set(KD)):
    selA = half & (KD == k); selB = ~half & (KD == k)
    if selA.sum() < 3 or selB.sum() < 3: continue
    gk = lambda r, sel: float(np.array(r["ok"])[sel].mean() - base_ok[sel].mean()); bk = max(rows, key=lambda r: gk(r, selA))
    heldout["by_kind"][k] = dict(block=[bk["i"], bk["j"]], gainA=gk(bk, selA), gainB=gk(bk, selB))
bestAm = max(rows, key=lambda r: r["gainA_m"]); heldout_m = dict(block=[bestAm["i"], bestAm["j"]], gainA=bestAm["gainA_m"], gainB=bestAm["gainB_m"], gainB_raw_same_block=bestAm["gainB"])
best_loss = min(rows, key=lambda r: r["dloss"]); s2 = [r for r in rows if r["s"] >= 2]; least = sorted(s2, key=lambda r: r["dloss"])[:4 if not SMOKE else 1]; most = sorted(s2, key=lambda r: -r["dloss"])[:4 if not SMOKE else 1]
print(f"[{name}] least damaging block {best_loss['i']}-{best_loss['j']} dloss {best_loss['dloss']:+.4f}; held-out probe gain of the block chosen on half A ({bestA['i']}-{bestA['j']}): {bestA['gainB']:+.3f} (half-B s.e. {heldout['se_halfB']:.3f}); with the function words masked, the block chosen on half A ({bestAm['i']}-{bestAm['j']}) {bestAm['gainB_m']:+.3f} ({time.time() - t0:.0f}s)", flush=True)
# single-pass predictors
caps = {}
hooks = [M["layers"][b].register_forward_pre_hook(lambda m, a, kw, b=b: caps.setdefault(b, []).append(_hs(a, kw)[:, 1:].detach().float().reshape(-1, D)), with_kwargs=True) for b in range(N)]
hooks.append(M["layers"][N - 1].register_forward_hook(lambda m, a, kw, o: caps.setdefault(N, []).append(_out(o)[:, 1:].detach().float().reshape(-1, D)), with_kwargs=True))
for s0 in range(0, W.shape[0], 8): M["model"](W[s0:s0 + 8], use_cache=False)
for h in hooks: h.remove()
X = {l: torch.cat(v) for l, v in caps.items()}; keep = torch.ones(X[0].shape[0], dtype=torch.bool, device=DEV)
for l in X: keep &= ~sinkmask(X[l])
idx = torch.nonzero(keep)[:, 0]; idx = idx[torch.randperm(idx.numel(), generator=torch.Generator().manual_seed(656))[:96 if SMOKE else 256].to(DEV)]
MU = {l: X[l][keep].mean(0) for l in X}; U = {l: unitr(X[l][idx] - MU[l]) for l in X}; XN = {l: X[l][idx].norm(dim=-1) for l in X}
def reader_gram(b):
    L = M["layers"][b]
    if M["fam"] == "neox": mods = [(L.attention.query_key_value.weight, L.input_layernorm.weight), (L.mlp.dense_h_to_4h.weight, L.post_attention_layernorm.weight)]; centre = True
    else: mods = [(getattr(L.self_attn, k).weight, L.input_layernorm.weight) for k in ("q_proj", "k_proj", "v_proj")] + [(getattr(L.mlp, k).weight, L.post_attention_layernorm.weight) for k in ("gate_proj", "up_proj")]; centre = False
    G = torch.zeros(D, D, device=DEV)
    for Wm, g in mods:
        Wf = Wm.detach().float() * g.detach().float()[None, :]
        if centre: Wf = Wf - Wf.mean(1, keepdim=True)
        Gm = Wf.T @ Wf; G += Gm / Gm.trace()
    return G
GR = {}; SUP = {}
def support(l, j):
    if (l, j) not in SUP: SUP[(l, j)] = omp(U[l], A[: j * DFF])[0].cpu().tolist()
    return SUP[(l, j)]
for r in rows:
    i, j = r["i"], r["j"]
    if i not in GR: GR[i] = reader_gram(i)
    G = GR[i]; ri = ((U[i] @ G) * U[i]).sum(1); rj = ((U[j] @ G) * U[j]).sum(1)
    si, sj = support(i, j), support(j, j)
    r.update(pred_cos=float((U[i] * U[j]).sum(1).mean()), pred_lognorm=float((XN[j] / XN[i]).log().mean()), pred_wdd_jaccard=float(np.mean([len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(si, sj)])), pred_reader=float((rj / ri.clamp_min(1e-9)).mean()))
PRED = ("pred_cos", "pred_lognorm", "pred_wdd_jaccard", "pred_reader")
def within(key, target):
    vals = [spearman([r[key] for r in rows if r["s"] == s], [target(r) for r in rows if r["s"] == s]) for s in SIZES if sum(r["s"] == s for r in rows) >= 5]
    vals = [v for v in vals if np.isfinite(v)]; return float(np.mean(vals)) if vals else float("nan")
predictors = {k: dict(rho_loss=spearman([r[k] for r in rows], [-r["dloss"] for r in rows]), rho_loss_within_size=within(k, lambda r: -r["dloss"]), rho_gain=spearman([r[k] for r in rows], [r["gain"] for r in rows]), rho_gain_within_size=within(k, lambda r: r["gain"])) for k in PRED}
for k, v in predictors.items(): print(f"[{name}] predictor {k:17s} Spearman with -dloss {v['rho_loss']:+.2f} (within size {v['rho_loss_within_size']:+.2f}), with probe gain {v['rho_gain']:+.2f} (within size {v['rho_gain_within_size']:+.2f})", flush=True)
# junctions on the least and most damaging blocks
def e_norm(xi, xj, mlp1): return xj * (xi.norm(dim=-1, keepdim=True) / xj.norm(dim=-1, keepdim=True).clamp_min(1e-6))
def make_chord(Ab):
    def e(xi, xj, mlp1):
        d = (xj - xi).reshape(-1, D); n = d.norm(dim=-1, keepdim=True); sel, cof = omp(unitr(d), Ab, batch=512); return xi + ((cof[..., None] * Ab[sel]).sum(1) * n).reshape(xj.shape)
    return e
def e_strip(xi, xj, mlp1): return xj - mlp1
junctions = []
for r in least + most:
    i, j = r["i"], r["j"]; Ab = A[i * DFF: j * DFF]; Abr = unitr(rotate(Ab))
    out = dict(i=i, j=j, kind="least" if r in least else "most", plain_dloss=r["dloss"], plain_gain=r["gain"])
    for vn, ent in (("norm_matched", e_norm), ("native_chord", make_chord(Ab)), ("rotated_chord", make_chord(Abr)), ("mlp_stripped", e_strip)):
        with Repeat(M, rys_seq(N, i, j)), Junction(M, i, j, ent) as jn: loss = lm_loss(M, W, per_batch=jn.reset); ok, okm = probe_acc_both(M, P, STOP, per_batch=jn.reset)
        out[vn] = dict(dloss=loss - base_loss, gain=float(ok.mean() - base_ok.mean()), gain_m=float(okm.mean() - base_okm.mean()))
    junctions.append(out); print(f"[{name}] junction {out['kind']} {i}-{j}: plain dloss {r['dloss']:+.4f} | " + " | ".join(f"{vn} {out[vn]['dloss']:+.4f}" for vn in ("norm_matched", "native_chord", "rotated_chord", "mlp_stripped")) + f" ({time.time() - t0:.0f}s)", flush=True)
mid = lambda r: N / 3 <= (r["i"] + r["j"]) / 2 <= 2 * N / 3; bl2 = min(s2, key=lambda r: r["dloss"])
mostJ = [x for x in junctions if x["kind"] == "most"]
res = dict(model=name, N=N, base_loss=base_loss, base_acc=float(base_ok.mean()), base_acc_masked=float(base_okm.mean()), base_by_kind_masked={k: float(base_okm[KD == k].mean()) for k in sorted(set(KD))}, heldout_masked=heldout_m, base_by_kind={k: float(base_ok[KD == k].mean()) for k in sorted(set(KD))}, n_probes=NI, sizes=SIZES, rows=rows, heldout=heldout, predictors=predictors, junctions=junctions,
           least_damaging=[best_loss["i"], best_loss["j"], best_loss["dloss"]], least_damaging_s2=[bl2["i"], bl2["j"], bl2["dloss"]])
res["verdicts"] = dict(R1=best_loss["dloss"] > -0.005 * base_loss, R2=heldout["gainB"] >= 0.02, R2m=heldout_m["gainB"] >= 0.02, R3=mid(bl2),
                       R4=abs(predictors["pred_wdd_jaccard"]["rho_loss_within_size"]) > abs(predictors["pred_cos"]["rho_loss_within_size"]), R5=abs(predictors["pred_reader"]["rho_loss_within_size"]) > abs(predictors["pred_lognorm"]["rho_loss_within_size"]),
                       R6=all(x["norm_matched"]["dloss"] < x["plain_dloss"] for x in mostJ), R7=np.mean([x["native_chord"]["dloss"] for x in mostJ]) < np.mean([x["norm_matched"]["dloss"] for x in mostJ]),
                       R8=np.mean([x["native_chord"]["dloss"] for x in junctions]) < np.mean([x["rotated_chord"]["dloss"] for x in junctions]))
summ = (f"[{name}] repeat map over {len(rows)} blocks: least damaging {best_loss['i']}-{best_loss['j']} (dloss {best_loss['dloss']:+.4f} of base {base_loss:.3f}); held-out probe gain {heldout['gainB']:+.3f} (s.e. {heldout['se_halfB']:.3f}), with the function words masked {heldout_m['gainB']:+.3f}; "
        f"predictors within size (with -dloss): " + ", ".join(f"{k[5:]} {v['rho_loss_within_size']:+.2f}" for k, v in predictors.items()) + f"; verdicts {res['verdicts']} ({time.time() - t0:.0f}s)")
save(f"e656_rys_map_{name}" + ("_smoke" if SMOKE else ""), res, summ); print("done", flush=True)
