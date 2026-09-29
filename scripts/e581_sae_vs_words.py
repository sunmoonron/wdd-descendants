"""e581 (session 105): can the strongest existing method recover the same decomposition? EleutherAI's TopK sparse
autoencoder for Pythia-160m (32,768 latents, 32 active) at block 6, against the native words on the same states (block
6, the MLP rows of blocks 0-6). For each unit (a word with its row as direction and its eight largest projections as
class; a feature with its decoder row as direction and its eight largest activations as class; random directions with
their own extremes as the null): the producer coalition (the contribution of every MLP neuron of blocks 0-6, activation
times the projection of its row on the direction, at the class's positions), the within-unit coherence of the
contribution vectors, the effective number of producers, and, for 64 units each, necessity: the top-128 producers'
write columns zeroed and the unit's signal at its class re-measured (the projection for a word, the activation for a
feature) against 128 block-matched random neurons. Also the producer overlap between a feature and the word nearest
to it by decoder-row cosine. Features are taken with class sizes in the words' range (10-60 active positions), the 256
with the largest total activation. Pre-registered (probabilities are honest guesses):
 X1 (0.5) the features have private coalitions too (coherence above 0.3, random directions about 0.1): the coalition
    is a property of the cloud's spoken directions, not of native rows;
 X2 (0.6) the words' coherence exceeds the features' (native rows are the producer-aligned directions);
 X3 (0.7) the top-128 producers are necessary for both (relative drop over 0.3, random under 0.05);
 X4 (0.5) a feature and its nearest word share their producers (Jaccard of the top-128 sets over 0.3)."""
from s101_common import *
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import json as _json
t0 = time.time(); L = 6; KX = 8; NU = 256; NN = 64; KP = 128
model, tok, fam = load_model("pythia160"); arch = Arch(model, fam); D = arch.D; DFF = arch.DFF; ids = pile_ids("pythia160")
REPO = "EleutherAI/sae-pythia-160m-32k"; HP = f"layers.{L}"; cfg = _json.load(open(hf_hub_download(REPO, f"{HP}/cfg.json"))); sd = load_file(hf_hub_download(REPO, f"{HP}/sae.safetensors")); TOPK = int(cfg.get("k", 32))
def pick(sub, nd, size=None):
    for k, v in sd.items():
        if sub in k and v.dim() == nd and (size is None or size in v.shape): return v.float().to(DEV)
    raise KeyError(sub)
enc = pick("enc", 2); enc = enc if enc.shape[1] == D else enc.T; dec = pick("dec", 2); dec = dec if dec.shape[1] == D else dec.T; NL = enc.shape[0]; benc = pick("enc", 1, NL); bdec = pick("dec", 1, D)
def encode(x):
    z = (x - bdec) @ enc.T + benc; top = z.topk(TOPK, dim=1); out = torch.zeros_like(z); out.scatter_(1, top.indices, torch.relu(top.values)); return out
dimc = lambda M: M - M.mean(-1, keepdim=True)
def capture():
    """block-L states and the MLP activations of blocks 0..L at positions 1:, kept positions"""
    ACT = {}; hs = [arch.mlp_lin(b).register_forward_pre_hook((lambda b_: lambda m, a: ACT.__setitem__(b_, a[0].detach().float()[:, 1:].reshape(-1, a[0].shape[-1])))(b)) for b in range(L + 1)]
    try: X = block_states(model, arch, ids, [L])[L].reshape(-1, D)
    finally: [h.remove() for h in hs]
    return X, torch.cat([ACT[b] for b in range(L + 1)], 1)
X, act = capture(); keep = ~sinkmask(X); Xk = X[keep]; act = act[keep]; N = Xk.shape[0]
rms = dimc(Xk).pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6); cands = {"raw": Xk, "dimcentred": dimc(Xk), "layernorm": dimc(Xk) / rms}; fv = {cn: float((xin - (encode(xin) @ dec + bdec)).pow(2).sum(1).mean() / (xin - xin.mean(0)).pow(2).sum(1).mean()) for cn, xin in cands.items()}
coord = min(fv, key=fv.get); xin = cands[coord]; fa = encode(xin); fcount = (fa > 0).sum(0); log(f"SAE on {coord} (fvu {fv[coord]:.2f}); {int((fcount >= KX).sum())} features active at {KX}+ positions")
Rows = torch.cat([arch.wdir(b) for b in range(L + 1)]); norms = Rows.norm(dim=1); A = Rows / norms[:, None]; m = A.shape[0]; U = unitr(Xk - Xk.mean(0)); st = stats(U, A, K); words = torch.nonzero(wordset(st["usage"]))[:, 0]; ratio = st["ratio"].float()
okf = torch.nonzero((fcount >= 10) & (fcount <= 60))[:, 0]; feats = okf[fa[:, okf].sum(0).topk(min(NU, okf.numel())).indices]; decu = unitr(dec)
g = torch.Generator(device=DEV).manual_seed(5); Wr = unitr(torch.randn(NU, D, device=DEV, generator=g))
def units(kind):
    if kind == "words": return [(int(r), A[r], ratio[:, r].topk(KX).indices.to(DEV)) for r in words.tolist()]
    if kind == "features": return [(int(f), decu[f], fa[:, f].topk(KX).indices) for f in feats.tolist()]
    return [(q, Wr[q], (U @ Wr[q]).abs().topk(KX).indices) for q in range(NU)]
def coalition_stats(us, tag):
    cos_in, cos_x, eff, own, last, tops = [], [], [], [], None, {}
    for uid, dvec, pos in us:
        C = act[pos] * ((A @ dvec) * norms)[None]; Cu = unitr(C); cs = Cu @ Cu.T; cos_in.append(float((cs.sum() - KX) / (KX * (KX - 1))))
        if last is not None: cos_x.append(float((Cu @ last.T).mean()))
        last = Cu; a = C.abs(); eff.append(float(((a.sum(1) ** 2) / (a ** 2).sum(1).clamp_min(1e-12)).mean())); tops[uid] = C.clamp_min(0).sum(0).topk(KP).indices
        if tag == "words": own.append(float((a > a[:, uid][:, None]).sum(1).float().mean()))
    r = dict(n=len(us), within=med(cos_in), across=med(cos_x), effective=med(eff), own_rank=med(own) if own else None); log(f"{tag} ({len(us)}): within-unit coherence {r['within']:.3f}, across units {r['across']:.3f}, effective producers {r['effective']:.0f}" + (f", own row's rank {r['own_rank']:.0f}" if own else "")); return r, tops
res = dict(coord=coord, n_positions=N, class_size_words=float((ratio[:, words] > 1).sum(0).float().median()), class_size_features=float(fcount[feats].float().median()), groups={}); TOPS = {}
for kind in ("words", "features", "random_directions"): res["groups"][kind], TOPS[kind] = coalition_stats(units(kind), kind)
# producer overlap between a feature and its nearest word
cosfw = decu[feats] @ A[words].T; near = cosfw.argmax(1); jac = []
for q, f in enumerate(feats.tolist()):
    a_, b_ = set(TOPS["features"][f].tolist()), set(TOPS["words"][int(words[near[q]])].tolist()); jac.append(len(a_ & b_) / len(a_ | b_))
res["feature_word_overlap"] = dict(median_nearest_cos=float(cosfw.max(1).values.median()), producer_jaccard=med(jac), producer_jaccard_cos_ge_03=med([j for j, c in zip(jac, cosfw.max(1).values.tolist()) if c >= 0.3]) or None, n_cos_ge_03=int((cosfw.max(1).values >= 0.3).sum()))
log(f"feature -> nearest word: cosine {res['feature_word_overlap']['median_nearest_cos']:.2f}, producer Jaccard {res['feature_word_overlap']['producer_jaccard']:.2f} (pairs at cosine >= 0.3: {res['feature_word_overlap']['n_cos_ge_03']}, Jaccard {res['feature_word_overlap']['producer_jaccard_cos_ge_03']})")
# necessity: zero the top producers' write columns and re-measure the unit's signal at its class
blk = torch.arange(m, device=DEV) // DFF
def ablate(rows):
    saved = []
    with torch.no_grad():
        for r in rows.tolist():
            b, j = r // DFF, r % DFF; w = arch.mlp_lin(b).weight; saved.append((w, j, w[:, j].clone())); w[:, j] = 0
    return lambda: [w.__setitem__((slice(None), j), v) for w, j, v in saved]
def signal(kind, uid, dvec, pos, Xn):
    if kind == "features":
        xn = {"raw": Xn, "dimcentred": dimc(Xn), "layernorm": dimc(Xn) / dimc(Xn).pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6)}[coord]; return float(encode(xn[pos])[:, uid].mean())
    return float((unitr(Xn - Xn.mean(0))[pos] @ dvec).mean())
gg = torch.Generator().manual_seed(1); res["necessity"] = {}
for kind in ("words", "features"):
    us = units(kind); sel = [us[i] for i in torch.randperm(len(us), generator=gg)[:NN].tolist()]; drops, rdrops = [], []
    for uid, dvec, pos in sel:
        base = signal(kind, uid, dvec, pos, Xk); top = TOPS[kind][uid]; hist = torch.bincount(blk[top], minlength=L + 1)
        rnd = torch.cat([torch.nonzero(blk == b)[:, 0][torch.randperm(DFF, generator=gg)[:int(hist[b])].to(DEV)] for b in range(L + 1) if hist[b] > 0])
        for rows, store in ((top, drops), (rnd, rdrops)):
            restore = ablate(rows); Xn = block_states(model, arch, ids, [L])[L].reshape(-1, D)[keep]; restore(); store.append((base - signal(kind, uid, dvec, pos, Xn)) / max(abs(base), 1e-6))
    res["necessity"][kind] = dict(n=len(sel), drop_top=med(drops), drop_random=med(rdrops), share_drop_over_03=mean([float(x > 0.3) for x in drops])); log(f"necessity {kind}: relative drop of the class signal with the top-{KP} producers zeroed {med(drops):.3f} (over 0.3 in {mean([float(x > 0.3) for x in drops]):.2f}), random neurons {med(rdrops):.3f}")
G, Nn, Fw = res["groups"], res["necessity"], res["feature_word_overlap"]
summ = (f"SAE features vs words on Pythia-160m block 6 ({N} positions; features of class size 10-60, words' median class {res['class_size_words']:.0f}): within-unit coherence words {G['words']['within']:.3f}, features {G['features']['within']:.3f}, random directions {G['random_directions']['within']:.3f} (across units {G['words']['across']:.3f} / {G['features']['across']:.3f}); effective producers {G['words']['effective']:.0f} / {G['features']['effective']:.0f} / {G['random_directions']['effective']:.0f}; "
        f"necessity: zeroing the top-{KP} producers drops the class signal by {Nn['words']['drop_top']:.2f} for words and {Nn['features']['drop_top']:.2f} for features (random neurons {Nn['words']['drop_random']:.2f} / {Nn['features']['drop_random']:.2f}); a feature and its nearest word (cosine {Fw['median_nearest_cos']:.2f}) share producers at Jaccard {Fw['producer_jaccard']:.2f} | {time.time() - t0:.0f}s")
log(summ); record("e581_sae_vs_words", res, summ)
