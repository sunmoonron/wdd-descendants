"""e557 (session 101): is the vocabulary conserved across scale, across a data twin, and to the end of training? A word is
a row plus its context set (the positions where its projection is over the floor). Rows of different models cannot be
compared, context sets on the same text can. Pythia-410m (block 12, rows of blocks 0-12) against Pythia-160m (block 6,
rows 0-6) on the same eight Pile sequences; Pythia-160m against pythia-160m-deduped (same architecture, different data);
Pythia-410m's final words against its words at step 16000 from the session-88 cache (row indices are shared). For every
word of one model the best Jaccard over the other model's words, on over-the-floor sets and on top-8 sets, against two
nulls: random sets of the same sizes, and the 256 most used atoms of the other model's rotated dictionary (generic
directions with their own context sets). Token purity (the share of a set at its modal token) says how much of a twin is
a token identity. Row-wise cosine by index between 160m and its deduped twin says whether they share an initialization.
Age (checkpoints of 1000-16000 at which a 410m row is a word) against having a 160m twin: are the conserved words the
old ones? Pre-registered (probabilities are honest guesses):
 H1 (0.6) over a third of 410m words have a 160m twin at Jaccard >= 0.25 on over-the-floor sets, under a tenth for the
    rotated null;
 H2 (0.7) twins are mostly token-pure words (purity > 0.5 for the majority of twinned words);
 H3 (0.5) 160m and its deduped twin do not share an initialization (median row cosine by index under 0.1);
 H4 (0.6) 410m's final word set overlaps its step-16000 word set at Jaccard >= 0.5;
 H5 (0.55) 410m words with a 160m twin are older (more checkpoints as a word) than words without."""
from s101_common import *
t0 = time.time(); KX = 8; JT = 0.25
def context(S, tag):
    st = stats(S["U"], S["A"], K); w = torch.nonzero(wordset(st["usage"]))[:, 0]; ratio = st["ratio"][:, w].float(); over = ratio > 1
    top = torch.zeros_like(over); top[ratio.topk(KX, dim=0).indices, torch.arange(w.numel())[None].expand(KX, -1)] = True
    tok = S["tok"].cpu(); purity = torch.tensor([float(torch.bincount(tok[over[:, j]]).max() / max(int(over[:, j].sum()), 1)) if over[:, j].any() else 0.0 for j in range(w.numel())])
    Ar = unitr(rotate(S["A"], seed=7)); sr = stats(S["U"], Ar, K); wr = torch.nonzero(wordset(sr["usage"]))[:, 0]; rr = sr["ratio"][:, wr].float(); overr = rr > 1
    topr = torch.zeros_like(overr); topr[rr.topk(KX, dim=0).indices, torch.arange(wr.numel())[None].expand(KX, -1)] = True
    log(f"{tag}: {w.numel()} words, median over-the-floor set size {float(over.sum(0).float().median()):.0f}, median purity {float(purity.median()):.2f}; rotated words' median set size {float(overr.sum(0).float().median()):.0f}")
    return dict(w=w, over=over, top=top, purity=purity, usage=st["usage"][w], S=st["S"][w], over_r=overr, top_r=topr, N=int(over.shape[0]))
def jac(P, Q):
    P = P.float(); Q = Q.float(); inter = P.T @ Q; return inter / (P.sum(0)[:, None] + Q.sum(0)[None] - inter).clamp_min(1)
def size_null(P, Q, reps=5, seed=0):
    g = torch.Generator().manual_seed(seed); N = P.shape[0]; vals = []
    for r in range(reps):
        Qn = torch.zeros_like(Q)
        for j in range(Q.shape[1]):
            k = int(Q[:, j].sum()); Qn[torch.randperm(N, generator=g)[:k], j] = True
        vals.append(jac(P, Qn).max(1).values)
    return torch.stack(vals).mean(0)
def compare(a, b, tag):
    out = {}
    for kind in ("over", "top"):
        P, Q, Qr = a[kind], b[kind], b[kind + "_r"]; best = jac(P, Q).max(1).values; bestr = jac(P, Qr).max(1).values; null = size_null(P, Q)
        tw = best >= JT
        out[kind] = dict(median_best=float(best.median()), share_ge_025=float(tw.float().mean()), share_ge_05=float((best >= 0.5).float().mean()), rotated_median=float(bestr.median()), rotated_share_ge_025=float((bestr >= JT).float().mean()),
                         size_null_median=float(null.median()), size_null_share_ge_025=float((null >= JT).float().mean()), purity_twinned=float(a["purity"][tw].median()) if tw.any() else None, purity_untwinned=float(a["purity"][~tw].median()) if (~tw).any() else None,
                         share_twinned_pure=float((a["purity"][tw] > 0.5).float().mean()) if tw.any() else None, twin_rate_pure=float(tw[a["purity"] > 0.5].float().mean()) if (a["purity"] > 0.5).any() else None, twin_rate_impure=float(tw[a["purity"] <= 0.5].float().mean()) if (a["purity"] <= 0.5).any() else None,
                         usage_twinned=float(a["usage"][tw].median()) if tw.any() else None, usage_untwinned=float(a["usage"][~tw].median()) if (~tw).any() else None, best=best)
        o = out[kind]; log(f"{tag} [{kind}]: best Jaccard median {o['median_best']:.2f}, share >= 0.25 {o['share_ge_025']:.2f} (>= 0.5 {o['share_ge_05']:.2f}); rotated null {o['rotated_median']:.2f}/{o['rotated_share_ge_025']:.2f}; size null {o['size_null_median']:.2f}/{o['size_null_share_ge_025']:.2f}; "
                        f"purity twinned/untwinned {o['purity_twinned']}/{o['purity_untwinned']}; twin rate pure/impure {o['twin_rate_pure']}/{o['twin_rate_impure']}")
    return out
S410 = lm_states("pythia410"); S160 = lm_states("pythia160"); S160d = lm_states("pythia160d")
keepc = S410["keep"] & S160["keep"] & S160d["keep"]   # sinks differ per model: the common kept positions
for S_ in (S410, S160, S160d): S_["U"] = unitr(S_["X"][keepc] - S_["X"][keepc].mean(0)); S_["tok"] = S_["ids"][:, 1:].reshape(-1)[keepc]
log(f"common kept positions {int(keepc.sum())} of {keepc.numel()}")
c410, c160, c160d = context(S410, "410m"), context(S160, "160m"), context(S160d, "160m-deduped")
res = dict(pairs={})
for tag, a, b in (("410m->160m", c410, c160), ("160m->410m", c160, c410), ("160m->160m-deduped", c160, c160d), ("160m-deduped->160m", c160d, c160), ("410m->160m-deduped", c410, c160d)):
    res["pairs"][tag] = {k: {q: v for q, v in d.items() if q != "best"} for k, d in compare(a, b, tag).items()}
# same initialization? rows by index
cos = (S160["A"] * S160d["A"]).sum(1).cpu(); blk = torch.arange(cos.numel()) // S160["arch"].DFF
res["deduped_row_cosine"] = dict(median=float(cos.median()), q90=float(cos.quantile(0.9)), share_over_05=float((cos > 0.5).float().mean()), by_block=[float(cos[blk == b].median()) for b in range(7)])
log(f"160m vs deduped rows by index: median cosine {res['deduped_row_cosine']['median']:.3f}, q90 {res['deduped_row_cosine']['q90']:.3f}, share > 0.5 {res['deduped_row_cosine']['share_over_05']:.3f}")
# words at the end of training vs step 16000 (cache); age of the final words
CD = "/workspace/wdd/cache/e524_pythia410"; steps = list(range(1000, 16001, 1000)); wsets = {}
for n in steps:
    c = torch.load(f"{CD}/step{n}.pt", map_location="cpu"); wsets[n] = wordset(c["blocks"][12]["usage"]); del c
wfin = torch.zeros(S410["A"].shape[0], dtype=torch.bool); wfin[c410["w"]] = True; age = torch.stack([wsets[n] for n in steps]).sum(0)
j16 = float((wfin & wsets[16000]).sum() / (wfin | wsets[16000]).sum())
tw = torch.zeros_like(wfin); best410 = jac(c410["over"], c160["over"]).max(1).values; tw[c410["w"][best410 >= JT]] = True
res["end_of_training"] = dict(jaccard_final_vs_16000=j16, share_final_words_that_were_words_at_16000=float(wsets[16000][wfin].float().mean()), share_final_words_ever_words=float((age[wfin] > 0).float().mean()),
                              age_final_words_mean=float(age[wfin].float().mean()), age_twinned_mean=float(age[wfin & tw].float().mean()), age_untwinned_mean=float(age[wfin & ~tw].float().mean()),
                              age_twinned_median=float(age[wfin & tw].float().median()), age_untwinned_median=float(age[wfin & ~tw].float().median()), share_16000_words_still_words=float(wfin[wsets[16000]].float().mean()),
                              age_hist_final=torch.bincount(age[wfin], minlength=17).tolist(), twin_rate_by_age=[float(tw[wfin & (age == a)].float().mean()) if (wfin & (age == a)).any() else None for a in range(17)])
e = res["end_of_training"]; log(f"final vs step 16000: Jaccard {j16:.2f}; final words that were words at 16000 {e['share_final_words_that_were_words_at_16000']:.2f}, ever words in 1000-16000 {e['share_final_words_ever_words']:.2f}; age (checkpoints as a word) twinned {e['age_twinned_mean']:.1f} vs untwinned {e['age_untwinned_mean']:.1f}")
p = res["pairs"]["410m->160m"]["over"]; q = res["pairs"]["160m->160m-deduped"]["over"]; pt = res["pairs"]["410m->160m"]["top"]
summ = (f"410m->160m twins on over-the-floor sets: median best Jaccard {p['median_best']:.2f}, share >= 0.25 {p['share_ge_025']:.2f} (rotated null {p['rotated_share_ge_025']:.2f}, size null {p['size_null_share_ge_025']:.2f}); top-8 sets {pt['share_ge_025']:.2f} (rotated {pt['rotated_share_ge_025']:.2f}); "
        f"twinned words' purity {p['purity_twinned']:.2f} vs untwinned {p['purity_untwinned']:.2f}, twin rate pure/impure {p['twin_rate_pure']:.2f}/{p['twin_rate_impure']:.2f}; 160m->deduped twins {q['share_ge_025']:.2f} (rotated {q['rotated_share_ge_025']:.2f}), rows by index median cosine {res['deduped_row_cosine']['median']:.3f}; "
        f"410m final vs step 16000 word set Jaccard {j16:.2f} (final words that were words at 16000: {e['share_final_words_that_were_words_at_16000']:.2f}); age of twinned vs untwinned final words {e['age_twinned_mean']:.1f} vs {e['age_untwinned_mean']:.1f} of 16 | {time.time() - t0:.0f}s")
log(summ); record("e557_cross_scale", res, summ)
