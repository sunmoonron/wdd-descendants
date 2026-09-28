"""e549: can an activation-side feature reader recover the row-context associations WDD finds? Sessions 94-97 reduced a
native word to an association between a row and a class of contexts, the row's extreme positions, and e548 found the
neuron's own statistics do not recover it. The fair competitor is a learned feature dictionary of the states. The only
public residual-stream autoencoders in the Pythia line are EleutherAI's TopK autoencoders for Pythia-160m (as in
e504), so this runs on Pythia-160m at the end of training, block 6, the 16 x 512 evaluation tokens: the native words
are the 256 MLP rows of blocks 0-6 most used by OMP over the rows, each word's context set its eight positions of
largest projection over the floor; the autoencoder's 32,768 latents (32 active per position) give each feature an
active-position set. For each word, the feature whose active set best matches the word's context set (Jaccard, ties by
recall; a first run matched by recall and was covered trivially by features active at thousands of positions), with its precision, Jaccard and the cosine of its decoder direction with the row; the same for 256
random non-word rows, 256 random directions and position-shuffled context sets. The reverse: for every feature active
at eight or more positions, the nearest row by decoder cosine, whether it is a word, and how much of that row's context
set the feature covers. The decisive summary: the share of words whose context set a feature recovers (recall 0.5 or
more), and among them the decoder's cosine with the row (does the feature also point to the output direction).
Pre-registered (honest guesses):
- a feature matches the context set of more than half of the words (Jaccard 0.25 or more) against under a fifth of
  random non-word rows (0.5);
- among recovered words the decoder's median cosine with the row is under 0.3 (the feature finds the class, not the
  direction) (0.5); the alternative, 0.5 or more (0.3);
- in reverse, fewer than a fifth of live features have a word as their nearest row (0.6).
Arguments: [block] (default 6)."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ex_common import *
from lr_common import eval_ids
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
try: MODELS["pythia160"] = ("EleutherAI/pythia-160m", "neox")
except NameError:
    import wdd_common as _w; _w.MODELS["pythia160"] = ("EleutherAI/pythia-160m", "neox")
name = "pythia160"; L = int(sys.argv[1]) if len(sys.argv) > 1 else 6; K = 16; KX = 8; NCTRL = 256; model, tok, fam = load_model(name); arch = Arch(model, fam); D = arch.D; DFF = arch.DFF
for p_ in model.parameters(): p_.requires_grad_(False)
REPO = "EleutherAI/sae-pythia-160m-32k"; HP = f"layers.{L}"; cfg = _json.load(open(hf_hub_download(REPO, f"{HP}/cfg.json"))); sd = load_file(hf_hub_download(REPO, f"{HP}/sae.safetensors")); TOPK = int(cfg.get("k", 32))
def pick(sub, nd, size=None):
    for k, v in sd.items():
        if sub in k and v.dim() == nd and (size is None or size in v.shape): return v.float().to(DEV)
    raise KeyError(sub)
enc = pick("enc", 2); enc = enc if enc.shape[1] == D else enc.T; dec = pick("dec", 2); dec = dec if dec.shape[1] == D else dec.T; NL = enc.shape[0]; benc = pick("enc", 1, NL); bdec = pick("dec", 1, D)
def encode(x):
    z = (x - bdec) @ enc.T + benc; top = z.topk(TOPK, dim=1); out = torch.zeros_like(z); out.scatter_(1, top.indices, torch.relu(top.values)); return out
dimc = lambda M: M - M.mean(-1, keepdim=True)
ids = eval_ids("pythia410")[:16, :512].to(DEV); X = block_states(model, arch, ids, [L], chunk=4)[L].reshape(-1, D); keep = ~sinkmask(X); Xk = X[keep]; N = Xk.shape[0]; del model; torch.cuda.empty_cache()
rms = dimc(Xk).pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6); cands = {"raw": (Xk, lambda M: M), "dimcentred": (dimc(Xk), dimc), "layernorm": (dimc(Xk) / rms, dimc)}; fv = {}
for cn, (xin, _) in cands.items():
    z = encode(xin); fv[cn] = float((xin - (z @ dec + bdec)).pow(2).sum(1).mean() / (xin - xin.mean(0)).pow(2).sum(1).mean()); del z
coord = min(fv, key=fv.get); xin, tr = cands[coord]; act = encode(xin); Fm = act > 0; fsize = Fm.sum(0).float(); live = fsize >= KX; log(f"{name} block {L}: {N} positions; reconstruction fvu " + ", ".join(f"{k} {v:.2f}" for k, v in fv.items()) + f"; using {coord}; {int(live.sum())} of {NL} features active at {KX} or more positions")
Wv = torch.cat([arch.wdir(b).float() for b in range(L + 1)]); norms = Wv.norm(dim=1); Ru = unitr(Wv); m = Ru.shape[0]; U = unitr(Xk - Xk.mean(0)); st = stats(U, Ru, K); ratio = st["ratio"].to(DEV).float(); Lf = st["L"].to(DEV); words = wordset(st["usage"]).to(DEV); S = st["S"].to(DEV); del st
wl = torch.nonzero(words)[:, 0]; g = torch.Generator(device=DEV).manual_seed(0); nonw = torch.nonzero(~words)[:, 0]; rl = nonw[torch.randperm(nonw.numel(), device=DEV, generator=g)[:NCTRL]]; Q = unitr(torch.randn(NCTRL, D, device=DEV, generator=g)); ratioQ = (U @ Q.T).abs() / Lf[:, None]
decu = unitr(dec); rows_tr = unitr(tr(Ru)); log(f"{name}: {int(words.sum())} words, median S {float(S[wl].median()):.2f}; random non-words' median S {float(S[rl].median()):.2f}; random directions' {float(ratioQ.max(0).values.median()):.2f}")
def match(ctx_sets, dirs):
    """for each context set (KX positions) the feature best matching it by Jaccard: recall, precision, Jaccard, decoder-row cosine, feature size"""
    out = {q: [] for q in ("recall", "precision", "jaccard", "cos_decoder_row", "feature_size", "cos_best_by_cosine", "recall_of_best_by_cosine")}
    for j in range(ctx_sets.shape[0]):
        P = ctx_sets[j]; cnt = Fm[P].sum(0).float(); rec = cnt / KX; prec = cnt / fsize.clamp_min(1); jacv = cnt / (KX + fsize - cnt).clamp_min(1); score = jacv + 1e-3 * rec; f = int(score.argmax()); inter = float(cnt[f]); jac = inter / (KX + float(fsize[f]) - inter)
        out["recall"].append(float(rec[f])); out["precision"].append(float(prec[f])); out["jaccard"].append(jac); out["cos_decoder_row"].append(float(decu[f] @ dirs[j])); out["feature_size"].append(float(fsize[f]))
        cs = decu @ dirs[j]; cs[~live] = -2; fb = int(cs.argmax()); out["cos_best_by_cosine"].append(float(cs[fb])); out["recall_of_best_by_cosine"].append(float(rec[fb]))
    return out
ctx_w = ratio[:, wl].topk(KX, dim=0).indices.T; ctx_r = ratio[:, rl].topk(KX, dim=0).indices.T; ctx_q = ratioQ.topk(KX, dim=0).indices.T; ctx_s = torch.randint(N, (wl.numel(), KX), device=DEV, generator=g)
res = dict(model=name, block=L, sae=REPO, coord=coord, n_positions=N, n_words=int(wl.numel()), n_live_features=int(live.sum()), groups={})
for tag, ctx, dirs in (("words", ctx_w, rows_tr[wl]), ("random_nonword_rows", ctx_r, rows_tr[rl]), ("random_directions", ctx_q, unitr(tr(Q))), ("words_shuffled_positions", ctx_s, rows_tr[wl])):
    o = match(ctx, dirs); rec = torch.tensor(o["recall"]); jc = torch.tensor(o["jaccard"]); cosd = torch.tensor(o["cos_decoder_row"]); recov = jc >= 0.25
    res["groups"][tag] = dict(n=int(rec.numel()), share_jaccard_ge_0_25=float(recov.float().mean()), share_jaccard_ge_0_5=float((jc >= 0.5).float().mean()), share_recall_ge_0_5=float((rec >= 0.5).float().mean()), share_recall_ge_0_75=float((rec >= 0.75).float().mean()), median_recall=float(rec.median()), median_precision=med(o["precision"]), median_jaccard=med(o["jaccard"]), median_feature_size=med(o["feature_size"]), median_cos_decoder_row=float(cosd.median()), median_cos_decoder_row_recovered=float(cosd[recov].median()) if recov.any() else None, share_cos_ge_0_5_recovered=float((cosd[recov] >= 0.5).float().mean()) if recov.any() else None, share_cos_ge_0_3_recovered=float((cosd[recov] >= 0.3).float().mean()) if recov.any() else None, median_cos_best_feature_by_cosine=med(o["cos_best_by_cosine"]), median_recall_of_best_by_cosine=med(o["recall_of_best_by_cosine"]))
    r_ = res["groups"][tag]; f2 = lambda x: "n/a" if x is None else f"{x:.2f}"; log(f"{name} {tag} ({r_['n']}): the best-matching feature has Jaccard 0.25 or more for {f2(r_['share_jaccard_ge_0_25'])} (0.5 or more {f2(r_['share_jaccard_ge_0_5'])}), recall 0.5 or more for {f2(r_['share_recall_ge_0_5'])}, median recall {f2(r_['median_recall'])}, precision {f2(r_['median_precision'])}, Jaccard {f2(r_['median_jaccard'])}, feature size {r_['median_feature_size']:.0f}; the covering feature's decoder cosine with the row {f2(r_['median_cos_decoder_row'])} (among recovered {f2(r_['median_cos_decoder_row_recovered'])}, 0.5 or more for {f2(r_['share_cos_ge_0_5_recovered'])}, 0.3 or more for {f2(r_['share_cos_ge_0_3_recovered'])}); the feature nearest by cosine: cosine {f2(r_['median_cos_best_feature_by_cosine'])}, its recall of the context set {f2(r_['median_recall_of_best_by_cosine'])}")
# reverse: from live features to rows
lf = torch.nonzero(live)[:, 0]; cs = decu[lf] @ rows_tr.T; best = cs.argmax(1); bestcos = cs.max(1).values; isword = words[best]; recs = []
for i_, f in enumerate(lf.tolist()):
    r_ = int(best[i_]); P = ratio[:, r_].topk(KX).indices; recs.append(float(Fm[P, f].float().mean()))
recs = torch.tensor(recs); res["reverse"] = dict(n_live=int(lf.numel()), median_nearest_row_cosine=float(bestcos.median()), share_nearest_is_word=float(isword.float().mean()), share_nearest_cos_ge_0_5=float((bestcos >= 0.5).float().mean()), median_recall_of_nearest_rows_context=float(recs.median()), share_nearest_is_word_when_cos_ge_0_5=float(isword[bestcos >= 0.5].float().mean()) if (bestcos >= 0.5).any() else None, base_rate_words=float(words.float().mean()))
rv = res["reverse"]; f2 = lambda x: "n/a" if x is None else f"{x:.2f}"; log(f"{name} reverse ({rv['n_live']} live features): nearest row by decoder cosine, median cosine {f2(rv['median_nearest_row_cosine'])}, at 0.5 or more for {f2(rv['share_nearest_cos_ge_0_5'])}; the nearest row is a word for {f2(rv['share_nearest_is_word'])} (base rate {f2(rv['base_rate_words'])}; when the cosine is 0.5 or more {f2(rv['share_nearest_is_word_when_cos_ge_0_5'])}); the feature covers the nearest row's context set at median recall {f2(rv['median_recall_of_nearest_rows_context'])}")
W_, Rr = res["groups"]["words"], res["groups"]["random_nonword_rows"]; g_ = lambda x: -9 if x is None else x
res["checks"] = dict(feature_recovers_words=W_["share_jaccard_ge_0_25"] > 0.5 and Rr["share_jaccard_ge_0_25"] < 0.2, class_not_direction=g_(W_["median_cos_decoder_row_recovered"]) < 0.3, direction_too=g_(W_["median_cos_decoder_row_recovered"]) >= 0.5, reverse_few_words=rv["share_nearest_is_word"] < 0.2)
summ = f"{name} block {L}, {REPO} ({coord}): " + " | ".join(f"{k}: Jaccard 0.25 or more for {f2(v['share_jaccard_ge_0_25'])}, median Jaccard {f2(v['median_jaccard'])}, recall {f2(v['median_recall'])}, decoder-row cosine {f2(v['median_cos_decoder_row'])} (recovered {f2(v['median_cos_decoder_row_recovered'])}, 0.5 or more for {f2(v['share_cos_ge_0_5_recovered'])})" for k, v in res["groups"].items()) + f" | reverse: nearest row a word for {f2(rv['share_nearest_is_word'])} (base {f2(rv['base_rate_words'])}), median cosine {f2(rv['median_nearest_row_cosine'])} | checks {_json.dumps(res['checks'])}"
log(summ); record(f"e549_feature_side_{name}_b{L}", res, summ)
