"""e574 (session 103): what is the partition? Three readings of a word's context set, tested on Pythia-410m (block 12,
2032 positions, 256 words). (a) Surface: how well the set is predicted from surface features of the text (the token,
the previous token, the position bucket, capitalisation, digit, punctuation, whitespace-only) by a five-fold logistic
regression, AUC per word. (b) Output: the homogeneity of the model's next-token predictions within the set (the
share of positions whose argmax prediction is the set's modal one; the mean pairwise Jensen-Shannon divergence of the
predicted distributions) against the input-token purity and against random sets of the same size. (c) The interface:
the read energy of a word's direction by every other block's readers (the MLP input weights and the attention
query-key-value weights after the layer norm's gain), for words, norm-matched non-word rows, random rows and the
rotated dictionary's most used atoms. Pre-registered (probabilities are honest guesses):
 R1 (0.6) surface features predict the sets at AUC 0.8 or more for most words (the partition is largely the corpus's
    surface structure);
 R2 (0.55) the sets are more homogeneous in the predicted next token than in the input token (an output partition);
 R3 (0.5) words are read more than norm-matched rows by later blocks (the vocabulary is the interface)."""
from s101_common import *
t0 = time.time(); B = 12; S = lm_states("pythia410"); model, arch, tok_ = S["model"], S["arch"], None
from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m"); ids = S["ids"]; keep = S["keep"]; st = stats(S["U"], S["A"], K); w = torch.nonzero(wordset(st["usage"]))[:, 0]; over = (st["ratio"][:, w].float() > 1); N = over.shape[0]
with torch.no_grad(): LG = model(ids).logits.float()[:, 1:].reshape(-1, model.config.vocab_size)[keep]
P = LG.softmax(-1); top = P.argmax(-1).cpu(); tok = ids[:, 1:].reshape(-1)[keep].cpu(); prev = ids[:, :-1].reshape(-1)[keep].cpu(); posn = torch.arange(ids.shape[1] - 1).repeat(ids.shape[0])[keep.cpu()]
# (a) surface features
strs = [tk.decode([int(t)]) for t in tok]; V = 256
topt = torch.bincount(tok).topk(V).indices; topp = torch.bincount(prev).topk(V).indices; fmap = {int(t): i for i, t in enumerate(topt)}; pmap = {int(t): i for i, t in enumerate(topp)}
Fs = torch.zeros(N, 2 * V + 8 + 5)
for i in range(N):
    if int(tok[i]) in fmap: Fs[i, fmap[int(tok[i])]] = 1
    if int(prev[i]) in pmap: Fs[i, V + pmap[int(prev[i])]] = 1
    Fs[i, 2 * V + min(int(posn[i]) * 8 // 255, 7)] = 1; s = strs[i]
    Fs[i, 2 * V + 8] = float(s.strip()[:1].isupper()) if s.strip() else 0; Fs[i, 2 * V + 9] = float(s.strip().isdigit()) if s.strip() else 0; Fs[i, 2 * V + 10] = float(bool(s.strip()) and not s.strip().isalnum()); Fs[i, 2 * V + 11] = float(s.strip() == ""); Fs[i, 2 * V + 12] = float(s.startswith(" "))
Fs = Fs.to(DEV); aucs = []
for j in range(w.numel()):
    y = over[:, j].float(); n1 = int(y.sum())
    if n1 < 5: aucs.append(None); continue
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(j)); sc = torch.zeros(N)
    for f in range(5):
        te = perm[f::5]; tr = torch.tensor(sorted(set(perm.tolist()) - set(te.tolist()))); sc[te] = logreg(Fs[tr], y[tr].to(DEV), Fs[te], l2=1e-1, iters=100).cpu()
    aucs.append(auc(sc, y.bool()))
av = [a for a in aucs if a is not None]; res = dict(surface=dict(n=len(av), auc_median=med(av), share_ge_08=mean([float(a >= 0.8) for a in av]), share_ge_09=mean([float(a >= 0.9) for a in av]), auc_q10=float(torch.tensor(av).quantile(0.1))))
log(f"surface features -> context sets: AUC median {res['surface']['auc_median']:.3f}, share >= 0.8 {res['surface']['share_ge_08']:.2f}, >= 0.9 {res['surface']['share_ge_09']:.2f}, tenth percentile {res['surface']['auc_q10']:.3f}")
# (b) output homogeneity
def js(Pm):
    n = Pm.shape[0]
    if n < 2: return None
    M = Pm.mean(0, keepdim=True); return float(0.5 * ((Pm * (Pm.clamp_min(1e-12).log() - M.clamp_min(1e-12).log())).sum(-1)).mean() + 0.5 * ((M * (M.clamp_min(1e-12).log() - Pm.clamp_min(1e-12).log())).sum(-1)).mean())
def homog(sets):
    out = {q: [] for q in ("input_purity", "pred_purity", "next_purity", "js", "size")}
    for pos in sets:
        if pos.numel() < 3: continue
        out["input_purity"].append(float(torch.bincount(tok[pos]).max() / pos.numel())); out["pred_purity"].append(float(torch.bincount(top[pos]).max() / pos.numel()))
        nxt = ids[:, 1:].reshape(-1)[keep].cpu(); nn_ = torch.cat([ids[:, 2:].reshape(-1), torch.full((ids.shape[0],), -1, device=DEV)]) if False else None
        out["js"].append(js(P[pos.to(DEV)])); out["size"].append(int(pos.numel()))
    return {q: (med(v) if v else None) for q, v in out.items()}
sets = [torch.nonzero(over[:, j])[:, 0] for j in range(w.numel())]; g = torch.Generator().manual_seed(1); rsets = [torch.randperm(N, generator=g)[:s.numel()] for s in sets]
res["output"] = dict(words=homog(sets), random_sets=homog(rsets)); ho, hr = res["output"]["words"], res["output"]["random_sets"]
log(f"context sets: input-token purity {ho['input_purity']:.2f} (random sets {hr['input_purity']:.2f}), predicted-token purity {ho['pred_purity']:.2f} ({hr['pred_purity']:.2f}), pairwise JS of the predictions {ho['js']:.3f} ({hr['js']:.3f}); median set size {ho['size']:.0f}")
# (c) read energy of directions
def readers(d):
    """total read energy of unit directions d [n, D] by the MLP inputs and attention QKV of all blocks except B, after each block's layer-norm gain"""
    e = torch.zeros(d.shape[0], device=DEV); late = torch.zeros(d.shape[0], device=DEV)
    for b in range(arch.NB):
        if b == B: continue
        l = arch.layers[b]; g1 = l.input_layernorm.weight.detach().float(); g2 = l.post_attention_layernorm.weight.detach().float() if hasattr(l, "post_attention_layernorm") else g1
        eb = (l.attention.query_key_value.weight.detach().float() @ (d * g1[None]).T).pow(2).sum(0) + (l.mlp.dense_h_to_4h.weight.detach().float() @ (d * g2[None]).T).pow(2).sum(0); e += eb
        if b > B: late += eb
    return e.cpu(), late.cpu()
m = S["A"].shape[0]; norms = S["norms"].cpu(); DFF = arch.DFF; blk = torch.arange(m) // DFF; used = torch.zeros(m, dtype=torch.bool); used[w] = True; nm = []
for r in w.tolist():
    cand = torch.nonzero(~used & (blk == blk[r]))[:, 0]; j = cand[(norms[cand] - norms[r]).abs().argmin()]; used[j] = True; nm.append(int(j))
nm = torch.tensor(nm); rnd = torch.nonzero(~used)[:, 0]; rnd = rnd[torch.randperm(rnd.numel(), generator=g)[:256]]; Ar = unitr(rotate(S["A"], seed=7)); sr = stats(S["U"], Ar, K); wr = torch.nonzero(wordset(sr["usage"]))[:, 0]
res["read"] = {}
for name, D in (("words", S["A"][w]), ("norm_matched", S["A"][nm]), ("random_rows", S["A"][rnd]), ("rotated_words", Ar[wr]), ("random_directions", unitr(torch.randn(256, arch.D, device=DEV)))):
    e, late = readers(D); res["read"][name] = dict(total=float(e.median()), late=float(late.median())); log(f"read energy of {name}: all blocks {float(e.median()):.3f}, blocks after {B} {float(late.median()):.3f}")
rd = res["read"]
summ = (f"what the partition is (Pythia-410m block 12): surface features predict the context sets at AUC {res['surface']['auc_median']:.2f} (share >= 0.8 {res['surface']['share_ge_08']:.2f}); within a set the input-token purity is {ho['input_purity']:.2f} and the predicted-token purity {ho['pred_purity']:.2f} (random sets {hr['input_purity']:.2f} / {hr['pred_purity']:.2f}), pairwise JS of the predictions {ho['js']:.3f} vs {hr['js']:.3f}; "
        f"read energy by other blocks: words {rd['words']['total']:.3f} (later blocks {rd['words']['late']:.3f}), norm-matched rows {rd['norm_matched']['total']:.3f} ({rd['norm_matched']['late']:.3f}), random rows {rd['random_rows']['total']:.3f}, rotated words {rd['rotated_words']['total']:.3f}, random directions {rd['random_directions']['total']:.3f} | {time.time() - t0:.0f}s")
log(summ); record("e574_what_partition", res, summ)
