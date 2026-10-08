"""e611 (session 114): class-conditioned descendants. The J-Lens (Gurnee et al. 2026) transports a mid-layer activation
to the output with a Jacobian averaged over contexts, and the J++ Lens found that one Jacobian per activation
cluster is far more faithful than the global average, most of all in early layers. WDD's descendants transport a
write the same way, and the word's class is the cluster for free. Pythia-410m, blocks 4, 8 and 12, the 256 words at
each block (rows of blocks 0..b) on 16 x 256 Pile windows, the 40 words with the largest classes. For each word
three predictions of the output change its write makes, as directions in logit space: the raw row through the
unembedding (the logit lens of the row), the row transported by the Jacobian averaged over every position (the
global map, estimated by perturbing every position by the row at once), and the row transported by the Jacobian
averaged over the word's class positions (the class map, estimated on half of the class and tested on the other
half). As the control for the estimate itself, the row transported by the Jacobian averaged over a random set of
positions of the same size as that training half (the random-subset map; a global map perturbs every position at
once and may carry more cross-position interference than a subset map). The ground truth at each test position is the actual logit change when that position alone is perturbed by
the row at block b, measured by two forward passes. Scores: the cosine between the predicted and the actual logit
change and the overlap of their top-10 tokens, averaged over eight held-out class positions per word, and the same
at eight positions outside the class. Argument: --smoke.
Pre-registered (honest guesses):
 J1 (0.6) at block 4 the class map's cosine with the actual change exceeds the global map's by 0.10 or more;
 J2 (0.6) at block 12 the gap is under 0.03 (the map is near-global late);
 J3 (0.7) the raw row is the worst of the maps at every block;
 J4 (0.5) the class map beats the random-subset map by 0.10 or more at block 4 (conditioning, not estimation)."""
from s101_common import *
from ma_common import Stop
SMOKE = "--smoke" in sys.argv; name = "pythia410"; t0 = time.time(); T = 256; NSEQ = 4 if SMOKE else 16; NW = 6 if SMOKE else 40; NTEST = 4 if SMOKE else 8; EPS = 0.25; torch.set_grad_enabled(False)
model, tok, fam = load_model(name); arch = Arch(model, fam); D = arch.D; ids = eval_ids("pythia410")[:NSEQ, :T].to(DEV); WU = model.get_output_embeddings().weight.detach().float()
def final_logits(hook=None):
    """logits [NSEQ, T, V] with an optional residual hook on a block"""
    h = hook() if hook else None
    try: lg = model(ids).logits.float()
    finally:
        if h: h.remove()
    return lg
def add_hook(b, vec, posmask):
    """adds vec at the given [NSEQ, T] positions to block b's output"""
    def hk(mod, inp, out):
        x = out[0] if isinstance(out, tuple) else out; y = x + posmask.to(x.dtype)[..., None] * vec.to(x.dtype); return (y,) + tuple(out[1:]) if isinstance(out, tuple) else y
    return lambda: arch.layers[b].register_forward_hook(hk)
base = final_logits(); res = dict(model=name, eps=EPS, n_seq=NSEQ, blocks={})
def cos(a, b_): return float((a * b_).sum() / (a.norm() * b_.norm()).clamp_min(1e-8))
def top10(v, k=10): return set(v.topk(k).indices.tolist())
for B in ((4, 12) if SMOKE else (4, 8, 12)):
    X = block_states(model, arch, ids, [B])[B].reshape(-1, D); keep = ~sinkmask(X); A, norms = rows_of(arch, B); U = unitr(X[keep] - X[keep].mean(0)); st = stats(U, A, K); R = st["ratio"].float(); words = torch.nonzero(wordset(st["usage"]))[:, 0]
    kidx = torch.nonzero(keep.cpu())[:, 0]; CLS = {int(w): kidx[torch.nonzero(R[:, w] > 1)[:, 0]] for w in words.tolist()}   # full flat indices (positions 1..T-1 of each sequence)
    sel = sorted(CLS, key=lambda w: -CLS[w].numel())[:NW]; scale = EPS * float(X[keep].norm(dim=1).median()); log(f"block {B}: {len(words)} words, testing {len(sel)} with classes of {CLS[sel[-1]].numel()}-{CLS[sel[0]].numel()} positions; perturbation {scale:.2f} ({time.time() - t0:.0f}s)")
    def flat_to_mask(flat):
        m = torch.zeros(NSEQ, T, dtype=torch.bool); m[:, 1:] = torch.zeros(NSEQ * (T - 1), dtype=torch.bool).index_fill_(0, flat.cpu(), True).reshape(NSEQ, T - 1); return m.to(DEV)
    allmask = torch.zeros(NSEQ, T, dtype=torch.bool); allmask[:, 1:] = keep.cpu().reshape(NSEQ, T - 1); allmask = allmask.to(DEV)
    rows_out = []
    for w in sel:
        vec = A[w] * scale; c = CLS[w]; g = torch.Generator().manual_seed(w); perm = c[torch.randperm(c.numel(), generator=g)]; test = perm[:NTEST]; train = perm[NTEST:]
        outside = torch.tensor([int(p) for p in kidx.tolist() if p not in set(c.tolist())]); outside = outside[torch.randperm(outside.numel(), generator=g)[:NTEST]]
        # the global map: every kept position perturbed at once, the mean logit change at the perturbed positions
        dg = (final_logits(add_hook(B, vec, allmask)) - base)[allmask].mean(0)
        # the class map: the training half of the class perturbed at once
        dc = (final_logits(add_hook(B, vec, flat_to_mask(train))) - base)[flat_to_mask(train)].mean(0) if train.numel() else None
        rsub = kidx[torch.randperm(kidx.numel(), generator=g)[:max(int(train.numel()), 1)]]; dr = (final_logits(add_hook(B, vec, flat_to_mask(rsub))) - base)[flat_to_mask(rsub)].mean(0)
        raw = WU @ A[w]
        rec = dict(word=w, n_class=int(c.numel()), cos=dict(raw=[], glob=[], cls=[], rsub=[]), top=dict(raw=[], glob=[], cls=[], rsub=[]), cos_out=dict(raw=[], glob=[], cls=[], rsub=[]))
        for grp, poss in (("in", test), ("out", outside)):
            for p in poss.tolist():
                m = flat_to_mask(torch.tensor([p])); da = (final_logits(add_hook(B, vec, m)) - base)[m][0]
                for k_, pred in (("raw", raw), ("glob", dg), ("cls", dc), ("rsub", dr)):
                    if pred is None: continue
                    (rec["cos"] if grp == "in" else rec["cos_out"])[k_].append(cos(pred, da))
                    if grp == "in": rec["top"][k_].append(len(top10(pred) & top10(da)) / 10)
        rows_out.append(rec)
    agg = {k_: dict(cos=mean([mean(r["cos"][k_]) for r in rows_out if r["cos"][k_]]), top10=mean([mean(r["top"][k_]) for r in rows_out if r["top"][k_]]), cos_outside=mean([mean(r["cos_out"][k_]) for r in rows_out if r["cos_out"][k_]])) for k_ in ("raw", "glob", "cls", "rsub")}
    wins = mean([float(mean(r["cos"]["cls"]) > mean(r["cos"]["glob"])) for r in rows_out if r["cos"]["cls"] and r["cos"]["glob"]]); wins_r = mean([float(mean(r["cos"]["cls"]) > mean(r["cos"]["rsub"])) for r in rows_out if r["cos"]["cls"] and r["cos"]["rsub"]])
    res["blocks"][str(B)] = dict(n_words=len(sel), agg=agg, class_beats_global=wins, class_beats_random_subset=wins_r, per_word=[dict(word=r["word"], n_class=r["n_class"], cos={k_: mean(v) if v else None for k_, v in r["cos"].items()}, cos_out={k_: mean(v) if v else None for k_, v in r["cos_out"].items()}) for r in rows_out])
    log(f"block {B}: cosine with the actual logit change at held-out class positions: raw row {agg['raw']['cos']:.3f}, global map {agg['glob']['cos']:.3f}, random-subset map {agg['rsub']['cos']:.3f}, class map {agg['cls']['cos']:.3f} (class beats global for {wins:.2f} of words, beats the random subset for {wins_r:.2f}); top-10 overlap {agg['raw']['top10']:.2f} / {agg['glob']['top10']:.2f} / {agg['cls']['top10']:.2f}; at positions outside the class {agg['raw']['cos_outside']:.3f} / {agg['glob']['cos_outside']:.3f} / {agg['rsub']['cos_outside']:.3f} / {agg['cls']['cos_outside']:.3f} | {time.time() - t0:.0f}s")
    del X, U, st, R; torch.cuda.empty_cache()
Bk = res["blocks"]; summ = "class-conditioned descendants (Pythia-410m): " + "; ".join(f"block {b}: cosine raw {v['agg']['raw']['cos']:.3f}, global {v['agg']['glob']['cos']:.3f}, random subset {v['agg']['rsub']['cos']:.3f}, class {v['agg']['cls']['cos']:.3f} (class beats global {v['class_beats_global']:.2f}, beats the random subset {v['class_beats_random_subset']:.2f}; outside the class {v['agg']['glob']['cos_outside']:.3f} / {v['agg']['rsub']['cos_outside']:.3f} / {v['agg']['cls']['cos_outside']:.3f})" for b, v in Bk.items()) + f" | {time.time() - t0:.0f}s"
log(summ); record("e611_class_descendants" + ("_smoke" if SMOKE else ""), res, summ)
