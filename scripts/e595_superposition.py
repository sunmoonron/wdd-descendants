"""e595 (session 110): superposition counts through training. With the classes defined without rows (spherical k-means
on the unit states, K clusters) and the words at each checkpoint: classes per word (the clusters at which the word is
over the floor at half the positions or more) and rows per class (the rows over the floor at half a cluster's
positions or more), at every checkpoint of e582's cache (Pythia-410m block 12, K = 512, with K = 128 and 2048 at
16000), on the final Pythia-410m, and on OLMo-1B's cache at 8000 and 16000 (K = 128). Pre-registered (probabilities
are honest guesses):
 X1 (0.5) at 16000 most words have exactly one class (two or more for under 0.3);
 X2 (0.6) classes per word rise through training;
 X3 (0.5) most clusters have at most one speaker row."""
from s101_common import *
t0 = time.time(); steps = list(range(1000, 16001, 1000))
def kmeans(X, Kc, iters=15, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed); C = X[torch.randperm(X.shape[0], device=DEV, generator=g)[:Kc]].clone()
    for _ in range(iters):
        lab = (X @ C.T).argmax(1); C = torch.zeros_like(C).index_add_(0, lab, X); cnt = torch.bincount(lab, minlength=Kc).float(); empty = cnt == 0; C = unitr(C / cnt.clamp_min(1)[:, None]); C[empty] = X[torch.randperm(X.shape[0], device=DEV, generator=g)[:int(empty.sum())]]
    return (X @ C.T).argmax(1)
def counts(U, A, Kc, tag):
    st = stats(U, A, K); words = torch.nonzero(wordset(st["usage"]))[:, 0]; ratio = st["ratio"].to(DEV); lab = kmeans(U, Kc); cnt = torch.bincount(lab, minlength=Kc).float().clamp_min(1)
    over = torch.zeros(Kc, A.shape[0], device=DEV).index_add_(0, lab, (ratio > 1).float()) / cnt[:, None]   # share of each cluster's positions at which each row is over the floor
    speaks = over >= 0.5; rows_per_class = speaks.sum(1).float(); classes_per_word = speaks[:, words].sum(0).float(); del ratio; torch.cuda.empty_cache()
    out = dict(K=Kc, n_words=int(words.numel()), classes_per_word_mean=float(classes_per_word.mean()), share_words_ge2=float((classes_per_word >= 2).float().mean()), share_words_0=float((classes_per_word == 0).float().mean()), share_words_1=float((classes_per_word == 1).float().mean()), max_classes_per_word=int(classes_per_word.max()),
               rows_per_class_mean=float(rows_per_class.mean()), share_classes_0=float((rows_per_class == 0).float().mean()), share_classes_1=float((rows_per_class == 1).float().mean()), share_classes_ge2=float((rows_per_class >= 2).float().mean()), share_classes_ge2_among_spoken=float((rows_per_class[rows_per_class > 0] >= 2).float().mean()) if (rows_per_class > 0).any() else None)
    log(f"{tag} (K={Kc}): classes per word {out['classes_per_word_mean']:.2f} (none {out['share_words_0']:.2f}, one {out['share_words_1']:.2f}, two+ {out['share_words_ge2']:.2f}, max {out['max_classes_per_word']}); rows per class {out['rows_per_class_mean']:.2f} (none {out['share_classes_0']:.2f}, one {out['share_classes_1']:.2f}, two+ {out['share_classes_ge2']:.2f}; two+ among spoken {out['share_classes_ge2_among_spoken']})"); return out
res = dict(pythia410={}, olmo1b={})
CD = "/workspace/wdd/cache/e582_pythia410"
for n in steps:
    c = torch.load(f"{CD}/step{n}.pt", map_location="cpu"); U = unitr((c["X"][c["keep"]].float() - c["X"][c["keep"]].float().mean(0)).to(DEV)); A = c["rows"].float().to(DEV)
    res["pythia410"][str(n)] = counts(U, A, 512, f"pythia410 step {n}")
    if n == 16000:
        for Kc in (128, 2048): res["pythia410"][f"16000_K{Kc}"] = counts(U, A, Kc, f"pythia410 step 16000")
    del U, A, c; torch.cuda.empty_cache()
S = lm_states("pythia410", B=12, ids=eval_ids("pythia410")[:24, :512].to(DEV)); res["pythia410"]["final"] = counts(S["U"], S["A"], 512, "pythia410 final"); del S; torch.cuda.empty_cache()
try:
    CO = "/workspace/wdd/cache/e550_olmo1b"
    for n in (8000, 16000):
        c = torch.load(f"{CO}/step{n}.pt", map_location="cpu"); U = unitr((c["X"][c["keep"]].float() - c["X"][c["keep"]].float().mean(0)).to(DEV)); A = c["rows"].float().to(DEV); res["olmo1b"][str(n)] = counts(U, A, 128, f"olmo1b step {n}"); del U, A, c; torch.cuda.empty_cache()
except Exception as e: log(f"olmo cache not available: {str(e)[:100]}")
P = res["pythia410"]
summ = (f"superposition counts (Pythia-410m block 12, K=512): classes per word at 1000/4000/8000/16000/final {P['1000']['classes_per_word_mean']:.2f}/{P['4000']['classes_per_word_mean']:.2f}/{P['8000']['classes_per_word_mean']:.2f}/{P['16000']['classes_per_word_mean']:.2f}/{P['final']['classes_per_word_mean']:.2f}, words with two or more {P['1000']['share_words_ge2']:.2f}/{P['4000']['share_words_ge2']:.2f}/{P['8000']['share_words_ge2']:.2f}/{P['16000']['share_words_ge2']:.2f}/{P['final']['share_words_ge2']:.2f} (none {P['16000']['share_words_0']:.2f} at 16000); rows per class at 16000 {P['16000']['rows_per_class_mean']:.2f} (none {P['16000']['share_classes_0']:.2f}, one {P['16000']['share_classes_1']:.2f}, two+ {P['16000']['share_classes_ge2']:.2f}); at K=128/2048 classes per word {P['16000_K128']['classes_per_word_mean']:.2f}/{P['16000_K2048']['classes_per_word_mean']:.2f}"
        + (f"; OLMo-1B (K=128) at 8000/16000 classes per word {res['olmo1b']['8000']['classes_per_word_mean']:.2f}/{res['olmo1b']['16000']['classes_per_word_mean']:.2f}, two+ {res['olmo1b']['16000']['share_words_ge2']:.2f}" if res["olmo1b"] else "") + f" | {time.time() - t0:.0f}s")
log(summ); record("e595_superposition", res, summ)
