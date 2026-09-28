"""e569b (session 102): the 'describe' level of e569 came out at 1.00 for every perturbation, including training from
step 1000 to 2000 where the word sets share 0.07: the old words' rows describe the new states as well as the new words
do at K=8. Is that a property of the words or of any 256 rows? Pythia-160m (block 6) and Pythia-410m at step 16000
(block 12): the FVU at K=8 of the states under the model's own 256 words, the 256 words of the other checkpoint or of
a noise-perturbed twin (old vectors), 256 random rows, 256 random rows matched to the words' norms, the 256 most used
atoms of the rotated dictionary, and 256 Gaussian atoms with the rows' covariance. Pre-registered: D4 (0.6) random rows
describe the states within 15% of the words at K=8 (the words are the rows the states use most often, not the rows that
span them), while the rotated and Gaussian atoms fall more than 15% behind."""
from s101_common import *
t0 = time.time(); res = {}
def panel(tag, S_new, A_old_words, extra=None):
    U, A = S_new["U"], S_new["A"]; st = stats(U, A, K); w = torch.nonzero(wordset(st["usage"]))[:, 0]; m = A.shape[0]; g = torch.Generator().manual_seed(0); tot = float(U.pow(2).sum())
    norms = S_new["norms"].cpu(); rnd = torch.randperm(m, generator=g)[:256]
    order = norms.argsort(); nm = []; used = torch.zeros(m, dtype=torch.bool)
    for r in w.tolist():
        cand = torch.nonzero(~used)[:, 0]; j = cand[(norms[cand] - norms[r]).abs().argmin()]; used[j] = True; nm.append(int(j))
    Ar = unitr(rotate(A, seed=7)); sr = stats(U, Ar, K); wr = torch.nonzero(wordset(sr["usage"]))[:, 0]
    ev, V = torch.linalg.eigh((A.T @ A / m).double()); Rm = ((V * ev.clamp_min(0).sqrt()) @ V.T).float(); G = unitr(torch.randn(256, A.shape[1], generator=torch.Generator().manual_seed(13)).to(DEV) @ Rm)
    D = {"own_words": A[w], "other_words_old_vectors": A_old_words, "random_rows": A[rnd], "norm_matched_random_rows": A[torch.tensor(nm)], "rotated_words": Ar[wr], "gaussian_covariance": G}
    if extra: D.update(extra)
    out = {}
    for k, Dd in D.items():
        sel, cof, err = omp(U, Dd, 8, batch=1024, record_err=True); out[k] = float(err[:, -1].sum() / tot)
    res[tag] = out; log(f"{tag}: FVU at K=8 " + ", ".join(f"{k} {v:.3f}" for k, v in out.items()))
# Pythia-160m: the other dictionary = the words of a noise-perturbed twin (relative 0.02) and of pythia-160m-deduped (old vectors)
m0, tok, fam = load_model("pythia160"); S0 = lm_states("pythia160", model=m0); st0 = stats(S0["U"], S0["A"], K); w0 = torch.nonzero(wordset(st0["usage"]))[:, 0]
g = torch.Generator(device=DEV).manual_seed(0)
for k, v in m0.state_dict().items():
    if v.is_floating_point() and v.dim() == 2: v.add_(torch.randn(v.shape, generator=g, device=DEV) * 0.02 * v.std())
S1 = lm_states("pythia160", model=m0); del m0; torch.cuda.empty_cache()
Sd = lm_states("pythia160d"); std_ = stats(Sd["U"], Sd["A"], K); wd_ = torch.nonzero(wordset(std_["usage"]))[:, 0]; del Sd["model"]; torch.cuda.empty_cache()
panel("pythia160 noise 0.02 states", S1, S0["A"][w0], {"deduped_words_own_vectors": Sd["A"][wd_]})
# Pythia-410m at 16000: the other dictionary = the words of step 8000 (old vectors) and of step 1000
m8, _, _ = load_model("pythia410", revision="step8000"); S8 = lm_states("pythia410", model=m8); st8 = stats(S8["U"], S8["A"], K); w8 = torch.nonzero(wordset(st8["usage"]))[:, 0]; A8w = S8["A"][w8]; del m8, S8; torch.cuda.empty_cache()
m1, _, _ = load_model("pythia410", revision="step1000"); S1k = lm_states("pythia410", model=m1); st1 = stats(S1k["U"], S1k["A"], K); w1 = torch.nonzero(wordset(st1["usage"]))[:, 0]; A1w = S1k["A"][w1]; del m1, S1k; torch.cuda.empty_cache()
m16, _, _ = load_model("pythia410", revision="step16000"); S16 = lm_states("pythia410", model=m16)
panel("pythia410 step16000 states", S16, A8w, {"step1000_words_old_vectors": A1w})
r1, r4 = res["pythia160 noise 0.02 states"], res["pythia410 step16000 states"]
summ = (f"FVU at K=8, Pythia-160m noise-0.02 states: own words {r1['own_words']:.3f}, the unperturbed words {r1['other_words_old_vectors']:.3f}, deduped's words {r1['deduped_words_own_vectors']:.3f}, random rows {r1['random_rows']:.3f}, norm-matched random rows {r1['norm_matched_random_rows']:.3f}, rotated words {r1['rotated_words']:.3f}, Gaussian {r1['gaussian_covariance']:.3f}; "
        f"Pythia-410m step-16000 states: own words {r4['own_words']:.3f}, step-8000 words {r4['other_words_old_vectors']:.3f}, step-1000 words {r4['step1000_words_old_vectors']:.3f}, random rows {r4['random_rows']:.3f}, norm-matched {r4['norm_matched_random_rows']:.3f}, rotated words {r4['rotated_words']:.3f}, Gaussian {r4['gaussian_covariance']:.3f} | {time.time() - t0:.0f}s")
log(summ); record("e569b_describe_baseline", res, summ)
