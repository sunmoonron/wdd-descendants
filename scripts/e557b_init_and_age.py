"""e557b (session 101): two follow-ups to e557. (1) Do Pythia-160m and pythia-160m-deduped share an initialization, and
how much of the initialization survives training? Rows by index at step 0 of both models (cosine 1 means the same
seed), the final rows against their own step-0 rows (retention of the initialization, by block), and the two final
models against each other (e557 found 0.07 by index, 0.19 in block 0). (2) e557's end-of-training panel used the
session-88 cache's usage, whose definition may differ from the rows-only OMP of sessions 89-101; here the step-16000
words are recomputed rows-only from the cached states and rows on the positions the final model also keeps, and
compared with the final words (step 143000) by index and by context-set twins (both directions, same text); the rows'
cosine by index from 16000 to the end for final words and for all rows; the age of every row over checkpoints
1000-16000 recomputed rows-only; the 160m twin rate of final words by age, and the twin rate of the step-16000 words.
Pre-registered (probabilities are honest guesses):
 I1 (0.5) the two 160m models share their initialization (median row cosine by index at step 0 above 0.99);
 I2 (0.6) the final rows keep under 0.3 cosine with their initialization at the median;
 I3 (0.6) the final vocabulary keeps under half of the step-16000 words by index, but context-set twins between the
    two checkpoints exceed 0.5 (the words move to other rows less than they persist);
 I4 (0.55) final words with a 160m twin are older than words without, on the rows-only ages."""
from s101_common import *
t0 = time.time(); res = {}
# (1) initialization
def rows_at(name, rev):
    m, _, fam = load_model(name, revision=rev); A, n = rows_of(Arch(m, fam), 6); del m; torch.cuda.empty_cache(); return A.cpu()
A0, A0d = rows_at("pythia160", "step0"), rows_at("pythia160d", "step0"); A1, A1d = rows_at("pythia160", None), rows_at("pythia160d", None); blk = torch.arange(A0.shape[0]) // 3072
def cs(P, Q): c = (P * Q).sum(1); return dict(median=float(c.median()), mean=float(c.mean()), by_block=[float(c[blk == b].median()) for b in range(7)])
res["init"] = dict(step0_by_index=cs(A0, A0d), retention_160m=cs(A0, A1), retention_deduped=cs(A0d, A1d), final_by_index=cs(A1, A1d), final_160m_vs_deduped_init=cs(A0d, A1))
for k, v in res["init"].items(): log(f"{k}: median cosine {v['median']:.3f} (blocks 0-6 {', '.join(f'{x:.2f}' for x in v['by_block'])})")
# (2) end of training
CD = "/workspace/wdd/cache/e524_pythia410"; steps = list(range(1000, 16001, 1000))
Sf = lm_states("pythia410"); c16 = torch.load(f"{CD}/step16000.pt", map_location="cpu"); b16 = c16["blocks"][12]
keepc = Sf["keep"].cpu() & b16["keep"]; N = int(keepc.sum()); Uf = unitr(Sf["X"][keepc] - Sf["X"][keepc].mean(0)); U16 = b16["U"][keepc].float().to(DEV); R16 = c16["rows"].float().to(DEV)
stf = stats(Uf, Sf["A"], K); st16 = stats(U16, R16, K); wf, w16 = wordset(stf["usage"]), wordset(st16["usage"]); tokc = Sf["ids"][:, 1:].reshape(-1)[keepc].cpu()
def csets(st, w):
    wi = torch.nonzero(w)[:, 0]; r = st["ratio"][:, wi].float(); return wi, r > 1
wfi, Cf = csets(stf, wf); w16i, C16 = csets(st16, w16)
def jac(P, Q):
    P = P.float(); Q = Q.float(); inter = P.T @ Q; return inter / (P.sum(0)[:, None] + Q.sum(0)[None] - inter).clamp_min(1)
bf = jac(Cf, C16).max(1).values; b16b = jac(C16, Cf).max(1).values; cosr = (Sf["A"] * R16).sum(1).cpu()
res["end"] = dict(common_positions=N, word_jaccard_by_index=float((wf & w16).sum() / (wf | w16).sum()), share_final_words_words_at_16000=float(w16[wf].float().mean()), twin_final_to_16000=float((bf >= 0.25).float().mean()), twin_16000_to_final=float((b16b >= 0.25).float().mean()),
                  twin_median_final_to_16000=float(bf.median()), row_cos_16000_to_end_all=float(cosr.median()), row_cos_16000_to_end_final_words=float(cosr[wf].median()), row_cos_16000_to_end_16000_words=float(cosr[w16].median()),
                  twin_rate_among_words_that_changed_row=float((bf[~w16[wfi]] >= 0.25).float().mean()) if (~w16[wfi]).any() else None)
e = res["end"]; log(f"final vs 16000 (rows only, {N} positions): word Jaccard by index {e['word_jaccard_by_index']:.2f}, final words that were words at 16000 {e['share_final_words_words_at_16000']:.2f}; context-set twins final->16000 {e['twin_final_to_16000']:.2f}, 16000->final {e['twin_16000_to_final']:.2f}; row cosine 16000->end all/final words/16000 words {e['row_cos_16000_to_end_all']:.2f}/{e['row_cos_16000_to_end_final_words']:.2f}/{e['row_cos_16000_to_end_16000_words']:.2f}; twins among final words that were not words at 16000 {e['twin_rate_among_words_that_changed_row']}")
# ages rows-only
age = torch.zeros(Sf["A"].shape[0], dtype=torch.long)
for n in steps:
    c = torch.load(f"{CD}/step{n}.pt", map_location="cpu"); b = c["blocks"][12]; kk = b["keep"]; U = b["U"][kk].float().to(DEV); st = stats(U, c["rows"].float().to(DEV), K); age += wordset(st["usage"]).long(); del c, U; torch.cuda.empty_cache()
# 160m twins of the final words and of the 16000 words (context sets on the common positions with 160m)
S160 = lm_states("pythia160"); keep3 = keepc & S160["keep"].cpu(); U160 = unitr(S160["X"][keep3] - S160["X"][keep3].mean(0)); st160 = stats(U160, S160["A"], K); w160i, C160 = csets(st160, wordset(st160["usage"]))
sub = keep3[keepc]   # common positions within keepc
tf = jac(Cf[sub], C160).max(1).values >= 0.25; t16 = jac(C16[sub], C160).max(1).values >= 0.25
res["age"] = dict(twin_rate_final_words=float(tf.float().mean()), twin_rate_16000_words=float(t16.float().mean()), age_final_words_mean=float(age[wfi].float().mean()), age_twinned=float(age[wfi][tf].float().mean()), age_untwinned=float(age[wfi][~tf].float().mean()),
                  age_twinned_median=float(age[wfi][tf].float().median()), age_untwinned_median=float(age[wfi][~tf].float().median()), share_final_words_age0=float((age[wfi] == 0).float().mean()), twin_rate_age0=float(tf[age[wfi] == 0].float().mean()), twin_rate_age_ge8=float(tf[age[wfi] >= 8].float().mean()) if (age[wfi] >= 8).any() else None,
                  age_hist=torch.bincount(age[wfi], minlength=17).tolist(), age_16000_words_mean=float(age[w16i].float().mean()), twin_rate_16000_by_age_ge8=float(t16[age[w16i] >= 8].float().mean()), twin_rate_16000_by_age_lt8=float(t16[age[w16i] < 8].float().mean()))
a = res["age"]; log(f"160m twins: final words {a['twin_rate_final_words']:.2f} (age twinned {a['age_twinned']:.1f} vs untwinned {a['age_untwinned']:.1f}; age-0 words {a['share_final_words_age0']:.2f} of final words, twin rate {a['twin_rate_age0']:.2f}, age >= 8 twin rate {a['twin_rate_age_ge8']}), step-16000 words {a['twin_rate_16000_words']:.2f} (age >= 8 {a['twin_rate_16000_by_age_ge8']:.2f} vs < 8 {a['twin_rate_16000_by_age_lt8']:.2f})")
i = res["init"]
summ = (f"initialization: 160m vs deduped rows by index at step 0 cosine {i['step0_by_index']['median']:.3f}; final rows keep {i['retention_160m']['median']:.3f}/{i['retention_deduped']['median']:.3f} cosine with their step-0 rows (block 0 {i['retention_160m']['by_block'][0]:.2f}, block 6 {i['retention_160m']['by_block'][6]:.2f}); final by index {i['final_by_index']['median']:.3f}; "
        f"end of training (rows only): final vs step-16000 words Jaccard by index {e['word_jaccard_by_index']:.2f}, context-set twins final->16000 {e['twin_final_to_16000']:.2f} and back {e['twin_16000_to_final']:.2f}, row cosine 16000->end {e['row_cos_16000_to_end_all']:.2f} (final words {e['row_cos_16000_to_end_final_words']:.2f}); "
        f"160m twin rate of final words {a['twin_rate_final_words']:.2f} vs step-16000 words {a['twin_rate_16000_words']:.2f}; age (checkpoints 1000-16000 as a word) of twinned vs untwinned final words {a['age_twinned']:.1f} vs {a['age_untwinned']:.1f}, age-0 share {a['share_final_words_age0']:.2f} | {time.time() - t0:.0f}s")
log(summ); record("e557b_init_and_age", res, summ)
