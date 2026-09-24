# 14. Self-description: the native vocabulary (S36-S39, S42-S44)

**Question.** How well do a model's own write directions describe its states, is this learned, and is it words or second-order geometry?
**Established.** WDD's sparse code is a re-description: 32-64 own words keep 90% of the loss at the middle depth, where the largest actual writes need 1024 or more (e391, e395). The advantage over a rotated copy of the dictionary is absent at initialisation in all five architectures and is learned (e395, e401). Early in Pythia's training it is a per-block second-order accent (0.82 of the gap at step 1000); words take over between steps 4000 and 16000, and at the end it is word-level in all five: per-block covA carries at most 0.33 of the gap, removing the lexicon leaves it, and 55-85% needs non-Gaussian state structure (e405, e426, e443, e439, e435). Session 44 removed two claims: the step-4000 "false friends" were mis-described sinks (e437), and Zipf-like usage is geometry (e435). The own words are not the best sparse vocabulary, a replacement model or sparse causal nodes (e399, e403, e392, e394).
**Start here:** e395, e399, e426, e443, e439 · **Sessions:** S36-S39, S42-S44 · **Scripts:** `scripts/e391_*.py` to `e443_*.py` (with `sd_common.py`)

## Experiments

| id | question | result | status | links |
| --- | --- | --- | --- | --- |
| e391 | Is WDD's code the few largest actual writes? | No: the largest writes recover 0.53-0.79 at k 64, 0.75-0.92 at 1024; 64 own words 0.91-0.99 (e388) | refuted | ← e388 · → e393 e395 e421 e436 |
| e392 | Can own-word codes replace every block at once? | No: errors compound (+2.05 to +5.32 nats at 64 atoms per block, +0.43 to +3.77 at 128); not via sinks (e392b) | refuted | ← e388 · → e392b e432 |
| e393 | Is WDD's edge over the largest writes only the refit? | Refit lifts the top-64 writes from 0.53-0.79 to 0.68-0.89, below WDD (0.91-0.99); 11-27% of WDD's MLP atoms are top writers | narrowed (support matters) | ← e388 e391 |
| e394 | Are WDD atoms sparse causal nodes? | No: effects spread over 13-16 of 32 atoms; linear attribution vs ablation Spearman 0.33-0.67 | refuted | ← e388 |
| e395 | Is self-description learned, and how long is it? | Own = rotated at step 0, ahead from step 256; 90% needs 32-64 own words, over 4096 actual writes at the end (e436: writes sparsest mid-training) | established | ← e388 e391 · → e396 e399 e401 e436 |
| e396 | Can one stage's words describe another's states? | Pythia: interchangeable from step 33000; later words read early states better than the reverse (accretion); step-4000 words below rotation (sinks, e437) | supported (Pythia) | ← e360 e395 · → e397 e398 e402 e431 |
| e397 | False friends, or an impoverished geometry? | Geometry ordinary (effective rank 848); step-4000 words -0.28 at k 4 on final states vs rotation 0.05. Were mis-described sinks (e437) | superseded by e437 | ← e396 · → e400 e404 e437 |
| e398 | Do seeds share a language up to shared tokens? | No (k 16): own 0.59, rotated 0.36, lexicon-translated 0.36 (lexicons not rotations); state-fitted maps 0.48 orthogonal, 0.66 linear | refuted | ← e396 · → e398b e445 |
| e398b | Is the linear map's gain only projection? | Mostly: random words through it recover 0.59, as own words do; translation adds 0.07 (0.11 orthogonal) | supported | ← e398 · → e445 |
| e399 | Is the advantage second moment, span, or words? | Pythia k 16: own 0.83 (step 1000), 0.60 (end); covA, mix8 near own early (0.78, 0.81), at rotation level late (0.33, 0.29 vs 0.30-0.32); covX beats own | established (e443 refines) | ← e395 e397 · → e405 e407 e426 e443 |
| e400 | Do Fisher-chosen words shorten the description? | No: 90% length unchanged; gap over rotation 0.04/0.09/0.24 at steps 1000/16000/end; M's Fisher per unit variance 0.09, not 0.017 (sinks, e437) | refuted (90% length) | ← e397 e399 · → e404 e408 e418 e437 |
| e401 | Is self-describability absent at initialisation everywhere? | Yes: own and rotated FVU agree within 0.004-0.014 at init (5 models, 3 depths); OLMo own beats rotation at every trained checkpoint | established | ← e395 · → e426 e447 |
| e402 | Where do false friends arise; does accretion hold? | Final words read step-256 states better than their own (0.97 vs 0.87); 6 cells below every rotation, sources 4000-8000: sinks (e437) | superseded by e437 (accretion stands) | ← e396 · → e405 e406 e410 e437 |
| e403 | Own words vs a learned SAE (GPT-2)? | SAE ahead at k 4-32 (0.79 vs 0.35 at k 4); own overtakes at k 64 (0.96 vs 0.93), more function per variance explained | supported (GPT-2) | ← e388 · → e408b |
| e404 | Are false friends mediated by the huge directions M? | With M exact, step-4000 words beat rotation (0.65 vs 0.56 at k 16); but M's 0.86 was one sink per sequence and the false friends sinks (e432, e437) | superseded by e437 | ← e397 e400 · → e406 e411 e419 e437 |
| e405 | When does the accent become words? | covA/mix8 carry 41-88%/45-134% of the gap at k 16 to step 8000, rotation level from 16000; per block 0.82 at 1000, 0.32 at 16000 (e443) | established (sharpened by e443) | ← e399 e402 · → e409 e426 e436 e443 |
| e406 | Do false friends and accretion hold in OLMo? | No false-friend cell; accretion to step 256000, but final words read early states worse than their own (0.61 vs 0.71) | narrowed (accretion) | ← e401 e402 e404 · → e410 e411 |
| e407 | Are readers word-level like writers? | No: writers' covA/mix8 shares fall 0.86/0.91 to -0.48/-0.09 (step 1000 to end); readers' stay second-order (1.29/1.48 at end) | refuted | ← e399 · → e412 |
| e408 | Do the weights alone know which directions matter? | Readers put 0.063 of trace on top-8 PCs holding 0.86 of variance (Pythia end; 0.86 was sinks, e432); as pursuit metric 0.43-0.74 of Fisher lift at k 4 | narrowed (pursuit metric) | ← e400 · → e408b e426 e432 |
| e408b | Can weight-only word selection approach the SAE? | No: read-strength and write-norm picks collapse onto token embeddings, below a random subset (0.37, 0.33 vs 0.46 at k 8, GPT-2) | refuted | ← e403 e408 |
| e409 | Do false friends and the accent shift hold at 70m, 160m? | No false-friend cell; accent becomes words at both (covA share 0.57, 0.51 at step 1000; 0.00, 0.04 at end) | supported (3 sizes) | ← e402 e405 |
| e410 | How are the intelligibility maps structured? | Accretion (null-relative) in 80% of pairs in Pythia, 100% in OLMo; symmetry 0.34 and 0.70 | supported | ← e402 e406 |
| e411 | Are false friends precursors of M's writers? | Top-100 words' share in final M: 0.03-0.05 to step 2000, 0.14 at 4000, 0.56 at 16000 (MLP rows); the false-friend account fits only sinks (e437) | narrowed (sink directions) | ← e404 e406 · → e437 |
| e412 | Are loss gradients described in the model's own words? | Pythia, over rotation: none at init; errors favour writers at step 1000 (0.059 vs 0.020), readers from 16000 (end 0.063 vs 0.019) | supported (Pythia) | ← e407 · → e426 e429 |
| e413 | Does the vocabulary cover in-context copying? | Own over rotation at k 16 on copied random tokens 0.40, 0.30 (step 1000, end), vs natural text 0.13, 0.29 | supported (Pythia) | ← e395 |
| e414 | How often is each native word used? | Zipf-like: slope -0.62 to -0.72 vs rotated -0.32 to -0.35 (Pythia, GPT-2); top words aim at M. Gaussian states reproduce the slope (e435) | narrowed (geometry, e435) | ← e400 e404 · → e422 e426 e435 e437 |
| e426 | Do the Pythia findings hold in all five models? | Final, all five: word-level (covA share -0.22 to 0.28); Zipf (geometry, e435); readers' trace on M 1-7x chance (M 10-26% at typical positions, e432); duality 3/5 (e429: 5/5) | established (word-level) | ← e405 e408 e412 e414 · → e429 e435 e437 e443 |
| e427 | Does self-description track capability (Pythia sizes)? | Final advantage 0.17 (70m) to 0.40 (1b) as loss falls 4.09 to 2.83 (rank corr -0.8); 1b and 1.4b have most of it by step 1000 | supported (two inversions) | ← e395 · → e447 |
| e428 | Is usage a free neuron-importance score? | Ablating top-used neurons costs more than magnitude, Taylor or weight-norm picks in SmolLM2, Pythia, Qwen-0.5B; not in GPT-2, OLMo (super weights) | mixed | ← e414 · → e428b |
| e428b | Does usage beat deviation and spread too? | In SmolLM2 and Qwen-0.5B; split in Pythia (spread ahead at N 64); not in GPT-2, OLMo | mixed | ← e428 |
| e429 | Does the duality hold at all depths, all readers? | With attention readers, states favour writers and errors readers at all three depths in all five; GPT-2's reversal was missing readers | established | ← e412 e426 |
| e430 | Is self-description a novelty or uncertainty signal? | No: random tokens least describable in four models, most in OLMo; shuffled text at or above natural in three; no negative link to loss | refuted | ← e395 |
| e431 | Does instruction tuning change the vocabulary? | No: median write-row cosine 0.997; each model's words describe the other's states as well as its own (within 0.02) | null | ← e396 · → e447 |
| e434 | Does the network keep its most-used words? | No: next-block retention is alike for most-used, never-used and rotated words (0.71-0.93); it follows variance (usage adds r 0.03-0.18) | refuted | ← e432 |
| e435 | Is the advantage just the states' covariance? | No: Gaussian states with the same covariance keep 15-45% of it (token means plus noise 31-68%), and reproduce the Zipf slope | refuted | ← e399 e414 e433 · → e439 |
| e436 | Are increments word-level before states? | Pythia step 4000: increment word-level (covA 0.13), state an accent (0.53); at the end less word-level than states in all five; writes sparsest mid-training (PR 374, 216, 322) | narrowed (step 4000) | ← e02 e399 e405 e426 · → e439 e443 |
| e439 | Is the native vocabulary the lexicon? | No: without token, position and block-0 MLP rows the gap over rotation stays at every checkpoint (final +0.23 to +0.57); current token named at 0-4% (OLMo 23%) | refuted | ← e433 e435 e436 · → e440 e448 |
| e440 | Does what context adds have native words? | Yes: non-lexicon words beat rotation on the state minus its token mean by +0.16 to +0.52, word-level (covA share -0.23 to 0.32), all five | established | ← e439 · → e448 |
| e442 | Does the advantage live on the broadcast path? | Both paths: self +0.14 to +0.52, broadcast +0.17 to +0.41 over rotation; broadcast larger in three of five | narrowed | ← e438 |
| e443 | Is the vocabulary a union of per-block accents? | Not at the end (per-block covA -0.14 to 0.33 of the gap, per-block mixtures 0.12-0.52); early yes: Pythia 0.82 at step 1000, 0.32 at 16000, 0.09 at end | refuted (at end) | ← e399 e405 e436 |

## How the results flow

- **What the code is.** `e388 → e391 → e393 → e392 → e394 → e395`: a re-description, not the largest writes; it compounds as a replacement model (not via sinks, e392b) and its atoms are not causal nodes. Hence the vocabulary view.
- **Control 1, rotation** (same Gram, no provenance). `e395 → e401 → e427`: own words win only after training, in five architectures, more in larger models; the win survived every later control and the sink check (e437).
- **Control 2, pooled covA and mix8** (session-38 review). `e399 → e405 → e407 → e409 → e426`: an accent early (0.62/0.85 at step 1000), word-level from step 16000 at 70m-410m and at the end in all five; readers stay second-order. covX and an SAE (e403) beat own words: "best vocabulary" fell, "word-level late" survived.
- **Control 3, per-block covA and mix8.** `e436 → e443`: the early accent is per block (0.82 at step 1000, 0.32 at 16000); words appear between 4000 and 16000, first in the block's own increment; at the end per-block covA carries at most 0.33. Late words survived; the early account was revised. e436 also finds writes sparsest mid-training, reconciling phase 1's "sparser" with e395's "denser".
- **Control 4, lexicon and state surrogates.** `e435 → e439 → e440 → e442`: Gaussian states keep 15-45%; the gap survives lexicon removal at every checkpoint; context is word-level; both paths carry it. "Covariance" and "lexicon" fell.
- **False friends.** `e396 → e397 → e400 → e402 → e404 → e406, e409 → e411 → e437`: step-4000 words below rotation on late Pythia-410m states: not geometry, deeper under Fisher, a 4000-8000 window, "mediated by M", none in OLMo or 70m/160m. e432/e437: M's 0.86 was one `\n\n\n` sink per sequence; with sinks exact the step-4000 words beat rotation (+0.15 vs +0.08 at k 4).
- **Usage.** `e414 → e426 → e435`; `e428 → e428b`; `e434`: Zipf-like usage is geometry; usage ranks neurons well in three of five; retention follows variance, not usage.
- **Duality and shared languages.** `e407 → e412 → e426 → e429`: errors in writers' words early, readers' from step 16000; 5/5 once attention readers count. `e396 → e402 → e406 → e410`: accretion, not universal (OLMo). `e398 → e398b`: seeds private; a fitted map's gain is mostly projection.

## Links to other areas

- [10 Sinks](10_sinks_huge_directions.md): e432 finds Pythia's and OLMo's M variance shares were sink positions; e437 reruns e397, e400, e404, e414, e426 with sinks exact; e392b clears e392; e433 feeds e435, e439; e438 feeds e442.
- [13 Readers](13_readers_readouts.md): e388 (own words as an SAE) is the baseline for e391-e395; e389 and e399 rule out token leakage.
- [15 Workspace](15_workspace_lenses.md): e418 finds self-description at 7B (0.79 vs rotation 0.26), without e400's Fisher gain; e419 repeats e404's test; e421 builds on e391, e422 on e414.
- [16 Vision](16_vision_round.md): e445 builds on e398b, e447 on e395 and e427, e448 on e439-e440; e444 applies covA to grokking.
- [03 Increments](03_increments_depth_targets.md), [12 Drift](12_circuits_coselection_drift.md): e02 motivates e436; e360's row drift feeds e396.
