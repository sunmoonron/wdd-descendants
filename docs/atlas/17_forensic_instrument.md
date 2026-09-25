# 17. WDD beyond description: forensics, steering, learning, grammar, self-consistency, interchange, life cycle (S47-S54)

**Question.** Beyond describing states, can WDD's provenance-labelled words serve as a forensic tool? The candidate uses are: telling what compression damaged, what a fine-tune changed, whether an intervention hit its target, and how content persists across generated tokens.

**Established.**
- Self-description does not detect compression damage. Own words keep their advantage over rotation even when the model is broken, because the states are still built from the model's rows (e456).
- WDD names what a fine-tune changed: a few words carry much of the usage change. But it does not compress the change; on chat text the change is low-rank and dense (e457).
- One concept word, injected at natural size across depth, moves 35-58% of translations. That is about as efficient per unit of displacement as a dense steering vector, and more efficient at 7B. This also corrected e455, whose swaps were 6-8 times natural size (e458).
- A downstream WDD checksum confirms that an intervention reached the intended word, but it predicts success no better than the intervention's size (e458).
- Native words persist across generated tokens only modestly more than rotated words (e459).
- Near-identical states differ in their futures mainly through context, and the WDD ledger adds nothing to the cosine in predicting the vector's effect. The write history is not an input (e460).
- A forged write-sized vector is barely detectable at its block, no better than a covariance detector, and invisible a few blocks later (e461).
- WDD is a forward coordinate, not a learning one. One batch's gradient on a word is unrelated to the word and does not predict its change between checkpoints; words form by rotation until about step 16000, then shrink in place; the most used words get no more gradient than others (e462).
- The native words have no grammar. A word's noun or verb role is carried by which words describe it, not by inflecting shared words (e463). A word written where it never fires is damped exactly as where it fires (e464).
- A description is not a self-sufficient state. After a 16-word splice the network does not regrow what the words left out, although it damps a random error of the same size (e465).
- The Kalman gap is large: 64 extra remainder directions still miss 7-11% of the loss (e466). The remainder holds the same kinds of information as the description (e467).
- Native words are good interchange coordinates. Across blocks, one to four native words carry a translated noun between contexts (Qwen 67-96% of answers switched), far better than rotated words or the task's principal directions at equal size. At a single block the principal directions win (e469). They work only across the band of blocks where the noun is written (Qwen 0-6, SmolLM2 about 6-12), and the carrier words turn over as it is re-written (e470).
- Over training, becoming a word and becoming important are separate: across eight Pythia checkpoints neither leads the other, and words that fall out of use keep their importance (e471).

**Start here:** e469, e458, e457, e460, e462 · **Sessions:** S47-S54 · **Scripts:** `scripts/e456_compression_autopsy.py`, `e457_model_diff.py`, `e458_steering_checksum.py`, `e459_generation_lineage.py`, `e460_markov_pairs.py`, `e461_forgery_forensics.py`, `e462_gradient_writers.py`, `e463_native_accents.py`, `e464_illegal_words.py`, `e465_description_healing.py`, `e466_kalman_gap.py` to `e471_word_lifecycle.py`

## Experiments

| id | question | result | status | links |
| --- | --- | --- | --- | --- |
| e456 | Does self-description detect compression damage? | No: GPT-2 at 4 bits loses 3.9 nats but keeps 96% of the advantage over rotation; own words stay 0.42-0.56 unexplained against 0.64-0.72 rotated at every level | refuted (not a damage detector) | ← e395 e401 |
| e457 | Can WDD describe what a fine-tune changed, write by write? | Names it: top 1% of words carry 38-59% of the chat usage change. Does not compress it: the chat change's own PCA leaves 0.27-0.32, native words 0.57-0.65 | narrowed (names, does not compress) | ← e431 |
| e458 | Native word against dense steering at natural size; does a WDD checksum predict success? | Native, 0.7x natural: 35-58% moved; dense: 82-100% with 4-6x the displacement; per displacement native equal or better. Checksum AUC 0.70-0.92, displacement 0.83-0.94 | mixed (corrects e455's size) | ← e455 · → VISION item 7 |
| e459 | Do native words persist across generated tokens? | Modestly: 1.3-1.8x rotated words' reuse at lags 4-32 in Qwen and GPT-2's natural text; no regeneration beyond usage rate; sampled text more self-similar | narrowed (modest persistence) | ← e440 e413 |
| e460 | Do near-identical states with different ledgers have different futures? | Different futures come from context: pairs (cos 0.95-0.998, always the same token) differ by KL 0.26-0.40, vector part 15-24%; ledger distance adds nothing (partial rho -0.07, -0.10) | refuted (history is not an input) | ← e131 e445 |
| e461 | Can WDD detect and locate a forged write-sized vector? | Weakly: AUC 0.60-0.64 at the forged block (Mahalanobis 0.62-0.75), correct block 0.14-0.35 (chance 0.06-0.14), chance a few blocks later | refuted (not an authenticity checker) | ← e131 e189 e195 |
| e462 | How does a gradient update reshape a native word, over training? | Per batch: along-row share 0.5-0.8x chance, sign a coin flip, no prediction of the net change (cos 0.000). Net: rows rotate and grow to step 16000, then shrink in place; usage vs gradient rho -0.63 to -0.04 | refuted (not a learning coordinate) | ← e443 e412 e429 e81 |
| e463 | Is a word's grammatical role carried by inflecting shared native words? | No: role decoded from role-specific words at 0.94-0.97, from shared words' coefficients 0.61 (no sign flips); rotated words decode it as well (0.88-0.89) | refuted (word choice, not inflection) | ← e146 e162 e440 |
| e464 | Is a native word written where it never fires corrected? | No: survival after 2 blocks 0.60 vs 0.64 (GPT-2), 0.48 vs 0.47 (Qwen); energy and re-description the same; generic contraction | refuted (no contextual syntax) | ← e214 e216 e246 |
| e465 | Does the network regrow what a 16-word description leaves out? | No: divergence stays 0.67-0.69 of the state at +4 (1.3-2x the omission); a same-size random error is damped to 0.49 and keeps 0.92-0.96 of the loss | refuted (omission is functional) | ← e388 e392 e221 |
| e466 | How many extra dimensions make a 16-word description dynamically sufficient? | Many: with 64 remainder directions loss recovered is 0.89 (GPT-2) and 0.93 (Qwen), never 95%; native plus remainder stays above rotated and PCA at every size | refuted (the gap is large) | ← e465 e388 e450 |
| e467 | What does a 16-word description leave out? | The same kinds of information it keeps: current, previous and next token and position are readable from both halves; the Qwen description gives the current token better than the whole state | refuted (no special remainder) | ← e465 e439 |
| e468 | Does a description keep what later tokens read from a position? | Later tokens barely depend on one middle state (mean-state KL 0.02-0.15). Native is best for the position itself; for later tokens best in Qwen, tied with PCA in GPT-2 | mixed | ← e465 e460 |
| e469 | Are native words interchange coordinates for a high-level variable? | Across blocks yes: 1-4 native words per block switch 67-96% of Qwen translations (SmolLM2 22-42%) against rotated 0-36%, PCA 8-66%. At one block PCA wins; whole-state patching there 78-99% | supported (across blocks; corrects e451's reading) | ← e455 e458 e460 |
| e470 | Where, and through which words, do native handles carry a variable? | In a band where it is written: Qwen native-4 over blocks 0-4 switches 85%, from block 8 on 15% (whole state 100%); carriers overlap 0.41-0.47 across blocks; global PCA as good as local | supported (a writing window) | ← e469 |
| e471 | Does a neuron become important before it becomes a native word, and does its function die with its usage? | Neither: lagged Spearman -0.01 and +0.06, co-move +0.04; dying words keep their importance; words in use fall from 14,115 to 9,435 after step 2000 (importance noisy) | refuted (independent processes) | ← e59 e462 e443 |

## How the results flow

- `e395 → e456`. Self-description is learned and absent at initialisation (e395, e401). Yet it survives compression that breaks the function, so it measures that states are made of the model's own rows, not that the model works.
- `e431 → e457`. Instruct tuning barely moves the rows (e431). It moves the states, above all on chat, where a few words carry much of the usage change. The change itself is dense and low-rank, so WDD is a naming tool for a model diff, not a compressor of it.
- `e455 → e458`. Adding the target word at every block accumulates. Measured at the middle block, e455 injected 6-8 times natural. At natural scale the native handle is weaker but about as efficient per unit of displacement as a dense steering vector. The WDD checksum sees the intended change but adds nothing over its size.
- `e440 → e459`. Native words carry context (e440) and persist a little longer than random directions across positions, with no special regeneration.
- `e131 → e460, e461`. Provenance is lost when writes accumulate (e131). A state's future depends on its vector and its context, not on its history (e460). A single forged write is lost in its block's other writes (e461).
- `e81, e443 → e462`. Rows keep rotating until late (e81), and words appear between steps 4000 and 16000 (e443). The rotation is accumulated drift: per batch the gradient on a word is noise relative to it, and after step 16000 the words mostly shrink in place.
- `e146, e440 → e463` and `e214 → e464`. A token's description mixes words for the token and words for its context (e146, e440). Roles are a matter of which context words appear, not an inflection of shared words. The contraction is direction-independent (e214), and it is also indifferent to whether a word belongs in its context.
- `e388, e221 → e465`. A 16-word description keeps much of the loss (e388), and random perturbations keep their energy while scattering (e221). What the description omits is not like a random perturbation: it persists as a functional deficit that later blocks do not re-derive.
- `e455, e458 → e469`. A concept word injected at every block accumulates (e458). An idempotent interchange that sets a few native words to the source's values at each block carries the noun almost as well as patching the whole state. It does so far better than rotated words or the task's principal directions at equal size, so the native words are the model's own interchange coordinates.

## Links to other areas

- [16 Vision round](16_vision_round.md): e458 corrects the size of e455's concept-word swaps; e457 extends e431's instruct comparison from rows to states.
- [14 Native vocabulary](14_native_vocabulary.md): e456 shows the self-description advantage (e395, e401) is robust to compression, and e459 tests its words over time rather than depth.
- [05 Cancellation](05_cancellation_contraction.md): e458's absorbed increments (at s = 1 only 0.29-0.36 of the target's typical coefficient is present at the middle block) look like the per-block contraction of e194 and e210 acting on an injected direction.
- [03 Increments](03_increments_depth_targets.md): e461 reads each block's increment, as the increment pipeline does (e02, e44), to look for a forged write.
- [12 Circuits, co-selection, drift](12_circuits_coselection_drift.md): e462 splits e81's atom drift into early rotation and late shrinkage, and finds single-batch gradients unrelated to it.
