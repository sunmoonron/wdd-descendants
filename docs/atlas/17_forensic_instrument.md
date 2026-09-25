# 17. WDD as a forensic instrument: compression, model diffs, steering, generation (S47)

**Question.** Beyond describing states, can WDD's provenance-labelled words serve as a forensic tool? The candidate uses are: telling what compression damaged, what a fine-tune changed, whether an intervention hit its target, and how content persists across generated tokens.

**Established.**
- Self-description does not detect compression damage. Own words keep their advantage over rotation even when the model is broken, because the states are still built from the model's rows (e456).
- WDD names what a fine-tune changed: a few words carry much of the usage change. But it does not compress the change; on chat text the change is low-rank and dense (e457).
- One concept word, injected at natural size across depth, moves 35-58% of translations. That is about as efficient per unit of displacement as a dense steering vector, and more efficient at 7B. This also corrected e455, whose swaps were 6-8 times natural size (e458).
- A downstream WDD checksum confirms that an intervention reached the intended word, but it predicts success no better than the intervention's size (e458).
- Native words persist across generated tokens only modestly more than rotated words (e459).

**Start here:** e458, e457, e456 · **Sessions:** S47 · **Scripts:** `scripts/e456_compression_autopsy.py`, `e457_model_diff.py`, `e458_steering_checksum.py`, `e459_generation_lineage.py`

## Experiments

| id | question | result | status | links |
| --- | --- | --- | --- | --- |
| e456 | Does self-description detect compression damage? | No: GPT-2 at 4 bits loses 3.9 nats but keeps 96% of the advantage over rotation; own words stay 0.42-0.56 unexplained against 0.64-0.72 rotated at every level | refuted (not a damage detector) | ← e395 e401 |
| e457 | Can WDD describe what a fine-tune changed, write by write? | Names it: top 1% of words carry 38-59% of the chat usage change. Does not compress it: the chat change's own PCA leaves 0.27-0.32, native words 0.57-0.65 | narrowed (names, does not compress) | ← e431 |
| e458 | Native word against dense steering at natural size; does a WDD checksum predict success? | Native, 0.7x natural: 35-58% moved; dense: 82-100% with 4-6x the displacement; per displacement native equal or better. Checksum AUC 0.70-0.92, displacement 0.83-0.94 | mixed (corrects e455's size) | ← e455 · → VISION item 7 |
| e459 | Do native words persist across generated tokens? | Modestly: 1.3-1.8x rotated words' reuse at lags 4-32 in Qwen and GPT-2's natural text; no regeneration beyond usage rate; sampled text more self-similar | narrowed (modest persistence) | ← e440 e413 |

## How the results flow

- `e395 → e456`. Self-description is learned and absent at initialisation (e395, e401). Yet it survives compression that breaks the function, so it measures that states are made of the model's own rows, not that the model works.
- `e431 → e457`. Instruct tuning barely moves the rows (e431). It moves the states, above all on chat, where a few words carry much of the usage change. The change itself is dense and low-rank, so WDD is a naming tool for a model diff, not a compressor of it.
- `e455 → e458`. Adding the target word at every block accumulates. Measured at the middle block, e455 injected 6-8 times natural. At natural scale the native handle is weaker but about as efficient per unit of displacement as a dense steering vector. The WDD checksum sees the intended change but adds nothing over its size.
- `e440 → e459`. Native words carry context (e440) and persist a little longer than random directions across positions, with no special regeneration.

## Links to other areas

- [16 Vision round](16_vision_round.md): e458 corrects the size of e455's concept-word swaps; e457 extends e431's instruct comparison from rows to states.
- [14 Native vocabulary](14_native_vocabulary.md): e456 shows the self-description advantage (e395, e401) is robust to compression, and e459 tests its words over time rather than depth.
- [05 Cancellation](05_cancellation_contraction.md): e458's absorbed increments (at s = 1 only 0.29-0.36 of the target's typical coefficient is present at the middle block) look like the per-block contraction of e194 and e210 acting on an injected direction.
