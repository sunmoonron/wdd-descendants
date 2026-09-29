# 38. Watching words being born, against a learned dictionary (S106)

**Question.** After session 105 one claim was left standing: that WDD keeps parameter identity across training for free, where a learned dictionary must establish it. The test is developmental: for every eventual word, when does its class exist, when does a per-checkpoint autoencoder have a feature for it, when is its row recruited, when does its coalition cohere, and when, if ever, does the autoencoder's feature point at the row? The autoencoder is allowed to win.

**Established.**
- The class is there from the start: every tracked class coheres in the cloud at step 1000, the rows are recruited at step 7.0 thousand at the median, the coalition coheres 5.0 checkpoints before recruitment for 0.84 of 241 tracked words (e584).
- The learned dictionary (4096 latents per checkpoint) has a persistent feature for 0.31 of the classes, sees those before the row (0.61), and does not date the parameter: at recruitment its decoder has cosine 0.11 with the row, and it points at the row 7.0 checkpoints after recruitment, for 33 of 241 words (e584).
- The claim, as sharp as the record allows: a native, parameter-indexed coordinate system for how representations form and how responsibility for them migrates through training.

**Start here:** e584 · **Sessions:** S106 · **Scripts:** `scripts/e582_longitudinal_cache.py`, `e583_sae_per_checkpoint.py`, `e584_longitudinal.py`

## Experiments

| id | question | result | status | links |
| --- | --- | --- | --- | --- |
| e582 | The longitudinal cache: Pythia-410m block 12 at steps 1000-16000 on 24 sequences of 512 tokens (states, sink masks, the rows of blocks 0-12) | 12219 positions kept at every checkpoint; sixteen checkpoints | tool (the cache for e583, e584) | ← e524 · → 23 |
| e583 | A TopK autoencoder (4096 latents, 32 active) trained on each checkpoint's states: the per-checkpoint learned dictionary without parameter identity | FVU 0.05-0.09, 4017-4096 latents alive | tool (the control for e584) | ← e549 · → 21 |
| e584 | For 241 eventual words, five events dated over sixteen checkpoints: the class present in the cloud (row-free), the checkpoint autoencoder's feature for it, the row recruited, the coalition coherent, the autoencoder's decoder pointing at the row; which comes first? | Every class present from step 1000; rows recruited at step 7.0 thousand at the median; a persistent autoencoder feature for 0.31 of classes, before the row for 0.61 of those; coalition -> row 5.0 (0.84); the autoencoder's decoder at cosine 0.11 with the row at recruitment, pointing at it 7.0 checkpoints later for 33 words | narrowed (the class is there from the start, the coalition precedes the row; the learned dictionary sees a minority of classes and never the parameter) | ← e546 e571 e581 · → 31 37 |

## How the results flow

- `e524 → e582 → e583 → e584`: the earlier cache's recipe on three times the text, an autoencoder per checkpoint, then the histories.
- `e546, e571 → e584`: the coalition's precedence and the row's turn, now dated against a row-free measure of the class and against the learned dictionary.
- `e581 → e584`: at a fixed checkpoint the dictionaries see the same structure; across checkpoints one of them names the parameter.

## Links to other areas

- [37 The instrument placed](37_instrument_placed.md): the comparison at a fixed checkpoint.
- [31 The coalitions](31_coalitions.md), [25 Training's hand](25_training_motion.md), [23 The onset of wordhood in training](23_training_sweep.md): the developmental results this area dates.
- [21 The provenance layer under SAEs](21_sae_bridge.md): the earlier bridge to learned dictionaries.
