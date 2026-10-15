# 38. Watching words being born, against a learned dictionary (S106)

**Question.** After session 105 one claim was left standing: that WDD keeps parameter identity across training for free, where a learned dictionary must establish it. The test is developmental: for every eventual word, when does its class exist, when does a per-checkpoint autoencoder have a feature for it, when is its row recruited, when does its coalition cohere, and when, if ever, does the autoencoder's feature point at the row? The autoencoder is allowed to win.

**Established.**
- The class is there from the start: every tracked class coheres in the cloud at step 1000, the rows are recruited at step 7.0 thousand at the median, the coalition coheres 5.0 checkpoints before recruitment for 0.84 of 241 tracked words (e584).
- The learned dictionary (4096 latents per checkpoint) has a persistent feature for 0.31 of the classes, sees those before the row (0.61), and does not date the parameter: at recruitment its decoder has cosine 0.11 with the row, and it points at the row 7.0 checkpoints after recruitment, for 33 of 241 words (e584).
- The claim, as sharp as the record allows: a native, parameter-indexed coordinate system for how representations form and how responsibility for them migrates through training.

**Start here:** e584 · **Sessions:** S106, S152, S153 · **Scripts:** `scripts/e582_longitudinal_cache.py`, `e583_sae_per_checkpoint.py`, `e584_longitudinal.py`, `e672_routes.py`, `e674_routes_rep.py`, `e676_center_out.py`

## Experiments

| id | question | result | status | links |
| --- | --- | --- | --- | --- |
| e582 | The longitudinal cache: Pythia-410m block 12 at steps 1000-16000 on 24 sequences of 512 tokens (states, sink masks, the rows of blocks 0-12) | 12219 positions kept at every checkpoint; sixteen checkpoints | tool (the cache for e583, e584) | ← e524 · → 23 |
| e583 | A TopK autoencoder (4096 latents, 32 active) trained on each checkpoint's states: the per-checkpoint learned dictionary without parameter identity | FVU 0.05-0.09, 4017-4096 latents alive | tool (the control for e584) | ← e549 · → 21 |
| e584 | For 241 eventual words, five events dated over sixteen checkpoints: the class present in the cloud (row-free), the checkpoint autoencoder's feature for it, the row recruited, the coalition coherent, the autoencoder's decoder pointing at the row; which comes first? | Every class present from step 1000; rows recruited at step 7.0 thousand at the median; a persistent autoencoder feature for 0.31 of classes, before the row for 0.61 of those; coalition -> row 5.0 (0.84); the autoencoder's decoder at cosine 0.11 with the row at recruitment, pointing at it 7.0 checkpoints later for 33 words | narrowed (the class is there from the start, the coalition precedes the row; the learned dictionary sees a minority of classes and never the parameter) | ← e546 e571 e581 · → 31 37 |

| e672 | Cammarata's recursive generalization over Pythia-160m's checkpoints: a neuron's top firings grouped into routes by exact upstream attribution (k-means against a unimodal null; the essay's own grouping by the main upstream neuron); are offshoot firings less costly when the neuron is removed, and do they become top firings after the core's? | in Pythia-160m block 6, 83 of 300 neurons have two distinct routes; removing a neuron costs its offshoot firings as much as its core firings (less in 0.45; by the essay's grouping offshoots cost more, p 0.045), and distinct routes form in no order (core first in 0.53, p 0.33); inside one route the bulk does become top firings first (0.60 of 217 neurons, found after the fact) (e672) | refuted (offshoots cost as much; distinct routes form in no order; a bulk-first gradient inside one route found after the fact) | ← e584, e595 |

| e674 | e672 replicated in eight runs (Pythia-160m blocks 3 and 9, Pythia-410m blocks 6, 12 and 18, OLMo-1B blocks 4, 8 and 12), with Gemini's SwiGLU leak filter: does the larger part of a one-route neuron's top firings become top firings first, do distinct routes form in order, do offshoots cost less? | in eight runs of Pythia-160m, Pythia-410m and OLMo-1B the bulk of a neuron's top firings became top firings before the rest in one-route (0.58) and two-route neurons (core 0.56, p 0.0017) alike, so e672's single-block 'no order' was underpowered; offshoot firings cost more, not less, when the neuron is removed (pooled p 0.0017); the SwiGLU leak is not the offshoots (the more gate-closed route in 39 of 74; 40 of 219 two-route neurons collapse on gate-open firings) (e674) | narrowed (the bulk-first order replicates in six of eight runs and holds for distinct routes too, pooled; offshoots are not cheaper; the leak is not the offshoots) | ← e672 |

| e676 | The control e674's one positive result needed: does the larger group's lead survive the token's and bigram's corpus frequency and position, and do firings nearer the centre of a neuron's top-firing attributions become top firings first (no k-means)? | the bulk-first lead survives token and bigram frequency and position in all six Pythia runs and two of three OLMo runs, and the firings nearest the centre of a neuron's input attributions became top firings first in 0.63 to 0.80 of neurons in all nine runs (p below 0.0001 in each): growth from a local origin, measured (e676) | supported (the lead survives frequency and position in eight of nine runs; growth from the centre in all nine) | ← e674 |

## How the results flow

- `e672 → e674`: the route measurement repeated on eight blocks of three models, with the SwiGLU leak filter.

- `e674 → e676`: the lead controlled for frequency and position, and growth from the centre measured without k-means.

- `e584, e595 → e672`: a neuron's routes dated over training, against Cammarata's recursive generalization.

- `e524 → e582 → e583 → e584`: the earlier cache's recipe on three times the text, an autoencoder per checkpoint, then the histories.
- `e546, e571 → e584`: the coalition's precedence and the row's turn, now dated against a row-free measure of the class and against the learned dictionary.
- `e581 → e584`: at a fixed checkpoint the dictionaries see the same structure; across checkpoints one of them names the parameter.

## Links to other areas

- [37 The instrument placed](37_instrument_placed.md): the comparison at a fixed checkpoint.
- [31 The coalitions](31_coalitions.md), [25 Training's hand](25_training_motion.md), [23 The onset of wordhood in training](23_training_sweep.md): the developmental results this area dates.
- [21 The provenance layer under SAEs](21_sae_bridge.md): the earlier bridge to learned dictionaries.
