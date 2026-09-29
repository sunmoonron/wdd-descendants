# 37. The instrument placed: the strongest existing method invited, and the detector benchmark (S105)

**Question.** After the object was separated (areas 35-36), is anything WDD shows beyond what existing methods show, and is its one practical lead, the label-free detector of generalisation, better than the classic signals? Two tests, both designed to let WDD lose.

**Established.**
- The producer-consumer decomposition is not the native rows'. An autoencoder's features on the same states have coalitions as coherent as the words' (0.692 against 0.672, random directions 0.154) and as necessary (the top 128 producers zeroed drop the feature's activation at its class by 0.92, the word's projection by 0.77); a feature and its nearest word share few producers (Jaccard 0.07) (e581).
- The native excess does not predict generalisation ahead of the classic signals: over 20 toy trajectories the best WDD statistic reaches AUC 0.824 a thousand steps ahead against 0.961 for the best baseline (emb_fourier_top5_d) (e580b).
- The placement: WDD's distinct element is parameter provenance of the association (the row behind a direction, which can be transplanted, frozen and followed through training), and that provenance buys the developmental questions, not the descriptive or the detector ones.

**Start here:** e581, e580b · **Sessions:** S105 · **Scripts:** `scripts/e580_detector_runs.py`, `e580b_detector_analysis.py`, `e581_sae_vs_words.py`

## Experiments

| id | question | result | status | links |
| --- | --- | --- | --- | --- |
| e580 | Twenty toy trajectories (five knobs times four seeds, 10000 steps) with the WDD statistics logged every 250 steps beside training loss, weight norms, the states' participation ratio and top-8 share, ReLU density and the embedding's Fourier share | 13 of 20 runs generalise; every run measured 41 times; the runs' final excess ranges from 1.15 to 2.62 | tool (the trajectories for e580b) | ← e567 · → 23 |
| e580b | Which label-free statistic at t predicts generalisation at t + delta (250-2000 steps), each alone, leave-one-run-out, level and change? | Delta 1000: native excess 0.809 (change 0.824), advantage 0.226, training loss 0.053, weight norm 0.691 (change 0.261), state PR 0.484, Fourier share 0.829 (change 0.961); best emb_fourier_top5_d 0.961; at delta 250 excess 0.951 vs weight norm 0.803 | refuted (the classic signals are not beaten) | ← e579 e444 · → 23 |
| e581 | The public TopK autoencoder for Pythia-160m at block 6 on the same states as the words: do its features have producer coalitions, are the producers necessary, and do a feature and its nearest word share producers? | Coherence words 0.672, features 0.692, random directions 0.154; effective producers 1837 / 3298 / 6145; necessity drop 0.77 / 0.92 (random 0.01 / 0.00); feature-word producer Jaccard 0.07 at cosine 0.25 | refuted (the coalition is any spoken direction's; the instrument's element is the parameter, not the decomposition) | ← e549 e570 · → 21 31 |

## How the results flow

- `e570, e545 → e581`: the coalition, measured for the words, then for an autoencoder's features by the same attribution: the same coalitions, different producers.
- `e579 → e580 → e580b`: the detector that never fired falsely, then twenty trajectories with the classic baselines logged beside it, then the leave-one-run-out benchmark.
- `e577, e578 → this area`: what the parameter carries (the association, not the computation) is what the autoencoder cannot reach and what WDD's provenance is for.

## Links to other areas

- [21 The provenance layer under SAEs](21_sae_bridge.md), [32 The necessity tests](32_necessity.md): the earlier comparisons with learned dictionaries (e549: a feature sees a word's association by half).
- [31 The coalitions](31_coalitions.md): the coalition as first measured.
- [36 The object](36_object.md), [35 Theory extraction](35_mechanism_map.md): the object and the map this area places.
- [23 The onset of wordhood in training](23_training_sweep.md): the toy and its clocks.
