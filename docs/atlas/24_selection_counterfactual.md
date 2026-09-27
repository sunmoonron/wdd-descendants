# 24. The boundary's carrier: the counterfactual chord and the criterion across texts (S73)

**Question.** e516b found that a row's largest projection over the extreme-value floor predicts its entry into the vocabulary at the next checkpoint. What carries that projection, the position's co-written chord or the rest of the state, and does the criterion hold across texts and models?

**Established.**
- The chord does not carry it: with the position's 64 largest writes replaced by random rows of the same blocks, permuted among themselves, or removed, at every position and with the floor recomputed, the prospective prediction of entry at the next checkpoint is within 0.04 of the real state's (Pythia block 12, origins 2000-32000: real 0.86-0.97, fake 0.85-0.95, permuted 0.86-0.97, none 0.86-0.96), and removing the row's own write costs 0.01-0.05; the projection that predicts wordhood is carried by the embeddings, attention's output and the crowd of small writes (e517, e517b).
- The entrants rise because their projections rise: from 0.135 to 0.215 of the state norm over training, against a floor that moves from 0.159 to 0.145; the projection's share of the rise is 0.80-1.14 in log terms at every origin (e517b).
- Across texts at the end of training the criterion picks a disjoint text set's new words at 0.86-0.91 in GPT-2, 0.95-0.97 in Pythia and 0.90-0.93 in OLMo, level with usage and ahead of magnitude except in OLMo, where the words are the large rows and magnitude nearly matches (e518).

**Start here:** e517b, e518 · **Sessions:** S73 · **Scripts:** `scripts/e517_chord_swap.py`, `e517b_chord_swap_prediction.py`, `e518_cross_text_selection.py`

## Experiments

| id | question | result | status | links |
| --- | --- | --- | --- | --- |
| e517 | Every row's largest projection over the floor under counterfactual states at thirteen Pythia checkpoints (real; fake chord; permuted chord; no chord; own write removed), the records e517b reads | At the end 9.5% of rows clear the floor somewhere under the real state at block 12, 10.6% under the fake chord, 8.9% permuted, 6.0% with no chord, 7.9% with the own write removed; the chord is 0.6 of the state | tool | → e517b |
| e517b | Does destroying the position's chord destroy the prospective prediction of entry (the relayed chain: chord makes the maximum, the maximum makes the word)? | No. At block 12, entry at the next checkpoint from origins 2000-32000: real 0.86, 0.88, 0.91, 0.95, 0.97, 0.97; fake chord 0.85, 0.89, 0.89, 0.94, 0.95, 0.93; permuted 0.86, 0.88, 0.90, 0.95, 0.97, 0.97; no chord 0.86, 0.88, 0.90, 0.93, 0.96, 0.96; own write removed 0.85-0.95; magnitude 0.55-0.66; block 6 the same. The entrants' ratio rises from 0.9-1.4 to 1.1-1.6 while non-entrants stay at 0.8; their projection rises from 0.135 to 0.215 of the state norm over training and the floor from 0.159 to 0.145, the projection's share of the rise 0.80-1.14 | refuted (the chord); the carrier is the state outside its largest writes | ← e516b e505 e511 · → 22 |
| e518 | Does the selection criterion hold across texts, at the end of training, in GPT-2, Pythia and OLMo? | Yes: quantities on one set of eight sequences, the word set on a disjoint eight; the largest projection over the floor picks the new words at 0.86/0.89/0.91 (GPT-2, blocks 3/6/9), 0.97/0.97/0.95 (Pythia 6/12/18), 0.90/0.92/0.93 (OLMo 4/8/12); usage 0.85-0.96, magnitude 0.67-0.93 (0.85-0.93 in OLMo), selectivity 0.47-0.60, kurtosis 0.58-0.69; the two word sets overlap by 0.50-0.74 | established (three models) | ← e516b · → 23 |

## How the results flow

- `e516b → e517, e517b`. The criterion's carrier: not the chord of e505-e511, which the counterfactual removes without loss, but the state's other content.
- `e516b → e518`. The criterion across texts and models, the substitute for checkpoints where there are none.

## Links to other areas

- [23 The onset of wordhood in training](23_training_sweep.md): the prospective test this area takes apart (e516b) and the turnover it explains (e515b).
- [22 The model tested](22_model_tested.md): the chord (e505-e511), which makes the maximum at a position but does not decide which rows become words.
- [05 Cancellation, contraction](05_cancellation_contraction.md) and [06 Descendants I](06_descendants_fate_transport.md): provenance persisting because other components write the same directions is the candidate carrier of the predictive projection.
