# 32. The necessity tests: what the neuron's own statistics and a feature dictionary recover of the row-context association (S97-S98)

**Question.** After sessions 94-96 a native word is an association between an MLP output row and a class of contexts, written there by a private coalition and preceded by the neuron's selectivity. Is WDD then a roundabout way of measuring something simpler, the selectivity of a neuron, or something an activation-side feature reader would find anyway?

**Established.**
- WDD is not a proxy for neuron selectivity: six ordinary statistics of a neuron and its row predict entry four intervals ahead at 0.74 out of sample, S and breadth at 0.88, the activation side added to WDD +0.001 and WDD added to it +0.142; S and breadth lead at every horizon (0.83 and 0.85 at eight intervals against at most 0.64); a word's extreme contexts are the coalition's, where the neuron fires selectively but not most (the row's cosine at its most active positions predicts entry at 0.64) (e548).
- A learned feature dictionary of the states sees the association by half (Pythia-160m at the end of training, block 6, EleutherAI's TopK autoencoder): no feature matches a word's context set as a set (the best reaches Jaccard 0.25 for 0.28 of words against 0.17 of random rows, a feature of 5 positions unrelated to the row, cosine 0.04); the feature nearest to the word's row by decoder cosine sits at 0.48 (random rows 0.16, random directions 0.13) and fires at 0.50 of the word's eight extreme positions (random rows 0.00); in reverse the nearest row to a live feature is a word for 0.20 of features against a base rate of 0.01, while only 0.10 of features have a row within cosine 0.5 (e549).

**Start here:** e548, e549 · **Sessions:** S97-S98 · **Scripts:** `scripts/e548_activation_vs_wdd.py`, `e549_feature_side.py`

## Experiments

| id | question | result | status | links |
| --- | --- | --- | --- | --- |
| e548 | Can ordinary neuron statistics computed without the projection predict future entry as well as S: ten features at t for entrants at five horizons and 4,000 random non-words per origin, single-feature AUCs, logistic regressions out of sample (even origins to odd and the reverse) on the WDD set, the activation set, with the output geometry and the coalition, and all; and whether family pairs share input selectivity? | AUC at H=4: S 0.86, breadth 0.88, mean positive activation 0.64, kurtosis 0.47, top-5% share 0.45, context coherence 0.67, norm 0.43, norm growth 0.55, the row's cosine at its most active positions 0.64, coalition coherence there 0.65; at H=8: S 0.83, breadth 0.85, the rest at most 0.64; logistic out of sample at H=4: WDD 0.88, activation 0.74, with geometry 0.74, with coalition 0.74, all 0.88; WDD adds +0.142, the activation side adds +0.001; families: activation correlation 0.26 against -0.01, peak overlap 0.06 | established (the projection carries what the activation side lacks); refuted (WDD as a proxy for selectivity) | ← e547b e535 e516b · → e549 |
| e549 | Can a learned feature dictionary of the states recover the row-context associations: Pythia-160m at the end of training, block 6, EleutherAI's TopK autoencoder; for each native word's context set (its eight positions of largest projection) the best-matching feature by Jaccard and the feature nearest to the row by decoder cosine, against random non-word rows, random directions and shuffled positions; in reverse the nearest row to every live feature? | Words: best-matching feature at Jaccard 0.25 or more for 0.28 (random rows 0.17, random directions 0.17, shuffled 0.00), median Jaccard 0.19 / 0.15 / 0.12 / 0.12, matching feature's size 5 positions, its decoder's cosine with the row 0.04; the feature nearest to the row: cosine 0.48 / 0.16 / 0.13 / 0.48, its recall of the context set 0.50 / 0.00 / 0.00 / 0.00; reverse (4626 live features): nearest row's median cosine 0.28, at 0.5 or more for 0.10, a word for 0.20 (base 0.01), the feature covering that row's context set at recall 0.00 | narrowed (the feature side sees the row at cosine about a half and half of its contexts; no feature is the word's context set) | ← e548 e504 e497 · → 21 |

## How the results flow

- `e547b → e548`. The necessity test against the neuron's own statistics: they predict entry at 0.74, the projection at 0.88, and add nothing to it; the extremes are the coalition's contexts, not the neuron's peaks.
- `e548 → e549`. The necessity test against a feature dictionary: it sees the row by half and the contexts by half; the association as a set, and the row's identity for most features, are WDD's.

## Links to other areas

- [31 The coalitions](31_coalitions.md): the association these tests try to recover by other means.
- [21 The SAE bridge](21_sae_bridge.md): the earlier bridge from features to native words, in the other direction.
- [27 The native axes](27_native_axes.md): the generic four fifths of the criterion that any reader shares.
