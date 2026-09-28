# 33. The second model: the chain of sessions 89-97 on OLMo-1B (S99)

**Question.** Everything from session 72 on was Pythia-410m. Do the one-way gate, the native fifth of the criterion, the persistence of the rows' extremes, the private persistent coalition that precedes the word, the word's own neuron's selectivity and the necessity test hold on a different architecture, tokenizer and corpus at the same token range?

**Established.**
- The gate, the criterion and the persistence replicate on OLMo-1B (steps 1000-16000, block 8 of 16, the MLP rows of blocks 0-8), most more strongly: criterion 0.94 against 0.80 for random atoms, retention 0.64 against 0.19, entries 63 against 182, words at S 1.70 against 0.98; leavers over the floor at exit 0.96 against 0.17; positions at 1.1-1.2 persisting at 1.10 against 0.99; conditional persistence rising from 0.886 to 0.974 against a flat 0.870 (e551).
- The coalition replicates: within-row coherence of the writer vectors 0.731 against -0.000 across rows and 0.042 for random directions, persistence 0.969 against a selection-matched 0.511, precedence 0.52 against 0.31 four intervals before entry; the word's own neuron does not (z 0.06 at k=-8, 0.68 at entry, own write 0.03), and the attention outputs supply 0.14 of an extreme rather than more than half (e552).
- The necessity test: S and breadth 0.92 against 0.78 for the activation side with its geometry and coalition, WDD adding +0.133 and the reverse -0.004 (e553).

**Start here:** e551, e552, e553 · **Sessions:** S99 · **Scripts:** `scripts/e550_olmo_cache.py`, `e551_olmo_exits.py`, `e552_olmo_coalitions.py`, `e553_olmo_necessity.py`, `olmo_cache_common.py`

## Experiments

| id | question | result | status | links |
| --- | --- | --- | --- | --- |
| e550 | The cache: OLMo-1B-0724 at steps 1000-16000 (about two billion tokens a step), block 8, the 8 x 256 evaluation tokens: states, sink mask, unit centred states, the MLP rows of blocks 0-8 with norms, every writer's activation, the summed attention outputs and the embedding; each checkpoint's weights removed after use? | Sixteen checkpoints cached; 73,728 rows of dimension 2048; 2040 positions, none a sink; the state reconstructed from its writes at relative projection 1.000 at every checkpoint | tool | → e551 e552 e553 |
| e551 | The gate, the criterion and the persistence on OLMo-1B, rows against a fixed isotropic random dictionary of the same size? | Rows / random atoms: criterion at 1 / 4 0.98 / 0.94 against 0.87 / 0.80; entries 63 / 182; retention 0.64 / 0.19; words' S 1.70 / 0.98; leavers over the floor at k=0 0.96 / 0.17; entrants under it at -2 0.27 / 0.89; S across exits / entries -0.04 / 0.20 against -0.07 / 0.07; maximum at the same position 0.47 / 0.35; level-matched persistence at 1.1-1.2 1.10 / 0.99; conditional persistence bulk to extremes 0.886 to 0.974 against 0.870 to 0.865 | established (all four replicate, most more strongly) | ← e538b e539 e541 e542 · → 29 30 |
| e552 | The coalitions, their precedence and the neuron's selectivity on OLMo-1B? | Step 8000: effective writers 3175, top-10 share 0.03, MLP / attention shares 0.86 / 0.14; within-row 0.731, extreme against typical 0.009, across rows -0.000, activations alone 0.162 / 0.009 / 0.005, random directions 0.042 / 0.000; persistence 8000-16000 real 0.969, null 0.511, activations 0.713. Entrants (275) at k=-8/-4/-1/0/+2: S 0.98 / 1.09 / 1.26 / 1.40 / 1.51, z 0.06 / -0.14 / 0.42 / 0.68 / 0.65, within-row at -4 / 0 0.52 / 0.74, coherence 3.92 / 6.56, own share 0.03; near misses: S 1.26 / 1.22 / 1.22, z 0.24 / 0.42, within-row 0.31 / 0.34, coherence 4.79 / 8.55; random non-words z -0.08 / 0.07 | established (the coalition, its persistence and its precedence); refuted (the neuron's selectivity preceding entry; attention's half) | ← e545 e546 e547b · → 31 |
| e553 | The necessity test on OLMo-1B: ten features at the origin, single-feature AUCs at five horizons, logistic out of sample? | AUC at H=4: S 0.90, breadth 0.91, mean positive activation 0.75, kurtosis 0.59, top-5% share 0.64, context coherence 0.70, norm 0.43, norm growth 0.57, row's cosine at its most active positions 0.71, coalition coherence there 0.73; at H=8 S 0.86, breadth 0.88, best of the rest 0.73; logistic at H=4: WDD 0.92, activation 0.77, with geometry and coalition 0.78, all 0.91; WDD adds +0.133, the activation side adds -0.004 | established (WDD is not a proxy for neuron selectivity on the second model either) | ← e548 · → 32 |

## How the results flow

- `e550 → e551, e552, e553`. One cache, three replications: the exits and persistence, the coalitions, the necessity test.
- What replicates is the phenomenon (the gate, the native fifth, the coalition, the necessity); what does not is the word's own neuron's part in it and the share of transport, both architectural.

## Links to other areas

- [29 How a word leaves](29_exits.md), [30 The tails](30_tails.md), [31 The coalitions](31_coalitions.md), [32 The necessity tests](32_necessity.md): the Pythia results replicated here.
- [23 The onset of wordhood in training](23_training_sweep.md): the earlier two-model comparisons of the two clocks (e498).
