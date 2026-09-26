# 23. The onset of wordhood in training, at fine resolution (S68-S69)

**Question.** The checkpoint runs of areas 20 and 22 were made at eight or nine Pythia checkpoints a factor of two to four apart. Is the onset of native wordhood a transition, what precedes it, what precedes the sparsification that precedes it, and what follows?

**Established.**
- Wordhood turns on in a fast phase and then creeps: at block 12 the writer-win rate is 0.02 at steps 512-1000, 0.09 at 2000, 0.23 at 4000, 0.43 at 8000, 0.50 at 16000, 0.61 at 32000 and 0.73 at the end; half of its total rise falls in the two doublings from 2000 to 8000, the most concentrated change of nine observables (the attention share the least); block 6 is the same a doubling earlier. A crossover in log-time, not a discontinuity (e512).
- The developmental order is the same at every block: the effective number of writes falls first (midpoint near step 1500-1700), the writer-win rate rises next (2800-6600), the crowd's gain along the chord decays through zero and the linear contraction grows (2400-15400), and the chord's amplitude grows last (14000-19000). Sparsification precedes wordhood; the contraction and the chord follow it (e512).
- The accent is in place before step 512 and flat after (the non-writer tail factor 1.24-1.33 from step 2000 on at block 12); the state's energy moves in two phases, an early attention takeover (0.14 to 0.49 by step 2000 at block 12) and a late one (0.55 to 0.75 from step 40000 to the end), and the late phase erodes the last block's words (block 18: writer wins 0.70 at step 64000 to 0.48 at the end, the tail factor 1.47 to 2.29) (e512).

- Before sparsification (steps 0-2000): the gradient concentrates transiently on a few rows in the first 16 steps; the state cloud collapses to 9-16 effective dimensions between steps 16 and 64, attention's share of the state collapses with it and the accent is born (tail factor 1.00 through step 32, 1.14-1.48 at step 64); a collapsed plateau to step 256; then sparsification (effective writes 6800 to 3400 at block 12) as every row turns selective (median activity 0.50 to 0.19, none dying), the cloud re-expands (16 to 133 dimensions), attention takes the state back (0.06 to 0.49), the ledger's tails heavy (excess kurtosis 3.6 to 14.5) and the contraction is born at warmup's end; the gradient is spread over two to three times more rows than at initialisation by then, so it does not initiate sparsification (e513).

**Start here:** e512, e513 · **Sessions:** S68-S69 · **Scripts:** `scripts/e512_transition_sweep.py`, `e513_early_sweep.py`

## Experiments

| id | question | result | status | links |
| --- | --- | --- | --- | --- |
| e512 | Is the onset of wordhood a transition, and in what order do the observables move (Pythia-410m, 18 checkpoints from step 512 to the end, blocks 6, 12, 18)? | A fast crossover with an order: the writer-win rate at block 12 is 0.02, 0.09, 0.23, 0.43, 0.50, 0.61, 0.73 at steps 512, 2000, 4000, 8000, 16000, 32000 and the end, half its rise between 2000 and 8000; effective writes fall first (5900 to 3400 by step 2000, flat near 2000 after 8000), wins rise next, the crowd's gain along the chord decays through zero near step 20000 and the contraction grows (0 to -0.22), the chord's amplitude grows last (0.13 to 0.60); the tail factor is flat after step 2000; the attention share moves early and again late, and the late phase takes block 18's wins from 0.70 to 0.48 | narrowed (a crossover, ordered) | ← e498 e503 e506 e509 e510 |
| e513 | What precedes sparsification (Pythia-410m at 13 checkpoints from initialisation to step 2000; e512's observables plus the gradient's concentration across rows, the rows' liveness, the ledger's kurtosis and the cloud's anisotropy)? | Four phases: a transient gradient concentration (participation ratio 855 to 575 of 4096 by step 16, top-64 share 0.18 to 0.23) with a transient near-dead share of rows (0.12 at step 16); the cloud's collapse (85 to 13 dimensions, steps 16-64) with attention's share falling to 0.09 and the accent born (tail 0.99 to 1.14; 1.86 at block 18 by step 128); a collapsed plateau to step 256 with the gradient spreading (to 1900); sparsification from step 256 (effective writes 6800 to 3400 by 2000) as rows turn selective (activity 0.50 to 0.19), the cloud re-expands (16 to 133), attention returns (0.06 to 0.49), kurtosis rises (3.6 to 14.5), the crowd's alignment peaks at 512 (+1.24) and the contraction is born at 1000-2000. The gradient is spread (2400-2900 rows) when the writes sparsify | established (the phases); refuted (gradient concentration as the initiator) | ← e512 e498 · → 05 |

## How the results flow

- `e512 → e513`. Below step 512 the order continues backwards: the cloud's collapse and the accent's birth at steps 16-64, the plateau, then sparsification as the cloud re-expands; the gradient's concentration is a transient of the first 32 steps, not the initiator.
- `e498, e503, e506, e509, e510 → e512`. The coarse checkpoint series of the earlier areas are recomputed at eighteen checkpoints without ablation, which is enough to order the observables' midpoints and to see the fast phase of wordhood between steps 2000 and 8000.

## Links to other areas

- [22 The model tested](22_model_tested.md): the observables' definitions and the end-of-training anatomy.
- [20 Classical calibration](20_classical_calibration.md): the two clocks at coarse resolution (e493, e498), here seen as an ordered sequence.
- [05 Cancellation, contraction](05_cancellation_contraction.md): the linear contraction's birth (e217) placed here at steps 1000-4000 and growing to the end.
