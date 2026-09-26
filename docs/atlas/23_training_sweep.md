# 23. The onset of wordhood in training, at fine resolution (S68)

**Question.** The checkpoint runs of areas 20 and 22 were made at eight or nine Pythia checkpoints a factor of two to four apart. Is the onset of native wordhood a transition, what precedes it, and what follows?

**Established.**
- Wordhood turns on in a fast phase and then creeps: at block 12 the writer-win rate is 0.02 at steps 512-1000, 0.09 at 2000, 0.23 at 4000, 0.43 at 8000, 0.50 at 16000, 0.61 at 32000 and 0.73 at the end; half of its total rise falls in the two doublings from 2000 to 8000, the most concentrated change of nine observables (the attention share the least); block 6 is the same a doubling earlier. A crossover in log-time, not a discontinuity (e512).
- The developmental order is the same at every block: the effective number of writes falls first (midpoint near step 1500-1700), the writer-win rate rises next (2800-6600), the crowd's gain along the chord decays through zero and the linear contraction grows (2400-15400), and the chord's amplitude grows last (14000-19000). Sparsification precedes wordhood; the contraction and the chord follow it (e512).
- The accent is in place before step 512 and flat after (the non-writer tail factor 1.24-1.33 from step 2000 on at block 12); the state's energy moves in two phases, an early attention takeover (0.14 to 0.49 by step 2000 at block 12) and a late one (0.55 to 0.75 from step 40000 to the end), and the late phase erodes the last block's words (block 18: writer wins 0.70 at step 64000 to 0.48 at the end, the tail factor 1.47 to 2.29) (e512).

**Start here:** e512 · **Sessions:** S68 · **Scripts:** `scripts/e512_transition_sweep.py`

## Experiments

| id | question | result | status | links |
| --- | --- | --- | --- | --- |
| e512 | Is the onset of wordhood a transition, and in what order do the observables move (Pythia-410m, 18 checkpoints from step 512 to the end, blocks 6, 12, 18)? | A fast crossover with an order: the writer-win rate at block 12 is 0.02, 0.09, 0.23, 0.43, 0.50, 0.61, 0.73 at steps 512, 2000, 4000, 8000, 16000, 32000 and the end, half its rise between 2000 and 8000; effective writes fall first (5900 to 3400 by step 2000, flat near 2000 after 8000), wins rise next, the crowd's gain along the chord decays through zero near step 20000 and the contraction grows (0 to -0.22), the chord's amplitude grows last (0.13 to 0.60); the tail factor is flat after step 2000; the attention share moves early and again late, and the late phase takes block 18's wins from 0.70 to 0.48 | narrowed (a crossover, ordered) | ← e498 e503 e506 e509 e510 |

## How the results flow

- `e498, e503, e506, e509, e510 → e512`. The coarse checkpoint series of the earlier areas are recomputed at eighteen checkpoints without ablation, which is enough to order the observables' midpoints and to see the fast phase of wordhood between steps 2000 and 8000.

## Links to other areas

- [22 The model tested](22_model_tested.md): the observables' definitions and the end-of-training anatomy.
- [20 Classical calibration](20_classical_calibration.md): the two clocks at coarse resolution (e493, e498), here seen as an ordered sequence.
- [05 Cancellation, contraction](05_cancellation_contraction.md): the linear contraction's birth (e217) placed here at steps 1000-4000 and growing to the end.
