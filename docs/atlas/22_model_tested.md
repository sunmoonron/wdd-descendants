# 22. The extreme-value model tested: the floor, the chord and the crowd (S64)

**Question.** THEORY 3m said an atom is a word when its projection clears the Gumbel level of the competing atoms, a writer through its own coefficient and a non-writer through the second-order alignment of the state cloud with the atoms. Do those two numbers, computed with nothing fitted, reproduce the observed maxima state by state, and if not, what carries the maximum?

**Established.**
- The floor is right: the rotated dictionary's maximum is 0.97-1.00 of the exact Gaussian extreme-value mean computed from the atoms' second moment along the state, in every model, block and checkpoint (27 cells) (e503, e506).
- The numerator is not: the non-writers' maximum stands 1.3-2.3 times above the level from their own second moment while the second-order alignment is a factor 1.02-1.24 (the accent is a tail, arriving between Pythia steps 256 and 512), and the writers' maximum is 1.5-2.2 times the largest write's own coefficient; prominence against the floor predicts writer wins at 0.45 against 0.78 (e503).
- A word's projection is its own coefficient (0.3-0.7) plus the position's other large writes through their cosines (0.3-0.7), the chord, less the crowd of small MLP writes (-0.05 to -0.5), plus attention and the centring; the top non-writer atom is the chord's shadow in GPT-2 and OLMo (0.56-0.93 cross terms, cosine 0.3-0.6 with a writer) and attention's in Pythia; at Pythia step 512 the maxima are head atoms carried by the dense sum of small writes, and by the end the sparse chord carries 0.8-0.9 (e505).
- Summed as written, the largest writes overshoot the maximum twofold where they are as large as the state (GPT-2, OLMo) and name the atom in 2-16% of states; the overshoot is the common direction they share, and centred the chord predicts GPT-2's maximum within 1-13%, ranks states at 0.56-0.75 and names the atom in 32-46% (top four 54-64%); in OLMo the crowd's contraction is needed (all writes 0.78-0.86 of the level) and in Pythia the small writes add (all writes within 1-28%); over training the chord reproduces the words clock and not the accent (e506).

**Start here:** e503, e505, e506 · **Sessions:** S64 · **Scripts:** `scripts/e503_who_wins.py`, `e505_anatomy_of_the_max.py`, `e506_chord_prediction.py`

## Experiments

| id | question | result | status | links |
| --- | --- | --- | --- | --- |
| e503 | Do the model's two halves predict the native maximum, prominence for writers and the second-order Gumbel level for non-writers (GPT-2, Pythia at 8 checkpoints, OLMo; 3 blocks each)? | No, by 1.5-2 each way, with the floor itself right (rotated calibration 0.97-1.00 in 27 cells): non-writers stand 1.3-2.3 above the level from their own second moment (second-order term 1.02-1.24); writers 1.5-2.2 above their own coefficient; predicted writer wins 0.45 against 0.78 at GPT-2's middle block. Over Pythia's training the non-writer tail arrives between steps 256 and 512, and the writers' excess over prominence peaks at step 4000 (3.3) and falls to 1.5 | refuted (both halves; the floor established) | ← e483 e493 e498 · → e505 e506 |
| e505 | What carries a winning projection: the own coefficient, the position's other large writes, the small writes, attention? | Own 0.3-0.7 plus the other top-64 writes' cross terms 0.3-0.7 (the chord); the crowd of small MLP writes subtracts (-0.05 to -0.5); attention and the centring 0.02-0.57. The top non-writer is the chord's shadow in GPT-2 and OLMo (cross terms 0.56-0.93; cosine with a writer 0.33-0.60) and attention's in Pythia (rest 0.53-0.70). At Pythia step 512 the maxima are head atoms (0.82-0.91) carried by the dense sum of small writes (0.5-1.0); by the end the chord carries 0.8-0.9 and the small writes have turned negative | established (the anatomy) | ← e503 · → e506 |
| e506 | Does the chord (the largest writes through the Gram) predict the native maximum, in level, rank and identity? | Summed as written, no: twice the observed level where the writes are as large as the state (GPT-2, OLMo; 30-60 times too many atoms above the floor; the atom named in 2-16% of states). The overshoot is the common direction the largest writes share; centred, the chord predicts GPT-2's maximum within 1-13% of the level at every block, ranks the states at 0.56-0.75 and names the atom in 32-46% (top four 54-64%; 19-21 atoms above the floor against 40-107 observed). In OLMo the crowd's contraction is needed (all writes centred 0.78-0.86 of the level, the atom in 28-45%); in Pythia the small writes add (the centred chord 0.95-1.60 under, all writes within 1-28%; at step 512 all writes name the atom in 22-53%). Over Pythia's training the chord grows from 0.15 to 0.63 of the state and its predicted factor reproduces the words clock, not the accent | narrowed (the centred chord predicts GPT-2's level; the crowd and the small writes matter elsewhere) | ← e505 e503 |

## How the results flow

- `e483, e493, e498 → e503`. The calibrated floor (e483) and the two-term decomposition (e493, e498) gave the model two numbers; e503 computes them per state and finds the floor right and both terms wrong by 1.5-2.
- `e503 → e505 → e506`. The exact split of the winning projection shows the chord and the crowd; the prediction from the ledger alone then fails for the chord as written, succeeds for the centred chord in GPT-2, and needs the crowd in OLMo and the small writes in Pythia.

## Links to other areas

- [20 Classical calibration](20_classical_calibration.md): the floor and the decomposition this area tests; e483's law is confirmed at every checkpoint here.
- [05 Cancellation, contraction](05_cancellation_contraction.md): the crowd of small MLP writes that erases writes there contracts words here; provenance persisting because other components write the same directions is the chord.
- [06 Descendants I](06_descendants_fate_transport.md): the chord's cross terms are later and earlier writers of the same direction, the transport this area described.
- [21 SAE bridge](21_sae_bridge.md): the same session's feature-level results (e502, e504).
