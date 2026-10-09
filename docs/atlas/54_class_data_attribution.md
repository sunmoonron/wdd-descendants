# 54. Class-based data attribution (S123)

**Question.** A word's class, the contexts where its row writes above the floor, is a free attribution of the feature to data (sessions 108 to 109 showed the class's positions supply the row's gradient before recruitment). Is it causal: does removing those contexts from the stream stall the row and no other?

**Established.**
- Of the 51 rows the real run recruits between steps 8000 and 10000 on the held-out windows, 50 have specific class bigrams and are split into A (25) and B (25); after 2,500 steps of 8 windows from step 8000 on the full stream 0.28 of A and 0.20 of B are words (mean class sizes 32.6 and 41.2 against 51.2 and 66.8 at step 10000); with the 15194 windows carrying A's contexts removed, A is at 0.20 and B at 0.24; with B's 13225 removed, B is at 0.32 and A at 0.28; with 15194 random windows removed, 0.20 and 0.32; substitutes under the full stream 0.04 / 0.00, under removal of the group's own contexts 0.08 / 0.00; validation loss 2.893 to 2.997 (full stream) (e628).
- The test is underpowered and shows no specific effect: the full stream recruits 0.24 of the real run's recruits (the design's gate was 0.25), and removing a group's contexts moves its recruitment by no more than removing as many random windows does (A 0.28 on the full stream, 0.20 with its contexts removed, 0.20 with random windows removed; B 0.20, 0.32, 0.32), differences inside one standard error of 0.09 at 25 rows per group; the targeted removal takes a third of the stream because the classes' bigrams are common, the continuation is five million tokens against the real run's four billion, and a fifth of the step-8000 vocabulary churns in it, so the class's contexts are neither shown nor refuted as what recruits the row at this scale (e628).

**Start here:** e628 · **Sessions:** S123 · **Scripts:** `scripts/e628_class_data_attribution.py`

## Experiments

| id | question | result | status | links |
| --- | --- | --- | --- | --- |
| e628 | Class-based data attribution with a causal handle: Pythia-160m continued from step 8000 for 2,500 steps on four streams (full, group A's class windows removed, group B's removed, as many random windows removed); the recruits are the real run's new words at step 10000; recruitment rate, class size, substitutes, validation loss per group and stream | of the 51 rows the real run recruits between steps 8000 and 10000 on the held-out windows, 50 have specific class bigrams and are split into A (25) and B (25); after 2,500 steps of 8 windows from step 8000 on the full stream 0.28 of A and 0.20 of B are words (mean class sizes 32.6 and 41.2 against 51.2 and 66.8 at step 10000); with the 15194 windows carrying A's contexts removed, A is at 0.20 and B at 0.24; with B's 13225 removed, B is at 0.32 and A at 0.28; with 15194 random windows removed, 0.20 and 0.32; substitutes under the full stream 0.04 / 0.00, under removal of the group's own contexts 0.08 / 0.00; validation loss 2.893 to 2.997 (full stream); the test is underpowered and shows no specific effect: the full stream recruits 0.24 of the real run's recruits (the design's gate was 0.25), and removing a group's contexts moves its recruitment by no more than removing as many random windows does (A 0.28 on the full stream, 0.20 with its contexts removed, 0.20 with random windows removed; B 0.20, 0.32, 0.32), differences inside one standard error of 0.09 at 25 rows per group; the targeted removal takes a third of the stream because the classes' bigrams are common, the continuation is five million tokens against the real run's four billion, and a fifth of the step-8000 vocabulary churns in it, so the class's contexts are neither shown nor refuted as what recruits the row at this scale | null (the continuation does not recruit; the test is underpowered at this scale) | ← e588 e589 e590 · → 25 26 |

## How the results flow

- `e588, e589, e590 → e628`: the forecast, the gradient's source and the short continuation, turned into a data intervention.

## Links to other areas

- [25 Training motion](25_training_motion.md) and the developmental areas that follow it: the recruitment law this test puts a data handle on.
