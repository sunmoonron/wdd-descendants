# 53. The pinning checks (S122)

**Question.** Is programmable feature placement (area 52) real? The ways it could be hollow, each with its check: the dial could be damage; any pin could make a row powerful; it could be one lucky row; the concept could leave when the pin is lifted; and two concepts might not coexist on two rows.

**Established.**
- Three designated rows (seeds 0, 1, 2) take the concept at vote shares 1.00, 1.00, 1.00 with dials -0.029, +0.027, -0.006, each at a Pile cost on negation of +0.0009, -0.0019, +0.0017 nats and a norm that goes from 0.81, 0.48, 0.57 to 0.88, 0.55, 0.64; the handle's dial goes from +0.343 to +0.170, +0.165, +0.246 (control fine-tune +0.403); eight random rows move the pronoun by at most 0.004, 0.004, 0.002 (e627).
- The placebo pin (labels shuffled per pair) leaves its row at vote share 0.00 and dial +0.000 (e627).
- After 1,200 further steps of the next-token loss alone the pinned row's share goes from 1.00 to 1.00 and its dial from -0.048 to -0.018 (the handle's +0.175 to +0.220; validation loss 3.414 to 3.465) (e627).
- From the step-8000 checkpoint, where no handle exists (pronoun agreement 0.77 after the pin), the pin installs the row at share 1.00 and dial +0.128, and after release 1.00 and +0.092 (e627).
- Two rows pinned at once: the gender row 1542 takes share 1.00 and dial -0.004, the age row 291 share 1.00 and dial +0.035 (old over young on held-out nouns, 0.58 unedited); the cross-dials are -0.008 (gender row on age) and -0.009 (age row on pronouns); both negated give +0.008 and +0.017 (e627).
- The pin installs the description and not the behaviour: on the trained model every designated row takes the vote at share 1.00 while its dial stays near zero and the incumbent's dial falls, so the behaviour is redistributed rather than moved, and session 121's dial on row 218 was one row in four; from the step-8000 checkpoint, where no incumbent exists, the designated row takes both the vote and a dial that survives the pin's release, which is the narrower claim that stands, on one seed.

**Start here:** e627 · **Sessions:** S122 · **Scripts:** `scripts/e627_pinning_checks.py`

## Experiments

| id | question | result | status | links |
| --- | --- | --- | --- | --- |
| e627 | The pinning checks: damage (norm, Pile loss on negation, random rows), a placebo pin with shuffled labels, three seeds, a release phase with the next-token loss alone, the same from the step-8000 checkpoint, and two concepts on two rows with cross-dials and joint negation; Pythia-160m, block 4 | three designated rows (seeds 0, 1, 2) take the concept at vote shares 1.00, 1.00, 1.00 with dials -0.029, +0.027, -0.006, each at a Pile cost on negation of +0.0009, -0.0019, +0.0017 nats and a norm that goes from 0.81, 0.48, 0.57 to 0.88, 0.55, 0.64; the handle's dial goes from +0.343 to +0.170, +0.165, +0.246 (control fine-tune +0.403); eight random rows move the pronoun by at most 0.004, 0.004, 0.002; the placebo pin (labels shuffled per pair) leaves its row at vote share 0.00 and dial +0.000; after 1,200 further steps of the next-token loss alone the pinned row's share goes from 1.00 to 1.00 and its dial from -0.048 to -0.018 (the handle's +0.175 to +0.220; validation loss 3.414 to 3.465); from the step-8000 checkpoint, where no handle exists (pronoun agreement 0.77 after the pin), the pin installs the row at share 1.00 and dial +0.128, and after release 1.00 and +0.092; two rows pinned at once: the gender row 1542 takes share 1.00 and dial -0.004, the age row 291 share 1.00 and dial +0.035 (old over young on held-out nouns, 0.58 unedited); the cross-dials are -0.008 (gender row on age) and -0.009 (age row on pronouns); both negated give +0.008 and +0.017 | narrowed (placement fails one of its own checks) | ← e626 · → 52 |

## How the results flow

- `e626 → e627`: the placement result, against the checks that could dissolve it.

## Links to other areas

- [52 The cross-lingual dial and programmable feature placement](52_crosslingual_pinning.md): the result under test.
