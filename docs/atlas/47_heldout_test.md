# 47. The pre-registered held-out test (S116)

**Question.** Does the audit's rule, fixed before the runs, hold on domains, methods and an architecture it never saw; does it add anything to the forget rise, the output KL and the drift; and does silencing the writers do more than silencing the naive sets at the same budget?

**Established.**
- At the rules fixed before the runs, the writers' activation flags shallow unlearning on the held-out domains at 0.62 accuracy (AUC 0.85; majority 0.55; difference-selected 0.62, ratio-selected 0.71, probe 0.63), on the held-out methods at 0.51 (difference-selected 0.49, probe 0.50) and on the held-out architecture at 0.73 (difference-selected 0.67, probe 0.67) (e617b).
- The ranking travels where the cut does not: AUC 0.85 on the held-out domains, 0.91 on the held-out methods, 0.93 on the architecture, against the difference-selected set's 0.90 / 0.96 / 0.80 and the probe's 0.76 / 0.82 / 0.47 (e617b).
- Adding the writers' activation to a model of the forget rise, the output KL and the drift moves the leave-one-domain-out accuracy from 0.55 to 0.84 (the difference-selected set 0.85, the probe 0.65, the writers alone 0.81; 255 conditions, 7 domains) (e617b).
- Silencing the writers' inputs after unlearning to +2 adds +2 steps to the relearning on the held-out domains against +2 for the difference-selected set and +0 for a random set of the same size (42 conditions; WDD beats the difference-selected set in 0.12 and random in 0.48); after the gradient methods silencing the writers' inputs adds +36 (ga), +20 (gd), +35 (npo), +54 (scrub) steps, the difference-selected set +36, +8, +48, +23, a random set +0, +0, +0, +0; after RMU, dampening and task-vector negation nothing moves (e617b).
- By method: ga writers 0.98, recovery 0.88 in 40 steps; gd writers 0.96, recovery 0.74 in 95 steps; npo writers 0.96, recovery 0.85 in 35 steps; rmu writers 1.00, recovery 0.89, not back within 100 steps; scrub writers 0.99, recovery 0.94 in 30 steps; ssd writers 0.43, recovery 0.35, not back within 100 steps; tv writers 0.90, recovery 0.49, not back within 100 steps (e617).

**Start here:** e617b · **Sessions:** S116 · **Scripts:** `scripts/e617_heldout.py`, `e617a_prereg.py`, `e617b_heldout_analysis.py`

## Experiments

| id | question | result | status | links |
| --- | --- | --- | --- | --- |
| e617 | The held-out runs: the e602 bake-off on StackExchange, Wikipedia, USPTO, FreeLaw and DM Mathematics at 160m, on Qwen2.5-0.5B (PubMed, Github), at 410m (StackExchange), with the held-out methods tv and scrub, KL and drift recorded, and matched-budget silencing at +2 | 255 conditions; by method ga writers 0.98, recovery 0.88 in 40 steps; gd writers 0.96, recovery 0.74 in 95 steps; npo writers 0.96, recovery 0.85 in 35 steps; rmu writers 1.00, recovery 0.89, not back within 100 steps; scrub writers 0.99, recovery 0.94 in 30 steps; ssd writers 0.43, recovery 0.35, not back within 100 steps; tv writers 0.90, recovery 0.49, not back within 100 steps | tool (the conditions the held-out scoring reads) | ← e602 · → 43 46 |
| e617a | The pre-registration: the fixed cut of every candidate auditor on the 63 training conditions, and six honest guesses, written before the first held-out launch | Fixed 2026-10-08 22:08:39: WDD writers' activation above 0.782, difference-selected above -0.140, ratio-selected above 0.411, probe above 0.155, forget rise at or below 4.112 | tool (the rules e617b scores) | ← e616 · → 46 |
| e617b | The scoring: fixed-rule accuracy and AUC per held-out group, the incremental test against the forget rise, KL and drift (leave-one-domain-out), the silencing comparison | at the rules fixed before the runs, the writers' activation flags shallow unlearning on the held-out domains at 0.62 accuracy (AUC 0.85; majority 0.55; difference-selected 0.62, ratio-selected 0.71, probe 0.63), on the held-out methods at 0.51 (difference-selected 0.49, probe 0.50) and on the held-out architecture at 0.73 (difference-selected 0.67, probe 0.67); adding the writers' activation to a model of the forget rise, the output KL and the drift moves the leave-one-domain-out accuracy from 0.55 to 0.84 (the difference-selected set 0.85, the probe 0.65, the writers alone 0.81; 255 conditions, 7 domains); silencing the writers' inputs after unlearning to +2 adds +2 steps to the relearning on the held-out domains against +2 for the difference-selected set and +0 for a random set of the same size (42 conditions; WDD beats the difference-selected set in 0.12 and random in 0.48); P1 refuted, P2 refuted, P3 refuted, P4 confirmed, P5 refuted, P6 refuted | refuted (the rule does not travel at the fixed cut) | ← e617 e617a · → 43 46 |

## How the results flow

- `e602, e610, e616 → e617a`: the training conditions fix the cut of every auditor.
- `e617a → e617 → e617b`: the held-out conditions are scored at the fixed cuts; nothing is retuned.

## Links to other areas

- [46 The auditor against the state of the art](46_auditor_sota.md): the transfer test this one holds out from; [43 The audit's baseline, its causal test, and three new audits](43_audit_baselines.md): the training conditions.
