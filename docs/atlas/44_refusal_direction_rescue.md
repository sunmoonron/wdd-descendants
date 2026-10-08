# 44. The refusal direction, writer rescue, the causal test at 410m, and the baseline at 1B (S113)

**Question.** Session 112 left the audit with a route to a causal story and three loose ends: whether the refusal direction a chat model carries is written by the writers that fire on harmful requests, whether restoring silenced writers restores the forgotten domain, whether silencing after an unlearning slows relearning at a size where the gradient methods silence nothing, and whether the audit holds at a billion parameters.

**Established.**
- The refusal direction is real (ablation takes refusal from 0.93 / 0.99 to 0.04 / 0.03 on the 0.5B / 1.5B models, adding it induces refusal on harmless prompts at 0.88 / 0.98) and the harmful-class writers do not write it: rows' cosine 0.034 / 0.026 against 0.024 / 0.023 for any row, 0.11 / 0.01 of its MLP mass, and they fire at 0.99 / 0.88 under its ablation (e607).
- Restoring the writers after capped ascent recovers 0.09 / -0.01 / 0.01 of the rise and restoring every MLP neuron of blocks 0-B 0.02 / 0.26 / -0.20: the gradient methods' forgetting lives outside the MLP activations of those blocks, and the writers' silencing is a symptom (e609).
- At 410m silencing the writers after capped ascent adds 10 relearning steps (slower in 1.00 of 3 seeds; random rows 0), a fifth of the 160m effect, and after RMU nothing beyond random rows (e608); the activation-ratio audit and its naive siblings hold at 1B (writers 0.90, difference-selected 0.87, probe 0.82) (e610).

**Start here:** e607, e609 · **Sessions:** S113 · **Scripts:** `scripts/e607_refusal_direction.py`, `e608_separable_410m.py`, `e609_writer_rescue.py`, `e610_baseline_1b_summary.py`

## Experiments

| id | question | result | status | links |
| --- | --- | --- | --- | --- |
| e607 | The refusal direction and its writers on Qwen2.5-0.5B and 1.5B instruct: the difference-of-means direction, its ablation at every block, its addition to harmless prompts, the harmful-class writers' rows' alignment with it and share of its MLP mass, and their activation under ablation | Ablation: refusal 0.93 / 0.99 to 0.04 / 0.03; added: harmless refusal to 0.88 / 0.98; writers' rows absolute cosine 0.034 / 0.026 vs all rows 0.024 / 0.023, mass share 0.11 / 0.01; writers under ablation 0.99 / 0.88 | established (the writers mark the request; the refusal is carried by a direction they do not write) | ← e604 · → 43 |
| e608 | The separable interventions at Pythia-410m, PubMed, three seeds, with the writers' activation logged through the unlearning | Input silencing adds 10 relearning steps after ascent (slower in 1.00 of 3: +10, +5, +10) against 0 for random rows, after RMU 5 against 5; rows zeroed 10 / 5; difference-selected 5 / 5; at 160m the same silencing added 40 and 50 to 70 | narrowed (the effect shrinks to ten steps at 410m, consistent in direction after ascent, indistinguishable from random rows after RMU) | ← e603 · → e608b · 43 |
| e608b | The runs of e608 summarised over seeds, with the trajectories | The tables of area 44 | tool (summary) | ← e608 |
| e609 | Writer rescue: after capped ascent, NPO and RMU to +2 the forget loss with the writers', the naive sets', random or every MLP neuron's activations patched to the original's, and the reverse patch; Pythia-160m on two domains and Pythia-410m | Writers restored after ascent: 0.09 / -0.01 / 0.01 of the rise; every MLP neuron of blocks 0-B 0.02 / 0.26 / -0.20 (after RMU 0.75 / 0.57 / 0.83); reverse patch with the writers 0.02 | refuted (restoring the writers does not restore the domain; the forgetting lives in the attention and the later blocks) | ← e597 e603 · → 43 |
| e610 | The baseline of e602 at Pythia-1B (ascent, RMU, NPO; six neuron sets), summarised alone and pooled with the three sizes | 1B: writers 0.90, difference-selected 0.87, ratio-selected 0.75, random -0.08, still-writing -0.60, probe 0.82 with the recovery over 9 conditions; pooled over 63: 0.87 / 0.83 / 0.70 | supported (the audit and its naive siblings hold at a third size) | ← e602 · → 43 |

## How the results flow

- `e604 → e607`: the refusal audit found the harmful-request writers untouched by benign fine-tuning; here the direction that carries refusal, and whether those writers write it.
- `e603 → e608`, `e597 → e609`: the causal test at the size where no method silences the writers, and the patch that asks whether restoring them restores the domain.
- `e602 → e610`: the baseline at a third size.

## Links to other areas

- [43 The audit's baseline, its causal test, and three new audits](43_audit_baselines.md), [42 The unlearning audit, tested as its users would](42_audit_users.md): the claims these experiments complete.
- [31 The coalitions](31_coalitions.md): why the writers are a route and not a store.
