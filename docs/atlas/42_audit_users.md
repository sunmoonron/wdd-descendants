# 42. The unlearning audit, tested as its users would (S111)

**Question.** Session 110 found that bounded unlearning leaves the rows that wrote a domain's classes untouched while half of them stop writing there, and that the share still writing tracked relearnability over eight conditions of two crude methods. Would an unlearning researcher, a reviewer, a safety team or a continual-learning person believe it? Here the audit against the methods they use, the baseline they would propose, the attacks they run, the mirror question, and the tool question.

**Established.**
- Three levels, three numbers (e597). No training method touches the rows (cosine 0.9991 to 1.0000); ascent, difference and NPO silence the writers on Pythia-160m PubMed (activation 0.53 to 0.82) but not on Github or Pythia-410m (0.90 to 0.99); RMU never does (1.00), overrides the state, and is the shallowest (twenty steps recover 0.83 to 0.96); dampening alone reaches the rows (2.27 to 12.68 times rows of the same norm) and relearns slowest (0.38 to 0.49) at a retain cost of +0.684 to +2.646.
- Depth is read on the parameter side: over 39 conditions the writers' activation ratio correlates with the 20-step recovery at 0.82 on both model sizes (0.85 / 0.72), the state-side still-writing share at 0.39 and -0.68 on Pythia-410m, the domain probe at 0.67 and -0.03; the state-side readings call RMU the deepest when it is the shallowest. Benign relearning recovers a third to a half of ascent's rise on PubMed; 4-bit quantisation nothing at this scale.
- Fine-tuning recruits existing rows without rewriting them (25 / 29 entrants at cosine 0.99; attention-only fine-tuning gains 0.130 / 0.106 of 0.233 / 0.178) (e600); removing the writers the audit names does not unlearn (+0.08 nats), but removing them after capped ascent slows the relearning (back after 95 against 60 steps) (e601).

**Start here:** e597, e597b · **Sessions:** S111 · **Scripts:** `scripts/e597_unlearning_methods.py`, `e597b_audit_summary.py`, `e600_learning_audit.py`, `e601_guided_unlearning.py`

## Experiments

| id | question | result | status | links |
| --- | --- | --- | --- | --- |
| e597 | The unlearning audit across five methods (capped ascent, gradient difference, NPO, RMU, selective dampening) and a drift control, two forget domains, Pythia-160m and 410m, each stopped at a target forget rise; the audit with the writers' activation, the rows' change against same-norm rows, a logistic domain probe as baseline, and three attacks (relearning, benign relearning, 4-bit and 8-bit quantisation) | Rows untouched by every training method (cosine 0.9991 to 1.0000); ascent / difference / NPO silence the writers on 160m PubMed (activation 0.53 to 0.82, recovery 0.56 to 0.83) and not elsewhere (0.90 to 0.99, recovery 0.78 to 0.97); RMU activation 1.00, still writing 0.03 to 0.54, recovery 0.83 to 0.96; dampening rows' change 2.27 to 12.68 times same-norm rows, recovery 0.38 to 0.49, retain +0.684 to +2.646; Spearman with recovery: activation 0.82 (0.85 / 0.72 by model), still-writing 0.39 (-0.68 on 410m), probe 0.67 (-0.03); benign recovery 0.36 to 0.40 (ascent, PubMed), 4-bit -0.24 to 0.46 | narrowed (depth is read on the parameter side: the writers' activation and the rows' change, not the state's projection) | ← e592b · → e597b e601 · 31 41 |
| e597b | The runs of e597 summarised: per-method signatures and the Spearman of every candidate predictor with every attack outcome | The tables of area 42 | tool (summary) | ← e597 |
| e600 | The learning audit: Pythia-160m fine-tuned for one epoch on 640 Github or PubMed documents with all parameters, attention only, or MLP only; entrants among the domain's words, their rows' cosine, the collateral on control words | Entrants 25 / 29 with rows unchanged (1.00 / 1.00); attention-only gain 0.130 / 0.106 of 0.233 / 0.178; control words still writing 1.00 / 1.00 | established (fine-tuning recruits existing rows by feeding them differently) | ← e585 e586 · → 39 |
| e601 | WDD-guided unlearning: the forget writers' rows zeroed (against random rows, the retain writers, a wider net, a repaired version, and after capped ascent), with the audit and the attacks | Forget writers zeroed: +0.08 / +0.11 nats (random rows +0.00); wider net +0.16; the hybrid back after 95 / over 100 steps against 60 / 35 for ascent alone | superseded (the steps-back delay was a floor artefact: e618e, session 117) | ← e590 e591b · → 40 |

## How the results flow

- `e592b → e597 → e597b`: the audit of one model and two crude methods, then the methods people use, the probe and the attacks; the summary decides which number tracks depth.
- `e585, e586 → e600`: recruitment and substitution in training, then in fine-tuning.
- `e590, e591b → e601`: the removed speaker and the single row's small cost, then the writers removed as an unlearning method.

## Links to other areas

- [41 Six directions beyond the descendants](41_six_directions.md): the audit's first form (e592b).
- [31 The coalitions](31_coalitions.md): why a writer can be silenced or overridden without being touched, and why removing it does not remove the class.
- [39 Responsibility migration read directly, and the forecast](39_migration.md), [40 The mechanism of the turn, the removed speaker, the paper](40_turn_and_paper.md): recruitment in training, which e600 sees again in fine-tuning.
