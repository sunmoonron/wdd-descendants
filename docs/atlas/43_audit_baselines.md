# 43. The audit's baseline, its causal test, and three new audits (S112)

**Question.** Session 111 left one threat and one lead. The threat: any set of domain-selective neurons might predict relearnability as well as the WDD writers. The lead: removing the writers after a silencing method slowed relearning on one seed. Here the baseline with seeds, the causal test with seeds, and the audit turned on three new subjects: safety training and jailbreaks, model merging, and compression.

**Established.**
- Baseline (e602): a difference-selected set overlapping the writers by Jaccard 0.06 correlates with the 20-step recovery at 0.83 against the writers' 0.89 over 54 conditions (random -0.01, still-writing 0.36, probe 0.73); seeds agree at 0.98. The audit's substance is that domain-selective neurons still fire; the dictionary is one way to name them.
- Causal test (e603): silencing the writers' inputs adds 55 relearning steps after RMU (slower in 1.00 of six runs) and 40 after ascent; random rows 5 / 0.
- Refusal (e604): safety training left the harmful-request writers' rows at cosine 0.9973 with the base and made them fire 2.03 times more; the prefix jailbreaks do not move refusal on this model; benign fine-tuning lowers refusal to 0.37 with the writers still firing (0.96). Merging (e605): different rows for different domains (Jaccard 0.02), the averaged model keeps 1.00 / 0.90 of the entrants. Compression (e606): damage tracks the writers' silencing across compressions (Spearman -0.94) through severity, not within one compression across domains.

**Start here:** e602b, e603b, e604 · **Sessions:** S112 · **Scripts:** `scripts/e602_selection_baseline.py`, `e603_separable.py`, `e604_refusal_audit.py`, `e605_merging_audit.py`, `e606_compression_audit.py`

## Experiments

| id | question | result | status | links |
| --- | --- | --- | --- | --- |
| e602 | The naive-selection baseline with seeds: the five methods rerun on three model-domain pairs with six neuron sets tracked side by side (WDD writers at their classes and at all forget positions, WDD retain words, difference-, ratio- and magnitude-selected neurons, a random set, all neurons), the same attacks | Spearman with the 20-step recovery over 54 conditions: WDD writers 0.89, difference-selected 0.83, magnitude-selected 0.56, random -0.01, still-writing 0.36, probe 0.73; overlap of the difference-selected set with the writers Jaccard 0.06; seed agreement 0.98 | narrowed (a naive selection predicts relearnability about as well; the audit's substance is domain-selective neurons still firing) | ← e597 · → e602b e603 · 42 |
| e602b | The runs of e602 summarised: Spearman tables by model, domain and method; overlaps; seed agreement | The tables of area 43 | tool (summary) | ← e602 |
| e603 | Separable interventions, three seeds and two domains: capped ascent and RMU to +2 nats, each alone and followed by the writers' write columns zeroed, their inputs zeroed, random rows zeroed, or the difference-selected neurons' inputs zeroed; the attacks | Added relearning steps over the method alone: inputs zeroed 55 after RMU (slower in 1.00), 40 after ascent; rows zeroed 55 / 40; random 5 / 0; difference-selected 50 / 25 | superseded (the steps-back delay was a floor artefact: e618e, session 117) | ← e601 · → e603b · 42 |
| e603b | The runs of e603 summarised over seeds | The tables of area 43 | tool (summary) | ← e603 |
| e604 | The refusal audit on Qwen2.5-0.5B base and instruct: harmful-request writers' rows and activation across safety training, two prefix jailbreaks, one epoch of benign fine-tuning, a domain probe | Rows at cosine 0.9973 with the base (all rows 0.9972), activation in the base 0.49 of the instruct's; refusal 0.93 to 0.96 / 0.93 under the jailbreaks with the writers at 0.97 / 0.98; benign fine-tuning to 0.37 with the writers at 0.96 | established (safety training recruits existing writers; benign fine-tuning erodes refusal with them still firing; the prefix jailbreaks did not move this model) | ← e597 e600 · → 42 |
| e605 | The merging audit: two domain fine-tunes of Pythia-160m averaged, added in full, each alone and at half strength, and sequential; the entrants followed | Entrants' Jaccard 0.02; averaged model keeps 1.00 / 0.90 of the entrants writing, sequential 1.00 of Github's; losses 1.102 / 2.652 against 1.000 / 2.575 alone | established (different rows for different domains; merging keeps both) | ← e600 · → 42 |
| e606 | The compression audit on Pythia-410m: six domains, six compressions (8-bit, 4-bit in groups of 64 and 32, 3-bit, pruning 30% and 50%); per-domain damage against the domain's writers' activation and silenced share | Spearman of damage with the writers' activation -0.94 over 36 pairs (silenced share 0.82); pruning 50% silences 0.85 for +4.091 nats, 3-bit 0.78 for +2.265, 4-bit g64 0.04 for +0.355 | narrowed (across compressions the damage tracks the silencing through severity; within one compression, across domains, it does not) | ← e597 · → 42 |

## How the results flow

- `e597 → e602 → e602b`: the audit's number, then the naive sets that could replace it, with seeds.
- `e601 → e603 → e603b`: the one-seed hybrid, then the separable interventions with three seeds and the selection control.
- `e597, e600 → e604, e605, e606`: the audit's questions asked of safety training, of merging, and of compression.

## Links to other areas

- [42 The unlearning audit, tested as its users would](42_audit_users.md): the claims these experiments test.
- [31 The coalitions](31_coalitions.md): why a writer can be overridden or silenced without being touched.
- [40 The mechanism of the turn, the removed speaker, the paper](40_turn_and_paper.md): the recruitment picture that safety training and fine-tuning repeat.
