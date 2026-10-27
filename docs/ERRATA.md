# Errata

## 1. The doubled final norm in the readout of sessions 155 to 164 (found 2026-10-20, session 165)

**The flaw.** The shared helper `scripts/c11.py`, used by every experiment from e680 to e724 (42 scripts), computed next-token log-probabilities with `logprobs_at(M, h)`, which applied the model's final norm to `h` before the unembedding. Every caller passed `M["base"](...).last_hidden_state`, which the Hugging Face base model (`GPT2Model`, `GPTNeoXModel`, `Qwen2Model`) returns after that same norm. The unembedding was therefore fed a doubly-normalised state. For a LayerNorm with a non-uniform gain and for an RMSNorm with a large gain the second pass changes the logits.

**The size of the error** (`scripts/check_norm.py`, `results/e725/check_norm_float32.json`; eight windows of 256 tokens of wikitext-2 test): the KL from the model's own next-token distribution to the pipeline's, and the share of positions where the two agree on the top token.

| model | base loss, pipeline | base loss, reference | KL (reference to pipeline), nats | top-1 agreement |
| --- | --- | --- | --- | --- |
| GPT-2 small | 6.55 | 3.91 | 2.51 | 0.24 |
| GPT-2 medium | 6.54 | 3.60 | 2.85 | 0.35 |
| Pythia-160m | 4.59 | 4.02 | 0.50 | 0.61 |
| Pythia-410m | 3.59 | 3.53 | 0.04 | 0.91 |
| Qwen2.5-0.5B | 3.63 | 3.02 | 0.62 | 0.54 |

Applying the unembedding once to `last_hidden_state` reproduces the reference exactly (KL 0.00000). The pipeline's own base losses (GPT-2 small 6.6 nats, SmolLM2-135M 13.9, above the uniform distribution's 10.8) had been printed in e681's result files since session 155 and were not read as the alarm they were.

**What it touches.** Every KL-based number of sessions 155 to 164: the cost of the weak activations (e681, e720), the brake's KL ratios and off-class shares (e697, e716, e719), the fields (e703, e706, e713, e717, e723), the bypass and the ladder (e693, e696), read energy and the Fisher pursuit (e707, e714), the thermostat (e708), the dictionary comparisons (e715, e718, e724), and the bootstrap intervals derived from them (e722). The ablation and the edited model were read through the same distorted readout, so each recorded number is a KL between two distorted distributions; its ordering relative to another such number is not guaranteed to match the true ordering, which is why the core results were re-measured rather than rescaled.

**What it does not touch.** Results computed from the ledger alone, without a readout: the crowd's net sums and shares (e685b, e695), the brake's formation during training (e701, e721), the readers' concentration (e704), the field overlaps (e705), the clocks and the edge program (e709, e712); e711, which trains with the model's own loss; e682, e688 and e690, which read `.logits`; and every phase before e680, whose scripts applied the unembedding to a hooked pre-norm residual.

**The fix.** `c11.logprobs_at(M, h, normed=True)` applies the unembedding directly to a normalised hidden state and the norm only when told the residual is pre-norm; `wdd/model.py` has the same change; `c11.save` honours `WDD_TAG_SUFFIX` so that corrected results sit beside the recorded ones as `*_v2.json`.

**The re-measurement.** Session 165 (atlas area 73, `docs/atlas/73_corrected_readout.md`; review section 29) re-ran the core KL-based experiments with the corrected readout and the same scripts, neurons, windows and seeds, with sixteen statements pre-registered in `results/e725_prereg.json` about which recorded claims would survive. The recorded files are kept; every corrected number stands beside its recorded one in area 73 and in the table below.

**Which recorded claims survive.** Of the sixteen pre-registered statements, fourteen held and two failed (`docs/REVIEW_POSITIVE_CLAIMS.md` section 29). Survive: the cost of the weak activations in every model (0.32 to 8.06 nats, the mean replacement 0.14 to 0.98); the brake's excitatory runaway in every block (4 to 503 times the ablation damage) and its repair by the inhibitory wires; the off-class share where it was strong (Pythia-410m, Qwen2.5-0.5B, SmolLM2, unchanged to two decimals) and on the Pile; the fields halving a pruned neuron's error at its firings against a random subspace; a trained dictionary beating the native rows on a state and the fields adding nothing there. Fall: GPT-2 medium's off-class share, 0.64 recorded, 0.40 and 0.46 corrected, with 42 percent of its neurons above half; the signed wires beating the magnitude wires at the firings, now five blocks of eight; the cost at 7B, 1.57 recorded, 0.32 corrected; every eight-perturbation intervention fidelity, which fell by 0.05 to 0.3, so that the native rows' advantage over jbloom's L1 encoder under intervention holds at one block of two; the side-claim of e681 that dial neurons cost twice detector neurons, reversed at GPT-2 small. Not yet re-measured: e693, e696, e707, e708, e714 and the 7B brake and fields (e713), which need the 80 GB card.

**The corrected numbers** (the full comparison is `docs/atlas/73_corrected_readout.md`; the corrected files are `results/<id>/*_v2.json`, the intervals `results/e722/e722_intervals_v2.json`).

| experiment | run | statistic | recorded | corrected |
| --- | --- | --- | --- | --- |
| e681 | GPT-2 small | KL, strongest quarter kept (nats) | 0.24 | 2.10 |
| e681 | GPT-2 small | KL, mean replacement | 0.07 | 0.19 |
| e681 | GPT-2 large | KL, strongest quarter kept (nats) | 3.25 | 1.97 |
| e681 | GPT-2 large | KL, mean replacement | 0.19 | 0.14 |
| e681 | GPT-2 medium | KL, strongest quarter kept (nats) | 0.95 | 3.96 |
| e681 | GPT-2 medium | KL, mean replacement | 0.40 | 0.29 |
| e681 | OLMo-1B | KL, strongest quarter kept (nats) | 0.93 | 0.93 |
| e681 | OLMo-1B | KL, mean replacement | 0.31 | 0.31 |
| e681 | Pythia-160m | KL, strongest quarter kept (nats) | 1.54 | 3.37 |
| e681 | Pythia-160m | KL, mean replacement | 0.47 | 0.98 |
| e681 | Pythia-1B | KL, strongest quarter kept (nats) | 5.38 | 4.45 |
| e681 | Pythia-1B | KL, mean replacement | 0.36 | 0.33 |
| e681 | Pythia-410m | KL, strongest quarter kept (nats) | 8.68 | 8.06 |
| e681 | Pythia-410m | KL, mean replacement | 0.73 | 0.69 |
| e681 | Qwen2.5-0.5B | KL, strongest quarter kept (nats) | 0.99 | 0.89 |
| e681 | Qwen2.5-0.5B | KL, mean replacement | 0.61 | 0.53 |
| e681 | Qwen2.5-1.5B | KL, strongest quarter kept (nats) | 0.51 | 0.44 |
| e681 | Qwen2.5-1.5B | KL, mean replacement | 0.39 | 0.29 |
| e681 | SmolLM2-135M | KL, strongest quarter kept (nats) | 2.22 | 0.93 |
| e681 | SmolLM2-135M | KL, mean replacement | 2.31 | 0.72 |
| e681 | SmolLM2-360M | KL, strongest quarter kept (nats) | 1.31 | 0.73 |
| e681 | SmolLM2-360M | KL, mean replacement | 0.67 | 0.43 |
| e720 | Qwen2.5-7B | KL, strongest quarter kept (nats) | 1.57 | 0.32 |
| e720 | Qwen2.5-7B | KL, mean replacement | 0.88 | 0.23 |
| e697 | GPT-2 medium b12 | off-class share of the 16-input damage | 0.64 | 0.46 |
| e697 | GPT-2 medium b12 | 16 excitatory wires alone, all positions (x ablation) | 6 | 4 |
| e697 | GPT-2 medium b12 | signed wires at the firings / 16 by magnitude | 0.97 / 0.75 | 0.98 / 0.82 |
| e697 | GPT-2 medium b6 | off-class share of the 16-input damage | 0.64 | 0.40 |
| e697 | GPT-2 medium b6 | 16 excitatory wires alone, all positions (x ablation) | 10 | 18 |
| e697 | GPT-2 medium b6 | signed wires at the firings / 16 by magnitude | 0.68 / 0.41 | 0.56 / 0.37 |
| e697 | Pythia-410m b12 | off-class share of the 16-input damage | 0.81 | 0.81 |
| e697 | Pythia-410m b12 | 16 excitatory wires alone, all positions (x ablation) | 45 | 45 |
| e697 | Pythia-410m b12 | signed wires at the firings / 16 by magnitude | 0.82 / 1.08 | 0.83 / 1.10 |
| e697 | Pythia-410m b18 | off-class share of the 16-input damage | 0.78 | 0.78 |
| e697 | Pythia-410m b18 | 16 excitatory wires alone, all positions (x ablation) | 27 | 26 |
| e697 | Pythia-410m b18 | signed wires at the firings / 16 by magnitude | 0.63 / 0.74 | 0.64 / 0.75 |
| e697 | Pythia-410m b6 | off-class share of the 16-input damage | 0.52 | 0.52 |
| e697 | Pythia-410m b6 | 16 excitatory wires alone, all positions (x ablation) | 14 | 14 |
| e697 | Pythia-410m b6 | signed wires at the firings / 16 by magnitude | 0.59 / 0.51 | 0.59 / 0.49 |
| e697 | Qwen2.5-0.5B b12 | off-class share of the 16-input damage | 0.88 | 0.88 |
| e697 | Qwen2.5-0.5B b12 | 16 excitatory wires alone, all positions (x ablation) | 561 | 503 |
| e697 | Qwen2.5-0.5B b12 | signed wires at the firings / 16 by magnitude | 1.06 / 1.95 | 1.07 / 1.97 |
| e697 | Qwen2.5-0.5B b6 | off-class share of the 16-input damage | 0.86 | 0.86 |
| e697 | Qwen2.5-0.5B b6 | 16 excitatory wires alone, all positions (x ablation) | 130 | 132 |
| e697 | Qwen2.5-0.5B b6 | signed wires at the firings / 16 by magnitude | 0.61 / 0.64 | 0.61 / 0.67 |
| e697 | SmolLM2-135M b15 | off-class share of the 16-input damage | 0.82 | 0.82 |
| e697 | SmolLM2-135M b15 | 16 excitatory wires alone, all positions (x ablation) | 211 | 191 |
| e697 | SmolLM2-135M b15 | signed wires at the firings / 16 by magnitude | 0.70 / 1.68 | 0.69 / 1.49 |
| e719 | GPT-2 medium b12 (Pile) | off-class share at 5 percent | 0.74 | 0.46 |
| e719 | Pythia-410m b12 (Pile) | off-class share at 5 percent | 0.81 | 0.82 |
| e719 | Qwen2.5-0.5B b12 (Pile) | off-class share at 5 percent | 0.86 | 0.85 |
| e706 | GPT-2 medium b12 | at the firings: 16 wires / with both fields / with a random subspace | 0.75 / 0.22 / 0.56 | 0.82 / 0.20 / 0.66 |
| e706 | GPT-2 medium b6 | at the firings: 16 wires / with both fields / with a random subspace | 0.41 / 0.21 / 0.40 | 0.37 / 0.27 / 0.44 |
| e706 | Pythia-410m b12 | at the firings: 16 wires / with both fields / with a random subspace | 1.08 / 0.30 / 0.63 | 1.10 / 0.30 / 0.62 |
| e706 | Pythia-410m b18 | at the firings: 16 wires / with both fields / with a random subspace | 0.74 / 0.19 / 0.27 | 0.75 / 0.19 / 0.27 |
| e706 | Pythia-410m b6 | at the firings: 16 wires / with both fields / with a random subspace | 0.51 / 0.25 / 0.40 | 0.49 / 0.25 / 0.39 |
| e706 | Qwen2.5-0.5B b12 | at the firings: 16 wires / with both fields / with a random subspace | 1.95 / 0.53 / 1.15 | 1.97 / 0.52 / 1.14 |
| e706 | Qwen2.5-0.5B b6 | at the firings: 16 wires / with both fields / with a random subspace | 0.64 / 0.35 / 0.61 | 0.67 / 0.36 / 0.62 |
| e706 | SmolLM2-135M b15 | at the firings: 16 wires / with both fields / with a random subspace | 1.68 / 0.12 / 0.27 | 1.49 / 0.13 / 0.27 |
| e715 | GPT-2 small b5 | recovered: rows / SAE top 16 / SAE by pursuit | 0.95 / 0.97 / 0.98 | 0.95 / 0.97 / 0.98 |
| e715 | GPT-2 small b5 | fidelity: rows / SAE top 16 / SAE by pursuit | 0.71 / 0.60 / 0.83 | 0.54 / 0.55 / 0.75 |
| e715 | GPT-2 small b8 | recovered: rows / SAE top 16 / SAE by pursuit | 0.92 / 0.95 / 0.97 | 0.90 / 0.94 / 0.97 |
| e715 | GPT-2 small b8 | fidelity: rows / SAE top 16 / SAE by pursuit | 0.65 / 0.53 / 0.83 | 0.50 / 0.41 / 0.76 |
| e718 | GPT-2 small b5 oai128k | recovered: rows / SAE top 16 / SAE by pursuit | 0.95 / 0.98 / 0.93 | 0.95 / 0.99 / 0.95 |
| e718 | GPT-2 small b5 oai128k | fidelity: rows / SAE top 16 / SAE by pursuit | 0.71 / 0.84 / 0.49 | 0.54 / 0.78 / 0.30 |
| e718 | GPT-2 small b5 oai32k | recovered: rows / SAE top 16 / SAE by pursuit | 0.95 / 0.98 / 0.92 | 0.95 / 0.99 / 0.92 |
| e718 | GPT-2 small b5 oai32k | fidelity: rows / SAE top 16 / SAE by pursuit | 0.71 / 0.83 / 0.45 | 0.54 / 0.73 / 0.23 |
| e718 | GPT-2 small b8 oai128k | recovered: rows / SAE top 16 / SAE by pursuit | 0.92 / 0.97 / 0.93 | 0.90 / 0.97 / 0.92 |
| e718 | GPT-2 small b8 oai128k | fidelity: rows / SAE top 16 / SAE by pursuit | 0.65 / 0.84 / 0.37 | 0.50 / 0.81 / 0.17 |
| e718 | GPT-2 small b8 oai32k | recovered: rows / SAE top 16 / SAE by pursuit | 0.92 / 0.97 / 0.93 | 0.90 / 0.95 / 0.90 |
| e718 | GPT-2 small b8 oai32k | fidelity: rows / SAE top 16 / SAE by pursuit | 0.65 / 0.83 / 0.34 | 0.50 / 0.81 / 0.06 |
| e718 | Pythia-160m b5 el32k | recovered: rows / SAE top 16 / SAE by pursuit | 0.72 / 0.93 / 0.93 | 0.77 / 0.94 / 0.94 |
| e718 | Pythia-160m b5 el32k | fidelity: rows / SAE top 16 / SAE by pursuit | 0.75 / 0.91 / 0.91 | 0.63 / 0.87 / 0.86 |
| e718 | Pythia-160m b8 el32k | recovered: rows / SAE top 16 / SAE by pursuit | 0.72 / 0.90 / 0.90 | 0.77 / 0.91 / 0.92 |
| e718 | Pythia-160m b8 el32k | fidelity: rows / SAE top 16 / SAE by pursuit | 0.89 / 0.95 / 0.95 | 0.81 / 0.88 / 0.89 |
| e723 | GPT-2 small b6 jbloom | at the firings, 16 directions: fields / SAE / random; at 192: fields / SAE / random | 0.24 / 0.31 / 0.25; 0.14 / 0.15 / 0.25 | 0.23 / 0.30 / 0.33; 0.12 / 0.15 / 0.26 |
| e723 | GPT-2 small b6 oai32k | at the firings, 16 directions: fields / SAE / random; at 192: fields / SAE / random | 0.24 / 0.23 / 0.25; 0.14 / 0.16 / 0.25 | 0.23 / 0.30 / 0.33; 0.12 / 0.14 / 0.26 |
| e723 | GPT-2 small b9 jbloom | at the firings, 16 directions: fields / SAE / random; at 288: fields / SAE / random | 0.46 / 0.42 / 0.51; 0.18 / 0.09 / 0.30 | 0.38 / 0.33 / 0.40; 0.12 / 0.08 / 0.26 |
| e723 | GPT-2 small b9 oai32k | at the firings, 16 directions: fields / SAE / random; at 288: fields / SAE / random | 0.46 / 0.41 / 0.51; 0.18 / 0.12 / 0.30 | 0.38 / 0.36 / 0.40; 0.12 / 0.11 / 0.26 |
| e724 | GPT-2 small b5 jbloom | recovered: SAE full / plus 16 field atoms; fidelity: SAE full / plus 16 field atoms | 0.988 / 0.993; 0.85 / 0.95 | 0.989 / 0.993; 0.79 / 0.90 |
| e724 | GPT-2 small b5 oai32k | recovered: SAE full / plus 16 field atoms; fidelity: SAE full / plus 16 field atoms | 0.989 / 0.995; 0.85 / 0.97 | 0.990 / 0.995; 0.80 / 0.92 |
| e724 | GPT-2 small b8 jbloom | recovered: SAE full / plus 16 field atoms; fidelity: SAE full / plus 16 field atoms | 0.980 / 0.985; 0.83 / 0.93 | 0.974 / 0.981; 0.78 / 0.91 |
| e724 | GPT-2 small b8 oai32k | recovered: SAE full / plus 16 field atoms; fidelity: SAE full / plus 16 field atoms | 0.976 / 0.982; 0.88 / 0.96 | 0.976 / 0.985; 0.84 / 0.93 |

## 2. The remaining re-measurements (session 166, 2026-10-27)

Session 166 re-measured the five KL-based results that session 165 had left (e693, e696, e707, e708, e714): 33 runs (e693 8, e696 7, e707 6, e708 6, e714 6), the same scripts, neurons, windows and seeds, the corrected readout, results as `*_v2.json` beside the recorded files; the comparison with the recorded verdicts is in `docs/atlas/74_zoo_through_wdd.md` and `FINDINGS.md`. Recorded verdicts that flip under the corrected readout: e693: GPT-2 medium (V5_svd16_halves_zero), Qwen2.5-0.5B (V1_token_halves_zero), Qwen2.5-1.5B (V1c_token_beats_nmean), SmolLM2-360M (V1_token_halves_zero); e696: none; e707: GPT-2 small b5 (R3_weak_less_read_than_strong), Pythia-160m b6 (R2_fisher_predicts_finite_kl); e708: none; e714: none. Not re-measured and still marked as measured through the doubled norm: e703 (the rows' field at the firings; superseded by e706's design), e713 (the 7B fields and brake; the 7B model in float32 does not fit a 40 GB card) and e717 (the SAE readers; superseded by e723). The recorded files are kept.

| experiment | run | statistic | recorded | corrected |
| --- | --- | --- | --- | --- |
| e693 | GPT-2 large | KL, weak three quarters zeroed (nats) | 3.29 | 1.99 |
| e693 | GPT-2 large | KL, neuron mean | 0.20 | 0.14 |
| e693 | GPT-2 large | KL, per-token table | 0.15 | 0.11 |
| e693 | GPT-2 large | KL, 16 principal directions | 0.12 | 0.08 |
| e693 | GPT-2 medium | KL, weak three quarters zeroed (nats) | 0.95 | 3.96 |
| e693 | GPT-2 medium | KL, neuron mean | 0.41 | 0.28 |
| e693 | GPT-2 medium | KL, per-token table | 0.41 | 0.31 |
| e693 | GPT-2 medium | KL, 16 principal directions | 0.20 | 0.12 |
| e693 | OLMo-1B | KL, weak three quarters zeroed (nats) | 0.97 | 0.97 |
| e693 | OLMo-1B | KL, neuron mean | 0.31 | 0.31 |
| e693 | OLMo-1B | KL, per-token table | 0.30 | 0.30 |
| e693 | OLMo-1B | KL, 16 principal directions | 0.20 | 0.20 |
| e693 | Pythia-1B | KL, weak three quarters zeroed (nats) | 5.37 | 4.44 |
| e693 | Pythia-1B | KL, neuron mean | 0.36 | 0.33 |
| e693 | Pythia-1B | KL, per-token table | 0.27 | 0.24 |
| e693 | Pythia-1B | KL, 16 principal directions | 0.26 | 0.24 |
| e693 | Pythia-410m | KL, weak three quarters zeroed (nats) | 8.62 | 8.00 |
| e693 | Pythia-410m | KL, neuron mean | 0.74 | 0.70 |
| e693 | Pythia-410m | KL, per-token table | 0.54 | 0.51 |
| e693 | Pythia-410m | KL, 16 principal directions | 0.53 | 0.50 |
| e693 | Qwen2.5-0.5B | KL, weak three quarters zeroed (nats) | 0.99 | 0.89 |
| e693 | Qwen2.5-0.5B | KL, neuron mean | 0.60 | 0.53 |
| e693 | Qwen2.5-0.5B | KL, per-token table | 0.48 | 0.47 |
| e693 | Qwen2.5-0.5B | KL, 16 principal directions | 0.25 | 0.31 |
| e693 | Qwen2.5-1.5B | KL, weak three quarters zeroed (nats) | 0.52 | 0.44 |
| e693 | Qwen2.5-1.5B | KL, neuron mean | 0.39 | 0.29 |
| e693 | Qwen2.5-1.5B | KL, per-token table | 0.29 | 0.29 |
| e693 | Qwen2.5-1.5B | KL, 16 principal directions | 0.18 | 0.21 |
| e693 | SmolLM2-360M | KL, weak three quarters zeroed (nats) | 1.33 | 0.73 |
| e693 | SmolLM2-360M | KL, neuron mean | 0.71 | 0.44 |
| e693 | SmolLM2-360M | KL, per-token table | 0.59 | 0.40 |
| e693 | SmolLM2-360M | KL, 16 principal directions | 0.30 | 0.24 |
| e696 | GPT-2 large | KL, zeroed (nats) | 3.29 | 1.99 |
| e696 | GPT-2 large | KL, 16 oracle directions | 0.12 | 0.08 |
| e696 | GPT-2 large | KL, 16 predicted directions | 0.14 | 0.10 |
| e696 | GPT-2 large | KL, full linear bypass | 0.12 | 0.08 |
| e696 | GPT-2 medium | KL, zeroed (nats) | 0.95 | 3.96 |
| e696 | GPT-2 medium | KL, 16 oracle directions | 0.20 | 0.12 |
| e696 | GPT-2 medium | KL, 16 predicted directions | 0.25 | 0.15 |
| e696 | GPT-2 medium | KL, full linear bypass | 0.25 | 0.14 |
| e696 | OLMo-1B | KL, zeroed (nats) | 0.97 | 0.97 |
| e696 | OLMo-1B | KL, 16 oracle directions | 0.20 | 0.20 |
| e696 | OLMo-1B | KL, 16 predicted directions | 0.29 | 0.29 |
| e696 | OLMo-1B | KL, full linear bypass | 0.27 | 0.27 |
| e696 | Pythia-1B | KL, zeroed (nats) | 5.37 | 4.44 |
| e696 | Pythia-1B | KL, 16 oracle directions | 0.26 | 0.24 |
| e696 | Pythia-1B | KL, 16 predicted directions | 0.29 | 0.27 |
| e696 | Pythia-1B | KL, full linear bypass | 0.20 | 0.19 |
| e696 | Pythia-410m | KL, zeroed (nats) | 8.62 | 8.00 |
| e696 | Pythia-410m | KL, 16 oracle directions | 0.53 | 0.50 |
| e696 | Pythia-410m | KL, 16 predicted directions | 0.56 | 0.54 |
| e696 | Pythia-410m | KL, full linear bypass | 0.35 | 0.35 |
| e696 | Qwen2.5-0.5B | KL, zeroed (nats) | 0.99 | 0.89 |
| e696 | Qwen2.5-0.5B | KL, 16 oracle directions | 0.25 | 0.31 |
| e696 | Qwen2.5-0.5B | KL, 16 predicted directions | 0.36 | 0.40 |
| e696 | Qwen2.5-0.5B | KL, full linear bypass | 0.28 | 0.30 |
| e696 | SmolLM2-360M | KL, zeroed (nats) | 1.33 | 0.73 |
| e696 | SmolLM2-360M | KL, 16 oracle directions | 0.30 | 0.24 |
| e696 | SmolLM2-360M | KL, 16 predicted directions | 0.58 | 0.36 |
| e696 | SmolLM2-360M | KL, full linear bypass | 0.49 | 0.30 |
| e707 | GPT-2 small b5 | read energy per unit energy against random: content | 1.71 | 1.64 |
| e707 | GPT-2 small b5 | strong write | 0.72 | 0.67 |
| e707 | GPT-2 small b5 | weak write | 0.72 | 0.69 |
| e707 | GPT-2 small b5 | rows | 1.93 | 1.82 |
| e707 | GPT-2 small b5 | SAE error | 1.53 | 1.63 |
| e707 | GPT-2 small b5 | read dimension k90 | 6.00 | 6.00 |
| e707 | GPT-2 small b8 | read energy per unit energy against random: content | 1.92 | 1.11 |
| e707 | GPT-2 small b8 | strong write | 0.83 | 0.89 |
| e707 | GPT-2 small b8 | weak write | 1.29 | 1.32 |
| e707 | GPT-2 small b8 | rows | 1.66 | 1.27 |
| e707 | GPT-2 small b8 | SAE error | 1.38 | 1.44 |
| e707 | GPT-2 small b8 | read dimension k90 | 5.00 | 6.00 |
| e707 | GPT-2 medium b12 | read energy per unit energy against random: content | 1.77 | 1.61 |
| e707 | GPT-2 medium b12 | strong write | 0.96 | 1.07 |
| e707 | GPT-2 medium b12 | weak write | 1.41 | 1.44 |
| e707 | GPT-2 medium b12 | rows | 1.81 | 1.97 |
| e707 | GPT-2 medium b12 | SAE error | n/a | n/a |
| e707 | GPT-2 medium b12 | read dimension k90 | 5.00 | 8.00 |
| e707 | Pythia-160m b6 | read energy per unit energy against random: content | 1.39 | 1.43 |
| e707 | Pythia-160m b6 | strong write | 0.70 | 0.69 |
| e707 | Pythia-160m b6 | weak write | 1.23 | 1.27 |
| e707 | Pythia-160m b6 | rows | 1.01 | 0.96 |
| e707 | Pythia-160m b6 | SAE error | n/a | n/a |
| e707 | Pythia-160m b6 | read dimension k90 | 7.00 | 6.00 |
| e707 | Pythia-410m b12 | read energy per unit energy against random: content | 1.17 | 1.15 |
| e707 | Pythia-410m b12 | strong write | 0.91 | 0.93 |
| e707 | Pythia-410m b12 | weak write | 0.97 | 0.98 |
| e707 | Pythia-410m b12 | rows | 1.09 | 1.08 |
| e707 | Pythia-410m b12 | SAE error | n/a | n/a |
| e707 | Pythia-410m b12 | read dimension k90 | 8.00 | 8.00 |
| e707 | Qwen2.5-0.5B b12 | read energy per unit energy against random: content | 0.88 | 0.81 |
| e707 | Qwen2.5-0.5B b12 | strong write | 0.99 | 1.00 |
| e707 | Qwen2.5-0.5B b12 | weak write | 1.05 | 1.08 |
| e707 | Qwen2.5-0.5B b12 | rows | 0.86 | 0.81 |
| e707 | Qwen2.5-0.5B b12 | SAE error | n/a | n/a |
| e707 | Qwen2.5-0.5B b12 | read dimension k90 | 8.00 | 7.00 |
| e708 | GPT-2 medium b12 | alpha 0.5: total compensation | 0.33 | 0.33 |
| e708 | GPT-2 medium b12 | alpha 0.5: KL ratio | 0.32 | 0.01 |
| e708 | GPT-2 medium b12 | alpha 2.0: total compensation | 0.34 | 0.34 |
| e708 | GPT-2 medium b12 | alpha 2.0: KL ratio | 0.30 | 0.01 |
| e708 | OLMo-1B b8 | alpha 0.5: total compensation | 0.76 | 0.76 |
| e708 | OLMo-1B b8 | alpha 0.5: KL ratio | 270 | 264 |
| e708 | OLMo-1B b8 | alpha 2.0: total compensation | 0.69 | 0.69 |
| e708 | OLMo-1B b8 | alpha 2.0: KL ratio | 233 | 226 |
| e708 | Pythia-160m b6 | alpha 0.5: total compensation | 0.73 | 0.73 |
| e708 | Pythia-160m b6 | alpha 0.5: KL ratio | 6.85 | 7.25 |
| e708 | Pythia-160m b6 | alpha 2.0: total compensation | 0.70 | 0.70 |
| e708 | Pythia-160m b6 | alpha 2.0: KL ratio | 5.89 | 6.07 |
| e708 | Pythia-410m b12 | alpha 0.5: total compensation | 0.50 | 0.50 |
| e708 | Pythia-410m b12 | alpha 0.5: KL ratio | 9.88 | 9.25 |
| e708 | Pythia-410m b12 | alpha 2.0: total compensation | 0.45 | 0.45 |
| e708 | Pythia-410m b12 | alpha 2.0: KL ratio | 11.28 | 10.60 |
| e708 | Qwen2.5-0.5B b12 | alpha 0.5: total compensation | 0.72 | 0.72 |
| e708 | Qwen2.5-0.5B b12 | alpha 0.5: KL ratio | 66.23 | 51.18 |
| e708 | Qwen2.5-0.5B b12 | alpha 2.0: total compensation | 0.69 | 0.69 |
| e708 | Qwen2.5-0.5B b12 | alpha 2.0: KL ratio | 61.52 | 46.88 |
| e708 | SmolLM2-135M b15 | alpha 0.5: total compensation | 0.44 | 0.44 |
| e708 | SmolLM2-135M b15 | alpha 0.5: KL ratio | 0.03 | 0.02 |
| e708 | SmolLM2-135M b15 | alpha 2.0: total compensation | 0.43 | 0.43 |
| e708 | SmolLM2-135M b15 | alpha 2.0: KL ratio | 0.03 | 0.02 |
| e714 | GPT-2 small b5 | recovered: Euclidean pursuit, 16 rows | 0.98 | 0.97 |
| e714 | GPT-2 small b5 | blended pursuit, 16 rows and fields | 0.97 | 0.97 |
| e714 | GPT-2 small b5 | pure Fisher pursuit | 0.47 | 0.32 |
| e714 | GPT-2 small b5 | 16 stream PCs | 0.92 | 0.89 |
| e714 | GPT-2 small b8 | recovered: Euclidean pursuit, 16 rows | 0.90 | 0.90 |
| e714 | GPT-2 small b8 | blended pursuit, 16 rows and fields | 0.92 | 0.92 |
| e714 | GPT-2 small b8 | pure Fisher pursuit | 0.36 | 0.01 |
| e714 | GPT-2 small b8 | 16 stream PCs | 0.83 | 0.68 |
| e714 | GPT-2 medium b12 | recovered: Euclidean pursuit, 16 rows | 0.93 | 0.93 |
| e714 | GPT-2 medium b12 | blended pursuit, 16 rows and fields | 0.93 | 0.92 |
| e714 | GPT-2 medium b12 | pure Fisher pursuit | 0.46 | 0.17 |
| e714 | GPT-2 medium b12 | 16 stream PCs | 0.82 | 0.70 |
| e714 | Pythia-160m b6 | recovered: Euclidean pursuit, 16 rows | 0.72 | 0.75 |
| e714 | Pythia-160m b6 | blended pursuit, 16 rows and fields | 0.68 | 0.71 |
| e714 | Pythia-160m b6 | pure Fisher pursuit | -0.19 | -0.23 |
| e714 | Pythia-160m b6 | 16 stream PCs | 0.46 | 0.44 |
| e714 | Pythia-410m b12 | recovered: Euclidean pursuit, 16 rows | 0.69 | 0.68 |
| e714 | Pythia-410m b12 | blended pursuit, 16 rows and fields | 0.68 | 0.67 |
| e714 | Pythia-410m b12 | pure Fisher pursuit | -0.09 | -0.15 |
| e714 | Pythia-410m b12 | 16 stream PCs | 0.46 | 0.44 |
| e714 | Qwen2.5-0.5B b12 | recovered: Euclidean pursuit, 16 rows | 0.88 | 0.90 |
| e714 | Qwen2.5-0.5B b12 | blended pursuit, 16 rows and fields | 0.86 | 0.88 |
| e714 | Qwen2.5-0.5B b12 | pure Fisher pursuit | -0.07 | -0.14 |
| e714 | Qwen2.5-0.5B b12 | 16 stream PCs | 0.69 | 0.68 |
