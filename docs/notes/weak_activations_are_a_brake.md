# The weak activations are a brake

**Six pre-registered results on what the sub-threshold activations of a transformer's MLP neurons do, from an exact ledger of each neuron's inputs.**

Ronish Bhatt, with Claude (Anthropic) as the experimenter. 20 October 2026. A technical note from the WDD descendants program; everything here is in this repository (`scripts_phase3/`, `results/`, `docs/atlas/` areas 62 to 69, the `wdd/` library), with the pre-registrations in `results/*_prereg.json`.

## Read this first

Six pre-registered results, models from 70M to 7B, one exact instrument. The question they answer is the one both sides of the "already clean" debate left open: do the weak activations of a transformer's MLP neurons matter, and what do they do?

1. **They matter.** Keep only the strongest quarter of each neuron's activations and next-token KL rises by 0.24 to 8.68 nats in eleven models; replacing them by each neuron's mean still leaves 0.07 to 2.31. (e681)
2. **They are a brake.** A neuron driven by its sixteen largest inputs alone misfires off its class, where the crowd of thousands of small inputs would have held it down: 52 to 88 percent of the damage is off-class, sixteen excitatory wires alone cost 6 to 561 times the neuron's ablation damage, and sixteen inhibitory wires bring that to 0.6 to 3. At 7B: 85 to 87 percent off-class, 59 to 284 times, 1.1 to 2.7 with the brake. (e697, e713)
3. **The brake is learned.** The crowd pushes with the wires at step 1000 of Pythia-160m (+0.3), crosses zero by step 2000 and reaches -0.6 to -0.8 at the end, monotone in three seeds; in OLMo-1B it crosses -0.5 at a third to two thirds of training. (e701)
4. **It is low-rank and linear in the input.** A block's weak write lives in sixteen directions, and a closed-form rank-16 linear map from the block's normalised input, with the weak neurons never run, recovers 0.56 to 0.96 of the zeroing cost, while explaining less than half the weak write's variance. (e696)
5. **Those directions are a second kind of atom.** Sixteen wires plus the weak field plus a strong-write field bring a pruned neuron's error at its firings from 0.41 to 1.95 of the ablation damage to 0.12 to 0.53, and from 0.9 and 2.9 to 0.5 and 1.0 at 7B. (e706, e713)
6. **The output grades by a quadratic form, not by energy.** The Fisher information of the model's own next tokens predicts the cost of removing any part of a state at Spearman 0.83 to 0.92; by it the fields are read at about twice a random direction and a public SAE's error at exactly its share of the energy. (e707)

What this settles: the weak activations are neither removable noise nor a store of individually meaningful facts. They are a collective, learned, low-rank, mostly linear brake on the model's own strong neurons, carried by a field that neither camp has named, and training does not defend any of these atoms one at a time: it defends the function (e711, section 10).

### The six results at a glance

| # | result | models | the number | pre-registered: held / failed |
| --- | --- | --- | --- | --- |
| 1 | removing the weak activations costs nats | 11 models, 70M to 1.5B | 0.24 to 8.68 nats; 0.07 to 2.31 with mean replacement | held: cost over 0.2 (11/11), mean cheaper (10/11); failed: cost under 0.05 (0/11) |
| 2 | the brake | 8 model-blocks, 410M to 1.5B; 2 blocks at 7B | 52 to 88 percent of the damage off-class; excitatory wires alone 6 to 561 times, with the inhibitory 0.6 to 3; at 7B 85 to 87 percent, 59 to 284, 1.1 to 2.7 | held: off-class share (5/8 at the 0.7 bar, 2/2 at 7B), inhibitory wires carry half the opposition (8/8, 2/2); failed: signed wires plus the bus under 0.25 (0/8, 0/2) |
| 3 | the brake is learned | Pythia-160m x3 seeds, OLMo-1B x3 blocks | +0.3 at step 1000 to -0.6 to -0.8 at the end; OLMo crosses -0.5 at a third to two thirds of training | held: no brake at step 1000 (6/8), brake in the final model (7/8); failed: brake by step 8000 (0/8) |
| 4 | low-rank and linear | 8 models, 410M to 1.5B | 16 directions recover 0.66 to 0.96; the linear map 0.56 to 0.96 | held: map halves the cost (8/8), map matches the oracle (7/8); failed: map explains half the variance (0/8) |
| 5 | the fields | 8 model-blocks; 2 at 7B | 0.41 to 1.95 of the damage with 16 wires, 0.12 to 0.53 with both fields; at 7B 0.9 and 2.9 to 0.5 and 1.0 | held: both under 0.5 (7/8), both beat the weak field (5/8), beat random (4/8 small, 2/2 at 7B); failed: with attention under 0.25 (5/8 small, 0/2 at 7B) |
| 6 | read energy | 6 model-blocks, 160M to 500M; SAE on GPT-2 small | Fisher predicts finite KL at Spearman 0.83 to 0.92; SAE error read at 0.14 of the read energy against 0.15 to 0.19 of the energy | held: prediction (6/6), SAE error costs more than random (2/2); failed: fields read at three times random (0/6), SAE error unread (0/2) |

### Reproduce in a minute

The `wdd/` folder in this repository holds the instrument. On one GPU:

```
python -m wdd.demo_gpt2_small --fast
```

prints, for GPT-2 small at block 6 in under a minute: the rows over the floor at a position, the exact split of a neuron's pre-activation into its upstream contributions, the crowd's net sum off the neuron's class (the brake), the K-input simulation with and without the fields, and the read energy of the content, the weak write and the strong write. Every number in this note is in `results/`, every script in `scripts_phase3/`, every pre-registration in `results/*_prereg.json`.

### How to read this note

Sections 1 and 2 give the question and the instrument. Sections 3 to 8 give the six results, each with its figure and its pre-registered statements, the probability I gave each before the run, and how many runs it held in. Section 9 is the 7B check. Section 10 is what it means for the debate, including the one result that cautions both sides. Sections 11 and 12 are the limits and the reproduction map.

## 1. The question

In "Neural networks are already clean" (Cammarata, Goodfire, 15 October 2026) the already-clean position holds that a neuron's job is read off its strongest activations and that its weak activations can be removed at little cost. Its critics (Nanda; Gao, "Neural networks are not already clean", 17 October) hold that the weak activations carry memorised facts in superposition. StefanHex's summary of the first week left one question explicitly open until Goodfire's technical companion appears: do weak activations matter, and how?

This note answers it for models up to 1.5B parameters, with the instrument described next, and checks the two central results at 7B in section 9.

## 2. The instrument

**The ledger.** A neuron's pre-activation is a linear function of its block's normalised input, and that input is a sum of the writes of every upstream unit: the MLP neurons of earlier blocks, the attention outputs, and the embedding. Folding the read row through the normalisation gives an exact split of the pre-activation into one contribution per upstream unit plus a constant. The split is exact to the precision of the forward pass: in fp32 the sum of the contributions reproduces the pre-activation with a relative error of 5e-6 at Qwen2.5-7B (e700), and simulating the neuron from every input reproduces its effect on the output to within 0.00 to 0.05 of its ablation damage in every run below.

**The K-input simulation.** To ask what a subset of a neuron's inputs does, the neuron's activation is replaced by the activation its pre-activation would have if only those inputs (plus the constant) were present, at every position of a window, and the output distribution is compared with the unmodified model by KL. The score is the ratio of that KL to the KL of mean-ablating the neuron, at the neuron's strongest firings and over all positions: 0 means the subset reproduces the neuron, 1 means it is as bad as removing it, and above 1 means worse than removing it.

**The weak activations.** For each neuron, its activations in the bottom three quarters by magnitude on the evaluation text (the threshold is per neuron). "The crowd" is the set of a neuron's inputs outside its sixteen largest at a position; at a mid-network neuron it has thousands of effective terms.

**Pre-registration.** Every experiment's predictions were written with probabilities before its full runs; smokes could revise them, and every revision is dated in the pre-registration file. Each claim below is followed by its pre-registered statement, the probability I gave it, and how many of the runs it held in.

## 3. Removing the weak activations costs nats (e681)

Figure 1. Zeroing all but the strongest quarter of each neuron's activations at every block, on 12240 tokens of wikitext-2 test, costs 0.24 (GPT-2 small) to 8.68 (Pythia-410m) nats of next-token KL across eleven models, while removing 0.03 to 0.37 of the activations' energy. Replacing the removed activations by each neuron's mean instead of zero leaves 0.07 to 2.31 nats, so most of the cost is not an offset the mean restores. Zeroing only activations below a tenth of each neuron's maximum costs 0.48 to 14.09 nats.

![fig1](figures/fig1_e681_cost.png)

Pre-registered (results/e680_e691_prereg.json): W1, "keeping only the strongest quarter costs under 0.05 nats" (the essay's claim), at 0.45: held in 0 of 11 models. W1b, "it costs over 0.2 nats", at 0.25: 11 of 11. W1c, "zeroing below a tenth of the maximum costs under 0.05 nats", at 0.45: 0 of 11. W5, "mean replacement is cheaper than zeroing", at 0.70: 10 of 11.

The cost is widely distributed: by neuron tertile it sits in the "dial" tertile (neurons whose top one percent of positions carry the least of their energy) at least as much as in the "detector" tertile (W2 at 0.5, 5 of 11), and the predictions it flips are mostly near-ties (W3 at 0.6, 7 of 11). Nothing here says what the weak activations do; the next three results do.

## 4. The weak activations are a brake (e685b, e695, e697)

A mid-network neuron's input is a crowd. At its strongest firings it has 1874 to 15033 effective upstream contributors, its sixteen largest inputs carry 0.02 to 0.05 of the input's energy, and the writes over the identification floor carry 0.02 to 0.19 (e685b, twelve model-blocks). Driven by its sixteen largest inputs alone, the neuron does 0.43 to 2.11 of its ablation damage. Yet those sixteen are the drive: in the net ledger, which keeps the signs, the sixteen largest inputs carry 0.50 to 1.00 of the net drive at the neuron's core positions (e695), while the crowd's net sum, summed over thousands of terms, opposes them by -0.12 to -0.99 of their size off the class and by -0.10 to +0.5 on it.

e697 made that causal (Figure 2). On eight model-blocks of GPT-2 medium, Pythia-410m, Qwen2.5-0.5B and SmolLM2-135M, forty neurons each:

- The sixteen-input simulation's damage comes from off the class. Swapped in only at positions outside the neuron's top five percent it carries 0.52 to 0.88 of the all-position damage; swapped in only at the top five percent it carries 0.11 to 0.47.
- Sixteen excitatory wires alone cost 5.9 to 561 times the ablation damage over all positions: without the crowd the wires fire the neuron everywhere its class is nearby. Adding the sixteen strongest inhibitory wires brings that to 0.6 to 3.0, and those sixteen carry 0.5 to 0.9 of the crowd's off-class opposition.
- Neither sixteen inhibitory wires nor the upstream weak-write bus restores the neuron at its firings (0.59 to 1.06 and 0.29 to 0.93 of the ablation damage): the brake is position-specific, a sum of thousands of small terms whose sign flips with the position.

![fig2](figures/fig2_e697_brake.png)

Pre-registered (results/e696_e701_prereg.json): B0, "every input reproduces the neuron", at 0.9: 8 of 8. B1, "the simulation swapped in only off the class carries at least 0.7 of the damage", at 0.6: 5 of 8 (the others 0.52 to 0.64). B5, "the sixteen inhibitory wires carry at least half of the off-class opposition", at 0.45: 8 of 8. B2, "signed wires halve the sixteen-by-magnitude ratio at the firings", at 0.3: 1 of 8. B4, "signed wires plus the bus bring the ratio under 0.25", at 0.2: 0 of 8.

The reading: a neuron's drive is sparse and its silence is dense. The sixteen wires say when to fire; the crowd says when not to, and it says it with a different composition at every position.

## 5. The brake is learned (e701)

Figure 3. The same net ledger at one block across a model's public checkpoints, for one fixed sample of the final model's neurons. In Pythia-160m (block 6, three seeds, 100 neurons) the crowd's net sum over the sixteen wires in the bottom half of positions is +0.28 to +0.30 of the wires' at step 1000, meaning the crowd pushes with the wires, crosses zero by step 2000, and reaches -0.60 to -0.76 in the final model, monotone in the step (Spearman -0.99 to -1.00); meanwhile the wires' share of the net drive at the core rises from 0.48 to 0.56 to 0.75 to 0.86. In OLMo-1B (SwiGLU, blocks 4, 8 and 12) the crowd is within 0.3 of zero through the first percent of training and the brake forms over the whole run: it crosses -0.5 at step 512000 at block 12 and at step 1000000 at block 8 (about a third and two thirds of the run by tokens), monotone at both (Spearman -1.00), and never at block 4 (-0.23 in the final model).

![fig3](figures/fig3_e701_training.png)

Pre-registered (results/e696_e701_prereg.json and e701_e704_prereg.json): T1, "no brake at step 1000", at 0.5: 6 of 8 runs. T2, "the brake is under -0.5 by step 8000", at 0.45: 0 of 8 (Pythia crosses at 16000, OLMo much later). T3, "it strengthens monotonically", at 0.5: 5 of 8. T4, "the wires' net share at the core rises", at 0.5: 6 of 8. T5, "the brake is under -0.5 in the final model", at 0.5: 7 of 8.

The reading: the brake is a product of training, it arrives after the drive, and the drive sparsifies as the brake forms. A description of the model as neurons plus their strongest inputs describes the drive; the brake is what training added to make the drive safe to use.

## 6. The brake is low-rank and mostly linear in the input (e693, e696)

Figure 4. The summed write of a block's weak activations, projected onto sixteen principal directions of itself, recovers 0.60 to 0.96 of the zeroing cost in eleven models (sixty-four directions: 0.75 to 0.97), where a per-token table recovers 0.15 to 0.95 and the per-neuron mean -0.04 to 0.94 (e693). Those sixteen directions are not native write rows, not the stream's dominant subspace, and not shared across blocks.

A rank-16 linear map from the block's normalised input to those sixteen coordinates, fitted in closed form on held-out text and used in place of the weak neurons, which are not run at all, recovers 0.56 to 0.96 of the zeroing cost across eight models from 410M to 1.5B against 0.66 to 0.96 for the oracle sixteen (e696); on ARC-Easy, zeroing drops accuracy by 4 to 30 points and the bypass by 1 to 9.

![fig4](figures/fig4_e696_bypass.png)

Pre-registered (results/e696_e701_prereg.json): L1, "the predicted sixteen recover at least 0.8 of what the oracle sixteen recover", at 0.45: 7 of 8. L2, "the predicted sixteen halve the zeroing cost", at 0.4: 8 of 8. L5, "zeroing drops ARC-Easy by more than 5 points", at 0.5: 7 of 8. L4, "ARC-Easy under the bypass stays within 2 points", at 0.4: 3 of 8. L6, "the linear map explains more than half the weak write's variance", at 0.5: 0 of 8.

**That last result is the twist.** The map explains under half the weak write's variance (0 of 8 at the half bar) and still recovers 0.56 to 0.96 of the cost (8 of 8 at the halving bar), because what the output needs from the weak write lives in sixteen directions and is linear in the input there, while most of the weak write's variance lies in directions the output does not read (section 8). A reconstruction score would have called this map a failure; the model does not. Two follow-ups closed the obvious alternatives for the gated models, where the bypass recovers less: a quadratic map on 256 random bilinear features recovers no more than the linear one (e705), and one linear map per branch of the gate's sign pattern recovers no more either (e710).

## 7. The field is a dictionary atom (e703, e704, e706)

WDD reads a position's state as native write rows over an identification floor plus a crowd. The sixteen directions per block of section 6 are a second kind of atom, a field: not a row that one neuron wrote but the collective write of a block's weak activations. Scored on a neuron's readers with the ledger (Figure 5), sixteen wires leave 0.41 to 1.95 of the ablation damage at the firings on eight model-blocks; the weak field of the upstream blocks brings that to 0.29 to 0.93, the strong write's own sixteen directions per block (the kept quarter has a field too) to 0.28 to 0.94, and both together to 0.12 to 0.53, with the attention units as exact wires adding nothing further (e703, e706). The two fields overlap by 0.28 to 0.54, so they are different subspaces each worth about as much to the readers. Pushed by one standard deviation, a field direction has downstream readers as concentrated and sign-consistent as a strong neuron's, and a class more lexical than a neuron's (e704): the crowd's atoms read like neurons.

![fig5](figures/fig5_e706_fields.png)

Pre-registered (results/e705_e706_prereg.json): H0 at 0.9: 8 of 8. H4, "both fields bring the ratio under 0.5", at 0.6: 7 of 8. H2, "both beat the weak field alone by 0.1", at 0.45: 5 of 8. H5, "both beat a random subspace of their dimension by 0.2", at 0.65: 4 of 8, the margin falling under 0.2 where both fields' dimension, thirty-two per upstream block, exceeds half the stream's width. H1 at 0.3: 3 of 8; H3 at 0.3: 5 of 8; H6 at 0.5: 5 of 8.

Two limits of the field are also results. It is specific to the readers and not to the state's energy: after sixteen OMP rows it removes no more of the state's energy than a random subspace of its dimension (e703). And its directions on wikitext and on the Pile overlap by only 0.58 to 0.71 (chance under 0.02), so it has a corpus-shared core and a corpus-specific rest (e705).

## 8. What the output reads (e707)

Both camps grade a decomposition by how much of the state it reconstructs. The model grades a state by what its output depends on, and that grade turns out to be a quadratic form: the Fisher information of the model's own next-token distribution over the next sixty-four tokens, with respect to the state after a block, pulled back to the position. Its quadratic form on a vector, the vector's read energy, predicts the finite KL of removing that vector from the state at a Spearman of 0.83 to 0.92 over positions and parts on six model-blocks (Figure 6), while every subspace cut from the same Fisher, by trace or by the content's Fisher-weighted energy, failed finite removals in the smokes, because LayerNorm couples the parts (the revisions in results/e707_e712_prereg.json record both failures). The same construction appears independently in Picozzi (2026, arXiv 2609.11063), who uses it to predict SAE-feature ablations; here it grades the native parts.

![fig6](figures/fig6_e707_read_energy.png)

Per unit of Euclidean energy, against a random direction: GPT-2's sixteen native OMP rows are the most-read part of its state (1.7 to 1.9 times random) and its own strong write the least (0.7 to 1.0); in Pythia and Qwen every part is read at about its energy. The weak field's sixteen directions are read at 1.2 to 2.5 times random and the strong field's at 0.9 to 2.7, the stream's top principal directions at 1.8 to 5.9. On GPT-2 small a public residual SAE's reconstruction error (jbloom's resid_pre SAEs at blocks 6 and 9) holds 0.14 of the content's read energy against 0.15 to 0.19 of its Euclidean energy, and removing it costs 1.7 to 3.0 times a random vector of the same norm: the dark matter is read exactly as much as the rest of the state, no more and no less.

Pre-registered (results/e707_e712_prereg.json, after two dated revisions): R2, "the Fisher quadratic form predicts the finite KL at Spearman 0.8", at 0.6: 6 of 6. R4 and R4b, "the fields are read above three times a random direction", at 0.65 and 0.6: 0 of 6 each. R5, "the SAE's error holds under half as large a share of the read energy as of the energy", at 0.5: 0 of 2. R5b, "removing it costs more than a random vector", at 0.65: 2 of 2.

The reading: the field is what a neuron's readers need (section 7) and a second-class citizen of the output (this section). Both are true, because readability by a neuron and readability by the logits are different properties of a state.

## 9. At 7B

The same two scripts at Qwen2.5-7B in fp32, blocks 14 and 21 of 28, forty neurons each, with smaller samples than the record's (32 evaluation and 48 held-out windows; 16 and 24 for the brake at block 21) because the stored log-probabilities and the upstream captures do not fit 80 GB at 7B otherwise. Pre-registered in results/e713_e714_prereg.json with the statements of sections 4 and 7.

The brake at blocks 14 and 21: every input reproduces the neuron (0.01 of the ablation damage over all positions); the sixteen-input simulation's damage comes 0.85 and 0.87 from off the class; sixteen excitatory wires alone cost 284 and 59 times the ablation damage over all positions, and 2.7 and 1.1 with the sixteen strongest inhibitory wires added, which carry 0.78 and 0.64 of the crowd's off-class opposition; at the firings the sixteen wires cost 2.9 and 1.2 of the damage, the signed wires 1.3 and 1.0 and the wires with the weak bus 1.4 and 0.7. Five of e697's nine pre-registered statements hold at both blocks, the three halving statements at the mid block only, and the one that asked signed wires with the bus to reach a quarter of the damage fails, as at every scale.

The fields at blocks 14 and 21: sixteen wires cost 2.90 and 0.90 of the ablation damage at the firings, with the weak field 1.37 and 0.65, with the strong field 1.29 and 0.52, with both 1.01 and 0.48, where a random subspace of both's dimension leaves 2.23 and 0.76; the two fields overlap by 0.24 and 0.28. H0, H2 and H5 hold at both blocks, H3 at neither, H1, H4 and H6 at one.

One verdict that was split at 410M to 1.5B is clean at 7B: both fields beat a random subspace of their own dimension at both blocks (1.01 against 2.23, 0.48 against 0.76), because thirty-two directions per upstream block no longer approach the stream's width when that width is 3584. The shape is the small models' with the sixteen wires worse at the strongest firings, which the fp32 ledger at 7B had already shown (e700): a 7B neuron's sixteen largest inputs carry a smaller share of its input, so the brake and the fields matter more there, not less. The transfer question is answered for the brake; the two-field dictionary transfers with its limit, half the damage left at the mid block.

## 10. What this means for the debate

The already-clean position keeps the drive. A neuron's strongest activations are read off its sixteen largest inputs, those inputs are native rows, the rows are the most-read part of the state in GPT-2, and the drive sparsifies with training. The superposition position keeps the fact that the rest of the state matters and is invisible to the native basis: the weak activations cost nats, and what they do is not a function of any one neuron. What neither position has is the object in between. The weak activations are a brake: learned, position-specific in composition, low-rank and linear in the input in what the output needs from it, and carried by a field that reads like a neuron to the next block and like a minor direction to the logits. A dictionary with two kinds of atom, rows and fields, is what the record supports; a dictionary of neurons alone leaves a quarter to a half of a pruned neuron's error, and a dictionary of learned features reconstructs energy the output grades no higher than the rest.

One result from the following round belongs here as a caution to both sides: training does not defend any of these atoms one at a time. The restoring gradient on a rotated write row, or on a swapped field, is three to five times the noise floor, and two hundred steps of continued training repair a swapped field's third of a nat in full without restoring the field, even with its writers frozen (e711). What the model keeps is the function.

## 11. Limits

Models from 70M to 1.5B for the eleven-model results and 410M to 1.5B for the ledger results, with two blocks at 7B in section 9; wikitext-2 as the evaluation text with the Pile's domains as the only transfer check; forty neurons per block sampled among the active ones; the field fitted per block on held-out text; the pre-registrations revised after smokes where the first construction failed, with every revision dated and no result from a discarded run in the record.

## 12. Reproduction

Each result is one script in `scripts_phase3/` on the shared helper `c11.py`, standalone on a single GPU: `e681_weak_ablation.py`, `e685_readers_ledger.py` (the ledger), `e695_shells.py`, `e697_brake.py` (also the 7B brake, with the sample-size overrides `WDD_NWIN`, `WDD_NWH`, `WDD_CHUNK`), `e701_brake_training.py`, `e693_weak_ladder.py`, `e696_bypass.py`, `e703_rows_field.py`, `e704_field_readers.py`, `e706_strong_field.py` (also the 7B fields), `e707_read_subspace.py`, `e711_elasticity.py`, `e714_fisher_dictionary.py`. The `wdd/` package holds the same primitives as a library with the GPT-2 small demo. The result files are under `results/<id>/` (the 7B runs under `results/e713/`), the pre-registrations under `results/*_prereg.json`, and every hypothesis with its fate in `docs/GRAPH.md` (H892 to H1137 for sessions 155 to 161). The atlas pages `docs/atlas/62_*.md` to `69_*.md` carry the readings in full, and `docs/REVIEW_POSITIVE_CLAIMS.md` sections 19 to 25 the claims checked against their pre-registrations.
