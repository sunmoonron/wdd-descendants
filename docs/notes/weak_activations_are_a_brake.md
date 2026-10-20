# The weak activations are a brake

**Six pre-registered results on what the sub-threshold activations of a transformer's MLP neurons do, from an exact ledger of each neuron's inputs.**

Ronish Bhatt, with Claude (Anthropic) as the experimenter. 20 October 2026. A technical note from the WDD descendants program; everything here is in this repository (`scripts_phase3/`, `results/`, `docs/atlas/` areas 62 to 67), with the pre-registrations in `results/*_prereg.json`.

## Summary

- Removing a neuron's weak activations is not free. Keeping only the strongest quarter of each MLP neuron's activations costs 0.24 to 8.68 nats of next-token KL in eleven models from 70M to 1.5B parameters, and the cost is not a mean offset: replacing the removed activations by each neuron's mean leaves 0.07 to 2.31 nats (e681).
- The weak activations act as a brake. At a mid-network neuron, the sixteen largest inputs carry the net drive at its firings, but when the neuron is driven by those sixteen alone its damage comes 0.52 to 0.88 from positions off its class, where the sixteen fire it and the crowd of thousands of small inputs would have held it down; sixteen excitatory wires alone cost 6 to 561 times the neuron's ablation damage, and adding the sixteen strongest inhibitory wires brings that to 0.6 to 3.0 (e697).
- The brake is learned, not built in. In Pythia-160m the crowd's net sum off a neuron's class is +0.3 of the sixteen wires' at step 1000 and crosses -0.5 by step 16000 (three seeds, monotone); in OLMo-1B it forms over the whole run and crosses -0.5 between a third and two thirds of training (e701).
- The brake is low-rank and mostly linear in the block's input. The summed weak write of a block lives in sixteen directions, which recover 0.66 to 0.96 of the zeroing cost, and a rank-16 linear map from the normalised block input, with the weak neurons not run at all, recovers 0.56 to 0.96 (e696).
- Those sixteen directions per block are a field: a second kind of dictionary atom beside the native write rows. Sixteen wires plus the weak field plus a strong-write field bring a pruned neuron's error at its firings from 0.41 to 1.95 of the ablation damage to 0.12 to 0.53 (e706).
- What the output reads is a quadratic form, not a subspace. The Fisher information of the model's own next tokens predicts the cost of removing any part of a position's state at Spearman 0.83 to 0.92; by that grade the fields are read at about twice a random direction, GPT-2's native rows above their energy, and a public SAE's reconstruction error at exactly its share of the energy (e707).

For the debate: the weak activations are neither removable noise (the essay's "almost all weak activations can be removed for almost no cost") nor a store of individually meaningful facts (the superposition camp's working picture). They are a collective, learned, low-rank, mostly linear brake on the model's own strong neurons, and the object that carries it, the field, is a dictionary atom that neither side has named.

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

That last result matters: the map explains under half the weak write's variance and still recovers most of the cost, because the part the output needs lives in sixteen directions and is linear there, while the rest of the variance is in directions the output does not read (section 8). Two follow-ups closed the obvious alternatives for the gated models, where the bypass recovers less: a quadratic map on 256 random bilinear features recovers no more than the linear one (e705), and one linear map per branch of the gate's sign pattern recovers no more either (e710).

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

(Running at the time of writing: e697 and e706 at Qwen2.5-7B in fp32, blocks 14 and 21, forty neurons each, pre-registered in results/e713_e714_prereg.json. This section will carry the numbers when they land. What is already known at 7B: the fp32 ledger is exact there, every input reproduces the neuron, and the sixteen largest inputs, carrying 0.02 of a neuron's input, cost 1.27 of its ablation damage at its strongest firings, as at 400M (e700).)

## 10. What this means for the debate

The already-clean position keeps the drive. A neuron's strongest activations are read off its sixteen largest inputs, those inputs are native rows, the rows are the most-read part of the state in GPT-2, and the drive sparsifies with training. The superposition position keeps the fact that the rest of the state matters and is invisible to the native basis: the weak activations cost nats, and what they do is not a function of any one neuron. What neither position has is the object in between. The weak activations are a brake: learned, position-specific in composition, low-rank and linear in the input in what the output needs from it, and carried by a field that reads like a neuron to the next block and like a minor direction to the logits. A dictionary with two kinds of atom, rows and fields, is what the record supports; a dictionary of neurons alone leaves a quarter to a half of a pruned neuron's error, and a dictionary of learned features reconstructs energy the output grades no higher than the rest.

One result from the following round belongs here as a caution to both sides: training does not defend any of these atoms one at a time. The restoring gradient on a rotated write row, or on a swapped field, is three to five times the noise floor, and two hundred steps of continued training repair a swapped field's third of a nat in full without restoring the field, even with its writers frozen (e711). What the model keeps is the function.

## 11. Limits

Models from 70M to 1.5B for the eleven-model results and 410M to 1.5B for the ledger results, with two blocks at 7B in section 9; wikitext-2 as the evaluation text with the Pile's domains as the only transfer check; forty neurons per block sampled among the active ones; the field fitted per block on held-out text; the pre-registrations revised after smokes where the first construction failed, with every revision dated and no result from a discarded run in the record.

## 12. Reproduction

Each result is one script in `scripts_phase3/` on the shared helper `c11.py`, standalone on a single GPU: `e681_weak_ablation.py`, `e685_readers_ledger.py` (the ledger), `e695_shells.py`, `e697_brake.py`, `e701_brake_training.py`, `e693_weak_ladder.py`, `e696_bypass.py`, `e703_rows_field.py`, `e704_field_readers.py`, `e706_strong_field.py`, `e707_read_subspace.py`. The result files are under `results/<id>/`, the pre-registrations under `results/*_prereg.json`, and every hypothesis with its fate in `docs/GRAPH.md` (H892 to H1112 for these sessions). The atlas pages `docs/atlas/62_*.md` to `67_*.md` carry the readings in full, and `docs/REVIEW_POSITIVE_CLAIMS.md` sections 19 to 24 the claims checked against their pre-registrations.
