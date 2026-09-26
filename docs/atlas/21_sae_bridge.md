# 21. The provenance layer under SAEs (S59-S62)

**Question.** A learned sparse autoencoder's features are directions with no provenance. Decomposed over the model's own dictionary, what are they made of, which rows write them, does a feature fire when its writers fire, and is any of this special to features rather than to any direction the native dictionary explains well?

**Established.**
- A feature's top native word is an MLP row for 76-93% of features at every depth of GPT-2, and that row comes from any block up to the SAE's (the block just before holds 12-30%); features are sparser in native words than in rotated ones at every depth (0.47-0.49 against 0.71-0.73 unexplained at sixteen words); a fifth to a quarter of the most used native rows are features at cosine above 0.5, rotated rows never; the frequent features are the nearest to single rows (e494, e495).
- A feature is state-like, not write-like: it is exactly as sparse in native words as a whole state (0.48 against 0.47), far sparser than a random direction (0.72, which is also the rotated dictionary's floor on anything) and sparser than a covariance-matched direction (0.63). "A sparse composition of native writes" is the wrong phrase; a feature has a state's provenance profile (e497).
- On an independently trained TopK family the provenance holds (top word an MLP row 0.86-0.89; features far sparser in native words than random or covariance-matched directions), the state-likeness is partial (features between covariance-matched directions and states), and the activation part is much stronger: top row AUC 0.73, eight-word ledger 0.83 with half of features above 0.8 (e499).
- A feature's firing follows its top row's activity partly: the row's write size separates the positions where the feature fires at AUC 0.62 (a random row 0.50), strongly for a sixth of features; the ledger from its 4-16 words adds nothing over the top row (0.59-0.63), and the same coefficients on random rows are at chance (e496).

**Start here:** e497, e494, e496 · **Sessions:** S59, S60, S61, S62 · **Scripts:** `scripts/e494_sae_provenance.py`, `e495_sae_depth.py`, `e496_feature_from_ledger.py`, `e497_feature_specificity.py`, `e499_sae_second_family.py`

## Experiments

| id | question | result | status | links |
| --- | --- | --- | --- | --- |
| e494 | Which weight rows write a learned SAE feature (GPT-2, block 7 input)? | Its top native word is an MLP row for 0.93 of features, from every block up to the sixth; features are sparser in native words than in rotated (unexplained 0.71 against 0.91 at k = 4, 0.48 against 0.72 at 16) but not one row; 0.23 of the 4096 most used rows are features at cosine above 0.5 (rotated 0.00) | supported (many-to-many bridge) | ← e403 e16 · → e495 e496 e497 |
| e495 | Is the provenance layer under SAEs the same at every depth, and are frequent features single rows? | Yes and yes (GPT-2, SAEs at five depths): unexplained by 16 native words 0.47-0.49 (rotated 0.71-0.73), top word an MLP row 0.76-0.93 from any block (the block before the SAE 0.12-0.30), used rows that are features 0.19-0.23; frequency against single-word unexplained rho -0.25 to -0.38 | established (5 depths) | ← e494 |
| e496 | Does a feature fire when its writer fires? | Partly: the top row's write size separates the positions where the feature fires at AUC 0.62 (random row 0.50; a sixth of features above 0.8); the ledger from 4-16 words 0.59-0.63, no better than the top row; the same coefficients on random rows 0.50 (v3, 2000 features, 8176 positions; v2's 0.70 was a smaller sample). A Spearman over all positions is uninformative for a feature that is zero at 98% of them | narrowed (the top row, partly) | ← e494 e495 |
| e499 | Does the bridge hold on an independently trained SAE family (OpenAI TopK, 32k latents)? | Provenance yes: top word an MLP row 0.86-0.89 (random 0.40-0.54); features 0.56 unexplained at 16 words against random 0.72 and covariance-matched 0.63 at block 6. State-likeness partial: features between covariance-matched directions and states (0.56 against 0.47), below the covariance level at block 2. Activation stronger: top row AUC 0.73, ledger 1/4/8/16 words 0.66/0.80/0.83/0.85, half of features above 0.8; random rows 0.49-0.50 | supported (provenance; activation family-dependent) | ← e496 e497 |
| e497 | Are SAE features special to the native dictionary, or is any direction? | Features are as sparse in native words as a state (0.48 against 0.47 unexplained at 16), far sparser than a random direction (0.72, the rotated dictionary's floor on anything) and sparser than a covariance-matched one (0.63); top word an MLP row 0.93 against 0.54 for random directions. State-like, not write-like | narrowed (the bridge reworded) | ← e494 e495 |

## How the results flow

- `e403, e16 → e494 → e495`. Own words and a learned SAE were compared as codes for the states (e403), and atoms were decomposed over atoms (e16); e494 decomposes the SAE's features over the atoms and finds every feature has an identifiable top writer while most need many rows; e495 finds the same at five depths and that the frequent features are the nearest to single rows.
- `e494 → e497`. The control the bridge needed: under the same dictionary, random and covariance-matched directions are far less explained than features, and states are explained exactly as well as features. What e494 measured is that a feature has a state's provenance profile.
- `e494 → e496 → e499`. In activation the top row predicts a ReLU feature's firing partly and the composition adds nothing over it; on a TopK family both are much stronger, so the activation claim depends on the feature's own sparsity.

## Links to other areas

- [20 Classical calibration](20_classical_calibration.md): the same dictionary and controls; e497's random-direction floor is the rotated dictionary's level of e483.
- [14 Native vocabulary](14_native_vocabulary.md): e403 compared own words with this SAE as codes for the states; the features' state-like sparsity is the self-description property (e388) seen from the feature side.
- [13 Readers, readouts](13_readers_readouts.md): e388's SAE test judged codes by loss recovered; here the SAE's own directions are the object.
