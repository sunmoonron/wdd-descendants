# WDD in brief: the theory, what it forced, and the routes it closes

A ten-minute read of the whole program (439 experiments, five small models, one GPU, September 2026). It is organised around the mathematics, so that a reader can see which results follow from the setup, which were discoveries, and which routes are closed for reasons that no further experiment will reopen. The details, every number and every correction are in [`ATLAS.md`](ATLAS.md) and its area pages; experiment ids are given in parentheses.

## 1. The object, and two facts it carries with it

Weight-Dictionary Decomposition (WDD) writes a centred residual-stream state as a sparse combination of the model's own write vectors: token and position embeddings, the rows of each MLP down-projection, the output bases of each attention head, and the biases.

```
x − μ = Σ_i c_i ŵ_i + r,     ŵ_i a unit write vector, chosen by OMP
```

Each atom carries a label, the component that wrote it. The label is the only thing that distinguishes this dictionary from any other of the same shape; the control for everything below is therefore the **rotated dictionary**, a random rotation of the same atoms, which keeps every angle and norm (the same Gram matrix) and loses the provenance. Anything the rotated dictionary also does is geometry, not WDD.

Two facts hold before any experiment is run.

- **A description is a projection.** The reconstruction lies in the span of the words that were chosen, so describing it again chooses the same words. Measured: after one application, 96-100% of descriptions are fixed points, for native and rotated words alike (e472). This settles, without a run, every proposal that iterates WDD on its own output: recursion, bootstrapping from one word, closure of combinations of atoms (sparse recovery, e63), interpolation in coefficient space (the map from coefficients to states is linear, e271), and reading "what WDD missed" by decomposing the residual (the residual holds no more own-atom structure than a rotated residual, e09, e122).
- **Identification is an extreme-value statistic.** OMP finds a write when its projection on the state beats the largest projection of the competing atoms, which for covariance-matched competitors is about sqrt(2 ln m / d_eff) of the state's norm. This is the prominence law (e56, e60, e127, e129); its one-shot form is close to a definition. Its consequences are the shape of the whole first area: readability is amplitude, a stable trait of a neuron (ICC 0.95-0.97) unrelated to its importance (e59, e130); competition, not budget, limits identification, so more atoms, aliases, twins, fillers and ridge tweaks do not raise it (e10, e41, e102, e181); and the edge of a trained dictionary over its rotation is an extreme-value edge carried by the first eight or so atoms (e03, e09), gone by 64 atoms, where any overcomplete dictionary fits (e388).

## 2. What happens to a write: first-order transport

The theory of the middle of the program is one line. Within the linear range, the footprint of a write at a later level is its transported image,

```
d_i(ℓ, t) = c_i(t) · J_t · ŵ_i + O(c²),     J_t = J̄_C + ΔJ_t,
```

with J_t the Jacobian of the downstream map along the token's own trajectory, split into a context-class mean and a token-specific fluctuation. Three measured properties make it concrete: J has a generic spectrum with no null space and a root-mean-square gain near one (e273, e282); the token-specific term dominates by mid depth (coherent fraction 0.16-0.33 at mid depth, 0.05-0.14 late, e277); and the response is linear up to the natural amplitude, distorting at three to ten times it (e278). Everything in areas 05-08 is either a consequence of this line or a measurement of J.

- **Fading is scattering, not cancellation.** Along any direction, every trained block is a contraction with gain −0.16 to −0.42, the normalisation gain times the trace of the MLP Jacobian, direction-independent, zero at initialisation and born as warmup ends (e194, e210, e217). The write's energy is kept while its direction spreads over thousands of later writers (e221-e225). Erasure, a write actually removed, is rare (1-6% of tokens) and is done by a crowd, never by attention (e08, e185). Provenance is lost at accumulation: block increments read what states lose (e131), and an identified atom names who wrote a direction, not what caused it (e211).
- **The descendant is the transported write.** It is determined by the vector, not the neuron; homogeneous and additive; angle-preserving for random components by concentration of measure, with the departures on the operator's top and bottom singular directions predicted within 0.07 (e238, e282, e285); a state variable that predicts its own future (e283); and the carrier of the write's function, reproducing its effect when injected (e269).
- **Provenance against decomposition, as an inequality.** With κ the coherent fraction, nearest-centroid identification among K classes has signal-to-noise about sqrt(κ)·Δ / sqrt((1 − κ)·K / D_eff) and survives down to κ of order K / D_eff, a few percent, whereas a reconstruction by any fixed dictionary leaves at least 1 − κ unexplained. That is why the descendant still names its neuron in 88-98% of tokens (e227) while a transported dictionary fails as a basis in five of five models (e236). Every plan to read WDD at depth through transported or learned atoms is closed by this inequality, not by a missing trick.
- **The one exception** is the massive-activation neuron of SmolLM2, where the second-order term wins at the write's own tokens (e284b, e290). It is the only such place found.

## 3. What part of the descendant matters: compressible, not structured

Causal observables (identity, the future, the logit footprint, the removal KL) need 8-32 directions of a descendant; reconstruction needs 512-1024 (e286), and this holds for every perturbation family tried, including ones built orthogonal to the natural manifold (e304, e310). WDD and this quotient are two coordinates of one linear map: the ledger-weighted sum of the atoms' functional coordinates predicts a joint ablation's coordinate (e313).

The negative result is as firm as the positive one. The compressed content has no privileged basis: a projection fitted to shuffled targets, a PCA, or the discarded complement decode as well as the quotient (e340, e346, e347); there is no null space or coset structure and the fibres are random (e336, e338); the WDD atoms are not a special basis of it (e332-e334). The reading is that the readout Jacobian, restricted to the perturbation cloud, is of moderate rank with a long tail and nearly isotropic, so any high-variance projection of a few tens of dimensions carries the function and none is distinguished. Functional units and attribution built from atoms failed for this reason (e316), and it will not be undone by a better decoder.

Two readout facts govern how any interaction is measured. Cross-entropy is convex in the logits, so effects that add on the logits interact on the loss by Jensen's inequality: 26-48% of pair signs flip between the two readouts, phase 2's growing "redundancy" was mostly readout convexity, and interactions must be read on the logits (e374-e384). Units that matter are robust to partial removal and fail only near full removal, so circuit structure is a large-move property that gradients, co-selection and curvature at the trained point cannot reproduce (e371, e372).

## 4. Self-description: where provenance shows up at scale

Judged not as an identifier but as a code the network must run on, the native vocabulary is functionally faithful: 32-64 own words at the middle depth keep 90% of the next-token loss, where the model's largest actual writes need a thousand or more (e388, e391), and at 8-32 words rotated words, PCA and embeddings alone fall far behind. The property is learned. It is absent at initialisation in all five architectures (e401); early it is a per-block accent, a second-order alignment that Gaussian words with each block's covariance reproduce, and it becomes word-level between Pythia steps 4000 and 16000 (e443, e395); it is written by training the writers, since writer rows frozen at random are barely used as words even when the function is held fixed (e444b, e449, e452); it is private across seeds (e398); and it holds concept words that are the same across four languages (15 of 24 nouns in Qwen2.5-0.5B, 22 at 7B, e448d). It is not the best sparse vocabulary: a learned sparse autoencoder wins at small k (e403). Several early numbers were inflated by attention-sink tokens and were corrected (e432, e437).

What self-description measures is that a state is made of the model's own rows, not that the model works: it survives compression that breaks the model (e456). It is a forward coordinate, not a learning one: a batch's gradient on a word is noise relative to the word (e462), and over training a neuron's usage as a word and its importance evolve independently (e471).

## 5. Where native words do what no control does: handles during construction

The positive results of the last sessions are of one kind. A concept word injected at natural size across depth moves 35-58% of translations, about as efficiently per unit of displacement as a dense steering vector (e458). Under interchange, setting one to four native words per block to another context's values carries a translated noun between contexts in 67-96% of Qwen's answers, against 0-36% for rotated words and 8-66% for the task's own principal directions at equal size (e469). This works only across the band of blocks in which the noun is being written (Qwen blocks 0-6); from block 8 on, four native words switch 15% while the whole state still switches 100%, and the carrier words turn over as the variable is re-written (e470). For a word with two attributes, the native words for gender and for generation act independently and combine: gender from one source word and generation from another give the doubly changed word in 74% of items, with additive effects, where rotated words fail (e473). The gender words are the vocabulary's, not the item's: four native words found on other kinship words, in other languages, flip only gender in 98% of items, better than the item's own, and one MLP row of block 4 sits in 91% of the items' own gender supports; the generation handle is only half shared (e477). The limits are as informative. The task's own principal directions do as well as native words on those attributes (e473); identity handles are not typed by grammatical role; and for an entity copied into the answer, native words are no better than rotated ones (e474).

The reading, a hypothesis and not a theorem: while MLP writers are building a semantic variable, their rows are its coordinates; once it is built, the variable is a distributed direction that any good basis finds and a few rows cannot move. Native words are a temporary vocabulary for the construction of variables, not a coordinate system for the finished state.

## 6. Open

- The one native-only signal that is not yet understood: at the deepest depth, how concentrated a description's coefficient profile is predicts the similarity of two positions' next-token distributions beyond activation distance, at partial correlation +0.14-0.15 in three models, while rotated profiles predict nothing (e475). Its audit (e476) found it is not an artefact of the normalisation or of k, and not a proxy for norm, position, confidence, token frequency or token identity. A third to a half of it is how well sixteen own words fit and how much one own write (an MLP row, in 86-98% of positions) dominates the state, which is prominence, a quantity a rotated dictionary cannot compute. The rest (+0.07 to +0.11) is open; in Qwen the PCA profile carries as much.
- Which variables native words carry better than generic coordinates: a translated concept, yes; two attributes of a word, as well as task PCA, and for gender as a single shared word (e477); a copied entity, no. The open question is what separates these: a guess is that native words win for variables that MLP writers build and lose for variables copied from the input.

## 7. The skip list: routes closed by the mathematics

| Route | Why it cannot work | Shown in |
| --- | --- | --- |
| Iterating, bootstrapping or recursing WDD; reading the residual for what WDD missed | A description is a projection; the own-atom edge is spent in the first ~8 atoms | e472, e09, e122 |
| Raising identification with budget, aliases, twins, fillers or ridge | Competition-limited: an extreme-value statistic | e10, e41, e102, e181 |
| Cancellation or erasure as the mechanism of forgetting | The contraction is the diagonal of an energy-conserving mixing map | e194, e210, e221 |
| Sparse reading at depth with transported or learned atoms | Coherent fraction falls to 5-15% while identification needs only a few percent | e236, e277 |
| A privileged functional basis; functional units or attribution from atoms; equivalence classes | The readout is near-isotropic on the cloud; no null space; random fibres | e336-e347, e316, e338 |
| Circuits, redundancy or interactions from loss-level or local (gradient, curvature) quantities | Jensen's inequality on the loss; large-move structure | e371-e384 |
| Sign flips, dose tests, observability against controllability | Effects are linear to 3-10 times natural amplitude | e271, e345 |
| Write history, forgery or a checksum as forensics | The forward pass is a function of the current states; history is not an input | e458, e460, e461 |
| A grammar of native words (inflection, accents) | Role is carried by which words are used; rotated words decode it as well | e463 |
| A self-sufficient description, or a small Kalman gap | What is omitted is spread over many directions and is functional | e465, e466, e467 |
| Word-for-word translation between models or seeds | Private languages; no row correspondence | e398, e445, e453 |
| Native words as a learning coordinate, or a life cycle of words | Gradients are noise relative to words; usage and importance are independent | e462, e471 |
| WDD as a geometry of states | Activation distance predicts behaviour better; the small surplus is shared with rotated words | e475 |
| Interpolation, closure and permutation tests | Coefficients map linearly to states; sparse combinations are recovered by construction; a permutation of coefficients is a random vector, already the rotated control | e271, e63, e388 |

## 8. Reading any number here

- The controls, in order of strength: the rotated dictionary; a random-init dictionary; covariance-matched Gaussian atoms and per-block versions; sums of eight atoms of one family; PCA; a learned SAE; sinks kept exact. A claim of provenance needs the first and, from session 38 on, the third.
- Corrections that supersede earlier text: the attention constant of e54 was N-fold too large; the M variance shares (86%, 48%, 97%) were single sink tokens, 10-26% at ordinary positions (e432, e437); e455's concept-word swaps were 6-8 times natural size (e458); e451's weak single-depth effects are one word being outvoted, not the concept having moved (e469). The atlas lists the rest.
- Nothing here is claimed beyond these five models.
