# Hypothesis graph of the WDD mapping sprint (sessions 1 to 4, 2026-09-19)

Format: `H<n>` hypothesis → experiments that tested it → status → what it spawned. Status words: SURVIVES (not falsified under the tests run), KILLED (falsified), NARROWED (true only in a restricted form), ARTIFACT (result retracted), OPEN (untested or pending). Experiment numbers refer to `scripts/eNN_*.py` and the matching lines in `results/FINDINGS.log`; FINDINGS.md has every number.

## Root

- H0 WDD is classical sparse approximation over a fixed, non-learned, provenance-labelled dictionary, and the label is what distinguishes it from any other sparse code. → e01 to e10 (reproduction gate, decoder comparisons, dictionary types), e63 (exact recovery condition), e158 (MDL) → SURVIVES. Spawned H1, H2, H3, H7, H9.

## Reading: what the decoder recovers

- H1 Identification is a prominence phenomenon: a write is read iff its projection on the state exceeds the level's competitor level. → e12 to e20 (law), e47/e119 (non-circular target), e129 (held-out, zero-parameter), e178 (dual law), e198 (across depth: Spearman 0.90 to 1.00) → SURVIVES, with the audit framing (definitional one-shot form; the content is that OMP tracks it and the level is covariance-determined). Spawned H1a, H1b, H6.
  - H1a The competitor level is sqrt(2 ln m / d). → audit → KILLED; replaced by the k-th order statistic of covariance-matched random directions (SURVIVES).
  - H1b The reading threshold is the k-th extreme of a per-atom Gaussian null (parameter-free). → e204 (k = 64 matches within ±0.5 z in 4/5), e205 (no collapse across k in 3/5) → KILLED as a law, NARROWED to "the per-atom z-score is the right statistic; the threshold is empirical".
- H2 Decoder hierarchy: OMP reconstructs best, projection/dual identifies provenance best, function follows reconstruction. → e02, e44, e102, e154 (reweighted l1), e144/e159 (increment-guided), e189 (support instability: OMP Jaccard 0.03 to 0.14, dual 0.57 to 0.82) → SURVIVES. Spawned H8.
- H3 Identifiability is a property of the neuron, orthogonal to importance. → e46, e130 (amplitude phenotype), e160 (per-write ablation), e170 (not the massive channels) → SURVIVES. Spawned H3a, H3b.
  - H3a Never-neurons are a static geometric class (large null std, small self-gain). → e206 → KILLED.
  - H3b Never-neurons are readable in their own increments; the loss is at accumulation. → e131, e147 → SURVIVES (4/5; GPT-2 loses within the block).

## Three quantities

- H4 Provenance match, causal footprint and functional importance are three different per-token quantities. → e152, e156, e160, e165 (function of identified vs unidentified: 4/5, OLMo reverses), e208 (footprint), e211 (reading uncorrelated with footprint, Spearman −0.09 to +0.03 in 4/5) → SURVIVES. Spawned H4a.
  - H4a An identified atom is the cause of its direction's presence. → e208 (8 to 19% of the direction at the mid layer depends on the write), e211 → KILLED. The label names who wrote a direction, not what caused its presence.

## Erasure → contraction (session 3, the main new line)

- H5 A direct-contribution "erasure matrix" reveals who cancels whom. → e173 (five models, Pile caches, random init, Pythia checkpoints), e141 → SURVIVES as a description (two topologies, learned, data-invariant). Spawned H5a to H5e.
  - H5a The next block erases the previous block's writes (a wired mechanism). → e183 (ablation never restores survival, 5/5), e186 (later blocks take over) → KILLED.
  - H5b Cancellation is carried by dedicated eraser neurons or negative aliases. → e185 (crowd: top neuron 1 to 3%, cos ~0), e187 (no anti-aligned-row signature) → KILLED.
  - H5c Cancellation is the MLP output bias. → e192 (2 to 13%) → KILLED.
  - H5d Cancellation is proportional to the write (gain-like). → e191 (constant relative cancellation in Qwen, OLMo, GPT-2; additive in Pythia; saturating in SmolLM2) → NARROWED. Spawned H6.
  - H5e Canceller blocks are the unreadable blocks. → e199 → KILLED.
- H6 Every trained block is a generic linear contraction of its input ALONG ANY GIVEN DIRECTION (diagonal gain −0.16 to −0.42, direction-independent, learned), while conserving the perturbation's energy (see H6g/H6h: a decorrelation, not a damping). → e194 (write direction = random direction, linear over 0.25 to 1×), e207 (zero at init, grows with training, fades late, block 0 expands), e210 (derived block by block from normalization gain × Jacobian trace, 5/5), e209 (anti-aligned read/write vectors in GELU models; sign in the active neurons for gated), e213 (gain ∝ 1/‖state‖, 5/5) → SURVIVES. Spawned H6a to H6e (session 4).
  - H6a The gain is the same for real writes, attention writes, principal directions, embeddings and random directions, and zero along the state's own direction; linear from 0.001 to 1× the norm. → e214 → NARROWED: independent within 0.2 in Qwen, OLMo, Pythia, SmolLM2; zero along the state direction (4/5, OLMo explained by its large attention writes); the lowest-variance principal direction is barely damped (the contraction lives in the occupied subspace); GPT-2 damps its own write directions and the top principal direction 3 to 4× more than random (write-specific cancellation). Linear over three decades (0.001 to 0.1× the norm), weaker at 1×. Spawned H6f.
  - H6f The contraction's strength follows the state's spectrum (strong along high-variance directions, absent along low-variance ones). → e220 → KILLED as a monotone law (Spearman −0.52 to +0.69). NARROWED: the top principal direction is damped hardest in GPT-2, Qwen and Pythia, the next few top directions weakly, the bulk moderately, and only the covariance's null direction escapes; OLMo damps uniformly.
  - H6b Multi-block survival of an arbitrary perturbation is the product of single-block gains. → e194 (two blocks), e215 → NARROWED: it holds over three blocks in 5/5 models (mean absolute log error 0.09-0.20; Qwen 0.52), and by seven blocks the product over-damps 1.5-6 times, leaving a floor of 0.1-0.2. (This line read "OPEN (running)" until session 55, although e215 had answered it.)
  - H6c The 1/‖state‖ dependence is caused by the normalization's per-token scale. → e216 → SURVIVES: with the scale frozen the correlation vanishes in 5/5 (+0.47 to +0.55 → −0.10 to +0.03), cleanly in the GELU models; in the gated models the frozen scale moves the MLP off its operating point (Qwen median gain +242, OLMo +18), so only the correlation is interpretable there.
  - H6d The contraction is born before step 1000 of training. → e217 (Pythia steps 1 to 512, 2000, 3000) → NARROWED: nothing up to step 512 (indistinguishable from init), cosine −0.037 at step 1000, −0.157 at 2000, −0.212 at 3000; born as the learning-rate warmup ends (1430 steps); the read/write anti-alignment leads the gain (70% vs 20% of final at step 2000). Pythia's dedicated block-5 canceller (e173) exists at step 1000, before the generic decorrelation: specific before generic.
  - H6e The residual toy reproduces the contraction. → e190 → KILLED (the six-block autoencoder only reinforces).
- H7 Provenance persists because later blocks re-write directions, not because writes are passive. → e188/e193 (survival 2 to 3× longer than passive decay), e208 (the write-dependent part decays as the contraction predicts; causal half-life 1.5 to 2 blocks), e212 (late-born writes keep 2 to 4× more) → SURVIVES.

## Certificates and the prior

- H8 Spurious atoms come in two kinds, noise-driven (OMP) and geometry-driven (dual), each with its own certificate. → e189 (known-null atoms selected at the clean rate), e196 (selection prior predicts real selection, dual up to 0.89), e202 (prior atoms are the workhorses), e195 (stability selection certifies OMP 5/5), e200 (per-atom z-score certifies the dual 5/5, transfers to the Pile), e203 (certified atoms track for two levels) → SURVIVES. Spawned H8a, H8b.
  - H8a A residual-based self-certificate separates real from spurious. → e180 → KILLED.
  - H8b Dropping the null-selected atoms is a certificate. → e200 → KILLED (recall collapses).
  - H8c Coarser provenance labels (block, head) are reliably observable where neuron labels are not. → e218 → KILLED for MLP block labels (block-level recall exceeds its control by 0.03 to 0.31, below the 0.2 threshold in 3/5 and above it only in GPT-2 and SmolLM2; misses are not right-block-wrong-neuron: the closest support atom is from the same block in only 2 to 5% of them). NARROWED for attention: the top head of block L is identified by its own OV atom in 73% (GPT-2) and 100% (Pythia) of tokens under OMP against controls of 5 to 10%.

## Where a write's energy goes (session 5)

- H13 A write's energy, conserved while its direction is lost (H6g/H6h), goes somewhere accountable. → e223 (exact residual identity, residual 0.000) → SURVIVES: footprint = −write + MLP responses of every later block + attention responses of every later block. Spawned H13a to H13d.
  - H13a It is redistributed among other known writers. → e223 → SURVIVES, in the most diffuse form: the delta-ledger has an effective number of writers of 400 to 1500 per intervening block growing linearly with depth (up to 5,000 to 11,000), top successor 1 to 8%, equal shares per block, half on already-active neurons. Attention carries a part of comparable size that partially cancels the MLP part.
  - H13b It is concentrated into a learned dynamical subspace (writes funnelled into shared directions). → e224 (distinct real writes' footprints stay orthogonal, cosine ≤ 0.11; random perturbations do converge, 0.3 to 0.5), e225 (the footprint cloud's effective dimension is as large as or larger than the writes' and comparable to the state's, growing with depth; random-direction footprints shrink with depth) → KILLED for real writes; off-manifold perturbations are funnelled.
  - H13c It is transformed into a functionally equivalent representation in different coordinates. → e223 (Mahalanobis footprint roughly conserved, 0.6 to 1.8, while Euclidean grows), e224 (logit effects of distinct writes only weakly related, 0.06 to 0.46) → PARTIAL: the write's distinguishability against the state distribution is conserved, and distinct writes stay functionally distinct, but no single 'functional direction' replaces the neuron.
  - H13d It is genuinely lost. → e223 → KILLED as energy (every unit is accounted for by named writers and heads); SURVIVES as identity (the ledger is near-maximal entropy, so no per-writer reading can recover the origin).

## Transport of descendants (session 6)

- H14 Descendants are transported linearly (additive across writes, homogeneous in the write's size). → e226 → NARROWED: linear in Pythia (errors 0.05 to 0.12) and GPT-2 (0.12 to 0.20), mildly nonlinear in OLMo (0.11 to 0.26), additive but inhomogeneous in Qwen (halving a write changes its footprint by 0.8 to 0.9), non-additive in SmolLM2 (1.3 to 1.9); nonlinearity follows the massive-activation neurons; opposing pairs interact more than reinforcing ones.
- H15 Transport is carried by one path, MLP chains or attention, additively. → e228 → KILLED: the interaction residual is 0.35 to 0.92 two blocks after the write and 1.1 to 2.8 at the mid layer; attention-only transport overshoots (1.4 to 9× the write) and the MLP responses cancel part of it; MLP-only transport is the closer approximation to the descendant's direction. Transport alternates and interacts.
- H16 A neuron keeps a recognisable descendant signature after its direction is gone (provenance is invertible from footprints). → e227 → SURVIVES: within-neuron footprint cosine 0.20 to 0.31 vs 0.01 to 0.12 between neurons; source neuron identified from the footprint in 88 to 98% of tokens (chance 3 to 6%), source block in 84 to 95% (chance 33%). Spawned H16a, H16b.
  - H16a The signature is readable from the state without intervention (a descendant dictionary). → e230 → SURVIVES with a split: 37 to 66% source identification from the state in four models (chance 3 to 4%), better than the neuron's own atom in the damped gated models (Qwen, OLMo, SmolLM2), worse in the GELU models where the write's own direction survives longer; ceiling from the footprint 79 to 99%.
  - H16b WDD's reading predicts a write's behavioural influence. → e227 (Spearman of removal KL with readability at L −0.16 to +0.24, with |coefficient| +0.06 to +0.30) → KILLED.
- H17 Dispersal is learned, not architectural. → e229 → SURVIVES: at initialisation a write is transmitted verbatim along its own direction (along-component 1.00, footprint cloud of the writes' own dimension); the along-component falls to 0.79, 0.53, 0.41, 0.33 at steps 2000, 16000, 64000 and final; Mahalanobis conservation and the absence of collisions hold at every checkpoint; the footprint cloud passes through a transient low-dimensional phase at step 2000 before expanding past the writes' dimension.

## Attacking the descendant signature (session 7)

- H18 The descendant signature is contextual activation statistics, not a property of the write. → e231 (transplants of the neuron's write vector into foreign states classify as that neuron in 0.89 to 0.97 of tokens at four blocks in GPT-2, Pythia, OLMo, SmolLM2, 0.55 in Qwen; 0.25 to 0.71 at the mid layer; magnitude-invariant; random directions at chance) → KILLED at four blocks, NARROWED at the mid layer, where about half of the natural identifiability is vector-driven and half comes from the states the neuron naturally fires in. Spawned H18a.
  - H18a The signature depends on the operating point (write magnitude). → e231 (0.5× and 2× give identical accuracy) → KILLED.
  - H18b Signatures superpose. → e231 (two injected vectors both recovered in the top two centroids in 0.55 to 0.83 of tokens at four blocks, 0.04 to 0.33 at the mid layer) → SURVIVES early, fades with depth.
- H19 There is a depthwise crossover from the native coordinate to the descendant signature, then to neither. → e232 → SURVIVES: crossover at two blocks (Qwen), four blocks (OLMo, Pythia), from the first block in GPT-2; SmolLM2 keeps the native read ahead until the end. The intervention footprint keeps 4.7 to 5.6 of 5.6 to 6.0 bits through the mid layer and loses about half in the last two blocks, where provenance is finally scrambled.
- H20 The signature is a brittle coincidence of the exact weights. → e233 → KILLED: identification is unchanged with every weight perturbed by 3% of its RMS (state moved 8 to 15%) and survives at 10% perturbation (state moved 28 to 56%, model degraded) in Pythia and OLMo (0.98), GPT-2 (0.87), partially in Qwen (0.56) and SmolLM2 (0.48).
- H21 The signature transfers across corpora without refitting. → e234 → SURVIVES for GPT-2 (WikiText-fitted centroids identify Pile sources at 0.95, chance 0.07; reverse 0.83); SmolLM2 weak: 0.35 / 0.27 against within-corpus 0.61 / 0.45, chance 0.08.
- H22 The identity is carried by one path (MLP chain or attention). → e235 → KILLED: both paths carry it; MLP-only transport preserves the full-footprint signature in 5/5 (0.68 to 1.00), attention-only preserves it in the GELU models and OLMo (0.91 to 0.94) and carries a different but self-consistent neuron-specific signature in Qwen and SmolLM2 (0.88 / 0.62 self, 0.15 / 0.10 against the full centroids).

## Transported coordinates (session 8)

- H23 Transporting the native dictionary through the network (one atom per token, averaged over contexts) restores state-level provenance at depth where native WDD fails. → e236 → KILLED in the models run so far: the transported dictionary decodes the dominant write worse than the native one at the mid layer in GPT-2 (dual 0.89 → 0.62) and Pythia (0.68 → 0.48) and no better in Qwen (0.51 → 0.49), much worse in OLMo (0.82 → 0.37) and slightly worse in SmolLM2 (0.78 → 0.74): killed in 5/5. e240: coherence rises only in the gated models (Qwen 0.15 → 0.23, SmolLM2 0.17 → 0.27) and not where the failure is largest (OLMo 0.16 → 0.13); the token-averaged image is a blurred version of each token's descendant, sharp enough for classification among tens of candidates, not for a 64-sparse competition among tens of thousands of atoms.
- H24 Transport is angle-preserving within a state (an isometry on injected directions). → e238 → SURVIVES in 5/5: descendant cosines equal initial cosines within ±0.02 from 0.99 down to 0 at every level to the mid layer; no resolution limit beyond the initial separation. The geometry of natural descendant centroids follows the write vectors' geometry only in the GELU models (Spearman +0.55, +0.31), not in the gated ones (context dominates the centroids).
- H25 Provenance capacity from the state saturates at a small number of sources. → e237 → KILLED up to 128 candidates: state bits keep growing with log2 K (Qwen 5.1 to 5.6 bits at K = 128 through four blocks, OLMo 4.9, Pythia 4.0, GPT-2 3.0 at K = 64); the footprint carries 6.0 to 6.4 of 7 bits at every depth to six blocks; only SmolLM2's mid layer collapses (0.8 bits).
- H26 The signature is imposed by the downstream network rather than carried by the vector. → e239 (Pythia final vs step 16000: vectors cos 0.74, centroids cos 0.54; final vectors through the step-16000 network classified by either checkpoint's centroids at 1.00; step-16000 vectors through the final network at 0.99 / 0.93) → KILLED: the signature follows the vector and the transport is stable from step 16000 on.

## Compositionality, prediction, training order (session 9)

- H27 The provenance code is compositional: several simultaneous writes remain individually decodable and their descendants add. → e241 → SURVIVES with limits: additive within 0.5 up to eight writes in 5/5 (sixteen in the GELU models and SmolLM2); eight injected identities recovered from the mixture at 0.61 to 0.88 two blocks out (chance 0.28 to 0.38) in 5/5, and at the mid layer at 1.5 to 2× chance in GPT-2, Pythia and OLMo only.
- H28 An unseen neuron's descendant is predictable from its write vector alone (the transport is the decoder). → e242 → SURVIVES: zero-shot identification from the transplant image equals the natural-centroid standard two blocks out in 5/5 and at the mid layer everywhere except the massive-activation block-2 neurons of Qwen and SmolLM2 (0.44 and 0.39 against 0.85 and 0.71).
- H29 Training orders the phenomena: transport isometry and identity at initialisation; readability and dispersal at warmup end; a transient funnel; an expanded descendant chart; late final-layer scrambling. → e243 → SURVIVES as measured on Pythia (native +4 readability 0.01 → 0.18 → 0.38 → 0.40 → 0.37 → 0.32; along-survival 1.00 → 0.97 → 0.64 → 0.53 → 0.41 → 0.34; pair cosine 0.75 at every step; footprint bits at +4 4.6 → 3.9 → 3.4 → 4.2 → 4.5 → 4.7; penultimate-block footprint bits 4.6 → 3.9 → 3.3 → 3.8 → 3.3 → 2.3; footprint dimension 70 → 73 → 58 → 162 → 186 → 223).
- H30 The GELU-versus-gated split in centroid geometry comes from a more context-variable MLP linearisation in gated models. → e244 → KILLED: the factor's relative deviation is 1.23 and 1.86 in the GELU models and 1.16 to 1.46 in the gated ones; the split follows the massive-activation neurons instead (e242).

## The recoverability frontier (session 10, self-directed)

- H31 Transport is compositional where decomposition is not: the identity of the largest of many superposed writes survives in the descendant while the native dictionary cannot read it. → e245 → SURVIVES in GPT-2 and Pythia (largest of 64 heavy-tailed writes identified from the mixture at 0.92 and 0.84 against chance 0.02; native dual read 0.00 to 0.01 at every k); Qwen 0.72 and SmolLM2 0.46 at k = 64 (native 0.00 to 0.03); OLMo unfinished at close.
  - H31a The native failure on injected writes is a prominence effect, not foreignness. → e247 → SURVIVES (Pythia: native recall 0.06 → 0.52 → 0.99 → 1.00 at 1× to 8× magnitude while identity stays 1.00; natural writes of the same prominence read the same).
  - H31b The transported code is neuron-specific rather than vector-specific. → e245 (random unit vectors behave exactly like neuron writes: top-1 1.00 at k = 1, 0.06 to 0.07 at k = 16) → KILLED: the code is vector-specific.
  - H31c The context-independent part of a descendant is large. → e245 (coherent fraction 0.46 → 0.29 in GPT-2, 0.35 → 0.07 in Pythia from two blocks to the mid layer, the same for random vectors) → NARROWED: small at depth, yet sufficient for 64-way identification.
- H32 The descendant chart can be built without data (random-token inputs). → e246 → SURVIVES on Pythia (random-token images identify natural footprints at 1.00 two blocks out and 0.93 at the mid layer, against 1.00 / 0.95 for text images; image cosine 0.91 / 0.77); the other models were queued at close.

## Characterising the transported object (session 11, self-directed)

- H33 The native-to-descendant crossover depth is predictable from single-block gains (passive survival prod(1+g) < 0.5). → local analysis of e215 and e232 → SURVIVES in Pythia (6 vs 6) and Qwen (4 vs 4), within a level in OLMo (4 vs 6); fails where a chart is anomalous (GPT-2's prior-dominated dual loses from the first block; SmolLM2's state-level signature never wins before the end).
- H34 Provenance survives in a few dimensions. → e249 → KILLED as 'tiny': 90% of full identification needs 32 to 64 random dimensions (8 to 32 principal components) two blocks out and 128 to 256 (32 to 128 components) at the mid layer.
- H35 The native coordinate is lost at a specific sub-operation. → e249 → SURVIVES: the along-direction component is unchanged by the attention residual add and drops at every MLP add; the descendant identity is unchanged by both through block 6 in 5/5.
- H36 Massive-activation neurons are a failure of transport. → e250 → KILLED: they are transported as linearly and angle-preservingly as matched normal neurons; what differs is that their natural descendant points away from their transplanted image (cos −0.45 to −0.75 in Qwen, OLMo, SmolLM2), because their natural effect runs through the sink mechanism that a transplanted copy does not trigger. They are the boundary of vector-determined transport.
- H37 An explicit linear transport operator, fitted by the analysis from random injections, predicts held-out descendants, gives zero-shot provenance for natural footprints, inverts descendants back to write space, composes across segments and preserves the Gram matrix; and the code transports vectors, not neuron identities (a+b test). → e248 → SURVIVES with a split: the random-fitted ridge operator predicts held-out random directions' descendants (cos 0.58 to 0.94 two blocks out), preserves the Gram matrix partially (Spearman 0.5 to 0.8), composes across segments as well as the direct operator, and the a+b test says vectors are transported, not neuron identities (0.83 to 0.96 in 5/5); zero-shot provenance and inversion from the random-fitted map work in the GELU models (Pythia 1.00, GPT-2 0.88 two blocks out) and not in Qwen, where real writes and random directions are transported differently (e224, e242). Decodable, not linearly reconstructible (inverse cosine 0.1 to 0.4).

## Recoding, regimes, function (session 12, from the six-experiment program)

- H38 The birth-transport-scrambling regimes change at one block (a representation phase transition). → e252 → KILLED: the observables are staggered (along-component halves at blocks 4 to 8, the state-level descendant chart at 10 to 16, the intervention footprint only in the last one to three blocks or never) and angle preservation never breaks, at any level, in 5/5. What looks like scrambling in the last blocks is a loss of the natural descendant's distinctness, not of the isometry.
- H39 Whitened footprint energy is a conserved quantity. → e252 → NARROWED: conserved to the mid layer, then growing in the amplifying models (GPT-2 2.6, Qwen 8.5, SmolLM2 9.6 at the last block), dipping and recovering in Pythia, flat in OLMo.
- H40 Downstream, the descendant is better coded by dense coordinates than by the sparse native dictionary (a sparse-to-distributed recoding). → e251 → SURVIVES for provenance at equal budget (32 coefficients at the mid layer: descendant basis 0.70 to 0.90 vs native sparse code 0.03 to 0.67 vs random 0.49 to 0.66; all 1.00 at birth) and NARROWED for reconstruction (no 8-coefficient code reconstructs the descendant, native and dense equal within 0.1).
- H41 Descendant geometry predicts functional geometry better than neuron identity does. → e254 → SURVIVES in 5/5 (Gram Spearman with the logit-footprint Gram +0.23 to +0.61 for descendants vs −0.02 to +0.37 for write vectors; per-token agreement with the functional identity 0.49 to 0.60 for the descendant vs 0.02 to 0.21 for the native atom in the gated models). The logit footprint itself identifies the source at 0.52 to 0.60.
- H42 A single contextual variable (the neuron's own natural activation, the sink route) explains the massive neurons' context-dependent transport. → e253 → KILLED: the transplant's match to the natural descendant is flat across the neuron's own activation quartiles and uncorrelated with it (Spearman −0.06 to +0.11), only mildly related to norm and position, and at the mid layer near zero or negative in every context; the massive neuron's natural effect is cross-token (the sink it creates changes other positions), not a within-token state variable.
- Not run: the residual-stream rotation with compensated weights (not a symmetry of pre-norm transformers with elementwise norm gains; the exact symmetry, neuron permutation, is the gauge invariance already on record), and the transport-defect map (covered by e226, e241, e250).

## Descendants and function (session 13, from the five-experiment list)

- H43 Descendant geometry organises neurons by downstream function better than write geometry. → e255 → SURVIVES: descendant neighbours share logit footprints more than write neighbours in 5/5 and more than random neighbours in 4/5; pairs with dissimilar writes but similar descendants are functionally similar in 5/5 (functional convergence without convergence at birth); held-out nearest-neighbour prediction of logit footprints transfers in Qwen, SmolLM2 and Pythia and not in GPT-2 or OLMo.
- H44 The functional footprint composes as the state descendant does. → e256 → SURVIVES: logit additivity errors equal the state's within 0.03 at every m in 5/5; identity is far less legible in function space (single-write logit identification 0.15 to 0.52; chance from an 8-write mixture).
- H45 The massive neurons' anomaly is a cross-token route (their effect migrates to receiver positions). → e257 → KILLED: 0.89 to 1.00 of their footprint energy stays at the source position and receivers do not identify them. Their anomaly is unexplained by any variable tested (own activation, norm, position, cross-token spread). Side result: cross-position provenance exists in GPT-2 (receivers identify a normal neuron at 0.46) and not in the other models.
- H46 Descendant-function coupling emerges during training after dispersal. → e258 → KILLED: the coupling is highest at initialisation (Spearman +0.84, a linear network), is broken by training (+0.14 at step 64000) and partly rebuilt by the end (+0.31).
- H47 One perturbation, three questions, three coordinates: who wrote it (native atom), what it became (descendant), what it affected (logit footprint) are answered best by three different coordinates. → sessions 6 to 13 → SURVIVES as the organising statement (table in SYNTHESIS.md).

## The descendant as functional coordinate (session 14)

- H48 Descendant similarity predicts functional similarity after controlling for write similarity. → e259 → SURVIVES in 5/5 (partial Spearman +0.29 to +0.60; the reverse −0.06 to +0.16; the 2×2 cells follow the descendant; same-descendant/different-writer pairs above the functional median in 5/5).
- H49 Functional geometry moves from write space into descendant space with depth. → e260 → SURVIVES in 5/5 (descendant-logit Spearman rises from the write-logit value at birth to +0.65 to +0.88 near the top; descendant-write decays over the same depths).
- H50 Function needs fewer descendant dimensions than identity. → e263 → SURVIVES (function saturates by 8 to 32 dimensions, identity needs 32 to 128); and the descendant predicts a token's functional footprint better than the neuron label does (above the identity oracle in 5/5).
- H51 Neurons with similar descendants are interchangeable under intervention. → e262 → KILLED (substitution worsens the removal effect in Pythia, GPT-2, OLMo and is neutral in Qwen, SmolLM2).
- H52 The neurons that compensate for a removed write are its descendant-space neighbours. → e261 → KILLED (compensators are a diffuse crowd only weakly aligned with either the write or its descendant; GPT-2 and Pythia lean to the write).

## What kind of object a descendant is (session 15)

- H53 Logit-footprint neighbours are descendant neighbours (few many-to-one maps). → e265 → SURVIVES in 5/5 (0.31 to 0.75 vs chance 0.18 to 0.25; many-to-one share 3 to 12%).
- H54 The coordinate change has an empirical depth. → e265 → SURVIVES: the write-space neighbour is lost in descendant space at levels 3 to 6 (never in GPT-2) and the logit-space neighbour is found there at levels 2 to 4.
- H55 A write's causal importance is predictable from its descendant in few dimensions. → e265 → SURVIVES moderately (Spearman with the removal KL 0.33 to 0.64, saturating by 32 dimensions; readability never predicted importance).
- H56 One operation (attention or the MLP) builds the functional organisation of descendants. → e266 → KILLED as universal: attention in Pythia, OLMo and SmolLM2, the MLP in GPT-2 and Qwen; the largest increment is right after the write or at the very end.
- H57 A write's successors form a stable lineage rather than a shared hub graph. → e267 → SURVIVES (within-neuron split Jaccard 0.53 to 0.76, between-neuron 0.01 to 0.08, hubs carry 0 to 3%).
- H58 Descendant-similar neurons have interacting or redundant effects. → e264 → KILLED: at co-active tokens their removal effects are independent (interaction ~0) and uncorrelated (cosine ~0), and their per-token descendants are near-orthogonal; aggregate descendant similarity is a centroid-level property, which explains H51's failure.

## Sufficiency and causal carriage (session 16)

- H59 The descendant is a sufficient statistic for the write's functional effect (adding the write vector does not improve prediction). → e268 → SURVIVES in 5/5 at every depth (gain at most +0.01; adding the write hurts by 0.02 to 0.10 in the last blocks).
- H60 Depth quotients write identity into functional classes (functional distance tracks descendant distance where write distance is uninformative). → e268 → SURVIVES modestly (high-D/low-W pairs: relative distance in write 1.01 to 1.08, descendant 0.81 to 0.98, function 0.80 to 0.99).
- H61 Identity bits exceed function bits exceed behaviour bits at depth. → e268 → SURVIVES broadly (identity dimensions grow to 128 to 256 with depth; function stays at 16 to 32; behaviour at 4 to 32).
- H62 The descendant causally carries the write's effect: injected directly, it reproduces the write's natural logit effect where the write vector in a foreign context does not. → e269 → SURVIVES in 5/5 (cosine 0.28 to 0.62 vs −0.29 to +0.16); injected-descendant effect geometry follows descendant geometry (Spearman +0.38 to +0.76).
- H63 The descendant geometry is stable across tokens, corpora and training. → e270 → SURVIVES across tokens (Jaccard 0.43 to 0.69, chance 0.11 to 0.15) and corpora (0.50, 0.53), moderately across checkpoints (0.40 vs step 64000, 0.27 vs 16000; write-space 0.77, 0.45).

## Parameterisation and the capacity ladder (session 17)

- H64 The descendant coordinate parameterises the intervention effect continuously (effects interpolate linearly along descendant segments, and effect distance tracks descendant distance). → e271 → SURVIVES in 5/5 (interpolant effects at cosine 0.99 to 1.00 to the linear interpolation; effect-distance vs descendant-distance Spearman +0.47 to +0.88 with slope about 1).
- H65 Reconstruction needs far more dimensions than identity, function or behaviour. → e272 → SURVIVES in 5/5 (reconstruction needs the full basis; identity 4 to 256 growing with depth; function 8 to 64; behaviour mostly 8 to 32).

## Kernel, nesting and the transport law (session 18)

- H66 Function is a quotient of the descendant: descendant space has functional null directions (equivalence classes). → e273, e273b → KILLED in 5/5 (relative functional response to a fixed descendant displacement is 0.31 to 0.53 along top, middle and bottom principal components and random directions alike, ratio 1.2 to 1.4; directions toward other descendants only 15 to 30% higher; GPT-2's bottom-PC response of 2.6× is the massive-activation channel 447, which carries 0.103 of the bottom PCs' energy against 0.004 chance).
- H67 The identity, function and behaviour subspaces of the descendant are nested. → e274 → NARROWED: identity and function share a leading core (function energy inside identity 0.48 to 0.76 at 8 dimensions, chance ≤ 0.014) and rotate apart beyond it (0.39 to 0.47 at 32 dimensions); the linear behaviour direction lies outside both in 4/5 (inside function in SmolLM2, 0.48 to 0.77), read with the caveat that the removal KL is quadratic in the logit change.
- H68 The massive neuron is the extreme tail of the one linear transport law. → e275, e275b → KILLED where testable (SmolLM2's massive neuron: descendant anti-parallel to sign(coef)·T w, cosine −0.64, rank 1/23, z +7.6 against a bulk within 0.77 to 0.98); the non-massive largest-coefficient neurons of the other four models sit inside the bulk (z −0.4 to +1.6).
- H69 The linear transport law fits best the neurons whose write direction survives most. → e275b → SURVIVES in 5/5 (Spearman between residual and along-survival −0.40 to −0.78); no relation with coefficient size or identifiability.
- H70 The corpus-averaged linear transport reproduces the natural descendant direction. → e248, e275b → NARROWED: at mid depth the sign-corrected T w meets the natural descendant at cosine 0.07 to 0.47 only (nearest-centroid provenance in e248 was relative alignment); the per-write transplant image (e242) is the better predictor (0.44 to 0.73).

## Leverage, budget, amplitude and rotation (session 19)

- H71 Downstream components are tuned to the descendants of the model's own writes (natural descendants out-lever random-vector descendants and covariance-matched directions). → e276 → KILLED in 5/5 as a magnitude claim (logit-response ratio 0.94 to 1.18 vs random-vector descendants, 0.65 to 1.14 vs covariance-matched; KL 1.0 to 2.0 and 0.36 to 1.32); the descendant's effect is direction-specific (H62) and magnitude-generic; the raw write direction at level L has random leverage (0.98 to 1.05).
- H72 Provenance survives on a small consistent component while the remainder is token-specific and not linearly readable from context, which is why no transported dictionary decomposes the state. → e277 → SURVIVES in 5/5 (coherent fraction 0.61 to 0.82 one block after birth, 0.16 to 0.33 at mid depth, 0.05 to 0.14 late, with identification 0.89 to 1.00 at mid depth and 0.49 to 0.83 late; remainder R² from the state ≤ 0.10 at mid depth in 4/5; SmolLM2 keeps 0.19 to 0.27 from the birth context).
- H73 The massive neuron lies beyond the linear range of the transport law (an amplitude effect). → e278 → KILLED (its transplant obeys the law at 0.1× to 10×, cosine 0.56 to 0.72, self-consistency 0.78 to 1.00 through operating-point shifts of 1.05 to 10.1; its natural descendant is anti-parallel at every amplitude, −0.56 to −0.90) → NARROWED to context-bound: the anomaly is the natural removal response at the write's own token, developing between +2 blocks (agreement 0.99, e250) and mid depth (−0.75), not the vector, its amplitude (e278), its activation or position (e253), or a cross-token route (e257).
- H74 Ordinary writes have amplitude-invariant descendants. → e278 → NARROWED to about the natural amplitude (self-consistency 0.78 to 0.99 at 1×; 0.08 to 0.80 at 10× with operating-point shifts of only 1.2 to 2.3).
- H75 There is a coordinate transition at depth. → e279 → NARROWED to two rotation zones, the 1 to 5 blocks after birth (identity subspace kept 0.38 to 0.69 per block) and the last block (0.43 to 0.70), with a stable middle (0.65 to 0.91) and one mild dip (SmolLM2 block 10, 0.46); identification collapses late without rotation, by loss of coherent energy (H72); the function subspace rotates in step with the identity subspace.

## The descendant as a dynamical state variable (session 20)

- H76 An interpolated descendant injected where neither endpoint is written follows the interpolation of the endpoint trajectories through the rest of the network (no snapping). → e280 → SURVIVES in 5/5 (cosine 0.97 to 1.00 at every later level and at the logits; fitted position equal to the true one within 0.01; 95 to 100% of the response in the endpoint plane).
- H77 The transport law composes functionally. → e281 → SURVIVES in 5/5 (composed prediction's effect vs the actual image's effect 0.43 to 0.81, direct fit 0.44 to 0.74, random 0.00; composed at least as good as direct in 4/5).
- H78 A whitened (Mahalanobis) metric makes transport more isometric than the Euclidean one. → e282 → KILLED (gain spread and pair-cosine preservation worse whitened at every level in 5/5).
- H79 Transport is an isometry specifically on the manifold of actual writes. → e282 → KILLED and reversed in gain (writes' gain spread 0.08 to 0.41 vs 0.02 to 0.07 for random directions, whose constancy is concentration of measure over an operator with singular spread 11 to 50); the isometry on writes is angular (pair cosines, e238, e282).
- H80 The descendant is a state variable: its future is predicted by its present through the data-free operator, not by its origin. → e283 → SURVIVES in 5/5 (per-token cosine to the future descendant 0.43 to 0.78 at +2 and 0.25 to 0.61 at +4 vs 0.03 to 0.12 for the write and 0.00 for a random vector; the state adds ≤ +0.16; the predicted future centroid reproduces the future effect at 0.59 to 0.85).
- H81 The massive neuron's context-bound reaction has a single carrier. → e284, e284b → SURVIVES for SmolLM2 (the MLP of block 10, frozen at the neuron's own tokens, flips the natural footprint from −0.67 to +0.52 against the transplant image; all attention frozen −0.66; both-frozen sanity 1.00 in 10/10); ordinary neurons' reactions are small and spread; block 10 is also SmolLM2's one mid-depth coordinate rotation (H75).

## Theory (session 21)

- H82 The transport's gain follows the fitted operator's spectrum: top singular directions > random > middle ≥ bottom, with random directions near the root-mean-square gain and no exact null direction. → e285 → SURVIVES in 5/5 (gains 0.33 to 1.12 / 0.26 to 0.57 / 0.22 to 0.52 / 0.16 to 0.35).
- H83 Pair-cosine preservation is gain concentration, not isometry: cos_out = cos θ · ρ / sqrt(cos²θ · ρ² + sin²θ) with ρ the gain ratio of the pair's base direction to a random direction. → e285 → SURVIVES quantitatively (15/15 within 0.07; random pairs preserved at ρ ≈ 1, top pairs raised to 0.81 to 0.93, bottom pairs lowered to 0.40 to 0.74).
- H84 The additivity error of superposed writes depends on the total injected norm, not on their number. → e285, e285b → OPEN (reference construction confounded by cross-token coherence; what is seen: direction agreement improves with the number of superposed writes at constant norm, 0.95 to 0.97 at sixteen, and degrades with amplitude, 0.50 to 0.77 at four times natural).

## The compression test (session 22)

- H85 Downstream observables of a write are functions of its descendant with progressively smaller required dimension: reconstruction > provenance > function > behaviour. → e286 → NARROWED to two rungs (reconstruction 512 to 1024 directions; future descendant 8 to 32, identity 16 to 32 with a ceiling of K − 1, function 8 to 32, behaviour 8 to 32, all from the same descendant with one decoder family; inside the causal rung the ordering is mild, behaviour smallest in 3/5).

## Shared versus token-specific read-out (session 23)

- H86 The low-rank causal read-out lives in the shared transport term (the span of the class centroids), and the token-specific term is causally redundant. → e287, e288 → NARROWED: the centroid span reaches the full score (e287), but so do a covariance-matched random K-dimensional subspace and a permuted-label span for function, future and KL (e288); the read-out lives in the high-variance, low-rank part of the descendant cloud, and only source identity is specific to the shared term. The token-specific remainder is redundant for the four observables but carries its own future (0.09 to 0.28).

## The kill-tree (session 24)

- H87 Identity, function, future and behaviour are read from one causal subspace. → e288 → NARROWED: identity, function and future share 0.35 to 0.73 of their 16-dimensional energy (chance ≤ 0.028) and each decodes all four observables near its own score; the KL subspace is less aligned (0.12 to 0.61); a covariance-matched random subspace matches them on everything but identity.
- H88 The causal coordinate is manipulable one direction at a time. → e289 → SURVIVES in 5/5 (paired logit response +0.14 to +0.34 on the diagonal, 40/40 positive, off-diagonal 0.00, random 0.03 to 0.04).
- H89 Adding the massive direction at its own tokens reproduces the natural footprint (the coherent-reaction reading). → e290 → KILLED and replaced: the own-token addition obeys the first-order law (+0.55 to the foreign addition, +0.66 to the linear law) and is anti-parallel to the natural removal footprint (−0.90); the response at the massive neuron's own tokens is even in the deviation from its natural amount (a set point).
- H90 The causal subspace is transported into itself. → e288 → NARROWED to stationarity: overlap over two blocks 0.30 to 0.82 without transport and never higher with it; the residual transport is near the identity on the subspace.
- H91 The two rungs are learned: at initialisation the descendant is low-dimensional and fully predictive; dispersal makes it full-dimensional within the first four thousand steps while the causal dimension grows slowly. → e291 → SURVIVES (reconstruction 64 → 256 → 1024 directions by step 4000; identity 4 → 32, function 2 → 32, future 2 → 16, KL 4 → 8 over training; span fraction 0.93 → 0.28).
- H92 The token-specific remainder is a transported state of its own. → e288 → SURVIVES weakly (its future predicted at 0.09 to 0.28, shuffled 0.00).

## The four questions (session 25)

- H93 The causal coordinate dictates the entire logit response to an intervention. → e292 → NARROWED: selective (paired directions hit at three to four times chance, and only they) but partial (full-vector cosine 0.09 to 0.33; 19 to 39% of the response energy in the paired span); generalises to unseen neurons with loss (0.05 to 0.24) and not to directions outside the cloud (0.01 to 0.08).
- H94 The causal core transports as a coherent object. → e293 → SURVIVES (fixed basis, two blocks: R² 0.89 to 0.96, gain 0.64 to 1.16, spread 1.3 to 2.7, Procrustes residual ≤ 0.17, off-diagonal ≤ 0.04, rotation ≤ 0.11).
- H95 The observables share one nested predictive hierarchy rather than separate latents. → e294 → SURVIVES (joint rank = largest individual rank in 4/5, far below the sum); with the caveat that the linear future needs about 128 directions, an intermediate scale.
- H96 The leftover dimensions of the descendant are a second state variable. → e295 → KILLED at this resolution (indistinguishable from a random vector under intervention on the logits, the KL, the later core and other tokens; carried by the near-identity transport, e288, but unread).

## The literature-derived program (session 26)

- H97 Under a causal metric the descendant cloud is low-rank and its geometry is the functional geometry. → e296 → SURVIVES in 5/5 (Spearman with held-out functional geometry up from 0.35 to 0.65 Euclidean to 0.52 to 0.82 causal; participation rank down 1.5 to 5×).
- H98 Individual causal coordinates persist as themselves across blocks (channels, not just a subspace). → e296 → SURVIVES (self-correlation 0.90 to 0.97 vs best other 0.23 to 0.34; self wins 16/16 in 5/5).
- H99 The physical dimension explodes within a few blocks of birth while the causal dimensions creep (the layerwise analogue of training). → e297 → SURVIVES (reconstruction 128 to 256 at block 3, 512 to 1024 by block 4 to 8; identity 8 → 32 slowly; function 8 to 32 mostly flat; future 8 to 16; KL 4 to 32).
- H100 The causal core predicts attention routing. → e298 → NARROWED to weak (R² 0.09 to 0.16 for per-head output changes, two to five times a random projection, below the descendant's top-64 PCs at 0.13 to 0.24; attention probabilities unavailable).
- H101 The perturbation propagates as a low-rank spatiotemporal wave. → e299 → NARROWED to the massive channel (rank one in Qwen, OLMo, SmolLM2; random-like spectrum in GPT-2, Pythia); the impulse response is mostly local (15 to 41% of the energy at later tokens by mid depth).
- H102 The core is a property of the network's receptor geometry, not of the writes' geometry. → e300 → SURVIVES (cores of 116 to 200 neurons in a 6 to 8-dimensional effective space with |cos| 0.22 to 0.28 against 0.02 to 0.04 for writes; write similarity does not predict core similarity, 0.00 to 0.08).
- H103 The massive neuron's own-token response is even in the deviation (a set point); all other responses are odd. → e301 → SURVIVES (evenness +0.88 vs −0.69 to −0.98 for every other neuron and context).
- H104 Causal coordinates are linear, independent channels. → e302 → SURVIVES (linear R² ≥ 0.82 in 39/40, saturation 0.88 to 1.26, interactions weak, uniform and rank one, self-interaction the largest).
- H105 The core coordinate of a write is computable from the write. → e303 → SURVIVES in the GELU models (cosine 0.53 to 0.71, identification 0.52 to 0.66 at chance 0.03 to 0.05), WEAK in the gated models (0.25 to 0.32).
- H106 The dimensionality hierarchy is special to WDD writes. → e304 → KILLED (the same ladder for random, covariance-matched, attention and later-MLP directions).

## What P_ℓ is (session 27)

- H107 The causal core is an invariant or singular subspace of the transport operator (a dynamical subspace). → e305 → KILLED (overlaps with T's top input and output spaces, the update's spaces, the slowest and fastest modes all at chance 0.008 to 0.028; gain of core directions equal to random directions').
- H108 The causal core is the read-out's high-gain subspace (a read-out bottleneck). → e305, e308 → KILLED (chance overlap with the forward-estimated G's top space and with the true backward sensitivity subspace).
- H109 Physically unrelated perturbation families share one functional coordinate system. → e306 → SURVIVES with depth (cross-family function-core overlap 0.02 to 0.06 two blocks after injection, 0.14 to 0.32 at mid depth, 0.44 to 0.57 near the end against within-family reliability 0.51 to 0.67); identity cores stay family-specific.
- H110 Channels keep their identities across all depths. → e307 → SURVIVES (identity assignment at every block in 4/5, gain 0.96 to 1.07, self-R² 0.83 to 0.89, mixing 0.02 to 0.06, no births or deaths outside SmolLM2's massive channel).
- H111 Forward transport modes and backward sensitivity modes coincide. → e308 → KILLED (overlap 0.03 to 0.07); the first-order identity with the true Jacobian holds at per-token cosine 0.58 to 0.67 (correlation 0.36 to 0.45; SmolLM2 0.45 and −0.02).
- H112 P_ℓ is the dominant-variance subspace of the model's own perturbation cloud, carried by near-identity transport and read by a near-isotropic read-out. → e288, e296, e305, e306, e307, e308 → SURVIVES as the only reading consistent with all of them.

## The quotient picture (session 28)

- H113 The causal coordinate is the subspace of high residual variance or of a component's output variance. → e309 → KILLED (core in the state's top-64 at 0.21 to 0.34, in components' at ≤ 0.22, in its own cloud's at 0.93 to 0.95).
- H114 The perturbation cloud is a curved manifold, locally low-dimensional. → e309 → SURVIVES in 3/5 clearly (local rank 26 to 30 vs global 105 to 130), mildly in GPT-2 (22 vs 31), artifact in SmolLM2.
- H115 The low-dimensional functional coordinate requires proximity to the natural manifold. → e310 → KILLED (orthogonal, heavy-tailed, sparse, low-rank and shuffled perturbations show the same ladder; proximity changes gain, image rank and future coherence only).
- H116 The network collapses physically different perturbations into functional equivalence classes independent of physical proximity and family. → e311 → SURVIVES (physical-functional Spearman rising from 0.06 to 0.17 to 0.37 to 0.66 with depth; functionally closest pairs physically far at 38 to 48% against a 50% baseline and cross-family at 63 to 68% against 67%; no contraction through the middle, 0.88 to 0.95 at the last step).
- H117 The converged functional coordinate system transfers across perturbation distributions. → e312 → SURVIVES near the output (cores 85 to 90% of own, metrics equal to own), PARTIAL at mid depth (60 to 75%, metrics at the Euclidean level); identity never transfers.
- H118 Functionally equivalent but physically unrelated perturbations interact when superposed. → e312 → KILLED (cosine 0.98 to 1.00, error 0.09 to 0.19, same as dissimilar pairs).

## WDD and the quotient (session 29)

- H119 The functional coordinate of a joint perturbation is the WDD-ledger-weighted sum of the atoms' coordinates (Z c). → e313 → SURVIVES (cosine 0.54 to 0.68 in GPT-2, Pythia and OLMo, 0.30 to 0.34 in Qwen and SmolLM2, random −0.03 to +0.04; better in the coarse coordinate than in full dimension, 0.16 to 0.43); the quotient is additive on unrelated atoms (0.78 to 0.97).
- H120 The coordinate predicts observables it was not built from. → e314 → NARROWED (attention-norm and cross-token observables at the full-descendant level and twice random; entropy modestly; behavioural scalars no).
- H121 The coordinate's dynamics under intervention are predicted without refitting. → e315 → SURVIVES for the state (future core coordinate 0.95 to 0.99), PARTIAL for the logits (0.13 to 0.32), with the KL even in 40/40 direction-signs.

## The application study (session 30)

- H122 Neurons with the same functional coordinate form functional units whose joint ablation is coherent. → e316, e316b → KILLED (class coherence 0.15 to 0.41 against 0.18 to 0.24 random; predicted-vs-actual no better than random).
- H123 Coordinate-equivalent neurons are substitutable, within or across blocks. → e316, e316b → KILLED (logit restoration −0.19 to −0.59; core restoration −0.20 to +0.12).
- H124 The quotient improves attribution of joint effects at the output over direct logit attribution and an averaged-Jacobian lens. → e316, e316b → KILLED (atlas 0.00 to 0.14, direct 0.01 to 0.16, lens 0.00 to 0.02; all poor). What survives: the ledger predicts the joint coordinate (0.21 to 0.38 for arbitrary neurons, 0.30 to 0.68 for candidates; H119).

## The quotient program, part 1 (session 31)

- H125 The quotient is universal across observables. → e317, e335 → SURVIVES (cross-observable overlap 0.37 to 0.45 at reliability 0.38 to 0.51; transfer 0.89 to 1.13 of own; a blind PCA-16 basis matches own for logits, future and KL).
- H126 The quotient's dimension is a property of the map, not of the observable's resolution. → e319 → SURVIVES (dimension min(r, 8 to 16)).
- H127 The birth-level quotient carries the same function as the mid-level one in a different subspace. → e322 → SURVIVES (decoding 0.24 to 0.42 vs 0.21 to 0.45; overlap 0.05 to 0.12).
- H128 Channels persist over the whole depth. → e320, e321 → NARROWED (in a fixed basis, GPT-2, Pythia, OLMo yes; Qwen and SmolLM2 no; with refitted bases only the subspace persists).
- H129 The read-out from the coordinate is nonlinear while the coordinate is linear. → e323 → SURVIVES (kNN on 16 coordinates ≥ linear on the full descendant).
- H130 The quotient is homogeneous, additive and flat. → e325, e326, e327, e337 → SURVIVES in the coordinate (homogeneity 0.75 to 1.00, additivity error 0.08 to 0.15, midpoints on the average 0.73 to 0.86); at the logits equivalence is weak.
- H131 Equivalence is preserved by transport. → e328 → NARROWED (0.55 to 0.79 two blocks on, decaying toward random at the last block).
- H132 The causal dimension expands beyond the linear range. → e329, e330 → KILLED (it collapses and the quotient rotates; all responses converge at 8×).
- H133 Massive-channel perturbations share the ordinary quotient. → e331 → KILLED (own quotient, overlap 0.11 to 0.25; even response at 4× in OLMo).
- H134 WDD atoms are a privileged basis of the quotient. → e332 → KILLED (rank, sparsity and concentration equal to random directions).
- H135 The sparse ledger determines a block's functional coordinate. → e333 → KILLED (top atoms −0.24 to +0.30; the full ledger 0.14 to 0.40).
- H136 Functional classes carry vocabulary meaning. → e334 → KILLED.
- H137 There is a null space and equivalence classes are its cosets. → e336 → KILLED (no null directions; differences of equivalent atoms are full-strength).
- H138 Fibres of the quotient have geometric structure. → e338 → KILLED (equal to random sets).

## The quotient program, part 2 (session 31)

- H139 The coordinate is largely implicit in the state and the source. → e339 → SURVIVES (R² 0.54 to 0.67 from the state alone, 0.45 to 0.74 from the source identity, 0.65 to 0.73 combined; SmolLM2 excepted).
- H140 The supervised quotient adds target-specific information beyond the cloud's dominant directions. → e340, e347 → KILLED (shuffled-target quotient scores equal; PCA = PLS = CCA = kernel PCA).
- H141 The later MLPs read the coordinate; attention barely does. → e341 → SURVIVES (MLP activation-change PCs R² 0.48 to 0.62; head norms 0.09 to 0.16).
- H142 A perturbation family exists for which the quotient fails. → e342 → KILLED at 1× and 4× (minimum advantage 1.11).
- H143 The quotient is context-general. → e343 → SURVIVES across positions and texts; NARROWED for token class (punctuation-and-number tokens have a narrower, asymmetric quotient).
- H144 The response along core directions is linear to well beyond the natural amplitude. → e345 → SURVIVES (3× to 10×, Pythia beyond 30×; negligible curvature).
- H145 The discarded dimensions are causally empty. → e346 → KILLED (they decode identity, function, future and token identity as well as the core).
- H146 The core component carries the logit effect. → e348 → KILLED (1 to 6% restoration vs 15 to 18% for the physical descendant).
- H147 The quotient, the cross-family convergence and the off-manifold funnel are born together in training. → e349 → SURVIVES (all by step 4000; function dimension 2 → 8 → 16 → 32).
- H148 Attention transports a token's own image; MLPs damp it. → e351 → NARROWED (attention frozen: image unchanged at 0.90 to 0.96; MLPs frozen: gain 1.3 to 2.7 and image changed; both frozen: the identity path).
- H149 The cross-token spread is attention alone and widens to the whole context. → e352 → SURVIVES (0.00 with attention frozen; 0.36 to 0.75 of the energy at other tokens by the last level, width 168 to 320 tokens).
- H150 The coordinates form an algebra beyond pairs. → e353 → NARROWED (third-order residuals small; dense superposition leaves the linear range through the total norm).
- H151 The low dimension is a kernel phenomenon. → e355 → NARROWED (functional kernel rank 4 to 22 with a long tail; coordinates align at 0.09 to 0.47).
- H152 The same direction has one coordinate across contexts. → e356 → KILLED (coherence 0.25 to 0.41, unexplained by the state).

## Phase 2: circuits, co-selection and drift (session 32)

- H153 The unit statistics used by the program (pairwise ablation interaction, joint non-additivity, interaction clustering) detect known circuits where they are known to exist. → e357 (induction), e358 (IOI) → SURVIVES (top heads 1.4-9x random interaction, non-additivity 2-25x random, in all five models; GPT-2's IOI top-12 holds 10 published heads), with the limit that interaction clusters do not recover attention-defined classes (ARI -0.01 to 0.24).
- H154 The same statistics are blind to those circuits in natural-text population averages. → e357, e358 → SURVIVES, graded (natural interaction 1.3-4x random for induction heads, at most 2x and in Qwen below random for IOI heads), and it retroactively limits H-level kills built on natural-text averages (e264, e302; see KILLED).
- H155 The blind spot is an averaging artefact: scoring only where the circuit applies restores it. → e366 → NARROWED (the top heads' interaction is 1.3-2.2x higher at applicable than at matched positions in all five, most in SmolLM2, never reaching task level; in Qwen and OLMo the induction-task top heads matter more away from induction positions).
- H156 Batch-level co-selection (the post's proposal) finds known circuits. → e359, e363 → NARROWED (induction heads co-select at z +1.5 to +8.5, strong only in SmolLM2 and Qwen; batch induction content tracked only there; the dominant co-selection structure is layer).
- H157 Co-selection at token resolution finds them. → e368 → SURVIVES in three of five (AUC 0.87-0.96 SmolLM2, Qwen, OLMo; token z two to three times window z in SmolLM2 and OLMo), FAILS in GPT-2 (0.41) and final-step Pythia (0.52).
- H158 First-order selection measures use. → e369, e370, e370b → KILLED (at an optimum a head's mean selection is zero while its use is not; GPT-2's induction heads have selection -0.002 against ablation effect +0.018 at the positions they serve; ablation co-effects find the circuit where the gradient does not).
- H159 Co-selection is strongest while a circuit is being built. → e363 and e368 across Pythia and OLMo training → SURVIVES at batch level in both (Pythia: absent at 512, rising from 1000, peaking at 3000-16000 at z 11-18, fading to 2.4-4.8 at the end; OLMo: +5.9 at 16000 fading to +1.5), NARROWED at token level in OLMo (no monotone fade, z 4.0-8.4).
- H160 Co-selection structure persists across training. → e364b → KILLED (agreement with the final matrix 0.00-0.24 through step 33000, communities never persist).
- H161 Local quantities at the trained point (gradient, curvature, co-selection) reproduce ablation-level circuit structure. → e371, e372 → KILLED (interactions 2.4-48x the local curvature, ranked at Spearman 0.11-0.53; second order ranks single effects at 0.63-0.94 but misses magnitudes; the loss along a top head's scale is flat near 1 and rises near 0, shape index 0.04-0.30 against 0.24-0.57 for random heads, and the quadratic model captures 3-65% of single and 1-54% of joint effects).
- H162 A neuron's function (basis-free logit signature) is more stable across training than its implementation (write vector): Bushnaq's first kind of drift. [Session 33: the companion claim that implementation settles first so lower parts are swappable is killed by stitching, H180.] → e360, e360d → KILLED for Pythia (reliability-corrected signatures reach the final at 0.34-0.48 when writes are at 0.60-0.72; implementation settles first, function follows as the downstream network settles); OLMo agrees (e362d: corrected signatures 0.23-0.66 when writes are 0.49-0.76).
- H163 Early-layer neurons' logit effects are context-free. → e360d, e360e, e362d → KILLED (two token samples agree at 0.93 at step 1000 and 0.28 at the end in Pythia block 2, 0.53 at step 32000 and 0.17 at the end in OLMo block 2; late blocks stay at 0.80-1.00).
- H164 Neuron importance turns over late in training. → e360c, e362 → SURVIVES (Pythia: 35-40% of the final top-5% enter after step 33000; OLMo: median entry at step 400000-550000 of 1.45M).
- H165 Features speciate (clusters split more than they merge) over training. → e360c → KILLED at this resolution (split and merge entropies equal within 0.1 bit at every step, both models).
- H166 The quotient's core results are a block-2 artefact. → e361 → KILLED (all five findings hold at four birth depths in all five models).
- H167 The functional coordinate is a persistent object across training once basis rotation is removed. → e365b → KILLED (its logit image overlaps the final at 0.24-0.39, no closer to the final quotient image than to a random-direction image; the generic image stabilises at 0.70).
- H168 The quotient's birth coincides with the induction transition, and the induction circuit carries it. → e365, e373 → SURVIVES in time (dimension 2 to 8, gain over random directions peaks 1.9x, loss 6.7 to 5.3 and the largest neuron-signature reorganisation, all between steps 512 and 1000) and NARROWED in mechanism (removing the induction heads removes a third of the descendant cloud's participation rank at steps 1000-2000 and none at the end, random heads none; the later growth of the quotient does not depend on them).
- H169 Circuits keep adding redundancy after they work. → e365 → SURVIVES on the loss (induction top-8 signed interaction +0.08 to +0.94 and joint non-additivity 0.38 to 2.0 between steps 1000 and 16000, with nine more prefix-matching heads); → e380 → KILLED in logit space (see H176).

## Phase 3: readers, readouts and toys (session 33)

- H170 WDD, read from the state and weighted by a reader's input map, recovers known circuit edges without intervention. → e375 → SURVIVES (previous-token head at rank 1 in 14 of 20 induction readers and rank 2 in 6, five models; S-inhibition head first for two of three GPT-2 name movers; plain WDD worse), with shares underestimated 5-30x and near-twin credit leakage.
- H171 Reader-weighted WDD is a reliable general edge finder. → e381 → NARROWED (strong edges found at 0.14-0.69 top-1 in four models against chance 0.005-0.034, 0.02 in OLMo).
- H172 The removal knee is a degenerate (higher-order) direction of the loss. → e374 → KILLED (even-part exponent 2.0-2.1 everywhere).
- H173 The removal knee is made by the softmax readout on moving logits. → e374 → SURVIVES for most heads (linear-logit prediction 0.10-0.34 against measured 0.04-0.30), with internal sub-linearity in GPT-2 and Qwen and internal prominence crossings matching the linear prediction in OLMo.
- H174 Near-twin writes make super-additive (backup) pairs, composition makes sub-additive (series) pairs. → e376 → KILLED on the loss (write cosine rho -0.11 to +0.22; composition rho positive in 9 of 10).
- H175 The sign of a pairwise ablation interaction is a property of the circuit. → e377, e382 → KILLED (26-48% of pairs change sign between loss and logit; with additive logits the loss alone makes 97-100% of pairs super-additive; the readout/circuit split is path dependent).
- H176 Induction circuits add redundancy after they work (phase 2's H169). → e380 → KILLED as stated under zero-ablation (joint over sum on the logit 0.49-0.93 at every Pythia checkpoint; the loss's 0.69 to 3.61 growth is readout convexity); → e384 → under mean-ablation Pythia's final circuit keeps 1.72 on the centred logit, so some redundancy is real in Pythia, several-fold smaller than the loss shows.
- H177 Redundancy is manufactured by training noise. → e379 → SURVIVES in toys (head dropout 0.2: joint over sum 3.79 loss / 1.31 logit; every converged no-dropout variant 0.20-0.62 on the logit; contested by prior work, see H184).
- H178 First-order selection vanishes at convergence while use does not. → e379 → SURVIVES under manipulation (converged t -0.1 to +2.2 against use 1.3-5.1 nats; unconverged t +5 to +8; selection-use rho +0.89 to +0.94 far from an optimum).
- H179 Co-selection marks circuits under construction. → e379 → SURVIVES under manipulation (high while building, near zero once converged, high throughout in runs that never converge).
- H180 Lower-layer implementation settles before function, so lower parts are swappable late in training. → e378 → KILLED (step-63000 lower part costs the final upper part 0.28-0.42 nats, more than the whole late gain; halves co-adapt; an affine map repairs about half).

- H181 The KL interaction of two ablations measures circuit additivity. → e383 → KILLED (it equals the Fisher inner product of the single logit effects, per-token correlation 0.73-0.98); e264's additivity survives in logit space (0.019-0.042).
- H182 Induction circuits are redundant (joint over sum above 1) on a linear readout. → e384 → NARROWED (centred logit 0.79-0.94 in GPT-2, SmolLM2, Qwen, OLMo; 1.72 in Pythia under mean-ablation, 0.91 under zero-ablation; the loss inflates the ratio 1.4-5.3x).
- H183 Readers draw preferentially on writes that WDD can identify in the state. → e385 → SURVIVES in aggregate (1.33-1.46x beyond write size in all five models, in every size quintile), not as a per-edge predictor (AUC 0.51-0.56).
- H184 Self-repair requires training noise. → e386 → SURVIVES in toys only (annealed no-dropout runs repair 0-19% or less; head dropout 32-49%; constant learning rate 26%); contested by prior work in real models trained without dropout (McGrath et al. 2023; Rushing and Nanda 2024) and consistent with Pythia's logit-space redundancy (e384).
- H185 Reader-weighted provenance is new. → literature check → KILLED for the exact hook-based split (standard; Franco and Crovella 2025), NARROWED to WDD's state-only approximation.

- H186 Readers' preference for WDD-identifiable writes is learned computational organisation. → e387 → NARROWED (raw preference present at initialisation, 1.71; after removing alignment with the state 1.05-1.27, and within alignment quintiles about 1 in SmolLM2, Qwen and Pythia; genuine only in GPT-2 and OLMo).
- H187 Reading is sparse (few writers per reader input). → e387 → NARROWED (half of an input from 8-14 components, 80% from 25-55; concentrated before the induction transition, 7 components at Pythia step 512, and spreading after).
- H188 WDD's k-sparse reconstruction preserves the model's computation (the SAE test). → e388 → SURVIVES (loss recovered 0.84-0.99 at k = 64 and 0.31-0.88 at k = 8 across five models and three depths, far above a rotated dictionary and PCA at small k).
- H189 WDD's functional advantage comes from the embeddings. → e389 → KILLED (MLP write rows alone match the full dictionary; embeddings alone 0.03-0.37, head bases -0.02 to 0.21).
- H190 WDD's functional advantage is provenance, not generic trained geometry. → e390 → SURVIVES (Pythia: rows from checkpoints with cosine 0.93 to the final rows match the model's own; rows at cosine 0.35 or less, trained or not, do no better than random).

## The vision chain and self-description (sessions 36 and 37)

- H191 WDD's sparse code is a sparse subset of the actual writes (the few largest writers). → e391, e393 → KILLED (the largest actual writes recover 0.53-0.79 at k = 64 against WDD's 0.91-0.99, and 0.75-0.92 at k = 1024; 20-37% of WDD's MLP atoms are among the true top-k neuron writes at k = 8, 11-27% at k = 64; the code is a re-description over the writer vocabulary).
- H192 WDD beats the largest actual writes only through its coefficients (the refit absorbs the tail). → e393 → NARROWED (refit raises 0.53-0.79 to 0.68-0.89 at k = 64; WDD's own support stays above: directions and coefficients both matter).
- H193 The model's own words give a training-free replacement model at every layer. → e392 → KILLED at k up to 64 (excess 2.05-5.32 nats), NARROWED at 128 (+0.43 GPT-2, +0.53 SmolLM2, +1.28 Qwen, +3.26 Pythia, +3.77 OLMo; rotated +1.82 to +6.07).
- H194 WDD atoms are sparse causal nodes with linear attributions. → e394 → KILLED (effects spread over 13-16 of 32 atoms; gradient-times-coefficient against ablation Spearman 0.33-0.67).
- H195 Self-describability (the own vocabulary beating its own rotation) is a property of any weight geometry. → e395 → KILLED (identical FVU at every k at step 0; the functional advantage is present by step 256 and grows through the induction transition).
- H196 Self-description length grows as the computation gets richer. → e395 → NARROWED (32 to 64 own words, while a rotated vocabulary goes from 32 to 128 and the largest actual writes fall short of 90% at 4096: the description stays short while the writing grows dense).
- H197 Each training stage speaks a private language. → e396 → KILLED for the late stages (from step 33000 the vocabularies are interchangeable within 0.03), NARROWED to an asymmetry (late words read early states; early words read late states worse) and one stage of false friends (step 4000).
- H198 The step-4000 anomaly is an impoverished geometry. → e397 → KILLED (effective rank 848, ordinary; its rotation behaves like every rotation; condition number 1.6; float64 identical). The false-friends reading SURVIVES at small k (below the mean state at 4-8 words while capturing three times the variance of random words; helpful again at 32-64).
- H199 Individuals trained on the same data share an internal language up to the rotation fixed by their shared tokens. → e398 → KILLED (lexicon-translated words at the random level; the two embedding matrices are not rotations of each other, R2 -0.10).
- H200 Individuals share their words up to a state-fitted map. → e398, e398b → NARROWED (a third to a half of the own-word advantage crosses: +0.11 over random words through the orthogonal map, +0.07 through the linear map).
- H201 What two individuals share linearly is where the function is. → e398b → SURVIVES at k = 16 (random words confined to the linearly shared subspace describe the state as well as the network's own words, 0.59 and 0.59; the translated state recovers 0.67-0.82 of the loss at state R2 0.28-0.38). Consistent with SVCCA (Raghu et al. 2017).

## The review's controls (session 38)

- H202 The own vocabulary's advantage over its rotation is second-order alignment with the states (second moment, span). → e399, e405, e400 → SURVIVES early and KILLED late.
  - Early: up to step 8000, covariance- and span-matched words carry 41-88% and 45-134% of the gap at k = 16, and under a functional pursuit only 0.04 of the gap remains at step 1000.
  - Late: from step 16000 both controls are at the rotation level, and the gap under the functional pursuit is 0.24 at the end.
- H203 The own words are the best sparse vocabulary for the states. → e399, e400, e403 → KILLED. Random words drawn from the states' covariance beat them at k = 16 at every checkpoint (by 0.11-0.14 Euclidean, 0.02-0.06 under the functional pursuit), and a learned SAE beats them at k = 4-32 in GPT-2. NARROWED to: the own words need no activations, overtake the SAE at k = 64, and keep more function per variance explained.
- H204 The advantage belongs to a coordinate system shared by writers and readers. → e399 → NARROWED. Downstream readers' input directions beat their rotation by 0.09 / 0.16 / 0.20 at k = 16 (steps 1000 / 16000 / end), against 0.15 / 0.21 / 0.27 for the writers; the writing blocks' own input directions do not beat theirs.
- H205 A function-aware pursuit shortens the self-description. → e400 → KILLED for the 90% length (32, 64, 64 unchanged), SURVIVES at small k (the own words at k = 4 rise from 0.24 to 0.41 at the end).
- H206 False friends come from Euclidean variance-chasing. → e400 → KILLED. They deepen under the Fisher metric: 0.27 against 0.54 at k = 16.
- H207 False friends are mediated by the target's few huge directions. → e404 → SURVIVES. With the top-8 principal subspace handled exactly, the step-4000 words beat their rotation (0.65 against 0.56 at k = 16) and nothing is below zero.
- H208 Self-describability is absent at initialisation in every architecture and learned in more than one family. → e401 → SURVIVES. At initialisation own and rotated FVU agree within 0.004-0.014 in five architectures at three depths; trained OLMo's own words beat the rotations at every checkpoint.
- H209 Misleading words come from a band of intermediate stages. → e402, e406 → SURVIVES in Pythia only. Sources 4000 and 8000 mislead on targets from step 33000, and sources 1000 and earlier are generic, not misleading. OLMo (steps 1000-1454000) has no false-friend cell, even where its states have huge directions.
- H210 Accretion: later vocabularies read earlier states at least as well as the earlier model's own words. → e402, e406 → SURVIVES in Pythia (step-256 states 0.97 with the final words against 0.87 with their own). NARROWED in OLMo: it holds through step 256000, but the final checkpoint's words read early states worse than their own (0.61 against 0.71).

## Round 2 and fresh angles (session 39)

- H211 Read alignment is word-level like write alignment. → e407 → KILLED. The readers' advantage stays second-order at every checkpoint (at the end, their second-moment and mixture controls beat the reader rows); only the writers turn from an accent into a vocabulary.
- H212 The weights alone know which directions of the state matter. → e408, e408b → SURVIVES for the huge directions: the readers and the unembedding put 0.9-6.3% of their trace on directions holding 52-86% of the variance. NARROWED as a pursuit metric: it recovers 0.83 of the Fisher lift at k = 4 in GPT-2 and 0.74 in Pythia at step 16000, but 0.43 at Pythia's end.
- H213 A weight-only selection of words approaches the SAE. → e408b → KILLED for the criteria tried (read strength, write norm): both pick the token embeddings and do worse than a random subset.
- H214 False friends are a Pythia-family property. → e409 → KILLED (none at 70m or 160m); 410m-only among the models tried.
- H215 The accent-to-vocabulary transition holds across sizes. → e409 → SURVIVES (70m, 160m, 410m).
- H216 False friends are precursors of the huge-direction writers. → e411 → SURVIVES. Aligned MLP rows first appear at step 4000 (top-100 mean share 0.14), grow at 8000 (0.33) and are formed by 16000 (0.56); the false-friend profile follows.
- H217 The false-friend window coincides with the cross-seed divergence phase. → literature timing → KILLED (divergence 256-2000; the window 4000-8000 is in early reconvergence).
- H218 Errors speak the readers' language and states the writers'. → e412 → SURVIVES from step 16000; at step 1000 the errors speak the writers' language; at initialisation neither.
- H219 The native vocabulary covers in-context computation. → e413 → SURVIVES (in-context copying on unseen sequences is kept at least as well as natural-text prediction, at steps 1000 and 143000).
- H220 Native-word usage is Zipf-like and more concentrated than rotated-word usage. → e414 → SURVIVES (slopes -0.62 to -0.72 against -0.32 to -0.35, in Pythia and GPT-2).
- H221 The most used native words are the writers of the huge directions, function words of the native language. → e414 → SURVIVES in part: they point into the top principal directions (29-47% of their norm) and cost 0.03 (Pythia) to 0.10 (GPT-2) when dropped.
- H222 Accretion is directional in both families. → e410 → SURVIVES (80% of pairs in Pythia, 100% in OLMo, null-relative), distinct from the superset property, which OLMo's final vocabulary lacks.

## The workspace agenda (session 40, WorkspaceBench-inspired, Qwen2.5-7B)

- H223 The native-word lens surfaces a two-hop bridge at middle depth where lenses cannot. → e415, e420 → KILLED at the final position (nothing reads the bridge there before block 22; late, the plain lens reads it at least as well). SURVIVES at the subject's last token (bridge in the top 20 for 0.11-0.21 of prompts at blocks 12-20 against 0.00-0.01 for the lenses; country first 0.50-0.63 against 0.41-0.44).
- H224 The native word that carries an intermediate is causally used. → e415, e420 → SURVIVES at the final position (removing it costs the answer 0.15-0.16 nats against 0.01-0.02). KILLED at the subject position (no effect; the information is redundant across positions).
- H225 Native words hold several workspace variables in separate words. → e415 → SURVIVES late (bridge and answer in distinct words in 0.25-0.49 of prompts at blocks 24-26).
- H226 The native lens resolves roles earlier than lenses. → e417 → KILLED (all readers at chance through block 20). A narrower claim SURVIVES: at the resolving blocks the other name sits on suppressing words and the answer name almost never does.
- H227 The native lens reads arithmetic intermediates and fabricates less. → e416 → UNTESTED in practice (the model solves 52 of 109 chains; noise). Fabrication is higher than the plain lens at most blocks.
- H228 Self-describability survives scale (7B). → e418 → SURVIVES (own 0.79 against rotation 0.26 at k = 16; 0.94 at k = 64).
- H229 The huge directions are functionally light. → e418, e419 → KILLED as a general statement. Locally light (0.3% of the Fisher trace at 7B, holding 97% of the variance), but kept exact alone they recover 29% of the loss. The local Fisher metric fails as a selection metric at 7B.

## Established lenses (session 41)

- H230 The native-word lens is the value-vector (sub-update) reading. → e421 → KILLED. The 16 largest actual writes read the two-hop bridge at the subject token worse than even the plain lens (country first 0.15 against 0.44); the native re-description reads it best (0.63).
- H231 The super weights are the native language's most used words. → e422 → KILLED in Qwen2.5-7B. The super neuron's write is a huge-direction word (0.92 in M) but is never used at positions after the first; the most used words are ordinary MLP rows.

## What WDD unlocks, tested on five models (session 42)

- H232 Native words read hidden content better than established readers, in general. → e421b, e423, e424, e425 → KILLED as a general claim. It SURVIVES for recalled entities in the larger gated models (OLMo-1B: top 20 0.51 against 0.21 lens and 0.40 actual writes; Qwen2.5-7B e421b); it is reversed in Pythia (actual writes 0.76 against 0.24); and no vocabulary-space reader surfaces context features (previous tokens) in any model.
- H233 At the end of training self-description is word-level in every architecture. → e426 → SURVIVES (second-order shares below one half in all five; GPT-2 weakest).
- H234 The directions holding the variance are not the ones readers weigh. → e426 → SURVIVES in all five (readers' trace at 1-7 times chance on directions holding 24-87% of the variance).
- H235 States speak the writers' language and errors the readers'. → e426 → NARROWED (states: all five; errors: three of five; GPT-2 reversed, Qwen-0.5B tied).
- H236 Native-word usage is Zipf-like and steeper than a rotated vocabulary's. → e426 → SURVIVES in all five.

## Self-description as an instrument, extended (session 43)

- H237 Self-description tracks capability. → e427 → SURVIVES across Pythia sizes (advantage 0.17 to 0.40 as loss falls from 4.09 to 2.83, rank correlation -0.8). NARROWED within runs: the largest models have most of it at step 1000.
- H238 Self-description usage is a free importance score. → e428, e428b → SURVIVES in SmolLM2 and Qwen-0.5B (above Taylor, spread, deviation and magnitude), split in Pythia. KILLED in GPT-2 (spread and weight norm better) and OLMo (weight norm finds the super-weight neurons; usage worst).
- H239 States speak the writers' language and errors the readers'. → e429 → SURVIVES in all five models at all three depths once attention readers are included (revises H235).
- H240 Self-description is a novelty or uncertainty signal. → e430 → KILLED (random tokens lowest in four models but highest in OLMo; shuffled text as describable as natural text; no negative correlation with loss).
- H241 Instruction fine-tuning rewrites the native vocabulary. → e431 → KILLED (row cosine 0.997; base and instruct words describe each other's states equally well); chat-specific drift in one of two models.

## The huge directions and the native vocabulary under new controls (session 44)

- H242 The huge directions' variance share is a sink artefact (the survey's top pick). → e432, e437 → SURVIVES for Pythia (one "\n\n\n" per sequence holds 0.83 of the variance; M is 0.17 at typical positions, not 0.86) and OLMo (0.48 → 0.10), and for Qwen2.5-7B (0.97 was one " series" token at 221 times the median norm in the fitting text; 0.11 without it, e437b). NARROWED: the subspace is the same with or without sinks (overlap 0.87-0.91). GPT-2, SmolLM2 and Qwen-0.5B have no sinks after position 0.
- H243 False friends are mediated by M (e404). → e437 → KILLED as stated. They were mis-described sinks: with the sinks exact, the step-4000 words are above their rotation at k = 4 (+0.15 against +0.08).
- H244 M is a set point, normalisation ballast, or a temperature knob. → e432 → KILLED, all three (odd part 16-29 times the even part; the norm-only change costs nothing; output norm within 5%).
- H245 M's non-local function is a property of the move's size, not of its directions. → e432 → KILLED. A random displacement of the same energy is quadratic and costs 4-12 times less; M's removal is 2-4 times its quadratic prediction, and the knee grades with variance rank.
- H246 The knee is the attention softmax. → e438 → KILLED. Freezing patterns removes 28-64% of the cost but not the knee.
- H247 The knee is MLP gating. → e441 → NARROWED. In GELU models the linear network is quadratic; in gated models the MLPs absorb large moves (linearising them costs 1.4-3.5 times more). No single seat.
- H248 M is the identity channel. → e433 → SURVIVES in part. It is the most lexical subspace (token 0.31-0.59 against 0.13-0.37 for the whole state; plus position in GPT-2) and the most predictable from block 0 (0.65-0.91). The token part carries 2.4-6.6 times more cost per unit energy. KILLED: the knee is not lexical (the within-token part has it too).
- H249 The network keeps what is said in its own words. → e434 → KILLED. Retention follows variance share; usage adds 0.03-0.18 partial correlation.
- H250 The self-description advantage is alignment with the states' covariance. → e435 → KILLED. Gaussian states with the same covariance keep 15-45% of it. SPAWNED: Zipf-like usage is geometry (reproduced by the Gaussian states).
- H251 The native vocabulary is the lexicon (token rows, position rows, block-0 MLP rows). → e439 → KILLED at every checkpoint and in all five; descriptions almost never name the current token.
- H252 What context adds to a token is described only to second order. → e440 → KILLED. Non-lexicon own words beat rotation by +0.16 to +0.52 on the context part, at the word level (covA share -0.23 to 0.32).
- H253 The word-level vocabulary is a union of per-block accents. → e443 → KILLED at the end of training (per-block Gaussian words carry -0.14 to 0.33). SURVIVES early: 0.82 at step 1000, 0.70 at 4000, 0.32 at 16000, 0.09 at the end. REVISES H(accent → vocabulary): the early accent is per block.
- H254 Increments are word-level before states are (e02 × e405). → e436 → SURVIVES at step 4000 in Pythia (the middle block's increment is word-level, shares 0.13/0.25, while the state is an accent, 0.53/0.66); KILLED at step 1000 (both accents). At the end the increment is less word-level than the state in all five. SPAWNED: the actual writes are sparsest mid-training (participation ratio 374 → 216 → 322).
- H255 The self-description advantage lives on the broadcast path. → e442 → NARROWED. Both paths: self +0.14 to +0.52, broadcast +0.17 to +0.41; larger on broadcast in three of five.
- H256 The replacement model compounds through the sinks. → e392b → KILLED in Pythia (excess at k 128 +3.53 with every position described, +3.51 with the sinks exact; GPT-2 has none, +0.49 either way). OLMo +4.11 either way (no sinks in its two sequences). The compounding is real, not a sink effect.

## The vision round: what a decade of WDD would need (session 45)

- H257 Native words carry meaning, not only function. → e446, e448, e448b-e448e → SURVIVES for concrete concepts. Qwen2.5-0.5B has a language-independent concept word (one MLP write row) for 15 of 24 nouns, Qwen2.5-7B for 22 of 24 (twelve in block 7), SmolLM2 for 7 (rotated 0 in all three). Sentence bags of native words retrieve translations at 0.55-1.00 top-1, with 2-4 times the continuous state's separation. Removing a concept word costs more than a matched other word in 4 of 4 languages (SmolLM2) and 3 of 4 (Qwen). NARROWED: token-level proxies cannot show meaning (rotated words score highest on them in 4 of 5 models).
- H258 The concept words are just multilingual neurons. → e448c → SURVIVES in Qwen-0.5B: the neuron fires (median 100th percentile) and supplies a median 54% of the component. NARROWED in SmolLM2: the neuron fires (98th) but supplies 10%, and other components write the rest. KILLED: 'invisible to activation analysis'.
- H259 Two models' vocabularies correspond word for word. → e445 → KILLED as a basis for translation (0.06-0.13 of function against 0.85-0.91 for a dense map). Partner words above chance (7-14% against 1-2%).
- H260 Self-description announces generalisation. → e444 → NARROWED. Generalising networks become sparse in own words; memorisers only second-order. The word-level part leads test accuracy by 500-750 steps in the two standard runs, but not in the frozen variants.
- H261 The vocabulary is spoken (activations organise around whatever writers exist). → e444, e444b, e449, e449b → KILLED. Writer rows frozen at random are barely used as words (0.69-0.75 against 0.74 rotated); trained rows are (0.08-0.25 against 0.89-0.93 in grokking); the description moves to trained writers.
- H262 Self-describability can be trained in cheaply. → e447, e447b → NARROWED. Euclidean self-describability can be trained in (fraction unexplained at k 16 down 9% for +0.008 nats, 41% for +0.28, 80% for +0.91). Functional self-describability does not follow (loss recovered by 16 own words 0.75 -> 0.67-0.73). The words do not move, only the states. SPAWNED: train on the functional objective.

## Causal tests of concept words and a re-implemented block (session 46)

- H263 Concept words are correlated provenance, not causal handles for their concept. → e451, e455 → KILLED for translation and, in the Qwen models, for the category.
  - At one depth, removal is 3-6 times as specific as removing a matched other word, and a swap raises the target 3-13 times as much as random, but answers change only in SmolLM2.
  - Swapped at every block up to the middle, one MLP row redirects 67-94% of translations (random 0-3%) and 39-78% of category answers in the Qwen models.
  - REVISES H257: native words carry meaning causally.
  - NARROWED by e458 (session 47). e455's swaps were 6-8 times the natural size. Near natural size (0.7 times) one word moves 35-58% of translations, still far above random.
- H264 One depth suffices to intervene on a concept. → e451, e454 against e455 → KILLED. The concept is re-written or read before the middle layer. This is e208's re-writing, now for a semantic direction. CORRECTED by e469: the explanation is not propagation. Whole-state patching at the last noun token at that one layer switches 78-99%; a single word is outvoted by the other directions carrying the noun there.
- H265 The concept word is the noun's only carrier. → e455 → KILLED in the Qwen models, where removal at every depth leaves accuracy at 0.88-0.90. SURVIVES in SmolLM2, where accuracy falls to 0.13.
- H266 A block's function determines its vocabulary. → e452 → KILLED. Re-implementations match the function within 0.02 nats with new rows: median best |cos| 0.26-0.28 with the originals, and 0.22-0.25 between two seeds.
- H267 Words are written by training the writers even when the function is held fixed. → e452 → SURVIVES.
  - Frozen random writer rows reproduce the function but are not words (advantage 0.000).
  - Trained writer rows are words, at 54-61% of the original's advantage.
  - Frozen random readers leave weak words (10-18%).
- H268 The re-implementation's half-sized advantage is the missing end-to-end objective. → e452b → KILLED. Next-token training of the block leaves 0.52-0.70 of the original's advantage, while more distillation gives 0.71-0.77. The gap is fit quality, and the rows drift toward the originals (median best |cos| 0.26-0.28 to 0.32) as the fit improves.
- H269 Concept words are the words two model sizes share. → e453 → NARROWED to the concept level. Identification among 14 shared nouns is 0.79 (reverse 0.86, chance 0.07), but the mapped words are nearly orthogonal (|cos| 0.08) and are the nearest atom for only 1 of 14.

## WDD as a forensic instrument (session 47)

- H270 Self-description detects compression damage. → e456 → KILLED. At 4 bits GPT-2 loses 3.9 nats and keeps 96% of the advantage over rotation. Own words stay at 0.42-0.56 unexplained against 0.64-0.72 for rotation at every level. The measure tracks that states are built from the model's rows, not whether the function survives.
- H271 A fine-tune's change is sparse in the model's own words. → e457 → NARROWED.
  - Usage changes are concentrated in a few named words (top 1% of words: 38-59% of the change on chat).
  - The change itself is low-rank and dense on chat: its own principal directions give 0.27-0.32 unexplained against native 0.57-0.65.
  - Native words do beat those directions on natural text (0.58-0.66 against 0.70-0.77).
- H272 A single concept word is a weaker steering handle than the dense difference of means. → e458 → MIXED. At natural size the dense steer moves more answers (82-100% against 35-58% at 0.7 times natural), but with 4-6 times the displacement. Per unit of displacement native is equal (Qwen-0.5B) or better (Qwen-7B).
- H273 WDD can certify an intervention before its effect is observed. → e458 → NARROWED. The checksum (the target word's component two blocks later) predicts success at AUC 0.70-0.92, but the displacement size alone predicts as well (0.83-0.94).
- H274 Native words are persistent objects across autoregressive positions. → e459 → NARROWED. They persist 1.3-1.8 times as much as rotated words at lags of 4-32 in Qwen and in GPT-2's natural text. They are not regenerated beyond their usage rate.

## Authorship labels as forensics (session 48)

- H275 Provenance predicts a state's future beyond its vector (the residual stream is not "Markov"). → e460 → KILLED. Near-identical states differ in their futures mainly through context (76-85%), and the ledger distance adds nothing to the cosine (partial Spearman -0.07 and -0.10).
- H276 WDD detects and locates forged write-sized contributions. → e461 → KILLED at the scale of one write. At the forged block the AUC is 0.60-0.64 (Mahalanobis 0.62-0.75). The correct block is found in 0.14-0.35 of cases, and detection is at chance a few blocks later.

## The learning signal in native coordinates (session 49)

- H277 A gradient update reinforces or shrinks a word along its own direction. → e462 → KILLED per batch. The along-row share is 0.5-0.8 times chance and the sign is a coin flip. SURVIVES for accumulated drift late in training, where half of the change is along the row (shrinkage).
- H278 One batch's gradient predicts how a word changes between checkpoints. → e462 → KILLED (cosine 0.000-0.001 at every stage).
- H279 The most used native words receive the most learning pressure. → e462 → KILLED (Spearman of usage with gradient norm -0.63 to -0.04). Forward use and backward pressure are different writer properties.

## A grammar over native words? (session 50)

- H280 Native words carry grammatical role by inflection (coefficient modulation of shared words). → e463 → KILLED. Role is carried by which words are used (0.94-0.97 decoding), not by shared words' coefficients (0.61, no sign flips). Rotated words decode role as well (0.88-0.89).
- H281 The network corrects native words written in contexts where they never occur. → e464 → KILLED. Survival, energy and re-description are the same in legal and illegal contexts: generic contraction, no contextual syntax.

## Self-consistency of native descriptions (session 51)

- H282 The network regrows what a native description leaves out (descriptions are dynamically self-sufficient). → e465 → KILLED. The omission persists at a constant share of the state (0.67-0.69 at +4) and grows 1.3-2 times in size. A random error of the same size is damped (0.49) and nearly harmless (0.92-0.96 of the loss kept). The omitted part is functional content.

## Systems-theory and causal-abstraction readings (session 52)

- H283 A few extra directions make a native description dynamically sufficient (a small Kalman gap). → e466 → KILLED. With 64 extra remainder directions 7-11% of the loss is still missing; the omitted content is spread out.
- H284 A description's remainder holds a different kind of information (unverbalised computation). → e467 → KILLED. Token identity, the neighbouring tokens and position are readable from both halves.
- H285 A native description keeps what later tokens read from a position better than generic codes. → e468 → MIXED. It does in Qwen (0.28 of the mean-state damage against 0.54-0.66); in GPT-2 it ties the principal components (0.10 against 0.09). Later tokens depend little on one middle-depth state in any case.
- H286 Native words are interchange coordinates for a high-level variable. → e469 → SUPPORTED across blocks. One to four native words per block carry the translated noun (Qwen 0.67-0.96, SmolLM2 0.22-0.42) far better than rotated words (0.00-0.36) or the task's principal directions (0.04-0.66) at equal k. NARROWED at a single block, where the principal directions win. REVISES H264's explanation.

## Where native words carry a variable (session 53)

- H287 Native words are causal handles only during the band of blocks in which a variable is written. → e470 → SUPPORTED. In Qwen, native 4 over blocks 0-4 switches 85% of answers, from block 8 onwards only 15%. SmolLM2's band is about blocks 6-12. Whole-state patching works at single blocks from early on.
- H288 The carrier words are persistent identities across blocks. → e470 → NARROWED. Consecutive-block overlap is 0.41-0.47, and a third to a half of the carriers are written by the last two blocks: partly persistent, partly re-encoded.
- H289 Native words beat task PCA because PCA is local to a block. → e470 → NARROWED. A basis shared across blocks is as good as or better than per-block bases at 16 dimensions (Qwen 0.95 against 0.89, SmolLM2 0.50 against 0.38). Native words lead at small k in both models.

## The life cycle of native words (session 54)

- H290 A neuron becomes causally important before it becomes a native word (a developmental ordering). → e471 → KILLED. The lagged correlations are near zero in both directions (-0.01, +0.06), and usage and importance do not co-move (+0.04).
- H291 When a native word falls out of use, its function dies with it. → e471 → KILLED. The importance change of dying words is about 0, like that of other neurons.
- H292 The vocabulary in use grows over training. → e471 → KILLED after step 2000. The distinct words used fall from 14,115 to 9,435.

## WDD applied to itself, word algebra, roles and induced geometry (session 55)

- H293 Repeatedly applying WDD's description reveals a native attractor, drift or collapse (the recursive fixed point). → e472 → KILLED. One application is already a fixed point for 96-100% of states, for native and rotated words alike; loss and energy do not change after the first step. The operator is a projection.
- H294 Perturbed descriptions flow back to a common native description. → e472 → KILLED. Each noisy trajectory freezes at its first description; two noise seeds share Jaccard 0.62-0.65 of their final words. NARROWED to stability: 5% noise keeps more native words than rotated ones (0.67-0.71 against 0.46-0.48).
- H295 Native handles for two attributes of one word act independently and compose (word algebra). → e473 → SUPPORTED in Qwen. With four words per block the gender handle alone changes only gender in 92%, the generation handle only generation in 75%; gender and generation taken from different words give the doubly changed word in 74% (83% with sixteen); behaviour is additive within 24%. Rotated words fail at four (6%, 25%, 16%). The task's principal directions do as well (91%, 80%, 91%). SmolLM2 underpowered (20 items).
- H296 Identity handles are typed by grammatical role. → e474 → KILLED. A handle found in the other role switches as many answers as the item's own (Qwen 0.18 against 0.17 for subjects, 0.32 against 0.31 for objects at four words; alike at sixteen and in SmolLM2), although the carrier words shared between roles fall to 0.61-0.66 at the middle block.
- H297 Native words are the better interchange coordinates for any variable. → e474 → KILLED for an entity copied into the answer: rotated words do as well or better (0.84-0.97 against 0.72-0.96 at sixteen words). NARROWS H286 to some variables (a translated concept, gender and generation).
- H298 WDD induces a geometry of states that predicts behaviour beyond activation distance. → e475 → NARROWED. Activation distance predicts next-token similarity best (Spearman +0.13 to +0.37, and the closest nearest neighbours); native coefficients or word sets add +0.04 to +0.09 beyond it, rotated words +0.03 to +0.07.
- H299 The identity-free part of a description (its coefficient profile) carries behavioural information. → e475 → SUPPORTED at the deepest depth only, and only for native words: +0.13 to +0.14 beyond activation distance and the entropy gap in all three models (rotated -0.02 to +0.04). It is not prediction confidence. OPEN: what it encodes.

## Audits that arose from the digest (session 56)

- H300 The native-only profile signal of e475 is an artefact of the coefficient normalisation or of k. → e476 → KILLED. L1 normalisation gives the same partial (+0.15), k = 8 and 32 give +0.10 to +0.16, the effective number of words +0.13; rotated profiles carry none (-0.01 to +0.04).
- H301 The profile signal is a proxy for a plain variable: norm, position, entropy, top probability, token frequency, copying, or the identity of the current token or top word. → e476 → KILLED for all of these (each removes at most 0.03). NARROWED: a third to a half of it is the description's fit and the top write's prominence (given all candidates +0.07 to +0.11 remains; per position the native top share tracks prominence at +0.56 to +0.86); the top word is an MLP row in 86-98% of positions. In Qwen the PCA-16 profile carries as much. OPEN: the remainder.
- H302 The attribute handles of e473 are item-specific (the vocabulary has no shared gender word). → e477 → KILLED for gender. Four native words from the other kinship words, in the other languages, flip only gender in 98% of items (own words 92%; at sixteen words 98% against 73% with leakage); one MLP row sits in 91% of items' own gender supports at block 4; rotated words fitted to the same shared direction give 7% at four words (81% at sixteen).
- H303 The generation handle is shared to the same degree. → e477 → NARROWED. 48% at four words against the own words' 75%; 69% at sixteen (own 70%). Generation is carried by more, and more word-specific, words.

## WDD as a workspace reader (session 57)

- H304 A native readout's own numbers (coefficient, provenance, a causal receipt) separate right claims from wrong ones better than the lens's confidence. → e478, e478b → NOT TESTABLE on two-hop bridges (73 of 73 native claims right at the subject token, 110 of 111 at the end); KILLED on chained arithmetic (every certificate at AUC 0.43-0.59; precision lens 0.63, native 0.48 against chance 0.50).
- H305 What the native reader surfaces at a writing position is an echo of the prompt token. → e479 → KILLED. Every surfaced concept rides on an MLP row (12 of 12), none on the concept's token embedding; native 0.30 against lens 0.10 under "think", 0.00 for all under "do not think".
- H306 The native reader loses nothing the lens captures (the benchmark's first desideratum). → e480 → KILLED for per-word pooling (0.40-0.50 against 1.00 on basic, multilingual and poetry); SUPPORTED for the reconstruction read as a whole (0.85-1.00) and the union (0.80-0.90).
- H307 A single native word can carry a multi-token concept as a unit. → e481 → SUPPORTED at the final position: both tokens of a two-token country in one word's top 10 in 0.31 of items (all MLP rows), the second token in 0.42 against the lens's 0.04, disambiguation 24 of 25. Not at the subject token (second token 0.08).
- H308 The signs of native words carry the direction of a described action without a question. → e482 → KILLED. At the period, mid blocks, every fixed rule is at 0.50-0.56; at the object token every reader reads the order from the current token; the lens reads it late by recency (0.88 at block 26).
- H309 The planned rhyme of a couplet is readable at the end of the first line. → e480 → KILLED at 7B for every reader (0.00).

## Calibration against classical theory (session 58)

- H310 WDD's detection threshold is an extreme-value statistic: the competitor maximum is Gumbel, scales as sqrt(2 ln m), and predicts the detection curve. → e483 → SUPPORTED in five: KS 0.02-0.07 (normal 0.06-0.11), scaling R^2 0.97-0.997, the fitted Gumbel's CDF within 0.1 of the measured detection at every injected size (within 0.11 in two), resolution 0.25-0.4 of the state's norm.
- H311 The scale of that law is set by the atoms' second moment. → e483 → KILLED for the native dictionary (slope 2.9-6.2 times the prediction), SUPPORTED for the rotated one (0.96-1.02): the excess is provenance.
- H312 The provenance factor is above 2 at every depth and falls with depth. → e492 → SUPPORTED: 6.2-13.7 at block 1 falling to 3.5-5.1 at the last blocks in GPT-2, Qwen, OLMo and SmolLM2; Pythia dips to 2.9 at the middle and rises again late. At step 1000 of Pythia it is already 2.2-3.0 (the accent of e443).
- H313 A native description is a redundant code. → e484 → KILLED: one or two erased words already cost a tenth of the intact value, and a random half refitted reaches only 0.59-0.80 of the chosen 8. NARROWED to correction: refitting native words recovers 0.02-0.09, refitting rotated words costs.
- H314 The correction comes from more coherent supports. → e490 → KILLED: native and rotated supports are equally near-orthogonal (mean |cos| 0.03-0.05). Within native supports the gain does follow coherence (Spearman +0.46 to +0.67).
- H315 The first native word carries more bits about the next token than the first rotated word. → e485 → SUPPORTED in five: 0.4-1.2 against 0.2-0.3 bits; below the current token's 0.9-1.4; and 1.6-4.4 bits about the current token (lexical).
- H316 The Heaps exponent of the vocabulary in use is a provenance signature. → e486 → KILLED: native 0.60-0.76, rotated 0.55-0.78, no consistent order. NARROWED: provenance concentrates usage (half of selections on 2-5% of the words used; rotated 16-23%).
- H317 The native word stream has sequential structure of its own. → e487 → SUPPORTED: the previous word predicts 0.2-0.9 bits of the next (shuffled about 0), about half of the text's own; the rotated stream has almost no entropy (1.0-3.5 bits). Most of the structure is lexical (the current token fixes 1.5-4.3 bits).
- H318 Native rows are the non-Gaussian directions of the state cloud (projection pursuit). → e488 → SUPPORTED in five: kurtosis 1.3-4.5 times rotated at the median, 2-10 times at the 90th percentile; usage follows it (+0.25 to +0.46).
- H320 Provenance loss has two timescales, a fast loss of the direct trace and a slow loss of the re-written part. → e491 → KILLED. A single exponential fits at R^2 0.983-0.996 (half-life 0.9-2.6 blocks, retention 0.71-0.88 per block, the contraction's); Prony's second mode gains at most 0.01 and carries no amplitude.
- H319 An unsupervised decomposition of the states rediscovers the rows. → e489 → SUPPORTED in part: FastICA components match native rows at median cosine 0.29-0.43 against 0.10-0.17 for rotated rows and 0.16-0.26 for principal directions; 27-31% above 0.5 in three models, 2% in two; the components are more non-Gaussian than any row, so they are built from several rows.

## The provenance factor decomposed, and the SAE bridge (session 59)

- H321 The provenance factor is write sparsity: the position's largest actual writes stand out above the extreme-value level. → e493 → SUPPORTED at early and middle depth in five models: the maximum is one of the 64 largest writes or the token embedding in 87-100% of positions at block 1 (0.73-0.91 at the middle in four), and the writers alone reproduce the whole factor. Across Pythia's training the writers' part rises from 1.0-1.5 (step 1000) to 1.6-2.6.
- H322 The rest is cross-alignment, rows resembling what other rows wrote (the accent of e443). → e493 → SUPPORTED: non-writers sit 1.3-1.9 above the rotated level in the middle blocks, are the whole factor at step 1000 beyond block 1 (writers 1.0-1.3), hardly grow at the middle, and lead in Pythia's late blocks (2.5-2.9), where the huge directions live.
- H323 A learned SAE feature is one native row. → e494 → KILLED: sixteen native words leave half of a feature unexplained (rotated 0.72). NARROWED: its top word is an MLP row for 93% of features, from every block; 23% of the most used rows are features at cosine above 0.5 (rotated 0%).

## The SAE bridge across depth and in activation; the accent's clock (session 60)

- H324 The provenance layer under SAEs depends on depth. → e495 → KILLED: at five depths of GPT-2 the numbers are the same (unexplained by 16 native words 0.47-0.49 against rotated 0.71-0.73; top word an MLP row 0.76-0.93; used rows that are features 0.19-0.23), and the top row comes from any block (the block just before the SAE holds 0.12-0.30).
- H325 Frequent features are the single-row ones. → e495 → SUPPORTED: Spearman of frequency with the single-word unexplained fraction -0.25 to -0.38 at every depth.
- H326 A feature fires when its top row fires. → e496 → NARROWED: AUC 0.62 for the top row's write size (random row 0.50), strongly (above 0.8) for a sixth of features; on the full sample (v3: 2000 features, 8176 positions) the 4-16-word ledger is 0.59-0.63, no better than the top row, and the same coefficients on random rows are at chance. v2's 0.70 for the ledger was a smaller sample.
- H327 The accent and the words form together. → e492, e493 on Pythia steps 256-3000 → KILLED: the non-writers' part forms between steps 256 and 512 and then plateaus (block 12: 1.15, 1.72, 1.47, 1.36, 1.34); the writers' part is below the rotated level at step 256 (0.82-0.90), crosses 1 near step 1000 and grows to the end (block 12: 0.82 to 1.58).

## The bridge's missing control (session 61)

- H328 SAE features are sparse compositions of native writes. → e497 → NARROWED. A feature needs as many native words as a whole state (0.48 unexplained at sixteen; states 0.47); it is far sparser than a random direction (0.72, which is also the rotated dictionary's floor on anything) and sparser than a covariance-matched direction (0.63 at block 7; 0.54 against 0.49 at block 3); its top word is an MLP row at the states' rate (0.93; random directions 0.54). Features are state-like directions in the native subspace, not write-like ones.

## The anatomy of the two clocks (session 62)

- H329 The writers' part of the provenance factor is write sparsity. → e498 → SUPPORTED across 45 checkpoint-block cells of Pythia: Spearman +0.86 with the top write's prominence, +0.82 with the top-64 energy share, -0.81 with the effective number of writes; at most 0.27 with any second-order quantity.
- H330 The non-writers' part is second-order alignment of the state cloud with the atoms. → e498 → SUPPORTED: +0.61 with the covariance-alignment ratio, +0.54 with the top-8 variance share, -0.56 with the effective dimension; at most 0.27 with any sparsity measure.
- H331 The accent is a prerequisite for the words (the former predicts the latter across checkpoints). → e498 → KILLED at this resolution: the non-writers' part at one checkpoint does not predict the writers' part at the next (+0.04), nor the reverse (-0.03).

## The two clocks in OLMo, the type of a word, a second SAE family (session 62, continued)

- H332 The SAE bridge holds on an independently trained SAE family. → e499 (OpenAI TopK, 32k latents) → SUPPORTED for provenance (top word an MLP row 0.86-0.89; features 0.56 unexplained at 16 words against random 0.72 and covariance-matched 0.63) and NARROWED for state-likeness (features between covariance-matched directions and states, 0.56 against 0.47; below the covariance level at block 2). In activation stronger than the ReLU family: top row AUC 0.73, eight-word ledger 0.83, half of features above 0.8.
- H333 The words are MLP rows at every checkpoint, whichever component carries the state. → e500 → SUPPORTED: MLP rows alone recover within 0.04 of the full dictionary from step 256 to the end at three blocks; the head bases' share of the words peaks at steps 512-1000 (0.31-0.37) and falls to 0.14-0.19 while attention's share of the state's energy rises to 0.75; the provenance advantage over rotation grows from 0.05-0.08 to 0.24-0.25.
- H334 The two clocks are Pythia-specific. → e498 on OLMo → KILLED: the writers' part rises with write sparsity through 256,000 steps (block 8: 1.27 to 2.70; effective writes 1,964 to 297) and the second-order alignment is largest at the first checkpoint and decays (1.28-1.65 to 1.01-1.07).

## What predicts row-following (session 63)

- H335 The TopK family's stronger row-following is a property of its parameterisation. → e501 → KILLED: it is its features' sparsity. Within both families the AUC falls with frequency (-0.40, -0.20) and rises with activation strength and the top-row cosine; at matched frequency the families are alike at both ends (0.79/0.77, 0.53/0.51), and the family effect is -0.09 given frequency, -0.19 given all controls (raw +0.18). Whether a feature follows its writers is a property of the feature.

## The extreme-value model tested (session 64)

- H336 The ledger's advantage over the top row is incremental information from several writers. → e502 → MIXED: real but modest (+0.05 ReLU, +0.02 TopK at the median feature; 40-50% of features gain over 0.05; nothing from random rows), concentrated in sparse ReLU features (+0.10 in the sparsest decile, zero in the densest), and the rows are partly redundant (the ledger without its top row does as well as the top row). The family gap in the ledger is mostly frequency (partial -0.09) but TopK leads by 0.1-0.25 at matched frequency in the middle deciles.
- H337 The SAE bridge holds on a residual-stream SAE of another model. → e504 → NARROWED: on Pythia-160m the top word is an MLP row at the states' rate (0.75) and the top row follows the feature (0.65; random 0.50), but the features are not state-like (between random and covariance-matched directions), the ledger adds nothing, and the sum of all MLP writes carries nothing of the firing (0.46) while the non-MLP part carries it (0.94).
- H338 The non-writers' excess over the Gumbel floor is the second-order alignment of the state cloud with the atoms. → e503, e505 → KILLED: the second-order term is a factor 1.02-1.24; the non-writers' maximum stands 1.3-2.3 times above the level computed from their own second moment; the excess is a tail, atoms aligned with the position's large writes (GPT-2, OLMo) or with attention (Pythia), and it arrives between Pythia steps 256 and 512.
- H339 A writer becomes a word through its own coefficient clearing the floor. → e503, e505 → KILLED as stated: the own coefficient is 0.3-0.7 of a winning projection; the position's other large writes supply 0.3-0.7 through their cosines; the crowd of small writes subtracts; prominence alone predicts writer wins at 0.45 against 0.78 (GPT-2 block 6).
- H340 The position's chord (its largest writes through the Gram) predicts the native maximum. → e506 → NARROWED: summed as written the chord overshoots twofold where the writes are as large as the state (GPT-2, OLMo) and names the atom in 2-16% of states, but the overshoot is the common direction the largest writes share; centred, the chord predicts GPT-2's maximum within 1-13%, ranks states at 0.56-0.75 and names the atom in 32-46% (top four 54-64%); in OLMo the crowd's contraction is needed (all writes 0.78-0.86 of the level) and in Pythia the small writes add (the centred chord under-predicts, all writes within 1-28%); over training the chord reproduces the words clock, not the accent.
- H342 The centred chord predicts the whole projection profile, not only the maximum. → e507 → NARROWED: a partial predictor in GPT-2 (calibrated within 13-25% on its own top 200, R^2 0.54-0.66, the top atom in 32-46% of states, a third of the top ten, the true top hundred ordered at 0.30-0.41); the centred sum of all writes is better there (R^2 0.73-0.77) and worse in OLMo (R^2 at or below zero), where the crowd removes 0.42-0.46 of the chord; the chord's ordering of the true top hundred rises through Pythia's training.
- H343 The crowd of small MLP writes contracts the chord throughout training. → e507 → KILLED as stated: the crowd's gain along the chord at Pythia block 12 is +1.37 at step 512, +0.63 at 4000, -0.08 at 16000, -0.50 at 64000 and -0.64 at the end; it amplifies early and contracts late, while attention's gain rises from 0.08 to 0.30-0.50. Whether the flip is area 05's learned contraction maturing is open.
- H344 The chord predicts the maximum better than the profile because the winner is selected where an exchangeable residual helped. → e508 → SUPPORTED in GPT-2 (a null that keeps the chord and permutes the residual reproduces the maximum's level within 2%, the winner's rank within 100 exactly and the winner's residual excess; the chord's top atom wins 0.10-0.13 more often than the null), and refuted elsewhere: in OLMo the residual contracts the chord's extremes specifically (the chord's top atom wins 0.13-0.19 against 0.60-0.79 under the null), and in Pythia mid-training it reinforces them (two to six times the null at steps 4000-16000).
- H345 The crowd's contraction of the chord is the network's response to the chord. → e509 → ESTABLISHED where the crowd contracts: removing the chord from what the MLPs see removes a gain 1.2-1.9 times the actual (GPT-2, OLMo, Pythia blocks 12 and 18 at the end); the early amplification is not a response (share 0.03-0.21 at Pythia step 512), and the response turns negative between steps 4000 and 16000 as area 05's contraction is born.
- H346 The chord is contracted as much as any direction. → e509 → KILLED: a random direction of the chord's size elicits -0.8 to -1.7 from the small writes, the chord -0.23 to -0.85, a third to a half of it; the vocabulary is what the learned contraction spares.
- H347 The chord's own rows are excited by the chord. → e509 → SUPPORTED: their response is +0.26 to +0.45 at the end of training in three models and grows through Pythia's training from +0.04-0.19 at step 512.
- H348 The learned contraction spares native atoms. → e510 → KILLED in the linear regime: a block's directional gain is isotropic within 25% across native rows, embeddings, head bases, random, covariance-matched and principal directions at the end of training in three models (native over random 1.01-1.17, used over unused 0.86-1.19), and it is born isotropic between Pythia steps 512 and 4000. What survives is a pattern-level, chord-amplitude effect: at a perturbation of the state's size the rows most used as words are contracted less than random directions in GPT-2 (0.63-0.80) and at Pythia's block 18 (0.66), and more in OLMo (1.02-1.19), matching e508's residual structure model by model. At step 512 the MLP amplifies on-manifold directions only (+0.03 to +0.07), the early anisotropy as a response.
- H341 The Gumbel floor is the right null for provenance-free atoms at every checkpoint. → e503, e506 → ESTABLISHED: the rotated dictionary's maximum is 0.97-1.00 of the analytic prediction in every model, block and checkpoint (27 cells).

## Attention, sinks, embeddings

- H9 Attention writes are unreadable by static atoms; per-head OV value atoms recover about half of a block's attention write inside its increment. → e11 v2, e132, e143, e145, e150 (hook-free joint recovery fails), e170 → SURVIVES as stated.
- H10 The centering mean's sink contamination changes identification. → e120, e126 → KILLED (gap −0.01 to 0.03, recall unchanged; the sink atom is a first pick in 38 to 60% of typical supports, a caveat not a change).

## Zero-sum computation

- H11 A measurable fraction of MLP write energy never reaches the state (within-block and cross-block cancellation). → e141, e173 (GPT-2 within-block 0.5 to 0.7), e219 → NARROWED to GPT-2: its blocks keep 31 to 51% of their neurons' write energy; Qwen, OLMo, Pythia and SmolLM2 are constructive (coherence ratios 1.08 to 1.18, block 0 and the last block 2 to 4).

## Training dynamics

- H12 Erasure structure, the contraction, and read/write anti-alignment are learned, early, and in a fixed order. → e173 (checkpoints), e207, e209 (alignment 0 at init, −0.04 at step 1000, −0.18 at 4000), e217 → SURVIVES for "learned and early"; the order is OPEN (e217).

## Artifacts retracted (never counted)

- S1c, S1d per-token partner atoms overwrote shared rows; S1b near-orthogonal cancellers needed 8× energy (replaced by S1e, S1f, S2 v2). e54 attention-DC magnitude N-fold. e61 linear null-space measure trivially ~97%. e116 own-span energy trivially 1.0. e207 "LN vs weights" split (the post-norm gain is the through-norm gain divided by the normalization scale; e210 is the right decomposition). e275 unsigned transport residuals (superseded by e275b, which carries the coefficient sign).
