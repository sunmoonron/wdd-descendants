# Hypotheses killed, attempts that failed, artifacts retracted (2026-09-19 sprint)

Kept on purpose. Each line: what was tried, what happened, what it rules out.

## Hypotheses killed by data
- Frame/second-order alignment explains the weight-vs-rotated gap: no; a random dictionary scores higher on it (e03). The advantage is an extreme-value property (max correlation), exhausted after ~8 atoms (e09).
- Survival (Janiak's projection ratio on the raw state) predicts identification: not in GPT-2 (AUC 0.43); the paper's survival is centered-state survival and the centering term carries the effect (e08).
- The identification threshold is the random-dictionary floor sqrt(2 ln m/d): a coincidence between the k=1 statistic of an isotropic null and the k=64 statistic of real states; the competitor level is set by the state covariance (e62, e118, e127, audit).
- Identification is budget-limited: no; k=16 to 64 changes recall by 2 to 3 points; competition (dictionary size) is what matters (e10).
- Dominant writes cancel each other: no; 59 to 80% of top-3 pairs are near-orthogonal and cancelling pairs are 0 to 3%; the crowd of small writes cancels them (e87).
- Attention is the eraser: no; the eraser is always later MLPs, in every model (e08).
- Depth decay of identification is erasure: not in GPT-2/SmolLM2, where the write's direction is amplified 3 to 13x while identification falls; it is erasure in Pythia and OLMo (e21, e76).
- Selected token-embedding atoms are attention copies of context tokens: no; they match context tokens less often than random tokens would (e23).
- There is a small set of filler atoms whose removal fixes the spurious support: no; the spurious support is diffuse over tens of thousands of atoms and pruning the most frequent 400 does not help (e102).
- Support co-occurrence reflects true co-firing: no (correlation -0.03 to 0.16, e101).
- Support-based (Cramer-Rao) error bars predict the coefficient error: no (2-sigma coverage 15 to 42%); the error is absorbed-write bias (e48).
- The "never identified" class is a magnitude effect: partly geometric; injected at identical dose, never-neurons are recovered at half the rate of always-neurons (e30, partial).
- Readability tracks causal importance: no relation in either direction (e59, e59b).
- Alias credit, twin merging, aggressive ridge, top-8 pruning raise identification or calibration meaningfully: no (e41 with the proper null, e42, e12).
- Vocabulary-space structure distinguishes readable from unreadable neurons at the mid layer: no (e40).
- WDD reads attention heads through the static SVD atoms: no (e13); OV value atoms do (e53).
- The linear null-space fraction quantifies unrecoverable provenance: trivially ~97% for any sparse ledger; uninformative (e61). The sparse statement (ERC, spark) is the right one (e03, e63).

## Artifacts retracted
- S1c/S1d: per-token partner atoms written into a shared dictionary overwrote other tokens' true atoms; "in-dictionary partners destroy identification" tracked the fraction of rows overwritten. Replaced by S1e/S1f (appended atoms).
- S1b: cancelling a write with near-orthogonal random atoms needs 8x its energy; those states were dominated by the cancellers. Replaced by anti-aligned partner sets.
- e11 v1: two variants subtracted raw sums from a centered state (constant offset); rerun re-centered (v2).
- e54: the attention-DC magnitude summed over rows (N-fold); corrected to 0.5 to 1.2x a centered state.
- e08 Pythia: computed before the NeoX MLP output bias was added to the caches; rerun (conclusions unchanged).
- e116: "energy in own attention span" is trivially 1.0 (a block's head subspaces span the whole space); ignore that column.
- e47: the "most prominent true write" target is the solver's own statistic restricted to true atoms; near-circular; the non-circular target (largest physically surviving write) is e119.
- e51/e102 early runs crashed on a token-key bug (block -1 with token index > DFF); fixed.
- e30 injection: killed for time after one dose.

## Framing corrected by the audit
- One-shot identification vs prominence is definitional; the content is that OMP tracks it, the competitor level is level-independent and covariance-determined, and the intervention is causal for OMP.
- e44 "equal" mode did not match the 64-atom budget for three models; recall and real-share conclusions stand, the FVU trade-off is overstated there.
- e46 never-groups are small (4 to 7 neurons).
- The centering mean includes sink states (inherited from the paper); typical-only centering shrinks the alignment gap by 0.01 to 0.03 and leaves identification unchanged (e120).

## Session 3 kills (2026-09-19, second free-exploration timer)
- "The next block erases the previous block's writes" as a wired mechanism: ablating the next block's MLP never restores survival (5/5, e183); later blocks take over (e186); the cancellation is a crowd (e185) with no anti-aligned-row signature (e187) and it equals the generic, direction-independent contraction of every trained block (e194, e207, e210).
- "Cancellation is the MLP output bias" (Pythia's constant-offset cancellation): the bias carries 2 to 13% (e192).
- "Dedicated eraser neurons / negative aliases": top canceller neuron carries 1 to 3%, cos ~0 with the write (e185).
- "Never-neurons are a static geometric class" (large null std, small self-gain): predicted own-write z does not order readability; never-read neurons have equal or higher predicted z (e206).
- "Canceller blocks are the unreadable blocks": no universal relation, Pythia's big cancellers are the most readable (e199).
- "The reading threshold is the k-th extreme of the null (parameter-free)": no collapse across k = 8 to 256 (e205); the z-score is the right statistic, its threshold is empirical (3.2 to 4 at k = 64).
- "Drop the atoms the null model selects" as a certificate: recall collapses because the prior atoms are the workhorses (e200, e202).
- "A residual-based self-certificate": flags 85 to 100% of tokens, precision equals the base rate (e180).
- "Identified atoms are causal provenance": reading is uncorrelated with the direction's dependence on the write (e211); provenance labels who wrote a direction, not what caused its presence.
- "Identified writes are more functional per energy" as a 5/5 claim: OLMo reverses it (e165); it is 4/5.
- "The residual toy reproduces erasure": the six-block autoencoder toy only reinforces (e190).
- "LN vs weights" split in e207 as designed: the post-norm gain equals the through-norm gain divided by the normalisation scale; the first-principles trace (e210) is the right decomposition.

## Session 4 kills (round four)
- "The contraction is a damping of writes": the perturbation's energy is conserved or amplified (0.65 to 2.2× after seven blocks) while its direction is scattered; the gain is the diagonal of a mixing Jacobian (e221, e222). Provenance fades by decorrelation, not by loss.
- "The long-range survival floor is rotation into weakly damped low-variance directions": the scattered perturbation drifts toward high-variance directions (e221).
- "The contraction follows the state's spectrum": no monotone relation across principal ranks (e220); only the covariance's null direction escapes, and the top direction is damped hardest in three models.
- "Direction-independent within 0.05" (the original kill rule): the operative band is 0.2, with structured exceptions (state direction, null direction, GPT-2's own writes at 3 to 4×) (e214).
- "The composition law holds at any depth": the product of single-block gains holds for three blocks (5/5) and over-damps beyond (e215).
- "Coarser labels make MLP provenance reliably observable": block-level recall gains 0.03 to 0.31 over a control, above 0.2 only in GPT-2 and SmolLM2; attention heads are the exception (e218).
- "Cancellation within blocks is a general zero-sum phenomenon": constructive in four models, destructive only in GPT-2 (e219).
- "The contraction is born before step 1000": nothing through step 512; born between 512 and 2000 as the warmup ends (e217).
- Frozen-scale intervention on gated models as a magnitude test: it moves SwiGLU off its operating point (Qwen median gain +242, OLMo +18); only the vanishing norm correlation is valid there (e216).

## Session 5 kills
- "A write's energy is genuinely lost": every unit of its footprint is accounted for by named later writers and attention heads (exact identity, residual 0.000; e223).
- "Provenance concentrates on a few successor writers": the delta-ledger has 400 to 1,500 effective writers per block and a top successor of 1 to 8% (e223).
- "Different writes merge into shared directions (provenance collisions)": real writes keep orthogonal footprints (cosine ≤ 0.11); only random, off-manifold perturbations converge (e224).
- "Writes are funnelled into a low-dimensional learned subspace": the footprint cloud's dimension grows with depth and exceeds the writes' own; random-direction footprints are the ones that shrink (e225).
- "Euclidean energy growth means amplification of the write": in Mahalanobis units the footprint is roughly conserved; the growth is drift into high-variance directions (e223).

## Session 6 kills
- "Transport is linear everywhere": linear in the GELU models, inhomogeneous in Qwen and non-additive in SmolLM2, tracking the massive-activation neurons (e226).
- "MLP chains and attention transport a write's descendants additively": the interaction residual reaches 1.1 to 2.8 at the mid layer; attention-only transport overshoots and MLPs cancel part of it (e228).
- "Dispersal is architectural": at initialisation a write is transmitted verbatim; dispersal grows monotonically with training (e229).
- "The descendant has no neuron identity (information loss)": source neuron identified from the footprint in 88 to 98% of tokens (e227).
- "The descendant dictionary beats the native atom everywhere": it wins in the damped gated models and loses in the GELU models (e230).
- "WDD's reading predicts a write's behavioural influence": no (e227).
- "History dependence can be tested by injecting the same vector later": not well-posed in a feedforward residual stream, where the descendant is a function of the vector and the current state (not run).

## Session 7 kills
- "The descendant signature is contextual activation statistics": transplanted write vectors classify as their neuron in foreign states at 0.89 to 0.97 four blocks out (e231); it transfers across corpora at 0.95 without refitting in GPT-2, weakly (0.35, chance 0.08) in SmolLM2 (e234).
- "The signature depends on the write's magnitude / operating point": 0.5× and 2× give identical accuracy (e231).
- "The signature is a brittle coincidence of the exact weights": unchanged under 3% weight noise, largely intact at 10% (e233).
- "One path carries the identity": both do; attention carries its own version in the gated models (e235).
- "Native atoms or descendant signatures work at any depth": both fade in the final two blocks, where the intervention footprint itself loses half its identity (e232).
- "History dependence" as a separate variable: what exists is state-conditioning, measured as the gap between natural and transplanted identifiability (0.03 to 0.10 at four blocks, up to 0.5 at the mid layer) (e231).

## Session 8 kills
- "Transported WDD restores provenance at depth": the transported dictionary is no better than the native one anywhere and worse at the mid layer in GPT-2, Pythia, OLMo and SmolLM2 (e236); coherence rises only in the gated models and not where the failure is largest (e240).
- "There is a provenance resolution limit below which nearby writes merge": descendants keep their initial cosine to within 0.02 in 5/5 (e238).
- "The state's provenance capacity saturates at a few dozen sources": bits keep growing to 128 candidates (e237).
- "The signature is imposed by the downstream network": final-model vectors through the step-16000 network are recognised by either checkpoint at 1.00 (e239).
- "Descendant centroid geometry mirrors write-vector geometry in general": only in the GELU models; context dominates in the gated ones (e238).

## Session 9 kills
- "Gating makes the MLP linearisation more context-variable, explaining the GELU-versus-gated split": the factor variability is as large in the GELU models (e244); the split follows the massive-activation neurons (e242).
- "Descendant decoding needs per-neuron templates": zero-shot identification from the write vector's transplant image matches fitted centroids (e242).
- "Superposed writes lose their identities": eight simultaneous writes are recovered two blocks out in 5/5 (e241).
- "Dispersal and the isometry are both learned": the isometry and identity transport exist at step 0; only dispersal, readability and final-layer scrambling are learned (e243).

## Session 10 kills
- "The transported code is neuron-specific": random unit vectors are transported and identified exactly like neuron writes (e245).
- "Injected writes are unreadable natively because they are foreign to the context": they are unreadable because they are not prominent; at 4× magnitude the native chart reads them at 0.99 (e247).
- "The descendant's context-independent part must be large for identification": 7% suffices for 64-way identification in Pythia (e245).

## Session 11 kills
- "Provenance survives in a handful of dimensions": tens to hundreds of dimensions are needed (e249).
- "Massive-activation neurons break vector transport": their transport is ordinary; their natural effect runs through the sink mechanism a transplant does not trigger (e250).
- "The native coordinate is lost by attention": it is lost at the MLP adds, block by block; attention leaves it unchanged (e249).
- "The crossover depth is unpredictable": the single-block gains predict it in the models whose charts behave (local analysis).
- "Neuron identity is what the network transports": a+b is classified as a+b, never as a or b (e248).
- "A single linear operator fitted on random directions transports real writes everywhere": true in the GELU models, false in Qwen, where real writes and off-manifold directions are transported differently (e248 with e224, e242).
- "The descendant is linearly invertible": the inverse identifies the source but reconstructs the vector at cosine 0.1 to 0.4 (e248).

## Session 12 kills
- "There is a single representation phase transition": the regimes are staggered across 4 to 8, 10 to 16 and the last blocks, and the isometry never breaks (e252).
- "Whitened energy is a conserved quantity": conserved only to the mid layer (e252).
- "The descendant is well coded densely": no 8-coefficient code reconstructs it; dense coordinates win only for provenance (e251).
- "Neuron identity predicts function better than the descendant": the reverse, in 5/5 (e254).
- "A single within-token state variable (the sink activation) explains the massive neurons": their transplant never matches their natural descendant in any context; the route is cross-token (e253).

## Session 13 kills
- "The massive neurons' effect migrates across positions (a cross-token route)": 0.89 to 1.00 of their footprint stays at the source and receivers do not identify them (e257).
- "Functional organisation emerges after dispersal": descendant-function coupling is highest at initialisation and is broken, then partly rebuilt, by training (e258).
- "Identity is as legible in function space as in the state": single-write logit identification 0.15 to 0.52 vs 0.88 to 1.00 (e256).
- "Provenance is token-local everywhere": in GPT-2 receiver positions identify a write at 0.46 (e257).

## Session 14 kills
- "Similar descendants are interchangeable parts": substitution worsens or does nothing for the removal effect (e262).
- "Compensators are the removed write's descendant neighbours": a diffuse crowd, weakly aligned with either coordinate, leaning to the write in GPT-2 and Pythia (e261).
- "Write similarity carries the functional relation": partial correlation given the descendant is −0.06 to +0.16 (e259).

## Session 15 kills
- "One operation builds functional organisation": attention in three models, the MLP in two (e266).
- "Descendant-similar neurons are redundant or coupled circuits": their effects at co-active tokens are independent and uncorrelated, and their per-token descendants near-orthogonal (e264).
- "Successors form a shared hub graph": stable per-neuron lineages, no hubs (e267).
- "Function maps many-to-one onto descendants": 3 to 12% of functionally similar pairs have dissimilar descendants (e265).

## Session 16 kills
- "The write vector adds functional information beyond the descendant": it adds nothing and hurts in the last blocks (e268).
- "The write vector, transplanted, carries the write's effect": it reproduces the natural logit effect at −0.29 to +0.16; the descendant does at 0.28 to 0.62 (e269).
- "Descendant geometry is as stable as write geometry across training": it is less stable (0.40 vs 0.77 at step 64000) (e270).

## Session 17 kills
- "Intervention effects are nonlinear in the descendant coordinate": interpolants along descendant segments give effects at cosine 0.99 to 1.00 to the linear interpolation (e271).
- "The descendant admits a small reconstruction basis": it needs the full dimension (e272).

## Session 18 kills
- "Function is a quotient of the descendant with a large kernel (functional equivalence classes)": the functional response to a fixed descendant displacement is the same along top, middle and bottom principal components and random directions (ratio 1.2 to 1.4); GPT-2's bottom-PC exception is the massive-activation channel (e273, e273b).
- "The identity, function and behaviour subspaces are nested": identity and function share a leading core (0.48 to 0.76 of the 8-dimensional energy) and rotate apart beyond it (0.39 to 0.47 at 32); the linear behaviour direction lies outside both in 4/5 (with the caveat that the removal KL is quadratic in the logit change) (e274).
- "The massive neuron is the extreme tail of one transport law": in SmolLM2 its descendant is anti-parallel to its linear transport (cosine −0.64, z +7.6 against a bulk within 0.77 to 0.98) (e275b).
- "Transport residual tracks coefficient size or identifiability": no consistent sign across the five models (e275b).
- Lesson: an unsigned transport prediction is wrong for gated models, where 43 to 75% of dominant writes carry negative coefficients (e275 superseded by e275b).

## Session 19 kills
- "Downstream components are tuned to the descendants of the model's own writes": natural descendants have the leverage of random-vector descendants (logit-response ratio 0.94 to 1.18, KL 1.0 to 2.0) and of covariance-matched random directions (0.65 to 1.14) (e276).
- "The raw write direction is read at depth": at level L it has the leverage of a random direction (0.98 to 1.05) (e276).
- "The incoherent part of a descendant is a linear function of the context": R² from the state ≤ 0.10 at mid depth in 4/5, ≤ 0.05 near the end; SmolLM2 alone keeps 0.19 to 0.27 from the birth context (e277).
- "The massive neuron lies beyond the linear range of the transport law": its transplant obeys the law at 0.1× to 10× (cosine 0.56 to 0.72, self-consistency 0.78 to 1.00 through a tenfold operating-point shift) while its natural descendant is anti-parallel at every amplitude (e278).
- "Ordinary writes are amplitude-invariant at every amplitude": invariant to about the natural amplitude (0.78 to 0.99), distorted at 3× to 10× (0.08 to 0.80) (e278).
- "A mid-depth coordinate transition exists": rotation is confined to the 1 to 5 blocks after birth and the last block; the middle is stable (0.65 to 0.91 kept per block); SmolLM2 has one mild dip at block 10 (e279).
- "The reviewer's local-versus-global transport split explains the massive neuron": its footprint is entirely local (e257); the split that holds is context-free versus context-bound (e278 with e250, e253, e257).

## Session 20 kills
- "The network snaps an interpolated descendant back to a natural regime": the trajectory stays on the interpolation of the endpoint trajectories at every later level and at the logits (cosine 0.97 to 1.00, fitted position equal to the true one within 0.01) (e280).
- "Level-specific operators do not compose functionally": the composed prediction reproduces the actual image's effect as well as or better than the direct fit (e281).
- "A whitened metric makes transport more isometric": gain spread and pair-cosine preservation are worse in the whitened geometry at every level in 5/5 (e282).
- "Transport is an isometry on the manifold of actual writes": writes vary in gain 3 to 6 times more than random directions; the isometry is angular only (e282).
- "The origin predicts the future descendant as well as the present does": write law 0.03 to 0.12 against descendant law 0.43 to 0.78 (e283).
- "Attention carries the massive neuron's context-bound reaction": all attention frozen leaves it at −0.66; the MLP of block 10 alone flips it to +0.52 (e284b).

## Session 21 kills and corrections
- "Transport is an isometry": the fitted operator's singular values spread 11 to 50×, the actual gain along its top directions is 2 to 5× that along its bottom directions, and the pair-cosine preservation of e238 follows quantitatively from gain concentration for random-component pairs (formula confirmed within 0.07 in 15/15) (e282, e285). The record's "isometric" is to be read as "angle-preserving for random-component pairs".
- "The additivity error depends on total norm, not on the number of superposed writes": not testable with the reference construction used (cross-token confound, floor 0.51 to 0.73 at m = 1); left open (e285b).
- "The operator's null directions are null in the network": the actual gain along the operator's bottom singular directions is 0.16 to 0.35 (e285).

## Session 22 kills
- "Required dimensionality descends rung by rung: reconstruction > provenance > function > behaviour": two rungs only. Reconstruction needs 512 to 1024 directions; identity, future descendant, function and behaviour all need 8 to 32 supervised directions at mid depth, within a factor of two to four of each other (e286).

## Session 23 kills
- "The token-specific part of the descendant is itself causally read": the span of the class centroids (28 to 46% of the energy) reaches the full score on identity, function, behaviour and the future in 5/5, the remainder adds nothing, and alone it carries what a random projection of the same dimension carries (e287).

## Session 24 kills and corrections
- "The causal read-out lives in the term shared across a neuron's tokens" (session 23): a covariance-matched random K-dimensional subspace and a permuted-label centroid span decode function, future and KL as well as the true centroid span; only identity is specific to the shared term. Corrected to: the read-out lives in the high-variance, low-rank part of the descendant cloud (e288).
- "The four causal subspaces are one subspace": identity, function and future share half to three quarters of their energy at 16 dimensions; the KL subspace is less aligned (e288).
- "There is an intrinsic causal dimension with a sharp elbow": the curves saturate gradually between 1 and 32 directions (e286 curves).
- "The transport operator carries the causal subspace into itself beyond stationarity": transported overlaps never exceed the untransported ones (e288).
- "Adding the massive direction at its own tokens reproduces the natural footprint" (the theory's stated prediction): the addition obeys the first-order law and it is removal that is reversed; the response at the massive neuron's own tokens is even in the deviation from its natural amount (e290).
- "The token-specific remainder is noise": it predicts its own future at 0.09 to 0.28 against 0.00 shuffled (e288).

## Session 25 kills
- "Intervening on a causal coordinate dictates the whole logit change": the full-vector prediction reaches cosine 0.09 to 0.33 and only 19 to 39% of the response energy lies in the paired span (e292). The selectivity stands; the completeness does not.
- "The steering map extends outside the descendant cloud": plain random directions are predicted at 0.01 to 0.08 (e292).
- "The leftover dimensions are a second state variable": under intervention they match a random vector on the logits, the KL, entry into the later core and cross-token reach (e295).
- "The future descendant is low-dimensional under linear reconstruction": it needs about 128 directions; the 8 to 32 of session 22 is the cosine-and-identification scale (e294).

## Session 26 kills
- "WDD writes are a privileged class of perturbation with respect to the dimensionality hierarchy": random, covariance-matched, attention-head and later-MLP directions show the same ladder (e304).
- "The causal core predicts attention routing": R² 0.09 to 0.16 for the per-head output change, a tenth to a sixth of the variance (e298).
- "The core is determined by the write's geometry across neurons": write similarity does not predict core similarity (Spearman ≤ 0.08) (e300); yet a single write predicts its own core coordinate at 0.53 to 0.71 in the GELU models (e303).
- "Causal coordinates interact with circuit-like structure": the interaction matrix is weak, uniform and rank one; the only strong nonlinearity is amplitude (e302).
- "The propagation is a low-rank spatiotemporal wave": only where the massive channel dominates (Qwen, OLMo, SmolLM2); in GPT-2 and Pythia the WDD tensor has the spectrum of a random injection (e299).
- "A sensitivity-dominant subspace exists": the ridge map's steepest directions are a low-variance artifact; the isotropic gain of e273 stands (e296, noted not claimed).

## Session 27 kills
- "The causal core is an invariant, singular, slow or fast subspace of the transport operator, or the update's": every overlap at chance, gain equal to random directions' (e305).
- "The causal core is the read-out's high-gain subspace": chance overlap with the forward-estimated read-out's top singular space and with the true backward sensitivity subspace (e305, e308).
- "The forward causal core and the backward sensitivity subspace coincide, or intersect in a small space that predicts the effect": overlap 0.03 to 0.07 (chance 0.01 to 0.03); the intersection is the forward core; the backward subspace decodes barely above random (e308).
- "Channels are born, die, merge or split along depth": outside the massive channel, none do; identity assignment at every block, unit gain, 2 to 6% mixing (e307).
- "The functional coordinate system is family-specific at every depth": it converges across unrelated families with depth to the within-family reliability (e306).

## Session 28 kills
- "Only perturbations near the natural residual manifold acquire the low-dimensional functional coordinate": perturbations orthogonal to the manifold, heavy-tailed, sparse, low-rank and shuffled all show the same ladder (e310).
- "The causal core is the subspace of high residual variance": 21 to 34% of it lies in the state's top-64 directions against 93 to 95% in the descendant cloud's own (e309).
- "The cloud is a linear subspace": local rank 22 to 30 against global 31 to 130; it is a curved manifold (e309).
- "The network contracts functional differences through the middle": consecutive-level ratios 0.96 to 1.00; mild contraction only at the last step (e311).
- "Functionally equivalent states interact when superposed": cosine 0.98 to 1.00 to the sum, the same as for dissimilar pairs (e312).
- "The converged coordinate system is only a within-family phenomenon": cross-family cores decode at 85 to 90% of own near the output and cross-family metrics match own (e312); at mid depth the transfer is partial.

## Session 29 kills and limits
- "The coordinate predicts behavioural scalars (next-token log-probability, top-1 probability)": 0.06 to 0.19, at the level of a random projection, and nothing predicts them above 0.22 (e314).
- "The entropy change is an odd function of the coordinate": sign flips in 38 to 100% of directions; it has an even part (e315).
- "The coordinate dictates the logit change under intervention": 0.13 to 0.32 without refitting, as e292 (e315).
- Discarded measurement: the cross-token reach in e315 (neighbouring tokens were also injected).

## Session 30 kills (the application study)
- "The atlas's functional classes are functional units": ablating a class is no more coherent than ablating random neurons (e316, e316b).
- "Coordinate-equivalent neurons are substitutable": a same-class atom with the matched coefficient worsens the logit deviation (−0.22 to −0.59) and does not restore the coordinate (−0.20 to +0.12) (e316b).
- "The quotient gives better output attribution than direct logit attribution or an averaged-Jacobian lens": all below cosine 0.2; the atlas is the worst at the logits (e316, e316b).
- "The data-free operator atlas approximates exact coordinates": cosine 0.28 to 0.58 only (e316).

## Session 31 kills, part 1 (the quotient program)
- "The quotient is observable-relative": nine observables' quotients overlap at the split-half reliability ceiling and transfer at 0.89 to 1.13 of own (e317).
- "Individual coordinates persist across depth when the basis is refitted": the composed matching returns 6 to 38% of coordinates; only a fixed basis keeps them (e321).
- "Local quotients decode better than the global one": no (e324).
- "The causal dimension expands in the nonlinear regime": it falls to 4 to 8 and the quotient rotates; all responses collapse toward a common direction at 8× (e329, e330).
- "WDD atoms are a sparse or concentrated basis of the quotient": no advantage over random 16-vectors or transported random directions (e332).
- "The top ledger atoms carry a block's functional coordinate": top-8 and top-32 predict it at −0.24 to +0.30; the full ledger at 0.14 to 0.40 (e333).
- "Functional classes of atoms are vocabulary-interpretable": no (e334).
- "Functional equivalence is a coset of a null space": the difference of equivalent atoms is a full-strength perturbation; no null directions (e336).
- "Fibres have structure": their diameter and dimension equal random sets (e338).

## Session 31 kills, part 2
- "The supervised quotient carries target-specific structure": a quotient fitted to shuffled targets scores the same; PCA, PLS, CCA and kernel PCA coincide (e340, e347).
- "The core is the carrier of the causal information": the discarded 64-dimensional complement, and a random 64-subspace, decode identity, function, future and token identity as well (e346).
- "The core component carries the logit effect": cancelling it restores 1 to 6%; cancelling the physical descendant 15 to 18% (e348).
- "Some perturbation family breaks the quotient": none of sixteen at 1× or 4× falls below an own-over-random advantage of 1.11 (e342).
- "Attention transports a token's own image": freezing attention leaves the image at cosine 0.90 to 0.96; freezing the MLPs changes it to 0.28 to 0.80 and doubles the gain (e351).
- "The context variation of a direction's coordinate is readable from the state": R² −0.16 to −0.36 (e356).
- "The functional kernel is fully captured by the coordinates": alignment 0.09 to 0.47, with a 41-to-71-eigenvalue tail (e355).

## Session 32 kills and reopenings (phase 2)
- "First-order selection (the gradient along a unit's own write) measures the unit's use": at a trained optimum every head's mean selection is about zero while its ablation effect is positive; GPT-2's induction heads: selection -0.002, ablation +0.018 at the positions they serve (e370, e370b).
- "Co-selection structure is a persistent object across training": agreement with the final co-selection matrix 0.00-0.24 through step 33000, communities never persist (e364b).
- "Co-selection communities are circuits": after the global mode, co-selection structure is mostly layer structure (top-100 correlations same-layer 0.34-0.74 against 0.03-0.08 chance) with modularity barely above shuffled (e359).
- "Local curvature and co-selection reproduce ablation interactions": interactions are 2.4-48x the Hessian's and ranked by it at 0.11-0.53 (e371).
- "A neuron's function is more stable than its implementation (drift with stable function)": in Pythia the write settles first and the reliability-corrected logit signature follows (e360, e360d).
- "Early-layer neurons have a context-free logit effect": two token samples agree at 0.28 at the end of training in block 2 (0.93 at step 1000) (e360d, e360e).
- "Features speciate during training": splits and merges balance at every checkpoint (e360c).
- "The quotient results hold only for block-2 writers": they hold at four birth depths in all five models (e361).
- "The functional coordinate persists through training in function space": its logit image is no closer to the final quotient's than to a random direction's (e365b).
- REOPENED, as blind-spot results: "Causal coordinates interact with circuit-like structure" (e302) and "descendant-similar neurons are redundant or coupled circuits" (e264) were measured as natural-text population averages, which e357, e358 and e366 show are largely blind to known circuits. They remain true of population averages; they no longer count as evidence against circuits.
- Retracted before counting: e359's mixed-batch induction test (circular head selection; superseded by e363); e360's raw late signature and late-dissociation statistics (below the reliability ceiling; replaced by the ceiling-corrected curves of e360d).

## Session 33 kills and corrections (phase 3)
- "The removal knee is a degenerate direction of the loss (the singular-learning-theory reading)": the loss around full strength is an ordinary quadratic, even-part exponent 2.0-2.1 in every model (e374).
- "Near-twin writes make backups and composition makes series": on the loss, write cosine predicts super-additivity at rho -0.11 to +0.22 and composition goes with super-additivity in 9 of 10 conditions (e376).
- "The sign of a pairwise ablation interaction measured on the loss reveals circuit topology (backup vs series)": 26-48% of pairs change sign between the loss and the correct token's logit; with additive logits the loss alone makes 97-100% of pairs super-additive; the readout/circuit attribution is path dependent (e377, e382).
- CORRECTION of phase 2 (H169, e365): "induction circuits keep adding redundancy after they work" holds only on the loss; on the logit the top-8 joint ablation is sub-additive at every Pythia checkpoint (0.49-0.93) (e380). Phase 2's super-additivity readings in e357 and e365 are loss-readout statements, not redundancy statements.
- "Lower-layer implementation settles before function, so lower parts are swappable late in training": the step-63000 lower part costs the final upper part 0.28-0.42 nats and old upper parts reject the final lower part (e378).
- "Reader-weighted WDD is a reliable general edge finder": top-1 on strong edges 0.02-0.69 across models (e381); it is reliable on the distinctive induction edge (e375).

## Session 34 kills and corrections (phase 3b)
- "The KL interaction of two ablations measures whether the circuit adds": it equals the Fisher inner product of the two single logit effects (per-token correlation 0.73-0.98) (e383). e264's additivity survives in logit space; its control-pair "redundancy" in GPT-2 and Pythia is retracted as a readout overlap.
- CORRECTION of the phase-3 correction: "the induction circuit is never redundant in logit space" holds under zero-ablation (e380) and for four of five models under mean-ablation; Pythia's final circuit keeps joint over sum 1.72 on the centred logit under mean-ablation (e384).
- "Reader-weighted provenance is a new circuit-discovery capability": the exact split of a head's query or key input over upstream components is standard (ARENA; Franco and Crovella 2025); only WDD's state-only approximation is new, and the exact split beats it.
- Scope note on "redundancy and self-repair come from training noise" (e379, e386): a toy-model statement; real models trained without dropout show self-repair (McGrath et al. 2023; Rushing and Nanda 2024), and Pythia keeps logit-space redundancy (e384).

## Session 35 kills and narrowings (fresh-slate pass)
- "Readers' preference for WDD-identifiable writes shows the model organises writes to be read": most of it is geometry shared with the state, present at initialisation (raw 1.71 in Pythia at step 0); a genuine part remains only in GPT-2 and OLMo (e387).
- "WDD's functional advantage over a random dictionary comes from the embeddings": the MLP write rows carry it (e389).
- "Any trained MLP rows would do as well as the model's own": rows from a trained checkpoint at cosine 0.35 to the final rows do no better than random (e390).

## Sessions 36 and 37 kills and narrowings (the vision chain; self-description and private languages)
- "WDD's code is the few largest writers": the largest actual writes recover 0.53-0.79 at 64 and 0.75-0.92 at 1024, against WDD's 0.91-0.99 at 64; most of WDD's MLP atoms are not among the top writers (e391, e393).
- "WDD's advantage over the true writes is only the refit": refitting the true top writes' coefficients closes part of the gap, not all of it; the support matters (e393).
- "A training-free replacement model from the model's own words": errors compound when every layer is replaced; +2.05 to +5.32 nats at 64 atoms per layer, +0.43 to +3.77 at 128 (e392).
- "WDD atoms are sparse attribution nodes": effects spread over 13-16 of 32 atoms, linear attribution weak (Spearman 0.33-0.67) (e394).
- "Any network's weights describe its states better than random directions": not at initialisation; the property is learned (e395).
- "Each training stage has a private language": false for the late stages, which are interchangeable (e396).
- "The step-4000 vocabulary fails because its geometry is degenerate": its effective rank and rotation are ordinary; the failure is carried by its orientation (e397).
- "The shared tokens fix a common coordinate system across seeds": the two embedding matrices are not rotations of each other (R2 -0.10) and the lexicon-fitted map translates nothing (e398).
- "A state-fitted linear map translates one network's words into another's, even better than the network's own words": random words through the same map already match the own words (0.59); the translation adds 0.07 (e398b). Feature transfers through fitted linear maps need random-direction controls.

## Session 38 kills and narrowings (the review's controls)
- "The own-word advantage is word-level at every stage": at step 1000 most of it is the vocabulary's second moment and span (0.62 and 0.85 of the gap at k = 16), and a functional pursuit leaves 0.04 of it (e399, e400, e405).
- "The network's own words are the best sparse vocabulary for its states": words drawn from the states' covariance beat them at k = 16 at every checkpoint, and a learned SAE beats them at k = 4-32 in GPT-2 (e399, e400, e403).
- "Choosing words by function shortens the self-description": the 90% length is unchanged (e400).
- "False friends are an artefact of Euclidean variance-chasing": they deepen under the Fisher metric. They are mediated by the target's few huge directions: handled exactly, the step-4000 words are partial friends (e400, e404).
- "Evaluation leakage explains self-describability" (review): not accepted as stated. The one concrete shortcut, the current-token embedding, is ruled out by MLP rows alone (e389, e399).
- "The translated vocabulary equals the target's" (review's reading of e398): the fitted linear map lifts random words to the own level; the translated words add 0.07 (e398b).
- "False friends are a general consequence of huge directions in the state": OLMo has huge directions at steps 64000 and 1454000 and no false-friend cells (e406); on the evidence so far the finding is Pythia-specific.
- "Accretion is universal": OLMo's final vocabulary reads early states worse than their own words (e406).

## Session 39 kills and narrowings (round 2 and fresh angles)
- "Readers carry atom-specific structure like the writers": their advantage is second-order at every checkpoint (e407).
- "A weight-only optimised vocabulary approaches the SAE": read-strength and write-norm selections collapse onto the token embeddings and do worse than a random subset (e408b).
- "The weights alone give the functional metric": they flag the huge directions correctly, but as a pursuit metric they recover 0.15-0.83 of the Fisher lift, less at the end of Pythia's training (e408, e408b).
- "False friends are a Pythia-family property": none at 70m or 160m (e409).
- "False friends coincide with the cross-seed divergence phase": the divergence is at steps 256-2000, the window at 4000-8000 (literature timing).
- "Readers describe the errors better than writers at every stage": at step 1000 the writers do (e412).

## Session 40 kills and narrowings (the workspace agenda)
- "The native-word lens reads a two-hop bridge earlier than the logit lens at the final position": nothing reads it there before block 22, and late the plain lens is at least as good (e415). It does at the subject's last token (e420).
- "The bridge word at the subject position is causally necessary": removing it there changes nothing (e420).
- "Native words resolve roles earlier": all readers are at chance through block 20 (e417).
- "The native lens fabricates less": for digits it fabricates more than the plain lens at most blocks (e416; the task itself was beyond the model).
- "The huge directions carry little function": locally yes (Fisher), but kept exact alone they recover 29% of the loss at 7B (e419).
- "A Fisher-metric pursuit gives a better functional description" (true in Pythia, e400): false at 7B, with or without the huge directions held exact (e418, e419).

## Session 41 kills (established lenses)
- "The native-word lens is just Geva et al.'s value-vector reading": the actual largest writes surface the hidden bridge worse than the plain lens; the native re-description surfaces it best (e421).
- "The super weights are the native language's function words": not in Qwen2.5-7B's middle-depth descriptions of positions after the first (e422).
