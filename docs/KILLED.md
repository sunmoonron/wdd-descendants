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

## Session 42 kills (what WDD unlocks)
- "WDD is a better lens in general": model-dependent for recalled entities (better in OLMo-1B and Qwen2.5-7B, worse than value vectors in Pythia, worse than the plain lens in GPT-2) and blind, like every vocabulary-space reader, to context features (e423, e424, e421b).
- "The previous-token word is causally used for induction": no such word exists in the native readout at key positions, so the test has no content (e425).
- "The forward-backward duality holds in every model": three of five; reversed in GPT-2 (e426).

## Session 43 kills
- "Self-description is a training-free novelty or uncertainty signal": random tokens are the most self-describable input in OLMo, shuffled text is as describable as natural text, and badly predicted positions are not described worse (e430).
- "Instruction fine-tuning changes the native vocabulary": median row cosine 0.997, full mutual intelligibility (e431).
- "Self-description usage is a better importance score everywhere": it fails in OLMo and GPT-2, where norm-dominant neurons (super weights and massive-activation writers) carry the damage (e428, e428b).
- "The forward-backward duality fails in GPT-2": an artefact of leaving out attention readers (e429).

## Session 44 kills (the survey's bridges)
- "The huge directions hold 86% of Pythia's variance (48% of OLMo's)": one sink position per sequence held it ("\n\n\n", 41 times the median norm; OLMo's " don", 46 times). At typical positions M holds 0.10-0.26 in all five models (e432).
- "At 7B the huge directions hold 97% of the variance": one token in the fitting text (" series", 221 times the median norm). Without it they hold 11% (e437b).
- "The replacement model compounds through the sinks": sinks exact changes nothing (e392b).
- "False friends are mediated by the huge directions": they were mis-described sinks. With the sinks exact, the step-4000 words beat their rotation at k = 4 (e437).
- "M is function-light": locally, per unit variance, yes in four models (Fisher about 0.1 of average), not in GPT-2 (0.57). Under ablation it costs the average per unit variance, 2-4 times its quadratic prediction (e432, e437).
- "M is a set point", "normalisation ballast", "a temperature knob" (e432).
- "The knee is the size of the move": a random move of the same energy stays quadratic (e432).
- "The knee is the softmax": freezing attention patterns removes 28-64% of the cost but not the knee (e438).
- "The knee is categorical identity decoding": M's within-token part has the knee too (e433).
- "The network keeps its own words": retention follows variance, not usage (e434).
- "Self-description is alignment with the covariance": Gaussian states with the same covariance keep 15-45% (e435).
- "Zipf-like usage is a property of the native language": Gaussian states reproduce the slope (e435).
- "The native vocabulary is the lexicon": removing token rows, position rows and block-0 MLP rows leaves the gap, at every checkpoint (e439).
- "Context is described only to second order": word-level in all five (e440).
- "The late vocabulary is a union of per-block accents": 0.09 or less of the gap in three models at the end, although 0.82 at step 1000 (e443).

## Session 45 kills (the vision round)
- "Two models' native vocabularies translate word for word": 0.06-0.13 of function against 0.85-0.91 for a dense map (e445).
- "The native vocabulary is spoken": writer rows frozen at random are barely used as words (0.69-0.75 unexplained against 0.74 for their rotation), in grokking and in a small sequence model (e444, e444b, e449b).
- "Native words are more interpretable than neurons by token-level proxies, and more than rotated words": more token-explainable than their neurons in five of five models, but rotated words score higher in four of five; such proxies reward lexicality (e446).
- "Self-description is a reliable early signal of generalisation": the word-level part leads in two of four generalising runs, not in the frozen variants (e444).
- "Rosetta words are invisible to activation analysis": their neurons fire on the concept (e448c).

## Session 46 kills (causal tests of concept words; a re-implemented block)
- "Concept words are correlated provenance, not causal handles": swapped at every block up to the middle, one MLP write row redirects 67-94% of translations and, in the Qwen models, 39-78% of category answers (random 0-3% and 11-13%) (e455).
- "One depth is enough to test a concept word causally": single-depth swaps change no answers in the Qwen models (e451, e454), and multi-depth swaps change most (e455).
- "The concept word is the noun's only carrier": removing it at every depth leaves Qwen's accuracy at 0.88-0.90 (e455; SmolLM2 0.13).
- "A block's function determines its vocabulary": re-implementations keep the loss within 0.02 nats with new rows (median best |cos| 0.26-0.28; two seeds 0.22-0.25) (e452).
- "Frozen readers keep the vocabulary when the function is fixed": trained writers behind frozen random readers keep 10-18% of the advantage (e452; the pre-registered guess was at least half).
- "The re-implemented rows lack the original's word advantage because they lack end-to-end training": next-token training of the block adds nothing beyond more distillation (0.52-0.70 against 0.71-0.77 of the original's advantage) (e452b).
- "Concept words are shared words between model sizes": a mapped concept word is the other model's nearest atom for 1 of 14 nouns, at |cos| 0.08, although it identifies the right concept in 11 of 14 (e453).

## Session 47 kills and corrections (WDD as a forensic instrument)
- "e455's swaps were at the target's typical coefficient": the per-block additions accumulated to 6-8 times natural size. Near natural size, 35-58% of translations move, not 67-94% (e458).
- "Self-description detects compression damage": at 4 bits GPT-2 loses 3.9 nats and keeps 96% of the advantage (e456).
- "A fine-tune's state change is sparse in native words": on chat it is low-rank and dense (its own principal directions 0.27-0.32 unexplained against native 0.57-0.65), although usage changes concentrate in a few words (e457).
- "A WDD checksum predicts an intervention's success beyond its size": AUC 0.70-0.92, against 0.83-0.94 for the displacement alone (e458).
- "Native words regenerate across generated tokens": no more than their usage rate predicts (e459).

## Session 48 kills (authorship labels as forensics)
- "Near-identical states with different write histories have different futures because of their histories": the difference is context (76-85%), and the ledger distance does not predict the vector's effect (e460).
- "WDD is an activation authenticity checker": a forged write-sized vector is detected at AUC 0.60-0.64, no better than a covariance detector, and not at all a few blocks later (e461).

## Session 49 kills (the learning signal in native coordinates)
- "A gradient update reinforces or shrinks a word along its own direction": per batch, the along-row share is 0.5-0.8 times chance and the sign is a coin flip (e462).
- "One batch's gradient predicts how a word will change": cosine with the net change between checkpoints is 0.000-0.001 (e462).
- "The most used native words receive the most learning pressure": Spearman -0.63 to -0.04 (e462).

## Session 50 kills (a grammar over native words)
- "Native words have accents (role carried by inflecting shared words)": role is carried by word choice; shared words' coefficients decode it at 0.61 and never flip sign (e463).
- "The network corrects a native word written where it never occurs": it is damped exactly as where it fires (e464).

## Session 51 kills (self-consistency of native descriptions)
- "The network regrows what a 16-word native description leaves out": the omission persists at a constant share of the state four blocks later, while a random error of the same size is damped (e465).

## Session 52 kills and corrections (systems-theory and causal-abstraction readings)
- "A few extra directions make a native description dynamically sufficient": 64 extra remainder directions still leave 7-11% of the loss unrecovered (e466).
- "What a description leaves out is a different kind of information": token identity, neighbouring tokens and position are in both halves (e467).
- "e451's single-depth weakness is because the concept has moved elsewhere by the middle layer" (session 46's explanation): whole-state patching at that position and layer switches 78-99% of answers; one word is outvoted by the other directions carrying the noun there (e469).

## Session 53 kills (where native words carry a variable)
- "A native word is a causal handle at any block where the variable is present": only across the band of blocks where it is written; from Qwen's block 8 onwards four native words switch 15%, while the whole state switches 100% (e470).
- "The carrier words are the same identities throughout": consecutive-block overlap is 0.41-0.47 (e470).

## Session 54 kills (the life cycle of native words)
- "Neurons become important before they become native words": no ordering; usage and importance evolve independently across Pythia's training (e471).
- "When a word falls out of use its function dies": its importance does not change (e471).

## Session 55 kills (WDD applied to itself, roles, induced geometry)
- "Iterating WDD's description reveals a native attractor or drift": one application is already a fixed point, for rotated words too (e472).
- "Perturbed states flow back to a common native description": each noisy trajectory freezes at its first description (e472).
- "Native identity handles are typed by grammatical role": handles found in one role work as well in the other (e474).
- "Native words are the better interchange coordinates for every variable": for an entity copied into the answer, rotated words do as well or better (e474).
- "WDD defines a geometry of states that predicts behaviour better than activation distance": activation distance is better, and WDD's extra information is mostly shared with rotated words (e475).
- "The identity-free signal in deep descriptions is prediction confidence" (a guess made during the session): it survives controlling for the entropy gap (e475).

## Session 56 kills (audits from the digest)
- "The native-only profile signal is an artefact of the normalisation or of k": it is the same under L1, at k = 8 and at k = 32 (e476).
- "It is a proxy for norm, position, confidence, token frequency, copying or token identity": none of their gaps removes more than 0.03 of it (e476).
- "The attribute handles of e473 are item-specific": four native words found on other kinship words, in other languages, flip only gender in 98% of items, better than the item's own (e477).

## Session 57 kills (WDD as a workspace reader)
- "The native reader's coefficients or causal receipts certify its claims": on chained arithmetic every certificate is at chance and no reader beats the 0.5 base rate by much (e478b).
- "The per-word native reader keeps everything the lens captures": it loses half of the finished answer; only the reconstruction read as a whole keeps it (e480).
- "Signs of native words carry who did what to whom": without a question, order is position and recency, for every reader (e482).
- "The planned rhyme is readable at the end of the first line": not at 7B, for any reader (e480).

## Session 58 kills (calibration against classical theory)
- "The competitor level is set by the atoms' second moment": true for the rotated dictionary (ratio 0.96-1.02) and false for the native one (2.9-6.2 times), so the excess is provenance (e483).
- "A native description is a redundant code": every word matters; a random half refitted reaches 0.59-0.80 of OMP's own 8 (e484).
- "Refitting compensates because native supports are more coherent": they are as near-orthogonal as rotated ones (e490).
- "The Heaps exponent of the vocabulary in use marks provenance": native 0.60-0.76 against rotated 0.55-0.78, no order (e486).
- "Provenance loss has a fast and a slow mode": one exponential fits, at the contraction's retention; Prony's second mode carries nothing (e491).

## Session 59 kills (the factor decomposed, the SAE bridge)
- "The provenance factor is a property of the dictionary's alignment as a whole": at early and middle depth it is the position's largest writes standing out; the collective alignment is a floor of 1.3-1.9 that is already there at step 1000 (e493).
- "A learned feature is one native row": sixteen rows leave half of it unexplained; a quarter of the most used rows are features, the rest are what features are built from (e494).

## Session 60 kills (the SAE bridge, the accent's clock)
- "The provenance layer under SAEs changes with depth": the same numbers at five depths of GPT-2 (e495).
- "A feature is its top row's activity": AUC 0.62 for the row, 0.70 for the eight-word composition; a Spearman over all positions is uninformative for a feature that is zero at 98% of them (e496).
- "The accent and the words form together": the accent by step 512, the words from about step 1000 (e492, e493).

## Session 61 kills (the bridge's control)
- "An SAE feature is a sparse composition of native writes": it needs as many native words as a whole state, half unexplained by sixteen; the rotated dictionary's numbers on features are the native dictionary's numbers on random directions (e497).

## Session 62 kills (the two clocks; the ledger)
- "The composition of a feature's rows predicts its firing better than its top row": on the full sample the 4-16-word ledger is 0.59-0.63 against the top row's 0.62 (e496 v3; v2's 0.70 was a small sample).
- "The accent is a prerequisite for the words": neither part predicts the other at the next checkpoint (e498).
- "SAE features are state-like in every family": on OpenAI's TopK family they sit between covariance-matched directions and states (e499).
- "The two clocks are a Pythia peculiarity": OLMo shows the same order and the same decay of the accent (e498).

## Session 63 kills (row-following)
- "TopK SAEs preserve the model's native write structure better than ReLU SAEs": the difference in row-following is the features' own sparsity; at matched frequency the families are alike, and the family label carries nothing (e501).

## Session 64 kills (the model tested)
- "The accent is the second-order alignment of the state cloud with the atoms": a factor 1.02-1.24 of a 1.3-2.3 excess; the rest is a tail of atoms aligned with the position's large writes or with attention (e503, e505).
- "A word is a write whose own coefficient clears the floor": the own coefficient is 0.3-0.7 of a winning projection; the other large writes and the crowd of small ones decide the rest (e503, e505).
- "The largest writes, summed as written, predict the maximum": twice the level where they are as large as the state and the wrong atom in 84-98% of states; the overshoot is the common direction they share, and the centred chord does predict GPT-2's level within 13% (e506).
- "The ledger is redundant with the top row" (e496): it adds where features are sparse, +0.10 in the sparsest ReLU decile (e502).
- "SAE features are state-like in every model": on Pythia-160m they sit between random and covariance-matched directions (e504).

## Session 65 kills (the profile)
- "The centred chord predicts the whole projection profile": it predicts a third to a half of the top atoms and orders the true top hundred at 0.3-0.4 in GPT-2, calibrated within 13-25%; the sum of all writes does better there and worse in OLMo (e507).
- "The crowd of small writes contracts the chord throughout training": it amplifies it early (+1.4 at Pythia step 512) and contracts it late (-0.64 at the end) (e507).

## Session 66 kills (selection and response)
- "The chord's clean maximum in GPT-2 shows the residual reinforces the winner": it is extreme-value selection over a nearly exchangeable residual; the null reproduces the level, the rank and the winner's residual excess (e508).
- "The early amplification of the chord by the small writes is feedback": at Pythia step 512 it is not a response to the chord (share 0.03-0.21); it is the anisotropy of the early state (e509).
- "The chord is contracted like any direction": a random direction of its size is contracted two to three times more; the learned contraction spares the vocabulary (e509).

## Session 67 kills (the contraction's anisotropy)
- "The network learns to preserve its vocabulary": as a linear response the contraction is isotropic within 25% and native rows are contracted slightly more than random directions, not less; the rows used as words are contracted like the rows never used (e510). The chord's relative sparing (e509) is a nonlinear, pattern-level effect at chord amplitude, present in GPT-2 and late Pythia and reversed in OLMo.

## Session 68 kills (the fake chord, the sweep)
- "The nonlinear response spares native-row chords in general": a fake chord of the block's most-used rows with the position's coefficients is contracted like a random direction (e511). Only the position's own writes are spared.
- "Wordhood turns on in a phase transition": a fast crossover over two doublings of steps followed by a slow creep; sparsification leads it and the contraction and the chord follow (e512).

## Session 69 kills (the origin of sparsification)
- "Sparsification begins with the gradient concentrating on a few rows": the gradient is most concentrated in the first 32 steps and spread by the time the writes sparsify; the rows turn selective together, none die (e513).
- "The accent arrives by step 512": it is born at step 64 with the state cloud's collapse to 9-16 dimensions, and peaks in the collapsed phase (e513).

## Session 70 kills (the lineage)
- "A future word is ordinary at step 256, selective at 1000, aligned at 2000 and a word later": nothing tells the future words from magnitude-matched non-words through step 1000, and against the population they are simply the larger rows from step 1000; among rows of one size selectivity and kurtosis lead usage by one doubling from step 2000 and the three grow together; direction never leads (e514).
- "The early writer wins are MLP rows": the token embedding through step 16, head atoms at steps 512-2000, MLP writers only from step 2000 (e514).
- "The early words are the later words": before step 2000 the future words are used less than their controls; the vocabulary turns over between steps 1000 and 8000 (e514).

## Session 71 kills (the turnover)
- "Wordhood is a fixed threshold rows cross": entrants are more selective and larger than leavers through the transition, and the words are never the largest rows (e515b).
- "The vocabulary is abandoned in an event": its retention tracks the churn of the largest rows at every pair (e515b).
- "The early words become the words": 0.04-0.05 of the step-1000 words survive to the end (e515b).

## Session 72 kills (the boundary)
- "Magnitude, selectivity or kurtosis govern entry into the vocabulary": prospectively they predict the next checkpoint's entrants at 0.5-0.7; the projection over the extreme-value floor does at 0.76-0.97 (e516b).
- "A row enters the vocabulary on its own write": its own write almost never clears the floor; the projection that crosses carries the position's chord, and the own write governs staying instead (e516b).

## Session 73 kills (the counterfactual chord)
- "The chord makes the maximum and the maximum makes the word": destroying, permuting or removing the position's largest writes leaves the prospective prediction of entry within 0.04 of the real state's (e517b). The projection that predicts wordhood is carried by the embeddings, attention and the crowd.
- "The floor's drift is what raises the entrants": the floor moves by a tenth over training; the entrants' projections rise by three fifths (e517b).

## Session 74 kills (the carrier)
- "Attention transports the projection that makes a row a word": attention's part alone ranks the entrants at 0.56-0.68, the weakest of the parts; the MLP writes together are the strongest part, and only the whole state reaches 0.86-0.97 (e519b).
- "One part of the state carries the vocabulary's selection": none does; the parts cancel along most atoms and the words are the atoms along which they agree (e519b).

## Session 75 kills (agreement beyond size)
- "Consensus among the state's parts is a mechanistic variable of wordhood": given the size of a row's projection over the floor, the agreement of the parts that make it adds nothing (within-size AUC 0.43-0.57; equal entry rates near the floor). The criterion is on the resultant alone (e520).

## Session 76 kills (the sources)
- "Writes elsewhere, transported by attention, construct a native direction here": other positions' writes supply 0.00-0.06 of a future word's projection; it is made by the writes at the same position (e521).
- "The vocabulary builds the vocabulary": current words supply 0.02-0.09; the largest sources are non-words from blocks below (e521).
- "A few descendants carry a row to wordhood": the sources are tens of thousands of small writes; the participation ratio is 30,000-120,000 (e521).

## Session 77 kills (the sources counted, the gradient, who moves)
- "Tens of thousands of writes carry a future word's projection" (session 76's own reading of the participation ratio): the ratio measured a cancelling mass sixty times the projection; causally the top thousand writes carry half and the top ten thousand nine tenths (e521b).
- "The loss gradient at the origin selects the entrants": the one-step flow reads entry at chance in every form, with 12k and 65k tokens, for its reliable part (e522, e522b). The row's own gradient direction is unmeasurable at this token count; that half is open.
- "The state moves onto the row": the rise of an entrant's alignment is the row moving toward the states, 0.64-0.78 of the rise at block 12 (e522).
- "A future word defends its maximum at one position": the best position persists for 38-60% of entrants and the rise at the old position is a fraction of the rise of the maximum (e523).
- "Given its size, nothing else about a row's projections predicts entry" (the over-reading of e520): composition at one position adds nothing, but breadth across positions does, the count over the floor at 0.71-0.85 within deciles of S (e523).

## Session 78 kills (the time course)
- "A row locks onto one occurrence": the row's own motion keeps its whole profile (Spearman 0.96-0.99) and its old best position first; the maximum moves because the states move (e524b).
- "Entry is an episode of accelerated rotation": the increments' size falls with training for entrants and matched rows alike; what changes at entry is the direction (the pull toward the best state doubles relative to matched rows) after a steady, directed drift of five thousand steps in which breadth rises first (e524b).
- "The typical row drifts": after step 6000 the typical non-word's consecutive increments are uncorrelated and then anti-correlated (-0.14 by 14000); only the rows near the floor keep a directed motion (e524b).

## Session 79 kills (the gradient at scale)
- "The drift is the integrated instantaneous gradient": at training-batch scale a row's gradient agrees with itself at cosine 0.08-0.18 and the thousand-step update follows it at 0.01; the whitened (Adam-like) direction is no better, the gradient noise being isotropic across a row's coordinates (e525, e525b).
- "The entrants' rows drift because their gradient is more consistent or better aimed": no difference from matched rows in consistency (AUC 0.47-0.59), in the pull toward the best state (cosine 0.00) or in following (e525).
- "The descendant of the row's own write is the object to follow" (the relayed framing): the entrant's own write is absent at its best position (e519) and the response part of the alignment is small (e521); the row's downstream image has nowhere to act. The testable core, the adjoint transport of the loss to the row, is what e525 estimated, and it does not steer the row.

## Session 80 kills (the input side and the curvature)
- "The drift runs in flat directions of the loss": the curvature along the thousand-step update is 1.13-1.15 times a random direction's, the entrants' no flatter than matched rows' (e527).
- "The row rotates toward the writes it co-occurs with": the co-active writes are not a target for any group (cosine 0.00-0.02) (e526).
- "The row's motion is the origin's geometry": the five input-side targets together explain 2-4% of an entrant's thousand-step increment (0.4% for random directions); most of the motion is growth along the row and unexplained (e526).

## Session 81 kills (the common flow, the geometry of drift and oscillation)
- "Vocabulary entry is which rows are positioned to gain from a common anisotropic flow on the row set" (this program's own idea, not the relayed one): the common flow carries a tenth of the motion early and a fortieth late, amplifies the cloud's principal subspace only in the first interval, and reproduces 0.11-0.26 of the entrants' rise before step 5000 and none after (e528).
- "The drift runs in flat directions, the oscillation in steep ones": killed: the eight-thousand-step displacement is 1.09-1.18 times as curved as random, the oscillation axis 1.15-1.17 (e529).

## Session 82 kills (the bound)
- "The direction of the row-specific drift is in the network's geometry at the origin": everything in hand predicts 0.049-0.072 of the eight-thousand-step displacement's energy on held-out rows, all of it growth and read repulsion; the non-radial part at 0.015 or less (e530).
- "The rows move toward where the states end up": the destination's geometry predicts the displacement no better than the origin's (0.035-0.066; 0.001-0.002 without the row) (e530).
- "The entrants' motion is the predictable part": entrants and matched rows are equally unpredictable, and rows of like alignment share no displacement (e530).
- The search for the drift's cause in local quantities, as a line of work: closed by the bound. What is identifiable is the criterion, the time course and the anatomy; the size of the motion but not its direction.

## Session 83 kills (the criterion's specificity)
- "S foretells entry because the native rows are moving into the cloud": the native dictionary frozen at step 4000 foretells entry as well as the moving rows (0.91 against 0.91), with the same turnover (e531).
- "The prospective criterion is a property of the native vocabulary": a fixed random dictionary shows it at 0.81 (four thousand steps) and 0.86 (one thousand); the native excess is 0.10 (e531). This program's own reading of sessions 72-82, corrected by its own control.

## Session 84 kills (what makes the native axes persistent)
- "The native rows' stability is their covariance": atoms with the dictionary's or the cloud's covariance keep 0.19-0.20 of their words over four thousand steps, isotropic atoms 0.18, the native rows 0.50 (e532).
- "A native word is a fine-tuned orientation that the row's drift would destroy": half the words survive a 26-29 degree rotation statically and 32-33 degrees prospectively, the rows' own four-thousand-step rotation being about 32 degrees; the basin is twice a random dictionary's (e533).

## Session 85 kills (the basin's anatomy)
- "The native basin is a local packing phenomenon": the nearest-neighbour cosine reads survival at 0.54-0.56, and the words survive their competitors' random motion (0.79-0.89 kept); only neighbours approaching the word matter (e534).
- "Native words have a wider basin than random words at the same margin": within quartiles of S the two survive alike; the width is the margin (e534).
- "A word's basin is isotropic": rotation toward the row's own future direction or its nearest atom is safer, away from the nearest atom worse, and into the cloud's principal subspace not a loss at all (e534).

## Session 86 (no kills; the record answering the relayed proposals)
- The margin's restoring tendency, the drift along the gradient of the row's own margin, and the shuffled-states control are e524b with e522 and e516b, e523 with e526 and e528, and e532's kernel respectively; the elasticity of retention with respect to margin is e534's records read again (e534b): at equal margin native words are, if anything, slightly less robust than random ones (-0.14--0.05 at 20 degrees).

## Session 87 kills (breaking the criterion)
- "The vocabulary is what OMP makes": under three other definitions of wordhood, one with no reconstruction at all, the prospective criterion, the generic level, the native retention advantage and the entry statistics are the same (e535).
- "The prospective criterion, breadth-first entry, turnover and retention need training": rows diffusing against a fixed Gaussian cloud show all four, at the generic levels of e531 (e536). What needs more than diffusion is the native excess, which a weak persistent pull into the cloud supplies.
- Of the relayed five, the floor under an injected projection is e483 and the state-row transplant is e532; the cross-layer causal state stays open.

## Session 88 kills (the factorial and the replay)
- "Trajectory memory makes the directed drift and the retention": memory with no relation to the cloud gives retention 0.24 against 0.31 for plain diffusion; only a motion directed into the cloud gives 0.52 (e537).
- "Covariance-shaped noise is a form of co-adaptation": it lowers retention (0.18); the geometry that matters is a drift, not a noise shape (e537).
- "The replay with a fixed permutation tests row-specific coupling": it cannot, the increments being larger than the rows; it tests whether the phenomenology rides on the sequences, and it does (e537).

## Session 89 kills (the coupling tests and the exits)
- "The rows' motion is intrinsically anisotropic, and would show the native excess against any cloud of the same shape": against the cloud turned by a fixed rotation the excess vanishes and the words sit at the floor (0.72, 0.13) (e538).
- "A pull that fades toward a target margin makes Pythia's plateau": it makes a slow approach from below, not a jump of 0.11 in one interval followed by a stop (e538).
- "Reversing time tests the direction of the drift": the alignment is with covariance directions that are stable through training, and the reversal pairs late rows with early states; it keeps most of the excess (0.86, 0.45) and says nothing about causes (e538).
- "Words leave as they enter, by the projection crossing the floor": they leave above it, 0.82 of them at the first checkpoint out, by losing usage and breadth with the projection kept (e538b).

## Session 90 kills (the takeover audit)
- "An exit is a competitive displacement": at the positions a leaver loses the field does not move (0.88 to 0.88) and its own projection falls (0.92 to 0.79), in Pythia, for a random dictionary and in the synthetic worlds (e539, e540).
- "The takeover is local, by a few competitors, the leaver's neighbours or an entrant taking its seat": the takers are at the null in every respect and entrants never displace a leaver (e539).
- "The one-way gate is generic": a random dictionary leaves through the floor (0.24 over it at exit), isotropic diffusion likewise (0.37); the gate is the persistence of the native maximum (e539, e540).
- "The erosion of a leaver's marginal positions is its own motion, as the rise of an entrant's maximum was": it is mostly the states' drift (-0.015 of -0.020), the row's part small and in the right direction (e539b).

## Session 91 kills (what holds the maximum)
- "Breadth alone holds the native maximum": independent churn from the isotropic kernel reproduces random words' maxima and under-predicts the native (1.19 against 1.28) (e541).
- "Peak memory": the states at native peaks persist as everywhere else (0.926 against 0.925), and the maximum relocates at the random rate (e541).
- "Usage erodes in chunks": the lost positions are the least coherent of a word's positions, not a chunk (e541).
- "A persistent maximum is a stable word": covariance-matched atoms have the most persistent maxima and churn as words below the floor (e541b).

## Session 92 kills (the cloud's persistence)
- "The persistent subspace is small and the native rows sit in it": 277 directions persist above 0.9, and a native row's whole profile persists like an isotropic atom's (0.905 against 0.899) (e542).
- "Slow dynamic modes behind the variance modes": the two top-32 subspaces overlap at 0.95 (e542).
- "The rows lead the cloud": their weight in the final subspace is below their contemporaneous weight early, and entrants enter the contemporaneous subspace (e542).
- "A Gaussian cloud with the real covariance and the real dynamics makes the native vocabulary": at any coupling it makes words at S 1.08 with retention 0.37-0.42 (e543).

## Session 93 kills (the tails)
- "The bulk of the real cloud carries part of the native fifth": with the top 16 projections per position Gaussianised the real states fall below a Gaussian cloud (retention 0.16 against 0.35) (e544).
- "The vocabulary needs the extremes to persist at their positions": moved to random positions at every checkpoint the grafted extremes give the same vocabulary (retention 0.51, gate 0.89) (e544b).
- "The shape of the tail predicts entry": kurtosis 0.58 within the band against breadth 0.86 (e544).

## Session 94 kills (the writers of the extremes)
- "An extreme is a few large writes": 9232 effective writers, the top ten 0.01 of the absolute contribution, a cancelling mass with more than half of the projection transported by attention (e545).
- "The writers behind a row's extremes are the big writers of any direction": random directions' ensembles cohere at 0.083 against the native rows' 0.489 (e545).
- "A few shared write programs make many words": the words-by-writers matrix has effective rank 63.6 of 247, each word its own ensemble (e545).

## Session 95 kills (the coalition before the word)
- "Entry is the ensemble forming": it is in place four intervals before (0.42 against a near miss's 0.20); what rises is coherence and alignment (e546).
- "The rise is the row's rotation alone": a quarter is the row, a quarter the arrival of transported writes, an eighth the local writers' motion (e546b).
- "The ensembles' persistence is a selection artefact of stable contexts": the selection-matched null gives 0.599 against the real 0.873 (e546).

## Session 96 kills (the channel and the neuron)
- "A coalition selected by its alignment with the row can tell which came first": random non-words give the same stable channel, the same approach and the same displacement cosine as entrants; my own construction, retired (e547).
- "The words' coalitions are drawn from a shared population of programs": 176 of 256 words are alone at cosine 0.3, and the families are near-duplicate rows at the same contexts (e547).
- "What separates a word from a near miss is the row's approach": both approach alike (0.174 and 0.140); the near miss's neuron loses its selectivity for its contexts (e547b).

## Session 97 kills (the necessity test)
- "WDD measures something simpler, the selectivity of a neuron": six ordinary neuron statistics predict entry at 0.74 out of sample, S and breadth at 0.88, and the activation side adds +0.001 to WDD (e548).
- "A word's extreme contexts are where its neuron fires most": the row's alignment at the neuron's peak contexts predicts entry at 0.64; the extremes are the coalition's contexts, where the neuron fires moderately (e548).

## Session 98 kills (the feature-side test)
- "A feature reader recovers WDD's row-context associations": no feature matches a word's context set as a set (Jaccard 0.19, 0.28 of words at 0.25 against 0.17 of random rows); the feature nearest to the row covers half of the contexts (e549).
- "The feature finds the class but not the direction" (the relayed take's cleanest case): the nearest feature is at cosine 0.48 with the row and at half of its contexts; both are seen in part, neither in full (e549).

## Session 99 kills (the second model)
- "The word's own neuron is selective for its contexts before entry, in general": on OLMo-1B the entrant's neuron is not (z 0.06 at k=-8, 0.68 at entry) and its own write is 0.03 of the extreme; the coalition alone makes the word (e552).
- "Attention supplies more than half of an extreme, in general": 0.14 on OLMo-1B against 0.54-0.57 on Pythia (e552).
- "The coherence of the contributions separates entrants from near misses, in general": 6.56 against 8.55 on OLMo-1B (e552).

## Session 100 kills (the anticipated critique of session 99, run)
- Session 99's own clause, "the word's neuron's part in the coalition is architectural": by absolute activation the OLMo entrant's neuron is selective eight intervals before entry (z 2.63) and is the top contributor to its extreme; the signed statistic misled (e554).
- "Attention's half of an extreme is about extremes": 0.27 at OLMo's extremes and 0.27 everywhere (e554).
- "A row above the floor is a word": 3312 non-words over the floor on Pythia at step 8000, 4252 on OLMo; words live at 1.1-1.3 times the floor (e554).
- "The native fifth's retention lives in the tails, in general": on OLMo the covariance keeps 0.58 of 0.64 (e555).

## Session 101 kills (GPT's map, the nine uncovered directions run)
- "The native vocabulary depends on the optimizer's geometry": AdamW, sign momentum and clipped SGD grok to the same advantage (0.76 / 0.73 / 0.72) (e556).
- "Cross-model twins are token identities": twinned words' purity 0.30, most twins impure (e557).
- "The final vocabulary is the step-16000 vocabulary, row for row": Jaccard 0.21 by index, 0.55 by context sets (e557b).
- "Entry is the cloud lifting the row": the 16000 direction already at 1.29 in the 8000 cloud; the row turns, the cloud was there (e558b).
- "The coalition precedes the word, so a transplanted neuron cannot bring it": the whole neuron makes 0.68-0.80 of entrants words (e558).
- "The ledger detects corruption": AUC 0.504 for the codes; no single state does beyond two tokens (e559).
- "The ledger sees factual errors": S right / wrong 1.41 / 1.42; the confidence does (0.85) (e560).
- "The advantage is smaller in a state-space model": Mamba's 0.258 against Pythia-160m's 0.188 (e563).
- "Fine-tuning protects the vocabulary (or targets it)": at matched loss damage noise and fine-tuning move the word set alike; the rows rotate by 1e-4 (e564, e564b).
- "A word set's Jaccard is a stable statistic": 0.86 under noise costing +0.03 nats; the usage correlation (0.99) is the stable one (e564b).

## Session 102 kills (theory extraction: the arrows tested before the map)
- "The neuron's input selectivity is what a transplant brings": the read row alone makes 0.18 of entrants words, the write direction alone 0.54 (e565).
- "The Pythia weight-seed variants are independently initialised": their step-512 rows match main's at 0.997 (e566d).
- "The invented-name row is a novelty detector": a generic function-word row that real people's names drive too; zeroing it changes nothing (e568).
- "The cloud comes to the row": the row turns toward the cloud's variance, the cloud does not grow along the row (e571).
- "The words span the states better than other rows": norm-matched random rows describe them equally at K=8 (e569b); what is the words' is the usage, the extremes and the twins (e572).
- "Width, decay, learning rate or initialisation scale set the association": only whether the run generalises does (e567).

## Session 103 kills (the object: class, direction or row)
- "The class is the object and the row its label": the direction carries the downstream effect (effect cosine tracks direction cosine at 0.86); a class that changes rows changes its effect (0.156 against 0.449) (e573, e573b).
- "The partition is an output partition": predicted-token purity 0.22 against input purity 0.24 (e574).
- "The vocabulary is the interface later blocks read": words are read no more than norm-matched or random rows (e574).
- "Two vision models share a partition of patches": 0.02 at the patch level; the spatial layout is what they share (e575, e575b).
- "The row can bring its class to a fresh network": frozen trained rows draw usage but not the base classes (profile correlation -0.01) (e576).
- "The transplanted direction carries computation": KL 0.0021, away from the later model (e577).

## Session 104 kills (what has to travel, and what the instrument is for)
- "The direction carries the function" without qualification: the coalition and the direction together close -0.022 of the gap at the class, the readout -0.873; the effect is the direction's, its producers' and its readers' together (e578).
- "The producer coalition is the transplantable computation": 128 neurons close -0.020 of the gap, 512 with the direction -0.066 (e578).
- "The vocabulary statistic predicts grokking": it fires at the step test accuracy crosses one half, not before (lead 0.0 steps), though it never fires in a memorising run (e579).

## Session 105 kills (the strongest existing method invited)
- "The coalition is what native rows have that a learned dictionary does not": autoencoder features have coalitions as coherent (0.692 against 0.672) and as necessary (0.92 against 0.77) (e581).
- "WDD is a better feature-discovery method": the descriptive decomposition is recoverable by attribution from an autoencoder's directions; what the autoencoder lacks is the parameter, which carries the association and not the computation (e581 with e577, e578).
- "The native excess is an early predictor of generalisation": at delta 1000 the best WDD statistic reaches AUC 0.824 against 0.961 for the best classic baseline (e580b).

## Session 106 kills (the developmental comparison)
- "Only the native dictionary sees the class early": where the per-checkpoint autoencoder has a feature for a class (0.31 of classes) it has it before the row is recruited (0.61); and the class is present in the cloud from the first checkpoint for either dictionary (e584).
- "The learned dictionary can date when a parameter becomes responsible": its decoder points at the row 7.0 checkpoints after recruitment, for 33 of 241 words (e584).

## Session 107 kills (responsibility migration, read directly)
- "Responsibility relays between rows": one speaker at the median, a gap before it, handoffs rare and passed through overlaps (e585).
- "Recruitment is a lottery among rows": the recruit is the first of 53,248 rows by its ratio over the class two checkpoints ahead for 0.52 of words, top ten for 0.84 (e587).
- "The class needs its row": with the eventual words frozen the toy still generalises (1.00) with a vocabulary on other neurons; a third of the classes re-appear on substitutes (e586).
- "One more autoencoder control is what the developmental claim needs": the three facts above need none.

## Session 108 kills (the forecast)
- "The forecast needs the future in the class": row-free clusters at t forecast recruitment at AUC 0.927 / 0.932 two checkpoints ahead (e588).
- "The class side adds to the row's own maximum": the combination adds +0.006 / +0.003 over S alone; the maximum already measures the pull (e588).

## Session 109 kills (the last mechanistic hole, the scale test, the paper)
- "One gradient step shows the turn": the step points at the class at mean cosine +0.020, consistent in sign and driven by the class's positions (0.215 of the gradient at 0.006 of positions), but a drift, not a jump (e589).
- "A removed speaker is replaced within a few million tokens": 0.17 of the classes have a substitute after 2.5M tokens against 0.30 unfrozen; the proxy cannot reach a real interval (e590).
- "WDD needs to be the hero of its own record": the paper is written with the network as the subject and WDD as the instrument (PAPER.md).

## Session 110 kills (the six directions beyond the descendants)
- "Recruitment is a performance event": the class drops -0.1231 across recruitment and its token-matched controls -0.1380; the direction's removal effect is 0.0047 nats two checkpoints before and 0.0109 two after (e591).
- "Unlearning reaches the rows": cosine 1.0000 after capped gradient ascent to +2 nats, the rows changed 0.89 times as much as rows of the same norm; what falls is their writing at the forgotten classes (0.47 still write) (e592b); e592's runaway run (forget loss to 137 nats) is a killed protocol, not a result.
- "S knows the recruits before recruitment": the eventual words sit among the lowest-S half for 0.62 at 1500 and 0.69 at 3500; pruning helps grokking whatever is pruned (e593, e593c).
- "Twins anchor a stitch": no better than random positions (e594).
- "A better refit fixes the magnitude bias": the bias is the support's (e596).

## Session 111 kills (the audit tested as its users would)
- "The still-writing share grades unlearning depth": Spearman 0.39 with relearning across five methods; RMU is lowest on it and shallowest (e597).
- "A linear probe would do the same job": it saturates after one nat of ascent and stays high under RMU (e597).
- "4-bit quantisation undoes shallow unlearning" at 160m with round-to-nearest weights: no recovery, the quantisation itself costs more (e597).
- "Fine-tuning recruits new writers by rewriting rows": entrants arrive with their rows unchanged (e600).
- "WDD can unlearn by removing the writers it names": +0.08 nats for the writers, the coalition carries the classes (e601); what survives is the hybrid, removal after silencing, which slows the relearning.

## Session 112 kills (the audit's baselines, causal test and three new audits)
- "The dictionary is what makes the audit work": a naive difference selection predicts relearnability at 0.83 against the writers' 0.89 (e602).
- "Safety training rewrites the writers of harmful requests": rows at cosine 0.9973 with the base (e604).
- "Jailbreaks silence the harmful-request writers": activation 0.97 / 0.98 under the two prefixes (e604).
- "Merged fine-tunes collide on the same rows": entrants' Jaccard 0.02 (e605).

## Session 113 kills (the refusal direction, writer rescue, 410m, 1B)
- "The harmful-request writers write the refusal direction": cosine 0.034 against 0.024 for any row, 0.11 of the mass (e607).
- "Restoring the silenced writers restores the forgotten domain": 0.09 of the rise recovered after ascent (e609).
- "The forgetting of a gradient method lives in the MLP activations of blocks 0-B": restoring all of them recovers 0.02 after ascent (e609).

## Session 114 kills (the J-Lens connection)
- "The logit lens of a write (the raw row through the unembedding) predicts what the write does": cosine 0.065 at block 4 against 0.283 for the class-transported row (e611).
- "A transport map can be estimated by perturbing every position at once": the all-positions map reads 0.071 where a random subset of the class's size reads 0.210 (e611).
- "Transport would rescue the writers' role in refusal": transported or not, silencing them leaves refusal at 0.89 (e612).

## Session 115 kills (the auditor against the state of the art)
- "The audit predicts how many steps a relearning takes": AUC 0.71 for back within 40 steps; the forget rise does that (0.81) (e616).
- "The still-writing share is a usable rule": leave-one-method-out accuracy 0.33 against a majority baseline of 0.57 (e616).
- "TOFU's forget set is a domain in the dictionary's sense": 2 of 256 words pass the 0.7 forget-share threshold on the fine-tuned model; the 32 most forget-leaning words average 0.62 where neutral is 0.52 (e613, e613b).
- "The writers' 0.76 with the recovery on TOFU is a result": the ranked writers read 0.99 to 1.00 in all 12 conditions; the ordering of RMU's 1.00 against 0.99 (e613b).
- "A membership test certifies depth": on TOFU the membership AUC correlates with the recovery at -0.57; RMU makes the forget pairs look like non-members (0.47) and relearns at 1.00 in 10 steps (e613b).
- "A page of the forgotten domain in the prompt undoes an override": in-context recovery after RMU 160m -0.02 / -0.12; 410m -0.04 / -0.22; 1B -0.13 / -0.37 (PubMed / Github), and the prefix deepens ascent's rise by up to +0.94 nats (e614).
- "The silenced writers wake up in context": 0.69 without context, 0.64 with it, after ascent on PubMed at 160m (e614).

## Session 116 kills (the pre-registered held-out test)
- "The writers' rule transfers to new domains at the fixed cut": 0.62 on 86 held-out-domain conditions against a majority of 0.55 (e617b).
- "The fixed rule holds on unseen methods": 0.51 on task-vector negation and SCRUB (e617b).
- "The fixed rule holds on another architecture": 0.73 on Qwen2.5-0.5B (e617b).
- "Silencing the writers is the intervention that delays relearning most": +2 steps against +2 for the difference-selected set (e617b).
- "RMU and task-vector negation leave the writers firing while the gradient methods silence them": ga writers 0.98, recovery 0.88 in 40 steps; gd writers 0.96, recovery 0.74 in 95 steps; npo writers 0.96, recovery 0.85 in 35 steps; rmu writers 1.00, recovery 0.89, not back within 100 steps; scrub writers 0.99, recovery 0.94 in 30 steps; ssd writers 0.43, recovery 0.35, not back within 100 steps; tv writers 0.90, recovery 0.49, not back within 100 steps (e617b).

## Session 117 kills (the statistics, the missing controls and the per-word test)
- "The threshold failure is a scale problem": the forget-over-retain normalised score tunes to 0.781 and scores 0.60 on the held-out domains (e618a).
- "The writers and the naive set are indistinguishable as rankers": on the held-out domains the difference-selected set is ahead by +0.05 [+0.02, +0.10] (e618a).
- "Silencing works because the neurons are selective, not because they are active": an active-but-not-selective set adds +31 steps against the writers' +56 (e618b).
- "The route back is distributed, so silencing two disjoint sets adds up": the union adds +56 against +56 and +56 (e618b).
- "The dictionary localises the failure": median per-word Spearman -0.06 (e618c).
- "The word structure is what localises": the difference-selected set read at the same positions reaches 0.08 against the writers' -0.06 (e618c).
- "Silencing the named writers after an unlearning slows the relearning" (H526, H528, H534; sessions 111 to 113): the steps-back criterion measures distance to the unsilenced original, and zeroing a set lifts the floor by +0.14 nats; against the floor a 20-step relearning recovers 0.76 of the recoverable rise with the writers silenced and 0.75 unsilenced over 72 held-out conditions (e618e).

## Session 118 kills (three parameter-side directions)
- "Chat tuning recruits rows without rewriting them, and few of them": 158 instruct-only words with row cosine 0.9973 (e619).
- "The instruct-only words carry refusal": zeroing their inputs leaves refusal at 0.92 against 0.85 for as many shared words (e619).
- "Chat tuning re-aims and amplifies the shared words at the harmful prompts": shift +0.001, amplification 0.75 / 0.79 (e619).
- "The ledger penalty cuts the effective writes by 30% at 0.05 nats or less": 0.96 at +0.203 (e621).
- "The penalty grows the own-word edge at k = 8": native +0.04, rotated -0.04 (e621).
- "The gain persists at k = 16 and 32": +0.02, +0.00 (e621).

## Session 119 kills (the attribute catalogue)
- "Each dominant row flips its attribute on held-out templates, above the naive neuron": held-out flips gender 0.28 against naive 0.07, generation 0.13 against naive 0.00, number 0.12 against naive 0.00, polarity 0.00 against naive 0.00 (e622).
- "The collateral is diagonal at five to one": gender +2.00 against 0.14, generation +1.09 against 0.03, number +0.26 against 0.06, polarity -0.03 against 0.01 (e622).
- "The dictionary's class predicts where a row edit damages general text, beyond the neuron's activation": the damage on Pile text lands where the neuron fires: ranking positions by the neuron's activation predicts the loss change at AUC 0.98 (gender), 0.99 (generation), 0.49 (number), 0.73 (polarity), while the rows have 0, 1, 60, 2 class positions on the 16 windows, so the floor-based prediction is n/a, 0.99, 0.56, 0.90; the collateral is tiny in any case (Pile loss +0.0003, -0.0002, +0.0005, +0.0003) (e622).
- "The effect scales with the row's coefficient": Spearman gender 0.03, generation 0.18, number -0.22, polarity -0.34 (e622).
- "The gender row exists and flips on the 1.5B model": on Qwen2.5-1.5B-Instruct the dominant rows are gender, generation, number; gender: row 8031 of block 7 (vote 0.75, rank 0 by activation difference) negated flips 0.00 of discovery and 0.00 of held-out items, shifts the score +0.58 on held-out items (naive neuron 0.00, +0.35; random 0.00), other attributes at most 0.02, Pile loss -0.0002; generation: row 8892 of block 7 (vote 0.60, rank 0 by activation difference) negated flips 0.10 of discovery and 0.14 of held-out items, shifts the score +2.16 on held-out items (naive neuron 0.00, -0.10; random 0.00), other attributes at most 0.03, Pile loss -0.0001; number: row 3756 of block 12 (vote 0.87, rank 1 by activation difference) negated flips 0.02 of discovery and 0.00 of held-out items, shifts the score +0.07 on held-out items (naive neuron 0.01, +0.59; random 0.00), other attributes at most 0.00, Pile loss -0.0001; polarity: row 54 of block 3 (vote 0.33, rank 0 by activation difference) negated flips 0.03 of discovery and 0.01 of held-out items, shifts the score +0.37 on held-out items (naive neuron 0.00, +0.30; random 0.00), other attributes at most 0.01, Pile loss +0.0004 (e622).

## Session 120 kills (blind edit detection and the dial in free text)
- "A sign-flipped word row can be found blind by its disagreement with its chord": the row-chord cosine is -0.004 on unedited rows, so there is nothing to disagree with; top-10 0.00 (e623).
- "The dictionary's detector beats the naive one": top-1 0.00 against 0.00 (e623).
- "The naive detector finds negated non-word rows": top-10 0.00 (e623).
- "Two attribute rows compose in the weights": both flipped in 0.05 against 0.34 and 0.17 alone (e624).

## Session 121 kills (the cross-lingual dial and programmable feature placement)
- "The 1.5B gender row reaches another language": on Qwen2.5-1.5B-Instruct the gender row negated moves agreement from 0.94 to 0.94 (French), 0.98 to 0.97 (Spanish), 0.95 to 0.95 (German); zeroed 0.94, 0.98, 0.95, doubled 0.94, 0.98, 0.95; the largest change from a random row is 0.00, 0.00, 0.00; Pile loss -0.0002 (e625).

## Session 122 kills (the pinning checks)
- "The pinned row's dial is not damage": three designated rows (seeds 0, 1, 2) take the concept at vote shares 1.00, 1.00, 1.00 with dials -0.029, +0.027, -0.006, each at a Pile cost on negation of +0.0009, -0.0019, +0.0017 nats and a norm that goes from 0.81, 0.48, 0.57 to 0.88, 0.55, 0.64; the handle's dial goes from +0.343 to +0.170, +0.165, +0.246 (control fine-tune +0.403); eight random rows move the pronoun by at most 0.004, 0.004, 0.002 (e627).
- "Any quiet row takes the concept": shares 1.00, 1.00, 1.00, dials -0.029, +0.027, -0.006 (e627).
- "The placed concept survives the pin's release": after 1,200 further steps of the next-token loss alone the pinned row's share goes from 1.00 to 1.00 and its dial from -0.048 to -0.018 (the handle's +0.175 to +0.220; validation loss 3.414 to 3.465) (e627).
- "Two designated rows take two concepts without cross-talk and compose": two rows pinned at once: the gender row 1542 takes share 1.00 and dial -0.004, the age row 291 share 1.00 and dial +0.035 (old over young on held-out nouns, 0.58 unedited); the cross-dials are -0.008 (gender row on age) and -0.009 (age row on pronouns); both negated give +0.008 and +0.017 (e627).

## Session 123 kills (class-based data attribution)
- "A short continuation shows the real run's recruitment": 0.28 / 0.20 of the recruits are words after 2,500 steps on the full stream (e628).
- "Removing a class's contexts stalls its row and no other": A 0.28 to 0.20 with B 0.20 to 0.24; B 0.20 to 0.32 with A 0.28 to 0.28 (e628).
- "Random removal does nothing": 0.20 / 0.32 against 0.28 / 0.20 (e628).
- "Removed classes get fewer substitutes": 0.08 / 0.00 against 0.04 / 0.00 (e628).

## Session 124 kills (literature review, no experiment)
- "The row is a unit the literature does not have" (session 121): the neuron is the literature's oldest unit and the down-projection column its write side (Lakretz 2019; Geva 2022; Gurnee 2023; Voita 2023).
- "The developmental ordering is unprecedented": positions-first, direction-later and early forecastability are in Stecher et al. 2026 and SAE-Track 2024 for autoencoder features; only the parameter-level parts survive (H593).
- "The audit is a new diagnostic": its substance is in Xu et al. 2025/2026, Song et al. 2026 and Goel et al. 2026; it is a replication with a transfer test (H596).

## Session 125 kills (the forecast head-to-head and the planted row)
- "the feature route that names a parameter by decoder cosine forecasts at least 0.15 below the class-side score at two ahead": decoder cosine 0.810 against 0.886 (e629).
- "adding the three feature-side scores to S and the class-side score raises the leave-one-checkpoint-out AUC by less than 0.02": 0.869 to 0.910 (e629).
- "two checkpoints before recruitment the recruit is in the top 10 of all rows by class ratio for 0.7 or more of recruits, and in the top 10 by the best feature's decoder for 0.3 or fewer": 0.79 / 0.33 (e629).
- "enriched stream: it rises by 0.2 or more over the no-training readout": 0.50 to 0.40 (e631).
- "plant_both: the designated row's mean share of its class over the floor rises to 0.5 or more and it is the substitute for 0.4 or more of the classes, while the designated rows in the unplanted enriched condition stay under 0.05 on both": over-floor share 0.00, substitute for 0.00; unplanted 0.00 / 0.00 (e631).
- "plant_frozen's rise in the designated row's over-floor share is less than half of plant_both's: the row's own motion is needed, the states' motion alone does not carry it": the pre-registered measure, the share of the class over the floor, never moved (frozen 0.00 against trainable 0.00); see H615 for the drift in the largest ratio (e631).
- "plant_dir rises less than plant_both, and plant_read rises less than plant_dir: the direction does most of the work and the read side adds to it": both 0.00, direction 0.00, read side 0.00 (e631).
- "under the enriched stream the loss at the class positions recovers more than half of its rise on removal": recovery n/a (e631).
- "where the designated row is recruited its own write's share of the state's projection on it rises from about 0.27 to 0.5 or more": median own share among recruited designated rows none recruited (e631).
- "The recruit is forecast from the class side at AUC 0.93 and that is the result" (PAPER.md step 2): the row's own usage forecasts better in every run (0.923 against 0.886 here; 0.964 / 0.968 against 0.927 / 0.932 in e588); the class side is the explanation of the recruit, not the better forecaster (e629, e588).

## Session 126 kills (literature review, no experiment)
- "WDD is a new instrument": Head Pursuit (Basile et al., NeurIPS 2025) decomposes head writes by simultaneous orthogonal matching pursuit onto the model's own unembedding rows, with no training, and the WDD paper did not cite it (H616).
- "The nearest-row law is a discovery": it is a prediction of small-initialisation theory and a theorem for autoencoder training; the program measured it in pretraining (H617).

## Session 127 kills (lottery or recruitment)
- "at step 0 the eventual row's rank among all rows by cosine with its final class direction is at chance on all three seeds: median normalised rank between 0.35 and 0.65 and top-100 share 0.02 or less": step-0 normalised rank 0.15 to 0.21, random 0.51 (e632).
- "at step 0 the eventual row's rank by activation selectivity for its class positions (the read side) is at chance by the same criteria": step-0 normalised rank by selectivity 0.33 to 0.38 (e632).
- "the lineage, the row staying in the top tenth of rows by cosine with the class direction, begins at step 1000 or later for most eventual words, not in the log-spaced steps up to 512": in the top tenth and staying by step 512 for 0.43 / 0.52 / 0.44 (e632).
- "the step-0 rank does not predict the recruitment checkpoint (Spearman between -0.15 and 0.15)": Spearman 0.07 / 0.19 / 0.02 (e632).
- "two checkpoints before recruitment the recruit's read-weight norm exceeds the median of its ten nearest competitors' for 0.6 or more of the eventual words (Wang 2026: the balance of input and output weights decides which aligned neuron takes the feature)": 0.49 / 0.48 / 0.38 against chance 0.5 (e632).
- "its write-column norm does not: the share is 0.55 or less (e572: a large row but not because it is large)": 0.65 / 0.53 / 0.52 (e632).
- "given the class, the recruit is in the top ten of non-word rows by mean class ratio for 0.7 or more two checkpoints before, and in the top ten by usage, which is class-blind, for 0.3 or fewer": top ten by class ratio 0.84 / 0.78 / 0.77, by usage 0.31 / 0.22 / 0.26 (e632).
- "with the row's own write subtracted from the class states, the step-0 normalised rank by cosine stays below 0.2: the ticket is real, not the row's own persistence": step-0 rank 0.32 / 0.32 / 0.33 (e632).
- "cross-seed: for seed 0's classes carried to seeds 1 and 2 by position, the speaker row found there has a step-0 normalised rank below 0.2 in its own seed, where the circularity cannot arise": speaker's step-0 rank 0.12 / 0.23, random 0.52 (e632).

## Session 128 kills (the ticket at initialisation)
- "Initialisation biases which row is recruited" (session 127): the observed initial alignment is what the row's persistence (0.13 at 16000) and the class direction's correlation with the final row produce, and for the own-write-subtracted direction persistence predicts a stronger ticket than was observed (H628).

## Session 129 kills (calendar rings)
- "phase coding: the twenty most day-selective rows per day write at phases aligned with that day's angle in the ring plane, mean cosine 0.5 or more (random rows give about 0)": Pythia-160m -0.04; present at 0.17 to 0.37 against random near 0 in GPT-2, Qwen2.5-0.5B, SmolLM2-135M and OLMo-1B, absent in both Pythias (e633).
- "the MLP writes carry less than half of the ring's between-day pattern at the best block; the embedding and attention carry the rest": they carry 0.69 to 1.00 of it; the embedding's own ring is abandoned by block 2 (e633).
- "the same holds on GPT-2 small and Qwen2.5-0.5B for M1 and M6 (cyclic order present), with the best block in the first half of depth": rings yes on both; GPT-2's best block is 10 of 12 by the tie-break, with every block in full order (e633).

## Session 130 kills (the calendar rings' edges)
- "it is a population, not a handful: the alignment decreases smoothly with the number of rows taken (top 5 above top 20 above top 100 above top 500) and the top 500 is still above the shuffle null." (e634).
- "the embedding's ring is outgrown rather than erased in Pythia-160m's weekdays (the absolute energy of the block-2 pattern projected on the embedding plane is at least 0.7 of the embedding pattern's own energy) and erased in Qwen2.5-0.5B's months (below 0.3)." (e634).
- "the day-arithmetic task (Today is X. Tomorrow is) is solved at 0.7 or better among the seven day tokens by Qwen2.5-0.5B and OLMo-1B, and at 0.3 or worse by Pythia-160m and GPT-2 small." (e634).
- "the population is causal: zeroing the write columns of the top-20 phase-coded rows per day (their union, at most 140 rows) halves the ring's radius at the best block and lowers day-arithmetic accuracy by 0.3 or more where the task is solved, while the same number of random rows changes neither by more than 0.05." (e634).
- "OLMo-1B's four month rows are real: selected on one half of the templates they carry 0.4 or more of the month pattern on the other half, and their write directions put month tokens among the top 20 of the unembedding reading." (e634).
- "ablating the rows that write half of the MLP ring pattern (their write columns zeroed) removes 40 percent or more of the ring's radius at the best block with the plane refitted, on Qwen2.5-0.5B, GPT-2 small, OLMo-1B and SmolLM2-135M." (e635).
- "dose: the radius falls monotonically with the attributed fraction removed (rows for 25, 50, 75, 90 percent)." (e635).
- "compensation: after the writers for half are removed, other rows' writes recover half or more of the lost pattern amplitude (the recomputed MLP pattern is at least 75 percent of the original), a self-repair of the manifold." (e635).
- "the month task on Qwen2.5-0.5B (0.92 at baseline) falls by 0.2 or more when the writers for half of the month ring are removed, and by 0.05 or less under random rows." (e635).
- "in the three Pythias the day-selective rows phase-code against the embedding plane's angles (0.15 or more, above the shuffle null) although not against the best block's plane." (e636).
- "in the sequential models the per-block phase coding against the best plane rises with depth toward the best block; in the Pythias it stays flat near zero at every block." (e636).
- "the selective rows sit mostly in the blocks just before the best block in the sequential models, and mostly in the first two blocks in the Pythias." (e636).
- "specificity: in Qwen2.5-0.5B months, GPT-2 small days and months, and SmolLM2-135M days, removing the writers for half lowers task accuracy by at least 0.15 more than removing the same number of rows matched on write magnitude at the same positions but with below-median ring attribution." (e637).
- "not general damage: the writers' increase in the general loss (on the templates and on neutral text) is within 0.1 nats of the activation-matched control's." (e637).
- "the activation-matched control leaves the ring's refit radius within 15 percent of baseline while the writers remove 30 percent or more, in those four cases." (e637).
- "top-norm rows raise the general loss by more than the writers do: their task damage is general." (e637).
- "with quantile-matched low-attribution rows the writers' forward-task drop exceeds the control's by 0.15 or more in GPT-2 small months, SmolLM2-135M days and months, and OPT-125m months, and the control's general loss change is within 0.05 nats of the writers'." (e638).
- "calendar-specific: removing the other calendar's writers lowers the forward task by less than half of what removing the own writers does, in those four cases." (e638).
- "the reverse task falls under the writers' removal by at least as much as the forward task, in those four cases." (e638).
- "the writers sit mostly outside block 0: fewer than half of them are in block 0, in all models." (e638).
- "scale: in GPT-2 medium and large and OPT-1.3b the writers' forward-task drop is 0.15 or more with a general loss change of 0.1 nats or less." (e638).
- Session 129's reading that the embedding's ring is abandoned by block 2: it is carried and amplified in place; the share fell because the state grew (e634).
- Session 129's reading that the blocks build the ring: in sequential models its plane is laid down by block-0 detectors (e636).

## Session 131 kills (the writers rotated)
- "the readout is angular and the writers set it: under the 90-percent forward rotation the forward task's +2 share rises to 0.3 or more from under 0.1 at baseline in GPT-2 small months, SmolLM2-135M days and OPT-125m months." (e639).
- "dose: the +2 share rises monotonically over the 25, 50, 75, 90 percent sets in those three cases." (e639).
- "backward: rotating the 90-percent set backward raises the 0 share (today) to 0.3 or more in those three cases." (e639).
- "dilution: the shift at the last block is at most half of the shift at the best block in GPT-2 small (later blocks and attention re-express the un-rotated class)." (e639).
- The reading that the ring is the medium of calendar arithmetic in these small models: its phase can be turned by about a class step at the parameter level and the answers mostly do not turn with it (e639).

## Session 132 kills (the name token's readout; classes defined in the prompt)
- "the complement of the class-mean subspace is followed 0.1 or less at every depth, in every pair." (e640).
- "the ring plane beats a random plane: the ring-plane patch is followed more than the mean of the two random planes inside the class-mean subspace, in at least two thirds of the pairs." (e640).
- "a circular readout: at the best block the angle sweep sends the answer to the class after the angle's class in 0.5 or more of the off-cue sweep, in at least three of the five causal pairs." (e640).
- "when the angle moves the answer it moves it around the cycle: among sweep points whose answer departs from the baseline answer, half or more land on the class after the angle's class, in the pairs where 0.2 or more of the sweep departs." (e640).
- "behaviour: asked 'The X is in group', each model picks the object's letter over the other at 0.9 or more." (e641).
- "gate at the second neutral mention: the identity code decodes the group at 0.75 or more and above the random-split null at half depth, in at least 4 of the 6 models." (e641).
- "borrowed: where the identity gate passes, the identity direction aligns with the letters' own contrast (median cosine 0.3 or more) at half depth." (e641).
- "slots: the ordinal directions agree across prompts above the 95th percentile of the null and their average's s exceeds 0.95 of its null, in at least 2 models." (e641).
- "graded: the ring plane's following rises by 0.2 or more from r = 1 to r = 4, in at least two thirds of the pairs where it is followed 0.3 or more at r = 4." (e642).
- The reading that the calendar arithmetic reads the ring's angle at the name token: the class-mean subspace carries the readout as a whole, random planes in it steer the answer too, the ring plane's lead is clear only for the months, and in half of the pairs the name token stops mattering before the ring's best block (e640, e642).

## Session 133 kills (stored or held)
- "storage works: both stored twins answer the held-out query 'The X is in group' with the object's letter at 0.9 or more, with the neutral-text loss rising by 0.1 nats or less, in every run." (e643).
- "the stored class has a native word: its s is above 0.95 of the random splits at half depth, in at least 3 of the 4 runs." (e643).
- The native-word statistic as a reliable thermometer of where a class lives at this scale: a stored class clears the word bar in 1 run of 4 (e643).

## Session 134 kills (the planted-change bake-off)
- "the WDD usage ranking erases more than random rows at k = 128, in at least 3 of the 4 runs." (e644).
- "the WDD and activation rankings agree: their top-128 rows overlap at a Jaccard index of 0.3 or more, in at least 3 of the 4 runs." (e644).
- The native dictionary as an immutable coordinate system for diffing fine-tunes (Gemini 4's pitch): the rows WDD's word usage flags carry less of a planted change than the plain activation diff or the weight change, and no more than random rows in two of four runs (e644).

## Session 135 kills (the subspace floor; stored, trained longer)
- "against covariance-matched planes it exceeds the 95th percentile in at most half of the pairs: the anchoring is mostly the variance." (e645).
- "against the spectrum-matched null the class-mean subspace's largest row projection exceeds the 95th percentile in at most half of the 22 pairs." (e645b).
- "excluding block 0, the class-mean subspace exceeds its spectrum-matched 95th percentile in fewer pairs than with block 0." (e645b).
- "the stored class's percentile at half depth rises by 0.2 or more from the 20-step snapshot to the 1280-step snapshot, in at least 3 of the 5 runs." (e646).
- "the neutral-text loss change at 1280 steps stays within 0.3 nats in every run." (e646).
- Session 133's reading that the stored class clears the word bar in one run of four: the twins had been stopped at unequal steps; at equal steps Pythia-410m clears it in every seed (e646).

## Session 136 kills (autoencoder blocks; read repulsion; is the calendar special)
- "at most half of the autoencoder blocks are anchored without block 0, at every d, in both models." (e647).
- "the anchored fraction is larger at d = 8 than at d = 2 in both models." (e647).
- "when their features fire, anchored blocks clear their rotation floor more often than unanchored blocks (median), at every d, in both models." (e647).
- "the blocks' activity is mostly real: the median block clears its rotation floor in half or more of the states where its features fire, at d = 2, in both models." (e647).
- "masking: the raw read cosine rises by 0.05 or more from the lowest to the highest pull quintile while the pull-removed cosine changes by less than 0.03, at all three origins." (e648).
- "in the top 5 percent by pull the pull-removed read cosine is negative at all three origins." (e648).
- "random-noun class-mean subspaces are anchored without block 0 in at least 80 percent of draws over all models." (e649).
- "the calendar's anchored share (20 of 22 pairs) exceeds the random nouns' by no more than 0.1." (e649).
- The subspace floor as the missing significance test for autoencoder-derived multi-dimensional features (Gemini 4's pitch): nearly every such block passes it, and in Pythia-160m the unanchored ones clear their state-side floor more often, not less (e647).
- Gemini 4's masking account of the entrants' exemption from read repulsion: the pull explains about a third of it (e648).

## Session 137 kills (the repulsion's split; the second row)
- "the static read-write cosine (raw read weights) is negative on average at all three origins, between -0.25 and -0.10." (e648b).
- "more than half of the summed repulsion comes from growth along an already anti-aligned write, at all three origins." (e648b).
- "at d = 4 and d = 8 at most a third of the autoencoder blocks have an anchored second direction, in both models." (e647b).
- "the weekday and month class-mean subspaces both have an anchored second direction in both models." (e647b).
- e647's weekday control as evidence: scored under wikitext bands, where random nouns pass too (e647b).

## Session 138 kills (what kind of class is anchored)
- "read in the same mention sentences, days and months are still anchored without block 0 in at least 8 of the 11 models each." (e650).
- "random nouns are anchored in at most a quarter of draws over all models." (e650).
- "colours and animals are anchored about as often as days: their counts of anchored models are within 2 of the days' count." (e650).
- "Functional manifolds are anchored in the write rows, arbitrary groupings are not" (Gemini 4's reading of sessions 135 to 137) as stated: in identical sentences the floor passes everything in three models and elsewhere favours ordered sequences, numbers above days above months and colours, animals almost never (e650).

## Session 139 kills (the edit detectors' coverage)
- "the read-write cosine flags a naive negate in at least half of both row kinds (a flip sends the cosine from about -0.2 to +0.2)." (e651).
- "the reflection evades every one of the four detectors (flagged-any at most 0.2) for both kinds." (e651).
- "the reflection still changes behaviour, its KL at least half the naive negate's, for both kinds." (e651).
- The read-write cosine as a standalone edit detector: it is GELU-only and blind to a norm-and-cosine-preserving reflection; only with the activation-consistency detector beside it is the behaviour-carrying reflection caught (e651).

## Session 142 kills (the answer as the readout of a running total)
- "the carrier's top-1 acceptance exceeds the runner-up of the model's own logits at p." (e653).
- "the best carrier reader (carrier, naming or pooled) exceeds the empirical bigram's top-1 acceptance overall." (e653).
- "the best carrier reader exceeds the empirical bigram's top-1 acceptance when the token after next is a subword continuation." (e653).
- "on continuations after an ambiguous token (the bigram's top-1 share under 0.3) the best carrier reader's top-5 exceeds the bigram's." (e653).
- "peeling the function-pushing words, with the mask, recovers at least three more wrong answers than the mask alone." (e654).
- "the function-word peel without the mask recovers more wrong answers than the mask alone." (e654).
- "peeling the words recurring across prompts, with the mask, recovers more wrong answers than the literal prompt-mean subtraction with the mask." (e654).
- "at M = 20, OMP in logit space recovers the reference ledger's largest word at 50 percent or more of positions." (e655).
- "at M = 20, OMP in logit space beats its other-position control by 0.10 or more in support Jaccard." (e655).
- "the sparse prior helps when underdetermined: OMP in logit space beats the minimum-norm reconstruction in Jaccard at both M = 64 and M = 256." (e655).
- "the ledger as the unembedding reads it agrees with the usual hidden-state ledger (Jaccard 0.5 or more)." (e655).
- The carrier word as a zero-pass draft of the token after next (Gemini 4's idea 1): it proposes spellings of the emitted token or topic neighbours, below a static bigram table on every model (e653).
- Peeling function-word words off a hedged answer to recover a known fact (Gemini 4's idea 2): masking the function words in the logits does it, and the peel matches a random peel (e654).
- OMP over restricted logit signatures as a black-box ledger reader (Gemini 4's idea 3): worse than the minimum-norm reconstruction at every M; above D+1 logprobs the state is exact by linear algebra, which is prior art (e655).

## Session 143 kills (repeating a block)
- "the 16-word WDD support Jaccard of x_i and x_j predicts the damage within block size better than their centred cosine." (e656).
- "the native-chord junction beats the rotated-chord junction on average." (e656).
- "at alpha 1 the native chord damages the loss less than the rotated chord on most bands." (e657).
- "the second-to-first MLP output cosine is lower in Pythia-410m (GELU, read repulsion) than in both gated models." (e658).
- The WDD support overlap as a predictor of which block to repeat (Gemini 4's idea 3b): the plain centred cosine predicts the damage map better in all three models; the weights-only reader compatibility wins only in Pythia-410m (e656).
- The WDD-filtered junction, x_i plus the band's native chord fed to the second traversal (Gemini 4's idea 3a): never the best fix and worse than the rotated chord on average (e656).
- Doubling the band's native chord as a better virtual repeat than scaling the raw increment (Gemini 4's idea 1): the raw increment reproduces the helpful repeat better (e657).
- Read repulsion making the second traversal cancel the first in GELU models (Gemini 4's prediction B): Pythia-410m cancels no more than the gated models, and friendly blocks amplify (e658).

## Session 144 kills (the relay chain and the clamp)
- "negating the gender chain flips more than 0.60 of held-out gender items at a wikitext loss change under 0.01 nats (Gemini's test A)." (e659).
- "negating the gender and generation chains together doubly flips at least 0.25 of the kinship items (Gemini's test B; e624's single rows 0.05)." (e659).
- "the clamp moves at least 0.3 of the other pole's held-out gender items to the target while keeping at least 0.9 of the target pole's, forcing either pole." (e660).
- "the chain rows' mean absolute activation on wikitext is under a tenth of that at the attribute words, for all three attributes (the self-gating premise)." (e660).
- The WDD vote as the selector of the rows to edit within a block: the top activation-difference row of each chain block does as well; the vote's contribution is which blocks (e659).
- Spreading an edit over the relay chain to make two attribute edits compose: two chains interfere as two rows did (e659).
- A sign clamp as a two-way pole setter: it forces the unmarked pole and never the marked one (e660).

## Session 145 kills (markedness and the dose response)
- "the chain rows are more active on the female, young and plural pole than on the other, for gender, generation and number." (e661).
- "the marked-over-unmarked activation magnitude ratio is 2 or more for all three attributes." (e661).
- "negating the WDD chain flips marked-pole items at least three times as often as unmarked-pole items, for all three attributes." (e661).
- "supplying the marked pulse at the word position moves at least 0.3 of the unmarked items to the marked pole, for all three attributes." (e661).
- "the WDD-free chain flips at least as many held-out items as the WDD chain for at least three of the four attributes (Gemini's block-selection question)." (e661).
- "the gender and generation pulses together turn king into princess in at least 0.25 of the items." (e661).
- "at the largest WDD dose, flips from the marked pole are at least three times those from the unmarked pole, for every attribute." (e662).
- Markedness as the account of every attribute: generation's rows are not one-pole detectors (e661).
- A block choice without the WDD vote: ranking blocks and rows by normalized activation difference lands in blocks that do not carry the attribute (e661, e662).
- Gemini's weight-space input shift as a way to force the marked pole: erratic for gender and number (e661).

## Session 146 kills (who writes an attribute)
- "negating the pole-0-active sub-chain flips at least 0.3 of singular items to plural." (e663).
- "each of the four target-quadrant edits puts at least 0.5 of all kinship items on its target (Gemini's four-quadrant composition)." (e663).
- "the WDD order's mean flips over doses exceed the edit-specific gradient attribution's for at least three of four attributes." (e664).
- "the WDD order beats attribution patching for at least three attributes." (e664).
- "the WDD order's mean flips exceed the direct-contribution order's by more than 0.05 for at least three of the four attributes." (e664b).
- "the base chain's rows keep a write cosine of at least 0.9 with the base model in the derivative." (e665).
- 'WDD broke the tie with one-line baselines' (Gemini 4, after e662): attribution patching, causal block patching and a direct-contribution ranking with no OMP localize the writers as well as the vote (e664, e664b).
- Two-way weight surgery by negating the unmarked pole's rows (Gemini 4's pole-split): the marked pole is still not set (e663).
- 'Jakobsonian markedness in silicon' as a new discovery: LSTM language models' masculine and singular defaults were shown by Jumelet et al. 2019; what is new is narrower (REVIEW_POSITIVE_CLAIMS.md section 11).
- '1000x cleaner erasure of marked behaviours': ablation over addition is known (Arditi et al. 2024), the fair comparisons are ablation, LEACE and AURA, and setting a mark costs what steering costs (e661).

## Session 147 kills (index against vector transfer)
- "on a target, ablating the base direction keeps less than half of the own direction's marked flips while negating at the base indices keeps at least three quarters of the own indices' flips, for gender and number (Gemini's co-rotation)." (e665b).
- Slot co-rotation as the reason base-model row edits survive heavy continued pretraining (Gemini 4): the attribute's residual direction is preserved, and transferred directions work as well as transferred indices (e665b).

## Session 148 kills (the provenance core against the activation side)
- "Gemini's second branch: the selectivity's or the direct write's top-10 share comes within 0.05 of the class ratio's, on all three seeds." (e666).
- The identification of the future recruit given its class is the class ratio's alone (e632, session 127): two checkpoints before recruitment the activation side names it too, and leads on a class defined by firing (e666).
- WDD's state projection is strictly necessary to identify the recruit before its own activation catches up (Gemini 4): the class ratio's lead over selectivity and the direct write is the home advantage of a class defined by the row's own projection (e666).

## Session 149 kills (the detectors' coverage rescored)
- "the artifact: pooled over the four models, consistency with the edited model's own states flags high-use reflections at most half as often as with the clean states." (e667).
- "the defender's reading fails: with the edited model's own states, high-use reflections are flagged by some detector in fewer than half of the rows on at least three of the four models." (e667).
- No single-row edit in e651's battery both changed behaviour and evaded every detector (e651): with the tampered model's own states, SmolLM2-135M's row 776 (KL 0.22 to 0.30) evades all three (e667).
- The single-row detector triangle closes the reflection blind spot and is robust to distribution across co-firing rows (e651, e652c's reading): the consistency catch it rested on was overstated by clean-state scoring; honest, it catches about a third of high-use reflections (e667).
- Gemini 4's demo claim that no behaviour-carrying single-row flip escapes the three blind detectors (session 149's demo, e667).

## Session 150 kills (Gemini's weight-only constructions)
- "Gemini's claim: pooled over the four models, C_perp (raw, two-sided) flags the high-use reflections in at least 0.6 of rows, against 0.33 for e667's honest consistency." (e667b).
- "Gemini's case: C_perp (raw, two-sided) flags SmolLM2-135M row 776's reflection, which evades all three detectors in e667." (e667b).
- "why it can fail: the high-use rows' clean C_perp sits inside the bulk, median clean percentile between 0.2 and 0.8 on every model, so the exact sign flip lands inside the bulk as well." (e667b).
- The pre-MLP orthogonal write-phase check catches SmolLM2 row 776 and restores the blind reflection catch rate (Gemini 4): it flags 0.29 of high-use reflections against 0.33 for honest consistency and misses row 776 (e667b).
- Weight alignment with a neuron's read vector predicts which upstream rows carry its attribute: Spearman -0.004 to 0.023 across all upstream rows (e668).
- OMP over the upstream writes improves on the one-line alignment ranking for naming feeders: it does worse (e668).

## Session 151 kills (frozen rows, the echo, signed poles)
- "norm alone: random directions at the trained norms reach test 0.5 at a median at least 2000 steps earlier than e576's initial-scale random rows." (e669).
- "the final position's own writers at k = 32: negating the top 32 by the final position's own direct contribution flips at least half as many held-out marked items as the oracle's top 32, on both attributes." (e670).
- The attribute relay rows as one-pole detectors whose unmarked form is silent (e661's reading under absolute activation): the unmarked form usually sits on the same rows' negative pole (e671).
- The final position's own active rows carry the word's attribute (the alternative to the echo): they flip almost nothing (e670).

## Session 152 kills (routes over training, edge consumption)
- "developmental, cluster split, multi-route neurons: the core's firings became top firings earlier than the offshoot's, final activation regressed out: lead positive in more than half, one-sided binomial p < 0.05 (REFUTED if the share is below half with p < 0.05 the other way; otherwise NOT SUPPORTED)." (e672).
- "causal, cluster split, multi-route neurons: removing the neuron moves the next-token distribution less at offshoot firings than at core firings (mean KL) in more than half, one-sided binomial p < 0.05; 'much less' when the median offshoot/core KL ratio is below 0.5." (e672).
- "replication of the smoke: top-upstream split, testable neurons, the residualized lead positive in more than half with p < 0.05, and still p < 0.05 with token typicality also regressed out." (e672).
- "descriptive: the share of sampled neurons that are multi-route under the cluster split lies between 0.05 and 0.25 (the essay's reference: about 90% of Qwen3-32B neuron poles have one main job)." (e672).
- "the control Gemini's comparison lacks: at block 17, rho(edge_ln) - rho(write) >= 0.05 with the CI above 0, i.e. reading the downstream gates adds to the write's plain size." (e673).
- Edge consumption by downstream gates as a better predictor of one neuron's ablation damage than the size of its write (Gemini 4's single-neuron transplant of the essay's edge motifs): the two rank 4,096 neurons alike at every block tested (e673).
- Node prominence (the write along the predicted token) as the natural score of a neuron's importance at mid depth: at blocks 12 and 17 of Pythia-410m it trails the write's plain size (e673).
- Offshoots much less causally important than a neuron's main route (the essay's first reason not to worry about recursive generalization), at Pythia-160m block 6: offshoot firings cost as much per firing, by the essay's own grouping slightly more (e672).
- A neuron's distinct routes forming in order, main route first (the essay's developmental prediction), at Pythia-160m block 6: no order among 83 two-route neurons (e672).

## Session 153 kills (routes replicated, the cross-position trace)
- "the contrast e672 implied but did not test: pooled over the eight runs, the share of one-route neurons whose larger group leads exceeds the share of two-route neurons whose core leads, one-sided Fisher exact p < 0.05." (e674).
- "the essay's causal prediction again: pooled over the eight runs, among two-route neurons the offshoot's mean KL is below the core's in more than half, one-sided binomial p < 0.05." (e674).
- "Gemini's leak hypothesis: in OLMo (the three runs pooled), among two-route neurons whose top 100 include gate-closed firings and whose two routes differ in their gate-closed share, the offshoot has the larger share in more than half, one-sided binomial p < 0.05." (e674).
- "Gemini's collapse: in OLMo (pooled), at least half of the two-route neurons become one-route when only gate-open firings are kept." (e674).
- "the essay's causal prediction on clean firings: in OLMo (pooled), among two-route neurons of the gate-open top 100, the offshoot's mean KL is below the core's in more than half, one-sided binomial p < 0.05." (e674).
- "Gemini's trace, part A: at each block the activation-free projection of the final state ranks neurons by their cross-position damage at Spearman 0.3 or more (either state)." (e675).
- "part A: the projection predicts cross-position damage better than the earlier positions' write size (bootstrap 95% CI of the difference above 0) at each block." (e675).
- "part B: negating each held-out marked item's own top 32 rows by its label-free projection flips at least half as many items as e670's supervised echo, for gender and for number." (e675).
- "part B: the attribute's oracle top row ranks in the top 100 of 63,232 rows under the label-free projection for at least half of the items, for gender and for number." (e675).
- e672's reading that the developmental order appears only inside a single route (session 152): pooled over eight runs distinct routes also show the core first (0.56, p 0.0017) and the two kinds of neuron do not differ (Fisher p 0.19) (e674).
- The SwiGLU double-negative leak as the source of apparent offshoots (Gemini 4): the offshoot is the more gate-closed route in 39 of 74 OLMo neurons whose routes differ, and gate-open filtering leaves 179 of 219 two-route neurons two-route (e674).
- The write dictionary as a label-free cross-position forensic trace (Gemini 4's reading of e670): the projection of a summary state is unrelated to which neurons' earlier writes matter there, and item by item it finds no attribute writer (e675).

## Session 154 kills (effect species, the double negative, splitting across scale)
- "fine-tuning keeps species: at Qwen2.5-0.5B blocks 3, 12 and 21, fewer than 10% of the neurons beyond the total null in either the base or the Instruct model change species between them." (e677).
- "the essay, every double-negative activation: zeroing them in every layer gives more repairs than damages on more than half of the eight non-generative benchmarks, in each model." (e678).
- "the essay, the leak-dominated neurons only: the same for leak0_dom, in each model." (e678).
- "specificity: zeroing the double negative raises wikitext loss less than the L2-matched weak gate-open control, in each model." (e678).
- "the essay's claim under the gentlest ablation: mean replacement of the double negative everywhere gives more repairs than damages on more than half of the eight non-generative benchmarks, in each model." (e678).
- "the best single Pythia-410m neuron covers a one-route neuron's top 100 better than a two-route neuron's (one-sided Mann-Whitney p < 0.05)." (e679).
- Zeroing SwiGLU's double negative across the network as a benchmark improvement (the essay's 'ups most benchmarks'): no model and no variant raises most of eight benchmarks, and zeroing costs 1.25 to 4.40 nats of wikitext loss (e678).
- The double negative as a small echo (the essay's word): it carries 0.15 to 0.19 of each layer's MLP activation energy, and zeroing it costs more than an L2-matched zeroing of weak gate-open activations (e678).
- A neuron's own block's readout as a guide to what it does to the output: early in the network 0.91 to 0.97 of the neurons that move it beyond the null do not move the output (e677).

## Session 155 kills (the already-clean debate tested)
- "at 2 floors, the position's own write scaled up costs more than 2x a random direction." (e680).
- "keeping only the strongest quarter of each neuron's activations by count (the essay's vision measure) costs under 0.05 nats of KL." (e681).
- "at K=16 the median KL ratio at strong positions is under 0.25." (e685).
- "over-floor upstream writes carry more than half of the MLP input energy at strong firings." (e685).
- "adding the crowd's mean halves the K=16 KL ratio at strong positions." (e685b).
- "with the crowd's mean the K=16 ratio at strong positions is under 0.25." (e685b).
- "with the crowd's mean the K=64 ratio is under 0.25." (e685b).
- "the crowd share of the resultant falls with d (Spearman < -0.5 within families)." (e689).
- "Spearman over pairs between token-bag similarity and reader-vector similarity exceeds 0.3." (e691).
- "a neuron's nearest neighbour by tokens is among its ten nearest by readers for more than 0.3 (chance 0.01)." (e691).
- "random neurons do so under 0.1." (e682).
- "the random codes' writers are related to a code-like field at most half as often as facts' writers to theirs." (e682).
- "chain enrichment (new contexts nearest a member that entered within two checkpoints, over the random expectation) exceeds 1.5 in at least 70% of neurons." (e683).
- "the recruited row gains readers (|share| > 1 std) from two checkpoints before to two after, more than norm-matched control rows on the same positions, in at least 65% of words." (e684).
- "sub-class specialists among later-block neurons increase across recruitment more than for random position sets in at least 60%." (e684).
- "it is at least 0.6." (e687).
- "such a rule has at most 10 entries for at least half." (e687).
- "a surface rule at precision 0.5 covers at least half of a neuron's held-out class at the middle block." (e687).
- "by release step 6000 at least one vote share falls below 0.5 in at least 3 of 5 seeds." (e688).
- "memorised writers sit more than one block deeper than both controls." (e690).
- The floor as a causal boundary for the readers: a native write injected below it costs the same as above it relative to a random direction, with no change of regime at one floor (e680).
- Native directions as privileged input for the readers: a foreign native row and its rotation cost exactly what a random direction of the same norm costs at every amplitude (e680).
- 'Almost all weak activations can be removed for almost no cost' in language models: keeping the strongest quarter of each neuron's activations costs 0.2 to 8.7 nats under zeroing (e681).
- Weak activations as a signal the readers filter: most of the zeroing cost is the removed activations' mean, which mean replacement recovers in the GELU and GPT-2 models (57 to 94 percent of it; 22 to 66 percent in the gated models) (e681).
- Recursive generalization as an order of arrivals (a branch of branches): no chain enrichment in three seeds of Pythia-160m, and jumps in 1 to 2 percent (e683).
- A dedicated reader arriving with a recruited row: no downstream neuron depends on a recruited row at a fifth of its spread before or after recruitment (e684).
- A pruned neuron attribution graph with 'almost no error' at variable resolution: in five of twelve model-blocks a neuron driven by its 1024 largest inputs still costs 0.57 to 1.55 of the damage of deleting it, and carrying the crowd's mean as the error node removes a seventh to a quarter of the sixteen-input error (e685, e685b).
- The negative pole as the nearest neighbour in general: it holds in SwiGLU models (0.91 to 0.97 of neurons at the middle block) and not in GELU ones (0.0 to 0.5) (e686).
- A neuron's explanation as a list of tokens or bigrams: 350-entry surface rules fitted to the class cover under 5 percent of it on held-out text (e687).
- Semantic gravity unpacking a planted knot: five seeds keep both pinned concepts through 6000 steps of the next-token loss, as single pins keep theirs (e688).
- The crowd shrinking with width: its share of the resultant along the most prominent row rises from 0.91 to 0.97 from 70M to 2.8B parameters in the GELU families and stays at 0.94 to 0.95 across Qwen2.5 from 0.5B to 7B (e689).
- Memorised text having special writers: depth, count, class coherence and field selectivity all match a low-loss control (e690).
- The activation label as a proxy for a neuron's function: across neuron pairs the token bag's similarity predicts the causal reader response at Spearman 0.00 to 0.04 (e691).

## Session 156 kills (the crowd read)
- "the weak write on the strong write's own 16 principal directions (the kept quarter's write, data, no weights) costs at most 1.2 times the weak write's own 16: the weak activations write where the strong ones write." (e693b).
- "the per-token table explains more than half the crowd's variance over all evaluation rows." (e692).
- "the crowd's write projected on the stream's 16 principal directions (held-out second moment) brings the 16-input ratio under 0.5." (e692).
- "the 200 largest contributors' top-1% classes contain the neuron's strong rows at more than five times chance (a district)." (e692).
- "with the per-token table the cost is under 0.05 nats (the essay's 'almost no cost' after a lexical correction)." (e693).
- "the bigram table beats the token table by a fifth." (e693).
- "the removed write projected on the stream's 16 principal directions (instead of its own) costs at most 1.2 times the own-16 rung: the weak write's low-rank part is the stream's dominant subspace." (e693).
- "the weak write's 16 principal directions are native rows (median max cosine with a write row of the block above 0.5)." (e693).
- "the 16 directions are shared across consecutive blocks (median subspace overlap above 0.5)." (e693).
- "the 16 directions carry over 0.8 of the weak write's second moment." (e693).
- "the weak write's 16 principal directions overlap the next block's 16 read directions by more than 0.3." (e693b).
- "and all later reads' 16 by more than 0.3." (e693b).
- "the best weights-only 16-direction subspace costs at most 1.2 times e693's data-derived 16." (e693b).
- "the 1 to 5 percent shell is adjacent to the core (max ledger cosine above the bottom half's 95th percentile) for more than 0.8 of its positions." (e695).
- "the 5 to 20 percent shell for more than 0.5." (e695).
- "the 5-20 shell's contribution-weighted upstream depth exceeds the core's by 0.05 of the network." (e695).
- "two sub-class directions explain 0.1 more of the read row than two random directions, beyond the class direction." (e695).
- "the per-token table recovers half of the zeroing cost." (e694).
- Guilds as the readers' unit: the sixteen largest clusters of upstream units by write row or by co-activation reproduce a neuron's input no better than random groups of the same sizes, and at group resolution not at all (e692).
- The crowd as a lexical baseline at depth: a per-token table halves a neuron's ablation damage only at the sixth blocks and in GPT-2 medium (e692).
- The stream's dominant subspace as the crowd's carrier: the crowd's write and the weak write projected on the stream's principal directions recover nothing, and often cost more than zeroing (e692, e693).
- The weight spectrum as the weak write's carrier: the block's dominant write directions recover 0.07 to 0.69 of the zeroing cost against 0.60 to 0.96 for sixteen data-derived directions (e693).
- Connected shells, a depth gradient, a read row of steps and a strong lexical seed as the residue of recursive generalization in the final model (e695).
- The crowd as the driver of a neuron's firing: its net sum at the firings is within -0.10 to +0.62 of the sixteen largest inputs' and it opposes them by 0.46 to 0.99 of theirs off the class (e695).
- The weights as the name of the weak write's directions: the next block's reads, all later reads and the unembedding overlap them by a tenth or less, and no weights-derived subspace recovers what their own sixteen directions recover (e693b).
- The K-input simulation as a 14B test in bf16: driven by every input a neuron still costs 0.95 of its ablation damage, so the ledger's KL ratios at that scale need fp32 or group ablations (e694).

## Session 157 kills (the brake and the bypass)
- "signed wires plus the bus bring the ratio under 0.25." (e697).
- "the brake (under -0.5) is present by step 8000." (e701).
- "the linear map explains more than half of the weak write's variance on the fitting data." (e696).
- A pruned neuron graph failing for lack of drive at the firings: at mid-depth 0.78 to 0.88 of its damage is off-class misfiring, and sixteen excitatory wires alone cost 6 to 561 times the ablation damage (e697).
- Sixteen inhibitory wires as the error node: they carry the net opposition at each position and still leave the firings at 0.59 to 1.06 of the ablation damage (e697).
- The brake as a property of the final weights only: at step 1000 the crowd reinforces the wires off the class and the brake forms between steps 2000 and 16000 (e701).
- The weak activations as a sparse or a superposed code: a closed-form rank-16 linear map of the block input replaces them for 0.56 to 0.96 of the zeroing cost (e696).
- The ledger's K-input simulation against a bf16 model: the residual stream's block-by-block rounding leaves the all-inputs check at 0.86 to 0.95, and only an fp32 model passes it (e700).

## Session 158 kills (the field-augmented dictionary)
- "the Taylor prediction (weights, the mean state and the weak shares, no regression) recovers at least 0.8 of what the fitted bypass recovers." (e702).
- "after 16 OMP rows, the field128 halves the energy the rows leave." (e703).
- "the field128 beats a random subspace of 128 by 0.1 of the energy." (e703).
- "and the stream's 128 principal directions by 0.05." (e703).
- "the field128 beats the field16 by 0.1 in ratio." (e703).
- "the attention units as exact wires alone halve the sixteen-wire ratio." (e703).
- "the first direction's readers number under 1 percent of the next three blocks' neurons (N_eff)." (e704).
- "the directions' own classes have a one-token share above 0.2." (e704).
- "the brake (under -0.5) is present by step 8000." (e701o).
- "the brake strengthens monotonically (Spearman under -0.8)." (e701o).
- A pruned neuron graph failing for lack of drive at the firings: at mid-depth 0.78 to 0.88 of its damage is off-class misfiring, and sixteen excitatory wires alone cost 6 to 561 times the ablation damage (e697).
- Sixteen inhibitory wires as the error node: they carry the net opposition at each position and still leave the firings at 0.59 to 1.06 of the ablation damage (e697).
- The brake as a property of the final weights only: at step 1000 the crowd reinforces the wires off the class and the brake forms between steps 2000 and 16000 (e701).
- The weak activations as a sparse or a superposed code: a closed-form rank-16 linear map of the block input replaces them for 0.56 to 0.96 of the zeroing cost (e696).
- The ledger's K-input simulation against a bf16 model: the residual stream's block-by-block rounding leaves the all-inputs check at 0.86 to 0.95, and only an fp32 model passes it (e700).

## Session 159 kills (the gated remainder and the strong field)
- "the quadratic bypass (256 random bilinear features) recovers 0.1 more of the zeroing cost than the linear one on wikitext." (e705).
- "out of domain the quadratic bypass beats the linear one by 0.1." (e705).
- "the brake is under -0.5 by step 128000 (nine percent of training) at both blocks." (e701l).
- A pruned neuron graph failing for lack of drive at the firings: at mid-depth 0.78 to 0.88 of its damage is off-class misfiring, and sixteen excitatory wires alone cost 6 to 561 times the ablation damage (e697).
- Sixteen inhibitory wires as the error node: they carry the net opposition at each position and still leave the firings at 0.59 to 1.06 of the ablation damage (e697).
- The brake as a property of the final weights only: at step 1000 the crowd reinforces the wires off the class and the brake forms between steps 2000 and 16000 (e701).
- The weak activations as a sparse or a superposed code: a closed-form rank-16 linear map of the block input replaces them for 0.56 to 0.96 of the zeroing cost (e696).
- The ledger's K-input simulation against a bf16 model: the residual stream's block-by-block rounding leaves the all-inputs check at 0.86 to 0.95, and only an fp32 model passes it (e700).

## Session 160 kills (what the model reads and defends)
- "the weak field's sixteen directions carry read energy above three times a random direction's." (e707).
- "the strong field's do too." (e707).
- "on GPT-2 small the public SAE's error holds under half as large a share of the content's read energy as of its Euclidean energy." (e707).
- "the expected gradient at the perturbed point points back toward the original rows: restoring cosine over 0.3 for rotated strong rows." (e711).
- "for the swapped field the restoring cosine on the swapped plane is over 0.3." (e711).
- "rotating every strong row of the block at once (up to 256) gives a restoring cosine over 0.3." (e711).
- "sixteen strong rows rotated by forty-five degrees return more than halfway (return fraction over 0.5) within 800 steps while unperturbed rows keep a cosine over 0.98 to their originals." (e711).
- "those rows return by over 0.2 of the way in 200 steps." (e711).
- "weak rows return less than strong rows by 0.2." (e711).
- "the swapped field is restored by the writers (writer restoration over 0.5)." (e711).
- "halved rows recover more than half their norm." (e711).
- "in the gated models a sixteen-branch mixture closes at least half the gap between the linear bypass and the oracle sixteen." (e710).
- "sixteen branches beat four by 0.03." (e710).
- "at alpha 1.5 the next three blocks' weak writes undo at least a fifth of the injected vector along its own direction." (e708).
- "the weak writes undo more than the strong writes do." (e708).
- "the undoing is asymmetric: larger at alpha 1.5 than at alpha 0.5 by 0.1." (e708).
- "and at least one and a half times the strong field's." (e709).
- "the weak field's coordinates jump across a document boundary by more than twice their jump inside documents." (e709).
- "a neuron's stable wires (in its sixteen at half its firings or more) hold at least half its wire mass." (e712).
- A pruned neuron graph failing for lack of drive at the firings: at mid-depth 0.78 to 0.88 of its damage is off-class misfiring, and sixteen excitatory wires alone cost 6 to 561 times the ablation damage (e697).
- Sixteen inhibitory wires as the error node: they carry the net opposition at each position and still leave the firings at 0.59 to 1.06 of the ablation damage (e697).
- The brake as a property of the final weights only: at step 1000 the crowd reinforces the wires off the class and the brake forms between steps 2000 and 16000 (e701).
- The weak activations as a sparse or a superposed code: a closed-form rank-16 linear map of the block input replaces them for 0.56 to 0.96 of the zeroing cost (e696).
- The ledger's K-input simulation against a bf16 model: the residual stream's block-by-block rounding leaves the all-inputs check at 0.86 to 0.95, and only an fp32 model passes it (e700).

## Session 161 kills (the brake at 7B and the Fisher-graded dictionary)
- "signed wires with the bus bring the ratio under 0.25." (e713).
- "both fields with the attention units bring the ratio under 0.25." (e713).
- "at k = 16 with rows and fields, the blended pursuit (lambda 1) recovers 0.1 more of the mean-ablation KL than the Euclidean pursuit." (e714).
- "it chooses field atoms 0.1 more often." (e714).
- "with the fields in the dictionary it beats rows-only blended pursuit at 16 by 0.05." (e714).
- "sixteen blended-chosen atoms match thirty-two Euclidean-chosen ones." (e714).
- "lambda 0.3 beats lambda 1." (e714).
- A pruned neuron graph failing for lack of drive at the firings: at mid-depth 0.78 to 0.88 of its damage is off-class misfiring, and sixteen excitatory wires alone cost 6 to 561 times the ablation damage (e697).
- Sixteen inhibitory wires as the error node: they carry the net opposition at each position and still leave the firings at 0.59 to 1.06 of the ablation damage (e697).
- The brake as a property of the final weights only: at step 1000 the crowd reinforces the wires off the class and the brake forms between steps 2000 and 16000 (e701).
- The weak activations as a sparse or a superposed code: a closed-form rank-16 linear map of the block input replaces them for 0.56 to 0.96 of the zeroing cost (e696).
- The ledger's K-input simulation against a bf16 model: the residual stream's block-by-block rounding leaves the all-inputs check at 0.86 to 0.95, and only an fp32 model passes it (e700).

## Session 162 kills (the dictionary against the SAE and the brake under other choices)
- "at k = 16, rows plus fields recover more of the mean-ablation KL than the SAE encoder's top sixteen latents." (e715).
- "at k = 16, rows plus fields recover 0.05 more than rows alone." (e715).
- "sixteen atoms from rows plus SAE directions beat the better of rows alone and SAE directions alone by 0.03." (e715).
- The fields as a better unit for reconstructing a position's state than native rows or SAE latents: at sixteen atoms they add 0.01 to the rows, and the SAE's directions beat both by 0.02 to 0.05 on the state's effect (e715).
- The SAE encoder's top latents as the right way to use an SAE's dictionary at a budget: pursuit over the same directions recovers more KL and responds to perturbations far more like the state (e715).
- The brake as an artefact of sixteen wires and a five percent class: it holds for k of 8 to 32 and class cuts of 1 to 20 percent (e716).

## Session 163 kills (the robustness round)
- "the encoder's top sixteen latents as the input misfire: all-position ratio above 1." (e717).
- "the native rows' intervention fidelity exceeds the encoder's top sixteen's in every run." (e718).
- "on Pythia-160m sixteen native rows recover at least 0.85." (e718).
- "no brake at step 1000: the crowd's net sum over the sixteen wires in the bottom half is within 0.3 of zero." (e721).
- "in every e697 model-block the lower bound of the median off-class share is above 0.4." (e722).
- "in every e706 model-block the upper bound of abs16_both at the firings is below the lower bound of abs16." (e722).
- "in every e697 model-block at least three quarters of the neurons have more than half their sixteen-input damage off the class." (e722).
- The L1 encoder's weakness as every SAE's: with OpenAI's and EleutherAI's TopK dictionaries the encoder's top sixteen beat pursuit over the same directions on reconstruction and under intervention (e718).
- The native rows' intervention advantage as general: it held over jbloom's L1 encoder and over no TopK encoder, which track a fixed perturbation at 0.83 to 0.95 against the rows' 0.65 to 0.89 (e718).
- The SAE's reconstruction error as the carrier of a neuron's brake: added to sixteen wires it leaves 0.30 to 0.32 of the ablation damage at the firings where both fields leave 0.14 to 0.18 (e717).
- The brake as wikitext's: on the Pile the same neurons show the same off-class share within 0.10 and the same runaway (e719).
- The weak activations as removable at 7B: keeping the strongest quarter costs 1.57 nats at Qwen2.5-7B, the mean replacement 0.88, with a bf16 noise floor of 0.002 (e720).
- The brake as a late property of training at 410m: it is under -0.5 by step 8000 of 143000 (e721).
- The GPT-2 medium off-class shares as settled numbers: their 95 percent intervals over neurons span up to 0.35 and reach below 0.4, where the other eight runs' lower bounds are 0.71 to 0.85 (e722).

## Session 164 kills (the matched-budget comparison)
- "the SAE's error projected on random directions improves it by less than 0.02." (e723).
- "at sixteen directions the projection on the SAE's directions beats the SAE's own coefficients on sixteen latents at the firings." (e723).
- "sixteen native rows add at least 0.02 to the SAE's full reconstruction." (e724).
- The fields as a better dictionary than an SAE's directions for a neuron's firings at the same budget: they tie at sixteen, at sixty-four and at the full dimension (e723).
- A fixed subspace as the SAE's best use on a neuron: the SAE's own coefficients on sixteen latents chosen per position beat every fixed sixteen-direction projection, the fields included (e723).
- The fields as adding to an SAE reconstruction's account of a state: sixteen field atoms add 0.002 to 0.006 of the mean-ablation KL, within the random atoms' range (e724).

## Session 165 kills (the corrected readout)
- "sixteen excitatory plus sixteen inhibitory wires cost less at the firings than the sixteen by magnitude in at least 6 of 8." (e725).
- "at sixteen directions the fields and the SAE's directions are within 0.05 of each other at the firings in 4 of 4." (e725).
- Every KL of sessions 155 to 164 as a measurement against the models' own next-token distributions: the readout applied the final norm twice (e725).
- The cost at 7B as 1.57 nats: 0.32 with the readout corrected, 0.23 with the mean replacement (e725).
- GPT-2 medium's brake as half off the class: 0.40 and 0.46 with the readout corrected (e725).
- The native rows' intervention advantage over jbloom's L1 encoder as a two-block result: one block of two with the readout corrected, every eight-perturbation fidelity lower than recorded (e725).
- The dial tertile costing twice the detector tertile as a general side-claim of e681: reversed at GPT-2 small, 0.66 against 3.24, with the readout corrected (e725).

## Session 166 kills (the zoo through WDD)
- "the top-64 neurons alone leave more than 0.5 nats of KL at both blocks." (e726).
- "the closed-form rank-16 field map recovers at least half of the top-64 KL at both blocks." (e726).
- "the trained d-by-d map recovers more than the closed-form rank-16 map at both blocks." (e726).
- "the trained rank-16 map recovers within 0.1 of the closed-form rank-16 map at both blocks." (e726).
- "the KL ordering of the trained full map and the field map differs from their FVU ordering at both blocks." (e726).
- "the weakest three quarters cost at least 0.2 nats in every pretrained model and in neither TinyStories model." (e727).
- "Reproduction: the evil pair at dose 1.0 hits the violent-term regex on >= 6/10 held-out prompts, and >= 5 of 7 personas hit their regex on >= 5/10 prompts at 0.8." (e728).
- "Overdrive: the zoo's 0.8 dose is >= 30 x q999(|a_n|) for every one of the nine zoo rows (the zoo steers far outside the row's natural range)." (e728).
- "Directions, not atoms: >= 6 of the 9 zoo rows are above the global median of R_dir while >= 6 of 9 are below the global median of R_nat." (e728).
- "Landmines predicted: both known landmines (L13 N142, L15 N143) are in the top 1% of R_dir." (e728).
- "Landmines caused: the median held-out NLL cost of the top-8 R_dir rows at 0.8 is >= 2 x that of the random-8, and the bottom-8 cost less than the random-8." (e728).
- "Any overdriven direction is a persona: the twenty random rows (stage 3 and 3b) at the zoo's 0.8 get blind consistency 2 on at least 6 of 20." (e728).
- "WDD-chosen rows (top R_nat, one per layer) at 3 x q999 get blind consistency 2 on >= 4 of 10; random rows at the same multiplier on <= 1 of 10." (e728).
- "The zoo's selection finds cheap directions: at 0.8 the median KL cost (on the model's own answers) of the eight zoo specs is below that of their random controls." (e728).
- "The readers' favourite direction of each layer (top R_dir within layers 3 to 21) at the zoo's 0.8 gets blind consistency 2 on >= 5 of 10 (p 0.45), and costs more KL on the model's own answers than the random row of its layer, median over the ten (p 0.7)." (e728).
- The zoo's persona rows as rows WDD would pick: by read energy the 43 cards are ordinary (direction percentiles 0.08 to 0.89, none in the top tenth), and WDD's own top rows carry no persona before a blind judge, 0 of 10 at the zoo's dose (e728).
- Read energy per unit norm as the landmine predictor at the zoo's dose: the two known landmines sit at percentiles 0.66 and 0.49, and the bottom rows by R_dir (all in layer 31) cost more than random rows at 0.8 because the zoo's dose scales with each layer's residual norm (e728).
- Any overdriven direction as a persona, at the pre-registered bar: 4 of 20 random rows at the zoo's dose get a nameable theme before the blind judge, under the 6 of 20 required (e728).
- The weak activations as a slice that costs more than its energy: per unit of energy removed the weak slice costs 0.73 to 2.27 times a random slice in eight runs (e727).
- The TinyStories models as the counter-example without a weak cost: on their own text the weakest three quarters cost 1.27 and 4.48 nats (e727).
- A trained d-by-d map on a sparsified MLP as the better repair: at 200 steps it recovers 0.22 and 0.09 of the top-64 KL on Qwen3-4B against 0.33 and 0.11 for the closed-form rank-16 field map; and the premise that top-64 per token costs half a nat, it costs 0.05 (e726).

## Session 167 kills (the dose-scaled landmine predictor)
- "Over the 24 stage-1 rows, the Spearman correlation of P with the held-out NLL cost at the zoo's 0.8 is at least 0.5." (e729).
- "The dose scaling repairs the inverted bottom group: the median P of the bottom-8 rows exceeds the median P of the random-8 rows." (e729).
- "Calibration: for at least 12 of the 24 stage-1 rows, P is within a factor of 3 of the measured KL on the model's own answers at the zoo's 0.8." (e729).
- "Over the 40 stage-3 and stage-3b rows at the zoo's 0.8, the Spearman correlation of P with the KL on the model's own answers is at least 0.5." (e729).
- "For at least 6 of the 8 stage-2 specs, the P of the zoo's rows (summed over a spec's rows) exceeds the P of its same-layer, same-sign random controls." (e729).
- "At least one of the two known landmines (L13 N142, L15 N143) is in the global top 10 percent of P over all 294,912 rows." (e729).
- The dose-scaled read energy as the landmine predictor at the zoo's dose: Spearman 0.30 with the measured cost over the 24 stage-1 rows against 0.35 for the unscaled energy, 0.15 against 0.43 over 78 rows (e729).
- The second-order KL as a description of the zoo's injection: P overpredicts the early-layer rows ten times (10.9 times the measured KL for the top-8) and underpredicts the late-layer rows ten times (0.11 for the bottom-8), 6 of 24 rows within a factor of three (e729).
- The late-layer floor as an artefact of the unscaled energy: scaled to the dose, the bottom-8 rows of layer 31 get the smallest P of any group (0.044 nats against 0.15 for random rows) and cost 0.73 nats, so the cost at the zoo's dose falls with depth (Spearman -0.49) while the predictor's depth profile is U-shaped (e729).

## Session 168 kills (the native-vocabulary finder)
- "Separation: the effective number of promoted tokens N_eff separates the 43 cards (at their steer sign) from 430 random (row, sign) candidates with the same layer distribution, AUC at least 0.7 with the cards lower." (e730).
- "Token-list judge: a blind judge of shuffled top-40 token lists gives consistency 2 to at least half of the 43 cards and to at most a fifth of 43 layer-matched random (row, sign)." (e730).
- "They are not merely broken: the judge flags degenerate text on at most 6 of 20." (e730).
- N_eff as the finder of personas: the 20 most concentrated rows loop 12 of 20 times and, loops excluded, carry a theme no more often than random rows of their layers, 7 of 20 against 6 of 20 (e730).
- The lookup as the finder at the zoo's dose: 5 of 8 retrieved rows loop on the retrieved words; one of eight produces the theme cleanly, the Hugs retrieval (e730).
- The zoo's cards as direct vocabularies: 12 of 43 token lists carry a theme before a blind judge, as many as random rows (11 of 43), and the pair's rows are not retrieved; most hand-found personas are made downstream of the row (e730).
- The raw bars of B1, C1 and C2 as tests of a finder: met by looping outputs; B3, written to catch that, failed (e730).

## Session 169 kills (switches from the weights)
- "mean generated tokens at +D at most 0.6 x base with at least 70 percent non-degenerate, and at -D at least 1.4 x base." (e731).
- "upper-case share of letters at least 0.5 at +D with at least 70 percent non-degenerate; base at most 0.15." (e731).
- "mean emoji characters per answer at least 2 at +D with at least 70 percent non-degenerate; base at most 0.3." (e731).
- "at -D the count of '**' per answer is at most a quarter of the base count with at least 70 percent non-degenerate answers; base at least 2 per answer." (e731).
- "at least 10 of 20 generations contain the think token at +D with at least 70 percent non-degenerate." (e731).
- "for at least 8 of 15 pairs both regexes hit in the same answer in at least 3 of 10 answers." (e731).
- "for at least 10 of 15 pairs each theme's hit rate is at least half its single-row hit rate measured in this run." (e731).
- "non-degenerate regex hit rate at least 0.5 for at least 4 of 8 specs." (e731).
- "the word direction's hit rate is at least the zoo row's at the same rule for at least 4 of 8 specs." (e731).
- "zeroing the 50 rows with the largest R_nat costs at least 1.0 nat of KL on the base answers." (e731).
- "at least 100 (row, sign) have p_max at least 0.9 and at least half of them sit in layers 30 and 31." (e731).
- The direct-path dose rule for one-token rows: at 16 nats the think, markdown and capitals rows spam their tokens and the Chinese row loops two characters, at 4 and 8 nats the last-layer rows move nothing (KL 0.005 to 0.08); the rule works only where the token is the control, the name and the end (e731).
- 'Controls on request from the weights' as a general claim: 2 of 7 switches held as written, one of them hollow; by the pivot rule the dose has to be measured, not read (e731).
- The vocabulary as the persona library: the theme word's own output direction reaches 1 to 4 of 10 for Jesus, Hugs, Galactic and Vegas against the rows' 7 to 10 at 0.8, and matches or beats the row only for the two word-themes, heavy and cat (e731).
- The read energy as the finder of load-bearing rows: zeroing the fifty largest-R_nat rows costs 0.015 nats, the fifty largest-R_dir 0.001, fifty random 0.001 (e731).
- An emoji row in the direct-path table: none; the best candidate is a diffuse row with the smiley on top by chance, and no (row, sign) has 12 of 40 emoji tokens (e731).
- The word-level repetition check as a loop detector for CJK text: it passed 16 of 20 answers that a character 6-gram check shows looping in 19 of 20 (e731).
- The opener switch: no first token reaches a 0.3 share over 20 prompts in Qwen3.5-4B, so there is no 'Certainly' to suppress (e731).
