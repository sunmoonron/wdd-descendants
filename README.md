# What a write becomes: the descendant and quotient program on Weight-Dictionary Decomposition

Ronish Bhatt ([ORCID 0009-0000-8835-5380](https://orcid.org/0009-0000-8835-5380)), September 2026. Companion to [Weight-Dictionary Decomposition](https://github.com/sunmoonron/weight-dictionary-decomposition) (WDD), which reads a transformer's residual state as a sparse combination of the model's own write vectors. This repository holds the follow-on program: 356 experiments, 1,430 recorded runs on five small models, that follow one WDD write through the network and ask what it becomes, plus the theory that the results support and the literature they sit in. No model was trained; every run is forward passes, ablations, injections and closed-form fits, one to five minutes each on one GPU.

The short version of the result: a neuron's write is the model's own perturbation of its residual stream. Downstream computation expands that perturbation physically (hundreds of dimensions) while its causally relevant content becomes compressible (tens of dimensions), and that compressibility is a property of the residual stream's response to any perturbation, not of the write. The compressed content is not carried by a privileged subspace: any moderate-dimensional high-variance projection of the perturbation cloud carries it, there is no null space and no equivalence-class structure, and the WDD atoms are not a special basis of it. WDD remains a clean birth coordinate and instrument; it is not the functional dictionary. The negative results are part of the result.

License: MIT for the code (`scripts/`), CC BY 4.0 for the result files and documents (`results/`, `docs/`). Cite with `CITATION.cff`.

## What is in the repository

| Path | Contents |
| --- | --- |
| `scripts/e01_*.py` to `scripts/e356_*.py` | One script per experiment (309 files). Each is self-contained: it loads a public checkpoint, reads the token cache, runs, and writes `results/<script>_<model>.json` plus one line to `results/FINDINGS.log`. The docstring at the top states the question, the design and what each number means. |
| `scripts/wdd_common.py`, `desc_common.py`, `func_common.py`, `quot_common.py` | Shared code: model loading, the dictionary and the cache (`wdd_common`), the runner with ablation and injection hooks (`desc_common`, `func_common`), the quotient helpers (`quot_common`). |
| `scripts/build_cache.py` | Builds the token cache and the dictionary for one model (see below). |
| `scripts/runjob.sh`, `wave2.sh`, `run.sh` | The launchers used on the box: resume-safe job runner with a skip guard, parallel wave runner, single launch. |
| `results/*.json` (1,458 files) | Every recorded result, one JSON per script and model or checkpoint revision, with the numbers behind every claim in the documents. |
| `results/FINDINGS.log` | One line per run in the order recorded: timestamp, experiment, and the headline numbers. |
| `results/e349_quotient_*.pt` | The 16-dimensional quotient bases of Pythia-410m at six training checkpoints. |
| `RUNLIST.txt` | Every run that produced a result, as `python <script> <model-or-revision>`, in recorded order (1,430 lines). Replaying it reproduces the repository. |
| `waves/` | The job files that were launched in parallel, for the record of what ran together. |
| `docs/FINDINGS.md` | The chronological narrative, session by session, with the numbers. |
| `docs/SYNTHESIS.md` | What each round established, the closing statements, and where everything is. |
| `docs/GRAPH.md` | The hypothesis graph, H0 to H152: each hypothesis, the experiments that tested it, and its status (survives, narrowed, killed, open). |
| `docs/KILLED.md` | Every hypothesis killed, with the experiment and the number that killed it, and the artifacts retracted. |
| `docs/THEORY.md` | The first-order transport theory, its derived predictions tagged by what they rest on, the tests of the new predictions, and the corrections the later rounds forced. |
| `docs/RELATED_WORK.md` | The literature placement and the priority check against the closest 2025 and 2026 work. |

The models: GPT-2 small (`gpt2`), Pythia-410m (`pythia410`), Qwen2.5-0.5B (`qwen05`), OLMo-1B-0724 (`olmo1b`), SmolLM2-135M (`smollm2`). Two GELU models, three gated (SwiGLU) models. Pythia's public training checkpoints are used for the training-dynamics runs.

## How to run and replicate

Environment: Python 3.11 or 3.12, `torch>=2.1`, `transformers>=4.46`, `numpy`, `scipy`, `datasets` (`pip install -r requirements.txt`). Checkpoints are downloaded from the Hugging Face Hub on first use; the wikitext-2 corpus is downloaded by `datasets`. One 80 GB GPU ran everything here with two or three scripts in parallel; the sub-1B models fit a 24 GB card one script at a time. Models are loaded in float32.

Paths are set by environment variables, with the defaults the box used:

```bash
export WDD_CACHE=/path/to/cache      # token caches and dictionaries, one folder per model tag
export WDD_RESULTS=/path/to/results  # where the JSON results and FINDINGS.log are written
export HF_HOME=/path/to/hf_cache     # optional, the Hugging Face cache
```

Build the cache for a model once (about a minute for the small models; the eval tokens are 32 sequences of 512 tokens of wikitext-2 test, the centering tokens are wikitext-2 train):

```bash
cd scripts && python build_cache.py gpt2
```

Model keys are the five above. A Pythia training checkpoint is cached under its own tag with `WDD_REV=step4000 python build_cache.py pythia410 pythia410_step4000` (the checkpoint scripts pass the revision themselves and read the base tag's tokens). A random-initialisation control is `WDD_RANDOM=1 python build_cache.py gpt2 gpt2_random`.

Run one experiment:

```bash
cd scripts && python e283_state_variable.py gpt2
```

It prints its log line and writes `results/e283_statevar_gpt2.json`. Replay a set in parallel with the skip guard (a run whose JSON exists is skipped, so the list is resume-safe):

```bash
cd scripts && ./wave2.sh 2 ../waves/waveAG.txt
```

The wave runner and `runjob.sh` hard-code the box paths (`/workspace/wdd/...`); edit the two paths at the top of each for another machine, or replay `RUNLIST.txt` directly with a loop over its lines. Every script's result name is the record name in `FINDINGS.log`, so any number in the documents can be traced to its JSON and to the script that wrote it.

Reading a result: each JSON has the fields named in the script's docstring and log line, plus `_exp` and `_time`. The `note` written to `FINDINGS.log` is the one-line summary the documents quote.

## The experiment map

The program ran as rounds, each answering the previous round's open questions. The rounds in brief (script ranges are approximate; the narrative in `docs/FINDINGS.md` has every one):

| Rounds | Scripts | Question |
| --- | --- | --- |
| 1 to 6 | e01 to e230 | What WDD reads: sparse approximation over the model's own atoms, prominence at birth, certificates, the erasure matrix, the learned per-block contraction, the exact residual identity, training dynamics. |
| 7 to 12 | e231 to e254 | The descendant: the footprint of a write at depth is neuron-specific, vector-determined, compositional, angle-preserving, robust, portable, zero-shot predictable; the transported dictionary fails as a basis; the linear transport operator; the massive-neuron boundary. |
| 13 to 17 | e255 to e272 | Function: descendant geometry predicts the write's effect better than write geometry; the descendant is a sufficient, causally active, continuously parameterising coordinate; the capacity ladder. |
| 18 to 25 | e273 to e295 | The descendant as a state variable: no functional null space, subspace nesting, the transport residual, interpolation without snapping, functional composition, the Euclidean metric, self-prediction of the future, the reaction carrier of the massive neuron, the compression test, the kill-tree. |
| 26 to 27 | e296 to e308 | The literature-derived program and the operator question: causal metric, persistent channels, layerwise dimensions, attention routing, propagation, cross-neuron geometry, second order, dose-response, family invariance; what the projection is not. |
| 28 to 30 | e309 to e316b | The quotient picture: the perturbation cloud's origin and curvature, the manifold sweep, the quotient by function, transfer across families, the WDD-to-quotient equation and its test, held-out observables, intervention without refitting, the functional atlas and the application test. |
| 31 | e317 to e356 | The quotient program: observable dependence, horizons, channels, nonlinearity, algebra, breaking, the functional dictionary, universal bases, null space and cosets, homomorphism, fibres, state-alone prediction, leakage audits, component internals, adversarial families, token topology, normalization, fine amplitude, discarded dimensions, compression baselines, cancellation, checkpoints, component transport, the light cone, kernels, one-to-many. |

## What was found, in order of confidence

Positive, replicated in five of five models unless noted:

1. WDD reads a write at birth (source identification 88 to 98% from the increment) and reading is a prominence phenomenon: it decays with a causal half-life of about two blocks and depends on the context re-writing the direction (rounds 1 to 6).
2. Every trained block is, along any direction, a learned diagonal contraction of a near-isometric mixing map: a write's energy is conserved or amplified while its direction is scattered into thousands of later writers and into attention, exactly accountably (the residual identity).
3. The descendant of a write, the footprint of ablating it at depth, is a vector-determined, compositional, magnitude-invariant, portable image of the write, identifiable to its source far past the point where sparse decomposition over transported atoms fails. Provenance survives on a coherent fraction that falls to 5 to 15% of the descendant's energy by depth; classification needs only that fraction, a decomposition would have to explain the rest.
4. The descendant is a state variable for the perturbation family: it predicts its own future two blocks later (cosine 0.43 to 0.78 through a data-free operator, against 0.03 to 0.12 for the original write) and reproduces the future logit effect; interpolated descendants follow the interpolation of the endpoint trajectories through the rest of the network; the transport law composes in state and in function.
5. The causally relevant content of the descendant is compressible: tens of directions suffice for identity, function, the future and the removal KL, about 128 for a linear reconstruction of the future descendant, the full width for the physical descendant. This holds for every perturbation family tried, sixteen of them, including perturbations built orthogonal to the model's natural residual manifold: the compressibility is a property of the residual stream's response, not of the write.
6. The compressed content is universal across observables (nine observables' quotients overlap at the split-half reliability ceiling), homogeneous and additive in its coordinate, linear along its directions to three to ten times the natural amplitude, read out nonlinearly, read by the later MLPs (which also damp the perturbation) and spread across tokens by attention alone, born by step 4000 of training together with the cross-family convergence, and it collapses rather than expands beyond the linear range.
7. The one exception is the massive-activation neuron of SmolLM2: its write transports by the ordinary law at every amplitude, but the natural response to removing it at its own tokens is anti-parallel to the response to adding it, an even (set-point) response carried by one MLP at block 10.

Negative, and decisive for how the positive results should be worded:

1. The compressed content has no privileged coordinate system. A projection fitted to shuffled targets scores the same as the real one; PCA, PLS, CCA and kernel PCA coincide; the discarded 64-dimensional complement decodes identity, function, the future and token identity as well as the 16-dimensional core; cancelling the core component restores nothing at the logits.
2. There is no null space of the response and no coset structure: the difference of two functionally equivalent perturbations is a full-strength perturbation; fibres of the quotient have the diameter and dimension of random sets.
3. The functional coordinate is not an eigenspace or singular space of the transport operator, not the read-out's high-gain subspace forward or backward, not the residual state's variance subspace, and not confined to the natural manifold.
4. WDD atoms are not a privileged basis of the functional content (no sparser, no more concentrated than random directions), their functional classes carry no vocabulary meaning, coordinate-equivalent atoms are not substitutable, and a block's functional coordinate is carried by the small-coefficient tail of its ledger, not its sparse head.
5. None of this yet explains the models' outputs better than direct logit attribution: the functional atlas built from the framework does not beat it for joint ablations.

## Theory

The documents keep two statements apart: the theorem-like object, which is mathematics conditional on assumptions, and the empirical discovery, which is what the five models showed. `docs/THEORY.md` tags every derived item by what it rests on.

### Objects

Residual stream of token t at level ℓ: x_ℓ(t) ∈ R^D, with a pre-norm block acting as x_{ℓ+1} = x_ℓ + f_ℓ(x_ℓ; X_{≤t}). The write of neuron i of block b at token t is c_i(t) ŵ_i, where ŵ_i is the unit row of the MLP down-projection and c_i(t) = a_i(t)·‖w_i‖ is the WDD coefficient. WDD writes the centred state at the birth level as a ledger over the model's own atoms,

```
x_{b+1}(t) − μ = Σ_i c_i(t) ŵ_i + r(t).
```

The descendant of a write at level ℓ is its natural footprint, d_i(ℓ, t) = x_ℓ(t)[with the write] − x_ℓ(t)[without it]; at ℓ = b + 1 the descendant is the write itself.

### The axiom (first-order transport)

Within the linear range,

```
d_i(ℓ, t) = c_i(t) · J_t · ŵ_i + O(c²),     J_t := ∂x_ℓ(t) / ∂x_{b+1}(t),     J_t = J̄_C + ΔJ_t,
```

with J_t the Jacobian of the downstream map along the token's own trajectory, split into a context-class mean and a token-specific fluctuation. Three measured properties make it concrete: (J1) a generic spectrum, singular values spread 11 to 50 times between the 90th and 10th percentiles, no exact null space, root-mean-square gain of order one; (J2) token dependence, the fluctuation term dominating by mid depth (coherent fraction 0.16 to 0.33 at mid depth, 0.05 to 0.14 late); (J3) a linear range up to about the natural amplitude, with distortion at 3 to 10 times.

The read-out is the un-averaged Jacobian lens, f_i(t) = G_{ℓ,t} · d_i(ℓ, t) with G_{ℓ,t} = W_U · ∂x_final(t)/∂x_ℓ(t), and G is nearly isotropic in gain on the descendant cloud.

### What follows, and what observed it

- Homogeneity, d(αŵ) = α d(ŵ), and additivity, d(Σ c_i ŵ_i) = Σ c_i d(ŵ_i), in the linear range (e250, e278, e241, e248, e337).
- Vector determination: d depends on ŵ, not on which neuron produced it, and every vector has a descendant (e236 to e248, e304).
- Sufficiency and causal carriage: the descendant is a sufficient statistic for the effect given the level, injecting it reproduces the effect, effects interpolate linearly along descendant segments and whole trajectories interpolate without snapping (e268, e269, e271, e280).
- The state-variable property: d(ℓ + k, t) = J_{ℓ→ℓ+k, t} · d(ℓ, t), so the present descendant predicts its future while the write does not (e283, e281).
- Angle preservation without isometry. For a pair (u, v_θ = cos θ·u + sin θ·r) with r random, cos(Ju, Jv_θ) = cos θ·ρ / sqrt(cos²θ·ρ² + sin²θ) with ρ = ‖Ju‖/‖Jr‖. Random bases have ρ ≈ 1 by concentration of measure, which is why pair cosines looked preserved; pairs on the operator's top and bottom singular directions have their cosines raised and lowered exactly as the formula says, within 0.07 in fifteen of fifteen cases (e238, e282, e285). "Isometric transport" in the record means this.
- Provenance versus decomposition as an inequality. With κ(ℓ) the coherent fraction, nearest-centroid identification among K candidates has signal-to-noise about sqrt(κ)·Δ / sqrt((1 − κ)·K/D_eff) and survives down to κ of order K/D_eff, while any reconstruction by a fixed dictionary has fraction of variance unexplained at least 1 − κ (e277, e272).

### The quotient and its equation

Reading a functional coordinate from any perturbation by a linear map on the linear range, z_ℓ(δ) = P_ℓᵀ δ, and composing with the ledger and the transport,

```
z_ℓ( Σ_i c_i(t) J_t ŵ_i ) = Σ_i c_i(t) z_i(t) ≈ Z_ℓ c(t),     Z_ℓ := P_ℓᵀ J̄ [ŵ_1 … ŵ_N].
```

Z_ℓ is the image of the WDD dictionary under transport and quotient. WDD reads c (which atom); the quotient reads Zc (what the atoms do). Tested by joint ablation of random subsets of atoms: the ledger-weighted sum of the atoms' coordinates meets the joint footprint's coordinate at cosine 0.54 to 0.68 in three models and 0.30 to 0.34 in two, against zero for random atoms, and the quotient is additive on unrelated atoms at 0.78 to 0.97 (e313).

### The correction the last rounds forced

The first-order model with a generic J and a near-isotropic G predicted the universality of the functional coordinate across observables and families and its additivity, and both held. It never said the coordinate is a privileged subspace, and the audits show it is not (e336, e338, e340, e346, e347, e348). The consistent reading is that G restricted to the descendant cloud is of moderate effective rank with a long tail (e355) and near-isotropic, so that any high-variance projection of the cloud of dimension a few tens suffices for prediction and none is structurally distinguished. P in the equations should be read as "any such projection". In one line:

```
x_ℓ(t)[with] − x_ℓ(t)[without] = c_i(t) [ J̄_C ŵ_i + ΔJ_t ŵ_i ] + O(c²),     f_i(t) = G_{ℓ,t} · d_i(ℓ, t),
```

with the causal information in d low-dimensional and its coordinates not unique. What the theory does not derive: the numerical values, which component reacts at the exception, why training shapes J as it does, and the sizes of the identity, function and behaviour subspaces beyond "function is low-rank because G is".

## Where this sits in the literature

`docs/RELATED_WORK.md` places the program against its nearest neighbours and states what is and is not new. First-order transport, Green-function propagation, composition and the local linear regime belong to Transformer Field Theory (Olivieri and Pérez Rodríguez, 2026); the learned spectral bottleneck of the residual Jacobian to Fernando and Guitchounts (2025, 2026); the causal-dimensionality wedge to Sarkar and Deka (2026); the averaged read-out to the Jacobian lens (Transformer Circuits, 2026); linear transport of learned features between layers to Activation Transport Operators (Szablewski, 2025); weight-space neuron description to ROTATE (2026) and parameter decomposition to SPD and VPD. What this program adds is the native-write anchor through that machinery, the separation of a write's identity, its transported descendant and its functional effect, the equivalence structure the read-out induces on arbitrary perturbations measured by held-out prediction and intervention, its acquisition off the natural manifold and its convergence across families, its independence from the operators' spectra, the equation z = Zc tested by joint ablation, and the negative results above, in the scope of five small models.

## Scope and caveats

Five models under 1.1B parameters, block-2 writers for most runs, one corpus (wikitext-2, with a Pile check), tokens in typical-norm range (attention-sink positions excluded), perturbations at the natural amplitude unless a sweep says otherwise, and decoders limited to nearest-centroid, kNN, ridge and closed-form kernels so that nothing is trained. Every number is a median over held-out tokens unless the script says otherwise. Retracted artifacts and superseded designs are listed in `docs/KILLED.md`; an unsigned transport residual (e275) and a mis-designed additivity reference (e285 part 3) were corrected by e275b and e285b, and a cross-token measure in e315 was discarded for dense injection. The one large outage of the run, two hours without network mid-program, did not lose results because the launcher is resume-safe.
