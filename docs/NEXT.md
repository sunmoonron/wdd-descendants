# Backlog: experiments not yet run (compiled after session 55)

This list was compiled when the session-55 box was retired. It gathers what the program left open: threads opened by sessions 53-55, the causal-abstraction test deferred in session 52, the unrun steps of [`VISION.md`](VISION.md)'s roadmap, and old items still open in [`GRAPH.md`](GRAPH.md). Proposals from the relayed reviews that repeat atlas results are not listed; each session's triage in [`FINDINGS.md`](FINDINGS.md) says why.

Each item gives the question, where it comes from, a design at the program's usual scale (a few minutes on one A100 unless marked), the controls, and what would count as an answer. The ids B1-B16 are backlog labels, not experiment numbers; the next experiment is e510. B1 and B3 were run in session 56 (e476, e477); their entries say what came out.

## Priority 1: threads opened by sessions 53-55

**B1. What does the native-only profile signal encode?** (H299, e475) — *Audited in session 56 (e476): not an artefact, not a proxy for norm, position, confidence, frequency or identity; a third to a half is fit and prominence; +0.07 to +0.11 remains open. Next: do positions dominated by one strong MLP write share a behavioural regime?*
- At the deepest depth, how concentrated a native description is predicts next-token similarity beyond activation distance and the entropy gap: partial +0.13-0.14 in GPT-2, Qwen and SmolLM2. Rotated words' profiles give nothing, and so do shallower depths.
- Design: at 3/4 depth, compute per position the native profile's top-word share, Gini and effective number of words. Correlate them with candidate causes:
  - the state's norm, and its share in the huge directions M (area 10);
  - the top prediction's probability;
  - whether the next token repeats an earlier token (copying or induction), and the token's frequency;
  - the position, and the distance to the nearest sink;
  - the block and type that wrote the top word.
- Then re-run e475's partial correlation with each candidate controlled for.
- Control: the rotated profile.
- Answer: a candidate that removes the +0.13, or none.

**B2. Which variables do native words carry better than generic coordinates?** (e469, e473, e474)
- So far, native words beat rotated words by far for a translated concept (e469). They beat rotated words for gender and generation but only tie task PCA (e473). They are no better than rotated words for an entity copied into the answer (e474).
- Design: one interchange battery using e469's recomputed interchange, blocks 0 to the middle, k = 1, 4 and 16, comparing native, rotated and task PCA. Variables:
  - grammatical number (is / are agreement);
  - a country's capital (a factual relation);
  - the ones digit of a two-digit sum (numeric);
  - tense;
  - sentiment;
  - e469's concept and e474's entity as anchors.
- Models: Qwen, SmolLM2, and GPT-2 where it answers.
- Hypothesis to test: native words win for semantic features written by MLPs, and not for lexical or positional ones. As a covariate, record the share of each variable's top carriers that are token embeddings against MLP rows.

**B3. Are attribute handles shared across words?** (e473) — *Done in session 56 (e477): the gender handle is shared across words and languages (98% at k = 4); the generation handle transfers only about half (48%).*
- Does the gender handle found on king/queen move father/mother?
- Design: e473's items. For each item, take a native basis from the other quadruples' gender differences (OMP on their mean at each block) and apply it with the item's own source.
- Compare with the task-PCA gender direction and with the item's own basis.
- Answer: whether the vocabulary has a shared "gender word" or only item-specific carriers.

**B4. Word algebra beyond Qwen.** (e473)
- SmolLM2 translated too few kinship words (20 items).
- Design: a two-attribute English task all five models can do, for example number and gender of a noun ("the king / kings / queen / queens ..."), read out separately (is / are for number, he / she / they for gender).
- The same G, A, GA and compose interventions as e473, with rotated and PCA controls.
- Also Qwen2.5-7B on e473's translation task.

**B5. Typed handles, redesigned.** (e474)
- For subject items, "same role, other co-entity" was the item's own basis, because the subject precedes the co-entity.
- Design: put a context noun before the subject ("Near the {Y}, the {e} chased the {X}.") so that subject items get a real other-context control.
- Also report the transfer when the interchange starts at later blocks, where the carrier words of the two roles have diverged. e470 suggests that late windows barely work even within a role, so keep this part small.

**B17. The native workspace reader, judged.** (session 57, e478-e482)
- Define the reader as the union of the whole-reconstruction lens (what is finished) and the per-word lens with provenance (what is being computed), and run it on WorkspaceBench's judged families with an LLM summariser that is told which words are computed and which copied. Needs the benchmark's repository, a judge, and a model that passes its gates (the 27B does not fit the 40 GB box in bf16; a quantised load or a smaller capable model would).
- Cheap first step on the box: the multi-token unit words of e481 on more entities (people, works, compounds), and whether the unit word's block is the same across entities of one kind.

**B18. The provenance factor as a training monitor.** (session 58, e492) — *Its decomposition into the writers' and the non-writers' part across five Pythia checkpoints is done (e493); the dense early checkpoints and OLMo's are done (sessions 60-62: the accent by Pythia step 512, the words from step 1000; the same order in OLMo); the toy models remain.*
- One number per checkpoint and block, from two dictionaries and 2,000 states, no splicing: the native over rotated slope of the competitor maximum. Run it on dense Pythia checkpoints (steps 256 to 4000) and on the toy models of the vision round (e444, e449) to see whether it rises with the induction transition and with the writer rows' training, as VISION item 3 asks.

**B19. What the independent components are made of.** (session 58, e489)
- Decompose each FastICA component over the native dictionary (OMP, k = 4): if a component is a few rows of one block, it is a co-writing group; report the blocks and the co-selection of those rows (area 12).

**B20. The provenance layer under SAEs, extended.** (session 59, e494) — *Other layers, the activation question and a second family are done (sessions 60-62, e495-e496, e499); other models need public SAEs, and the descendants question in reverse remains. The TopK-against-ReLU question is answered: the features' sparsity (session 63, e501).*
- Other layers and models with public SAEs; whether a feature's activation follows its top row's activation (does the feature fire when its writer fires?); and the descendants question in reverse, which features at a later layer are the transported images of a given row's write.

## Priority 2: the causal-abstraction test on a known algorithm (deferred in session 52)

**B6. Native words as coordinates for the variables of a known algorithm.**
- e469 showed that native words are interchange coordinates for a noun. The full test takes a task whose high-level algorithm is known and asks whether native words carry each of its variables where that variable is computed.
- Candidate tasks:
  - two-digit addition (variables: the carry, the ones digit);
  - indirect object identification in GPT-2 (the duplicated name, the indirect object's position; the circuit is known);
  - variable binding ("a = 3, b = 5; a + b =").
- Design: the interchange accuracy per variable and position, for native, rotated and task PCA words, with e470's window sweep. No training (no DAS).
- Answer: the positions and blocks at which each variable has native carriers.

## Priority 3: the roadmap in VISION.md, still unrun

**B7. Concept dictionaries at scale** (VISION 1).
- Mine concept words automatically: native words shared across translations of the same sentences and rare elsewhere, over a few hundred sentences in four languages, for all five models and Qwen2.5-7B.
- Report the number of concept words, and how stable they are across sentence splits.
- About 10-20 minutes.

**B8. A vocabulary card per model** (VISION 2).
- One page per model built from existing scripts: the self-description curve by depth (own, rotated, covA), the huge directions with sinks excluded, the concept-word list, and word frequencies.
- Minutes per model. This is a compilation, not a new question.

**B9. Vocabulary formation around the induction transition** (VISION 3).
- Pythia-410m at dense checkpoints (256, 512, 1000, 1500, 2000, 3000 and 4000).
- Measure the word-level advantage (own against covA and rotated) against an induction score.
- Answer: whether words appear with the induction heads. The quotient is born by step 4000, together with the induction transition (e349, e365, e373).

**B10. Freezing writers while pre-training on real text** (VISION 4; about 30-60 minutes).
- A 4-layer, d = 256 model trained on wikitext with its writer rows frozen at random, against a normally trained one.
- Follow the per-block accent turning into words (e443's measures) through training.

**B11. A functional self-description objective** (VISION 5; about 15-20 minutes).
- Fine-tune GPT-2 small for a few hundred steps with an auxiliary loss: the LM loss when the middle block's output is replaced by its 16-word description. Hold the supports fixed per batch, so the loss is differentiable through the least-squares refit.
- Measure the cost in LM loss against the gain in loss recovered.

**B12. An entropy-coded native codec** (VISION 6; CPU only).
- Entropy-code the indices of the native codes (e450) using their Zipf-like frequencies, per block.
- Measure bits per state at equal loss recovered, against quantised PCA.

**B13. Steering with combinations of native words** (VISION 7).
- Use e473's gender and generation words as natural-size steering vectors, added rather than interchanged, alone and combined.
- Compare with dense differences of means at equal displacement (e458's protocol).

## Priority 4: old open items

**B14. Additivity against total norm** (H84, e285b, open).
- e285b could not separate norm from write count because of a cross-token confound.
- Design: inject the sum of m writes at one token only, with the total norm fixed (m = 1 to 16), and at a fixed m with the norm varied.
- Measure the relative additivity error against the sum of single-write effects, measured the same way.

**B15. Memorisation fingerprints** (skipped in session 47 as not quick).
- Do memorised continuations in Pythia use a distinct native vocabulary?
- Needs EleutherAI's memorisation lists and loss-matched non-memorised controls, since memorised text is mostly boilerplate.

**B16. A write-up item, not an experiment:** the 20-30-paper novelty matrix deferred since session 31.

**Session 64 leaves (e502-e506):** (a) the count of words: the centred chord puts 11-21 atoms above the floor in GPT-2 where the state has 40-107, so attention's writes, in the head bases, add words; the full ledger through the Gram (MLP rows, head-basis coefficients from the attention outputs, embeddings, biases) would be the exact test; (b) the contraction's size: why the crowd removes about half of the chord in GPT-2 and OLMo and less in Pythia, and whether area 05's linear gain predicts it; (c) why Pythia-160m's features are not state-like (e504): coordinates, the massive dimensions, or the SAE's training data; (d) the chord at matched amplitude across models.

**Session 65 leaves (e507):** is the crowd's flip from amplifying the chord (+1.4 at Pythia step 512) to contracting it (-0.64 at the end) area 05's learned contraction (e194, e210, e217: a linear gain of -0.16 to -0.42 per block, born as the warmup ends)? The test is to measure the crowd's gain along the chord and the block's linear response gain at the same checkpoints and blocks and ask whether they move together and match in size.

**Session 66 leaves (e508, e509):** the learned contraction is anisotropic: it removes two to three times more of a random direction than of the chord. Which directions does it spare, and are they the vocabulary? The test is the MLP's directional gain (area 05's measure) along the native atoms against random directions, by atom type and by usage as a word, at the end of training and across Pythia's checkpoints; a second question is what makes OLMo's contraction target the chord's extremes (a saturation of the largest projections) where GPT-2's is exchangeable.
