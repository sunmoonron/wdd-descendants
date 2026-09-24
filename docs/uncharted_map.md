# Uncharted territory: a survey of e000-e431 (session 44)

Five read-only agents each skimmed about 90 experiments in groups of ten (index: `experiment_index.md`). For each group
they listed what it established, the results never linked to the current WDD picture ("orphans"), and one 1-5 minute
experiment to connect them, with a probability that it yields something new.

The orphans fall into five basins, ordered by how much probability mass the agents put on them.

## 1. The huge directions (M) as a hub (about 40% of the mass)

Orphans from every era point at the states' top principal directions:
- phase 1: sink tokens and massive channels (e020, e118, e126, e170). Phase 1 always dropped positions above 10x the
  median norm; phase 3 never did.
- phase 1: funnels of random perturbations (e224, e225, e229, e243).
- phase 2: the set point (e290, e301, e331), normalisation (e344), the common response direction (e329).
- phase 3: the knee (e372, e374), false friends (e404, e409, e411), Fisher lightness (e400, e419), the compounding
  replacement model (e392).
- model-specific anomalies: e381, e384, e387, e424, e428.

Agents' top picks: the sink confound (p 0.45), the knee of M (0.3), the set point (0.3), the junk funnel into M (0.3),
the replacement model with M exact (0.3).

Run this session: e432, e433, e437, e438, e441.

## 2. Where the vocabulary lives (about 25%)

Lexical or contextual (e117, e146, e162), increments or states (e02), the self path or the broadcast path (e352), and
the unreadable context features (e423, e425).

Picks: increments in own words (0.3), token-mean/context split (0.35), light cone (0.2).

Run this session: e435 (lexical surrogate), e436, e439, e440, e442.

## 3. Statistical or dynamical privilege (about 15%)

Own writes are generic levers (e304, e332, e282) yet privileged describers. The selection prior: is usage, Zipf
included, a matter of geometry (e189, e196, e202)?

Picks: the selection prior (0.3), the describer-by-lever 2x2 (0.15), the full-dimension e332 rerun (0.2).

Run this session: e434 (does the network keep its own words?), e435 (Zipf from covariance-matched states).

## 4. Development (about 12%)

The transient funnel at steps 2000-4000 against M_final (e229, e243), precursors as predictors (e411), late entrants
and usage (e360c), and stitching through the vocabulary (e378 against e396: same words, halves that do not compose).

Not run. It needs checkpoints the new box lacks, or long waits.

## 5. Applications (about 8%)

Model diffing by usage shift (e431), and a replacement model that does not compound (e392 with M exact).

Not run.

## What the agents' picks became

| Pick | Experiment | Outcome in one line |
| --- | --- | --- |
| sink confound | e432, e437 | Pythia's 0.86 was one sink token per sequence; M is 0.10-0.26 at typical positions; false friends were mis-described sinks |
| set point / ballast / temperature | e432 | all three killed |
| knee of M | e432, e438, e441 | real and specific to M's directions; attention carries a third to two thirds of the cost; no single seat |
| M as the identity channel | e433 | most lexical subspace, 65-91% predictable from block 0; the knee is not lexical |
| kept words | e434 | killed: retention tracks variance, not usage |
| selection prior / Zipf | e435 | Zipf shape reproduced by Gaussian states with the same covariance |
| sparse or second order | e435 | 55-85% of the advantage needs non-Gaussian joint structure |
| increments | e436 | words appear in a block's increment (step 4000) before the state (16000); at the end increments are less word-level than states |
| the lexicon | e439 | killed: removing the lexicon leaves the advantage; descriptions almost never name the current token (0-4%; OLMo 23%) |
| context vocabulary | e440 | exists: what context adds to a token is described word-level by non-lexical own words (all five) |
| light cone | e442 | the advantage holds on both the self and the broadcast path |
| per-block accents | e443 | not at the end (at most 0.33 of the gap), but 0.82 at step 1000: the early accent is per block |
