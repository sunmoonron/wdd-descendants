# 18. WDD applied to itself: fixed points, word algebra, role typing, induced geometry (S55)

**Question.** What does WDD show when it is applied to its own objects? The candidate uses are: repeating the description as an operator, combining native handles for separate attributes, moving an entity's handle between grammatical roles, and using WDD coordinates as a distance between states.

**Established.**
- WDD applied to itself is a projection. After one application 96-100% of 16-word descriptions choose the same words again, and loss and energy stop changing, for native and rotated words alike. There is no drift, collapse or attractor. Native descriptions are only more stable to small noise: 5% noise keeps about 13 of 16 native words and about 10 of 16 rotated ones (e472).
- Word algebra holds in Qwen. For kinship words that carry gender and generation, four native words per block for the gender difference change only the gender in 92% of items, and those for the generation difference only the generation in 75%. Gender taken from one word and generation from another give the doubly changed word in 74%, and behaviour is additive within 24%. Rotated words fail at this size; the task's principal directions do as well as native words (e473).
- Identity handles are not typed by grammatical role. A handle found with the entity as subject works as well with it as object, and the reverse, though the carrier words shared between roles fall to about two thirds at the middle block. On this copy-style task native words are no better than rotated ones (e474).
- WDD does not induce a better geometry of states than activation distance. Its distances predict next-token similarity worse and add only +0.04 to +0.09 beyond activation distance, nearly as much as rotated words. One native-only signal is left: at the deepest depth, the shape of a description's coefficient profile, without word identities, adds +0.13-0.14 beyond activation distance and the entropy gap (e475).

**Start here:** e473, e475 · **Sessions:** S55 · **Scripts:** `scripts/e472_wdd_fixed_point.py`, `e473_word_algebra.py`, `e474_typed_handles.py`, `e475_three_geometries.py`

## Experiments

| id | question | result | status | links |
| --- | --- | --- | --- | --- |
| e472 | Is the 16-word description a fixed point when WDD is applied again, and do perturbed descriptions converge (the recursive fixed point and attractor)? | A projection: after one application 96-100% of descriptions repeat, loss and energy unchanged, native and rotated alike; noisy runs freeze at their first description; 5% noise keeps Jaccard 0.67-0.71 of native words, 0.46-0.48 of rotated | refuted (a projection, not a native attractor) | ← e09 e122 e395 |
| e473 | Do native handles for two attributes of one word act independently and compose (word algebra)? | In Qwen, yes: with 4 words per block the gender handle alone flips only gender in 92%, generation 75%; gender from one word plus generation from another gives both 74% (16 words: 83%); rotated 6%, 25%, 16%; task PCA 91%, 80%, 91%. SmolLM2 underpowered (20 items) | supported (Qwen; task PCA equal) | ← e469 e470 e256 |
| e474 | Are native identity handles typed by grammatical role? | No: a handle from the other role switches as many answers as the item's own (Qwen 0.18 against 0.17 for subjects, 0.32 against 0.31 for objects at 4 words; SmolLM2 alike); carrier words shared between roles 0.61-0.66 at the middle block; rotated as good or better (16 words: 0.84-0.97 against 0.72-0.96) | refuted (untyped; no native edge here) | ← e469 e470 |
| e475 | Does WDD induce a geometry of states that predicts behaviour beyond activation distance? | Barely: activation cosine predicts next-token similarity best (Spearman +0.13 to +0.37); native coefficients or word sets add partial +0.04 to +0.09, rotated +0.03 to +0.07. Native only: the coefficient profile adds +0.13-0.14 at the deepest depth, beyond the entropy gap | narrowed (a coarse activation geometry) | ← e151 e169 e460 |

## How the results flow

- `e09, e122 → e472`. A second pass of WDD on the residual finds no new writes (e09, e122). Applied to the description itself, WDD returns the same description: OMP on a vector lying in the span of its chosen words picks those words again, whatever the dictionary's provenance.
- `e469, e470 → e473`. A few native words carry a translated noun across contexts during its writing window (e469, e470). For a word with two attributes, the words for each attribute can be set separately and combined from different source words. The task's principal directions do the same, so the composition belongs to the representation, which the native words read without supervision.
- `e469, e470 → e474`. The same interchange moves an entity's identity as well with a handle found in the other grammatical role. The carrier words diverge with depth, but the interchange acts in the early blocks where they still agree. Here rotated words do as well, so the native advantage of e469 depends on the variable.
- `e151, e169, e460 → e475`. Support overlap was known to predict the next token about as well as state cosine (e151, e169), and near-identical states with different ledgers behave alike (e460). With rotated and PCA controls, WDD's distances are a coarse version of activation distance whose small extra information is mostly shared with rotated words.

## Links to other areas

- [17 Forensic uses](17_forensic_instrument.md): e473 and e474 extend e469-e470's interchange coordinates to two attributes of one word and to grammatical roles; e475 re-runs e460's ledger-versus-cosine question with rotated and PCA controls.
- [14 Native vocabulary](14_native_vocabulary.md): e472 shows that the self-description advantage of e388 and e395 is settled in one application; repeating it adds nothing.
- [01 Method](01_method.md): e472's fixed points are OMP's recovery of a vector that lies in the span of its own support (e03, e63); e475 adds a rotated control to e151 and e169.
- [07 Descendants II](07_descendants_function.md): e473's behavioural additivity of native handles parallels the additivity of descendant effects (e256, e271).
