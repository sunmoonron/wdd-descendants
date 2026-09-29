# 39. Responsibility migration read directly: the relay, the geometry of recruitment, the blocked recruit, and the forecast (S107-S108)

**Question.** After session 106 the developmental claim waited on one more autoencoder control. Three experiments outside that loop instead, each able to end a thread: how responsibility for a class actually migrates between rows, with the class's own direction as the invariant; whether the row that will be recruited is already the nearest available row thousands of steps ahead; and, in the toy, whether a class blocked from its row is re-implemented in another.

**Established.**
- The latent object is the class with its direction, not the parameter: the class's own direction is at cosine 0.85 with its step-16000 direction at step 4000, when its eventual row is the speaker for 0.06 of classes, and at 0.86 at the final model, when the 16000 row is still the speaker for 0.38; a class has one speaker at the median after a gap of 8 checkpoints, and the rare handoffs pass through overlaps (e585).
- Recruitment is the geometry's: two checkpoints before recruitment the eventual row is the first of 53,248 rows by its ratio over the class for 0.52 of words and in the top ten for 0.84; eight checkpoints before, 0.62 (e587).
- The row is replaceable: with its 32 eventual words frozen the toy reaches test accuracy 1.00 with a full vocabulary on other neurons; the frozen rows are words at the end for 0.03; a third of the base classes re-appear on substitutes at profile correlation over 0.5 (median 0.38, random pairs 0.01), against half for the natural sharing of a class between two words in the unfrozen run (e586).
- The forecast (S108): with the classes at t from the states alone (no rows) the row that will be recruited two checkpoints later is picked at AUC 0.927 on Pythia-410m and 0.932 on OLMo-1B (eight ahead 0.869 / 0.848; norm 0.423 / 0.475); the row's own S forecasts at 0.926 / 0.965 and the class side adds +0.006 / +0.003; the clock at Spearman 0.04 / 0.04 (e588).

**Start here:** e588, e587, e585, e586 · **Sessions:** S107-S108 · **Scripts:** `scripts/e585_relay.py`, `e586_block_recruit.py`, `e587_recruit_geometry.py`, `e588_forecast.py`

## Experiments

| id | question | result | status | links |
| --- | --- | --- | --- | --- |
| e585 | The relay: for 253 classes over sixteen checkpoints and the final model, the speaker row at each checkpoint, handoffs, overlaps and gaps, and the class's own direction against the speakers' identity | One speaker at the median (0.31 of classes more), 8 checkpoints of gap, 187 handoffs of which 183 through overlaps; the class's direction at cosine 0.85 / 0.93 / 0.86 with its 16000 direction at 4000 / 8000 / the final model, the 16000 row the speaker for 0.06 / 0.35 / 0.38 | established (the class with its direction is the invariant; the row is its replaceable implementation) | ← e557b e584 · → 36 38 |
| e586 | Block the recruit in the toy: training resumed from step 1500 with the 32 eventual words frozen, with 32 random non-words frozen, and unfrozen; does it still grok, and do the classes re-appear on substitutes? | Test accuracy 1.00 / 1.00 / 1.00; frozen words that are words at the end 0.03; each base class's best substitute at profile correlation 0.38 (over 0.5 for 0.34; random pairs 0.01; unfrozen run's same row 1.00) | narrowed (substitution is partial) | ← e576 e444 · → 23 |
| e587 | Recruitment as geometry: the eventual row's rank among 53,248 rows by cosine with the class's direction, by its ratio over the class and by its count over the floor, 0-8 checkpoints before recruitment | Two before: first of all by ratio for 0.52, top ten for 0.84 (by cosine top ten for 0.78); five before top ten 0.72; eight before 0.62 | established (the recruit is the nearest available row, predictable thousands of steps ahead) | ← e571 e584 · → 25 31 |
| e588 | The forecast with no future in the class: row-free clusters at t (K = 512 / 128), each row's pull toward the class it would own, the target the words at t + k, against the row's S, norm and usage, on Pythia-410m and OLMo-1B; the clock from the pull's slope | AUC class-side 2 / 4 / 8 ahead 0.927 / 0.907 / 0.869 (OLMo 0.932 / 0.901 / 0.848), S 0.926 (0.965), combined 0.932 (0.967), norm 0.423 (0.475); precision at 100 0.05 / 0.10; clock Spearman 0.04 / 0.04 | supported (forecastable, below the pre-registered level) | ← e587 e548 · → 32 |

## How the results flow

- `e584 → e585, e587`: the histories of the eventual words, then the class's direction followed as the invariant and the recruit's rank followed backward in time.
- `e576 → e586`: frozen trained rows could not bring their classes to a fresh network; here the rows about to be recruited are frozen in the network that would have used them, and the classes take other rows.
- Together with area 36: the data gives the class and its direction, the nearest row is recruited to it, and the row is replaceable.
- `e587 → e588`: the post-hoc rank made a forecast, the future taken out of the class, and run on a second family.

## Links to other areas

- [38 Watching words being born](38_born.md), [36 The object](36_object.md): the histories and the object these three experiments read directly.
- [25 Training's hand](25_training_motion.md), [31 The coalitions](31_coalitions.md): the row's motion toward the cloud and the coalition's precedence.
- [23 The onset of wordhood in training](23_training_sweep.md): the toy.
