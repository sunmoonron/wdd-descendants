# 51. Blind edit detection and the dial in free text (S120)

**Question.** Two things the catalogue left open: whether the dictionary can find a one-row edit without the original model (the note's proposal), and whether the gender row's dial reaches free text and composes with another row.

**Established.**
- Blind, with no reference model, a negated row is not found by consistency: over 1280 block-4 rows with a floor-based class (89 vocabulary words) the row-chord cosine has median -0.004 on the unedited model, so a sign flip has nothing to disagree with; the word-chord detector ranks a negated word row at top-1 in 0.00 and top-10 in 0.00 of 10 negations (4 of 10 negated words drop out of the vocabulary), the naive top-activation detector at top-10 in 0.00, and negated non-word rows at top-10 in 0.00 / 0.00; a doubled row is found by the norm z-score at top-10 in 1.00 (words) and 1.00 (non-words) and by neither consistency detector (0.00 / 0.00) (e623).
- In free text the gender row (block 4, row 3019, vote 0.88) negated moves the probability of the subject's pronoun from 0.96 to 0.81 (zeroed 0.95, doubled 0.96; random rows 0.96 to 0.96) and the generated agreement from 0.99 to 0.88, with degenerate generations 0.00 to 0.00 and a Pile loss change of +0.0003 (e624).
- In the weights the two rows compose only partly: the gender row negated flips gender in 0.34 of the items and the generation row (block 4, row 599, vote 0.64) flips generation in 0.17, both negated give the doubly flipped word in 0.05, gender alone in 0.26 and generation alone in 0.14 (e624).

**Start here:** e624 · **Sessions:** S120 · **Scripts:** `scripts/e623_edit_detection.py`, `e624_free_text.py`

## Experiments

| id | question | result | status | links |
| --- | --- | --- | --- | --- |
| e623 | Blind detection of single-row edits: forty blinded edits of block-4 rows (word and non-word rows negated and doubled), three detectors that see only the edited model (the row-chord cosine over rows with a floor-based class, the naive top-activation cosine over every neuron, the norm z-score), scored by the edited row's rank and false positives | blind, with no reference model, a negated row is not found by consistency: over 1280 block-4 rows with a floor-based class (89 vocabulary words) the row-chord cosine has median -0.004 on the unedited model, so a sign flip has nothing to disagree with; the word-chord detector ranks a negated word row at top-1 in 0.00 and top-10 in 0.00 of 10 negations (4 of 10 negated words drop out of the vocabulary), the naive top-activation detector at top-10 in 0.00, and negated non-word rows at top-10 in 0.00 / 0.00; a doubled row is found by the norm z-score at top-10 in 1.00 (words) and 1.00 (non-words) and by neither consistency detector (0.00 / 0.00) | refuted (a sign flip is invisible to consistency; only scaling is found, by the norm) | ← e620 e622 · → 49 50 |
| e624 | The gender dial in free text (the subject's pronoun probability over he and she, and the first generated pronoun, on 72 prompts) and the composition of the gender and generation rows in the weights on the kinship items | in free text the gender row (block 4, row 3019, vote 0.88) negated moves the probability of the subject's pronoun from 0.96 to 0.81 (zeroed 0.95, doubled 0.96; random rows 0.96 to 0.96) and the generated agreement from 0.99 to 0.88, with degenerate generations 0.00 to 0.00 and a Pile loss change of +0.0003; in the weights the two rows compose only partly: the gender row negated flips gender in 0.34 of the items and the generation row (block 4, row 599, vote 0.64) flips generation in 0.17, both negated give the doubly flipped word in 0.05, gender alone in 0.26 and generation alone in 0.14 | supported (the dial reaches free text; the rows compose only partly) | ← e620 e622 · → 50 |

## How the results flow

- `e620, e622 → e623`: the edits the catalogue makes, hidden from the detector.
- `e620, e622 → e624`: the gender row, taken out of the translation template.

## Links to other areas

- [50 The attribute catalogue](50_attribute_catalogue.md) and [49 Three parameter-side directions](49_parameter_side.md): the rows and the edits.
