# 52. The cross-lingual dial and programmable feature placement (S121)

**Question.** Two uses of the row as a unit: does the gender dial, found on translation into English, reach agreement inside French, Spanish and German; and, since the carrier of a concept is a parameter, can the parameter be chosen by a loss.

**Established.**
- On Qwen2.5-0.5B the gender row negated moves agreement from 0.93 to 0.58 (French), 0.93 to 0.58 (Spanish), 0.84 to 0.53 (German); zeroed 0.89, 0.93, 0.70, doubled 0.94, 0.91, 0.84; the largest change from a random row is 0.00, 0.00, 0.00; Pile loss +0.0003 (e625).
- On Qwen2.5-1.5B-Instruct the gender row negated moves agreement from 0.94 to 0.94 (French), 0.98 to 0.97 (Spanish), 0.95 to 0.95 (German); zeroed 0.94, 0.98, 0.95, doubled 0.94, 0.98, 0.95; the largest change from a random row is 0.00, 0.00, 0.00; Pile loss -0.0002 (e625).
- Pythia-160m carries gender on row 2857 of block 4 (vote 1.00 over the contrast set), and negating that row drops held-out pronoun agreement by +0.343 from 0.86, a third family with the dial; the pinning loss on a designated quiet row (lambda 1) brings its vote share for the gender difference from 0 to 1.00 on the training subjects and 0.97 on held-out ones, at a validation cost of -0.008 nats over the control fine-tune (3.890), with the designated row's dial going from -0.000 to +0.203 and the handle's from +0.343 to +0.197; at lambda 5 the designated row's share is 1.00 / 1.00, its dial +0.179 and the handle's +0.159 at -0.020 nats over the control; the pinning term ends at 0.001 of the difference's energy unexplained; every fine-tune, the control included, raises the validation loss from 3.653 to about 3.890, so the costs are read against the control (e626).

**Start here:** e626 · **Sessions:** S121 · **Scripts:** `scripts/e625_crosslingual.py`, `e626_row_pinning.py`

## Experiments

| id | question | result | status | links |
| --- | --- | --- | --- | --- |
| e625 | The gender dial in French, Spanish and German: predicative adjectives and pronouns at the next token on twelve gendered subjects per language, the gender row negated, zeroed and doubled against eight random rows, on Qwen2.5-0.5B and 1.5B-Instruct | on Qwen2.5-0.5B the gender row negated moves agreement from 0.93 to 0.58 (French), 0.93 to 0.58 (Spanish), 0.84 to 0.53 (German); zeroed 0.89, 0.93, 0.70, doubled 0.94, 0.91, 0.84; the largest change from a random row is 0.00, 0.00, 0.00; Pile loss +0.0003; on Qwen2.5-1.5B-Instruct the gender row negated moves agreement from 0.94 to 0.94 (French), 0.98 to 0.97 (Spanish), 0.95 to 0.95 (German); zeroed 0.94, 0.98, 0.95, doubled 0.94, 0.98, 0.95; the largest change from a random row is 0.00, 0.00, 0.00; Pile loss -0.0002 | supported (the dial crosses languages on the 0.5B model) | ← e624 e622 · → 51 50 |
| e626 | Programmable feature placement: a pinning loss on a designated quiet row of Pythia-160m, asking the female-minus-male state difference to be carried by that row's write, against a control fine-tune; the vote share, the dial on held-out subjects, the validation loss | Pythia-160m carries gender on row 2857 of block 4 (vote 1.00 over the contrast set), and negating that row drops held-out pronoun agreement by +0.343 from 0.86, a third family with the dial; the pinning loss on a designated quiet row (lambda 1) brings its vote share for the gender difference from 0 to 1.00 on the training subjects and 0.97 on held-out ones, at a validation cost of -0.008 nats over the control fine-tune (3.890), with the designated row's dial going from -0.000 to +0.203 and the handle's from +0.343 to +0.197; at lambda 5 the designated row's share is 1.00 / 1.00, its dial +0.179 and the handle's +0.159 at -0.020 nats over the control; the pinning term ends at 0.001 of the difference's energy unexplained; every fine-tune, the control included, raises the validation loss from 3.653 to about 3.890, so the costs are read against the control | established (the carrier of a concept can be chosen) | ← e620 e622 · → 50 49 |

## How the results flow

- `e622, e624 → e625`: the gender row, taken into the languages the items came from.
- `e620, e622 → e626`: the dial as a property to install rather than find.

## Links to other areas

- [51 Blind edit detection and the dial in free text](51_edit_detection_free_text.md), [50 The attribute catalogue](50_attribute_catalogue.md), [49 Three parameter-side directions](49_parameter_side.md): the parameter-side line.
