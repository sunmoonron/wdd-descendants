# 40. The mechanism of the turn, the removed speaker at scale, and the paper (S109)

**Question.** Three things were left after session 108: the learning signal behind the row's turn toward its class, the replaceability of a speaker in a real model, and the write-up. All three here; the paper is [`PAPER.md`](../PAPER.md).

**Established.**
- The turn is a small, consistent drift driven by the class's own positions: two checkpoints before recruitment the eventual row's gradient step points toward its class's direction at mean cosine +0.020, 3.8 standard errors above zero and above matched non-recruits (+0.008), random rows (+0.002) and a class-shuffled null (-0.002); the class's positions supply 0.215 of the row's gradient at 0.006 of positions, and that part of the step points at the class (+0.018) (e589).
- At scale, with thirty speakers removed and frozen at step 8000 and 2.5M tokens of continued training, 0.17 of their classes have a substitute speaker (removal alone 0.23, unfrozen continuation 0.30); the loss at the classes goes 3.583 -> 3.588 -> 3.636 (e590).
- The paper states the record as one falsifiable sequence with the network as its subject: structure, alignment, recruitment, distributed implementation, replacement; WDD as the instrument that follows the parameter.

**Start here:** e589, e590, PAPER.md · **Sessions:** S109 · **Scripts:** `scripts/e589_gradient_turn.py`, `e590_removed_speaker.py`

## Experiments

| id | question | result | status | links |
| --- | --- | --- | --- | --- |
| e589 | The learning signal behind the turn: for 189 eventual rows two checkpoints before recruitment, the loss gradient of the write column decomposed by position, the step against the class's row-free direction and the row's 16000 direction, against matched non-recruits, random rows and a class-shuffled null | Step vs class: mean cosine +0.020 (3.8 s.e.; matched +0.008, random +0.002, shuffled -0.002); the class's positions supply 0.215 of the gradient at 0.006 of positions (matched 0.086, random 0.039); step vs the 16000 direction +0.002 | narrowed (a small consistent drift driven by the class's own positions; not a jump) | ← e571 e587 · → 25 26 |
| e590 | The removed speaker at scale: 30 words' write columns zeroed and frozen at step 8000, 600 proxy steps of Pile training (2.5M tokens), against random rows removed, nothing removed and the removal alone; substitutes, their rank at 8000, the loss at the classes | Substitute speakers: removed alone 0.23, removed and trained 0.17 (rank 2, top 10 1.00), random rows removed 0.30, unfrozen 0.30; loss at the classes 3.583 -> 3.588 -> 3.636 (unfrozen 3.636) | narrowed (replacement not reached within the proxy's 2.5M tokens) | ← e586 e587 · → 39 |

## How the results flow

- `e571, e587 → e589`: the row turns toward the cloud's variance and the nearest row is recruited; here the gradient behind the turn, decomposed by position.
- `e586 → e590`: the blocked recruit in the toy, then the removed speaker in the real model with a proxy continuation.
- Everything → PAPER.md.

## Links to other areas

- [39 Responsibility migration read directly](39_migration.md), [25 Training's hand](25_training_motion.md), [26 The drift's cause](26_drift_cause.md): the row's motion and its gradient at the origin, which e589 joins.
- [35 Theory extraction](35_mechanism_map.md): the map the paper condenses.
