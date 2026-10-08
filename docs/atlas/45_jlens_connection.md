# 45. The J-Lens connection: class-conditioned descendants and transported writers (S114)

**Question.** The J-Lens reads a mid-layer activation at the output through a context-averaged Jacobian, and the J++ Lens found that one Jacobian per activation cluster is far more faithful, most of all early. WDD's descendants transport a write the same way, and a word's class is that cluster, read off the weights. Two tests: is the class-conditioned map the faithful one, and does transport change the one conclusion of the program drawn from raw rows at a single layer, that the harmful-request writers do not write a chat model's refusal direction.

**Established.**
- At block 4 of Pythia-410m the class-conditioned map predicts a write's actual logit change at held-out class positions at cosine 0.283, a random-subset map of the same size at 0.210, the all-positions map at 0.071, the raw row at 0.065; at block 12 0.482 / 0.388 / 0.295 / 0.135; the class map is worse outside its class (0.191 against 0.271) (e611).
- With transport the harmful-class writers' cosine with the refusal direction is 0.101 (raw 0.034, random rows 0.103), and silencing them on the input side leaves the projection at 0.88 of the original and refusal at 0.89 from 0.93, random rows 1.00 / 0.93 (e612).

**Start here:** e611, e612 · **Sessions:** S114 · **Scripts:** `scripts/e611_class_descendants.py`, `e612_transported_writers.py`

## Experiments

| id | question | result | status | links |
| --- | --- | --- | --- | --- |
| e611 | Class-conditioned descendants on Pythia-410m, blocks 4, 8 and 12: the raw row, the globally averaged Jacobian and the class-averaged Jacobian as predictions of a write's logit change, against the actual change at held-out class positions and outside the class | Cosine at block 4: raw 0.065, all positions 0.071, random subset 0.210, class 0.283 (class beats the random subset for 0.78); block 12: 0.135 / 0.295 / 0.388 / 0.482; outside the class the class map 0.191 against the random subset's 0.271 | narrowed (the class map is the most faithful, by an estimate effect first and a conditioning effect second; a map of its class, worse elsewhere) | ← the descendants · → 45 |
| e612 | Who writes the refusal readout with transport, Qwen2.5-0.5B-Instruct: each row transported to the direction's block by a Jacobian-vector product, the transported cosine and projection share, and the harmful-class writers silenced on the input side against random rows and the harmless-class writers | Transported absolute cosine: harmful-class writers 0.101 (raw 0.034), harmless-class 0.083, random 0.103; share of the projection -0.05; silenced: projection 0.88, refusal 0.89 from 0.93 (random rows 1.00 / 0.93) | supported (e607's verdict stands with transport) | ← e607 · → 44 |

## How the results flow

- The descendants work `→ e611`: the transport map conditioned on the class, judged by the causal test the J++ Lens uses.
- `e607 → e612`: the raw-row measurement of who writes the refusal direction, redone with the writes transported to the direction's block and with input-side silencing.

## Links to other areas

- [44 The refusal direction, writer rescue, the causal test at 410m, and the baseline at 1B](44_refusal_direction_rescue.md): the raw-row verdict e612 re-examines.
- [31 The coalitions](31_coalitions.md): the writes a transport map carries forward.
