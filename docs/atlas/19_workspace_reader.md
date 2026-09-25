# 19. WDD as a workspace reader: WorkspaceBench's problems (S57)

**Question.** WorkspaceBench (LessWrong, 2026-09-23) scores activation-to-text readers on intermediates a model computes but never writes, and names the problems: single-token readers are a bag of words with no order and no multi-token concepts, expressive readers confabulate, and any reader can cheat by echoing the prompt. Area 15 had shown the native-word reader is not a better reader in general. Does what it has that no text generator has, provenance, a fidelity number per claim and a decomposition into separately readable parts, answer those problems? Qwen2.5-7B and its instruct variant.

**Established.**
- Trust. On two-hop bridges the native reader needs no certificate: 73 country claims at the subject token, all right, where the lens claims nothing; 0.99 at the final position for both (e478). On chained arithmetic no reader is reliable (precision 0.48-0.63 against a chance rate of 0.50) and no certificate, coefficient or causal receipt, separates fabricated digits from right ones (e478b).
- Prompt echo. A concept held in mind while copying a sentence is surfaced at the writing positions three times as often by native words as by the lens (0.30 against 0.10, precision 0.91), always on MLP rows and never on the concept's token embedding; under "do not think" nothing surfaces (e479). Provenance tells a computed representation from an echo.
- Losing nothing. Read word by word, the native description loses half of a finished answer (0.40-0.50 against the lens's 1.00); read as a whole it recovers it (0.85-1.00), and the union of the two readings keeps 0.80-0.90 within ten tokens (e480).
- Multi-token concepts. One native word, an MLP row, carries both tokens of a two-token country name in 0.31 of items and names the right second token in 24 of 25 cases, where the lens shows the second token in 0.04 (e481).
- Order. Without a question, the signs of native words do not carry who did what to whom; order is readable only from position and recency, by any reader (e482).

**Start here:** e481, e479, e480, e478b · **Sessions:** S57 · **Scripts:** `scripts/e478_certificate_lens.py`, `e478b_certificate_arith.py`, `e479_directed_modulation.py`, `e480_single_token_battery.py`, `e481_multitoken_concepts.py`, `e482_order_from_signs.py` (helper `ws_common.py`)

## Experiments

| id | question | result | status | links |
| --- | --- | --- | --- | --- |
| e478 | Do a native claim's coefficient, provenance and causal receipt separate right from wrong claims better than the lens's confidence (two-hop bridges)? | Nothing to filter: at the subject token native words make 73 country claims, all right, where the lens and rotated words make none (true bridge in the top 10 in 0.14 of items per cell); at the final position both readers claim 111 times at 0.99. All native claims ride on MLP rows | null (no wrong claims) | ← e420 e458 · → e478b |
| e478b | The same certificates where fabrications occur (chained arithmetic, e416's items)? | None works: digit-claim precision lens 0.63, native 0.48, rotated 0.55 against chance 0.50; AUC of z, rank, coefficient share, coefficient rank and receipts 0.43-0.59; keeping the better half by any of them leaves precision unchanged | refuted (no certificate) | ← e478 e416 |
| e479 | Is a concept held in mind while copying a sentence surfaced at the writing positions, and is it an echo of the prompt or computed? | Qwen2.5-7B-Instruct, 85 prompts: pass under "think" lens 0.10, native 0.30 (precision 0.91), MLP-row words only 0.30, token-embedding words only 0.00; under "do not think" 0.00 for all; carriers 12 of 12 MLP rows at block about 24 | supported (computed, not echo) | ← e420 e440 |
| e480 | Does the native reader keep what the lens captures on the benchmark's single-token families (basic, multilingual, typo, poetry), regex-scored? | 7B: lens 1.00 / 1.00 / 0.25 / 1.00; per-word native 0.50 / 0.40 / 0.33 / 0.40; the reconstruction read as a whole 0.95 / 0.85 / 0.25 / 1.00; union of top 5 and top 5 0.90 / 0.80 / 0.33 / 0.80; the planned rhyme at the end of the first line 0.00 for all. 0.5B passes few gates | mixed (whole reading keeps it, per-word loses half) | ← e415 e388 |
| e481 | Can a single native word carry a multi-token concept and disambiguate it? | Final position, 26 two-token countries: lens shows the second token in 0.04; native 0.42, both tokens in one word's top 10 in 0.31 (13 words, all MLP rows), the carrying word ranks the right second token in 24 of 25. Subject token: first token 0.19, second 0.08 | supported (final position) | ← e420 e478 |
| e482 | Do the signs of native words carry the direction of a described action without a question? | No: at the final period, mid blocks, every fixed rule is at 0.50-0.56 for every reader; at the object token all readers read the order from the current token (0.94-0.99); the lens reads it late by recency (0.88 at block 26); opposite signs in at most 0.62 of items, following position not role | refuted (position and recency only) | ← e417 e474 |

## How the results flow

- `e420 → e478 → e478b`. Native words surfaced the bridge at the subject token (e420); e478 asked whether their numbers can certify such claims and found no wrong claims to certify. Moved to the family where e416 had found fabrications, the certificates are flat: what the readers show there is not the chain's intermediate at all.
- `e440, e420 → e479`. Native words describe what context adds beyond the token (e440) and surface a hidden bridge (e420). Told to hold a concept while copying a sentence, the instruct model represents it on MLP rows at the writing positions, and the native reader sees it three times as often as the lens with no echo of the prompt token.
- `e388 → e480`. Sixteen own words keep most of the loss (e388), so the reconstruction read as a whole keeps the finished answer; it is the per-word pooling that loses it, because a finished answer is a distributed direction (the picture of e470 and e474).
- `e420 → e481`. The neuron that writes the bridge writes the whole name: one MLP row promotes both tokens, which no reading of the whole state through the unembedding can attribute to one source.
- `e417, e474 → e482`. e417 read the resolution of a role as suppression once a question fixed the answer; e474 found identity handles untyped by role. Without a question there is no role in the signs, only position and recency.

## Links to other areas

- [15 Workspace lenses](15_workspace_lenses.md): the same readers, items and helper; e478 and e481 extend e420, e478b extends e416, e482 extends e417.
- [17 Forensic uses](17_forensic_instrument.md): e478's certificates are e458's checksum turned on the reader's own claims, with the same outcome (no use beyond the base rate).
- [18 Recursion, algebra, geometry](18_recursion_composition.md): e474's untyped handles and e482's uninformative signs are two views of the same absence of role information in native words.
- [14 Native vocabulary](14_native_vocabulary.md): e479's computed-only reader is e440's context vocabulary put to a benchmark use.
