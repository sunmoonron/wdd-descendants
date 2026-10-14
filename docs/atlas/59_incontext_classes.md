# 59. Classes defined in the prompt (S132)

**Question.** With the weights frozen no row can be recruited. Does a class that exists only in the prompt get a native word, borrow its label's words, ride reusable slot rows, or travel without a word?

**Established.**
- A class defined only in the prompt (eight objects sorted into two groups named by arbitrary capital letters) is carried at the object's later, letter-free mention in 4 of 6 models (Pythia-410m, Pythia-1.4b, OLMo-1B, SmolLM2-360M; leave-one-out decoding of the group from twin prompts 0.96, 0.89, 0.88, 0.85 at the clearest depth, against null means of 0.49 to 0.50), and not at any tested depth in GPT-2 medium and Qwen2.5-0.5B, which still answer the group question (0.83, 0.84) (e641).
- Where it is carried the identity code has no native word: the class direction's nearest write row sits at a median percentile of 0.21 to 0.76 among the same prompt's random-split directions, and it is not the letters' own features (median cosine with the letters' contrast +0.04 to +0.12), while the same statistic finds the animal-object direction's row above every random split in 5 of 6 models, a row whose top tokens read animal words in several (e641).
- The ordinal code (which group was named first) is shared across prompts with different letters and objects in the carriers (mean pairwise cosine up to Pythia-410m 0.74, Pythia-1.4b 0.55, OLMo-1B 0.42, SmolLM2-360M 0.37, nulls 0.01 to 0.07), a binding-ID-like code; its average has a nearest row above its null in OLMo-1B, SmolLM2-360M at some depth, a lead only, because the null for an average of random splits is less structured than any real code (e641).
- Reading: in the models that carry an in-context class forward at all, it travels as a code that the model's own write rows do not speak, as the recruitment law predicts with the weights frozen; WDD, and any tool built on native words, is blind to what a model learns in context.

**Start here:** e641 · **Sessions:** S132 · **Scripts:** `scripts/e641_incontext.py` (standalone: torch, transformers, numpy; ran on a Quadro P4000)

## Experiments

| id | question | result | status | links |
| --- | --- | --- | --- | --- |
| e641 | Six models: eight objects sorted into two letter-named groups, twins with the letters swapped or the naming order swapped, the twin differences at later letter-free mentions; gate, nearest-row statistic against random splits, the letters' contrast, agreement across prompts; an animal-object control | a class defined only in the prompt (eight objects sorted into two groups named by arbitrary capital letters) is carried at the object's later, letter-free mention in 4 of 6 models (Pythia-410m, Pythia-1.4b, OLMo-1B, SmolLM2-360M; leave-one-out decoding of the group from twin prompts 0.96, 0.89, 0.88, 0.85 at the clearest depth, against null means of 0.49 to 0.50), and not at any tested depth in GPT-2 medium and Qwen2.5-0.5B, which still answer the group question (0.83, 0.84); where it is carried the identity code has no native word: the class direction's nearest write row sits at a median percentile of 0.21 to 0.76 among the same prompt's random-split directions, and it is not the letters' own features (median cosine with the letters' contrast +0.04 to +0.12), while the same statistic finds the animal-object direction's row above every random split in 5 of 6 models, a row whose top tokens read animal words in several; the ordinal code (which group was named first) is shared across prompts with different letters and objects in the carriers (mean pairwise cosine up to Pythia-410m 0.74, Pythia-1.4b 0.55, OLMo-1B 0.42, SmolLM2-360M 0.37, nulls 0.01 to 0.07), a binding-ID-like code; its average has a nearest row above its null in OLMo-1B, SmolLM2-360M at some depth, a lead only, because the null for an average of random splits is less structured than any real code | supported (a class defined in the prompt is carried in four of six models, with no native word and not as its letters' features; the ordinal code is shared across prompts) | ← 38 e584 · ← 56 e632 |

## How the results flow

- `e584, e632 → e641`: in pretraining the class forms before its row and the row's early lead is persistence; with no training at all the class forms and no row speaks it.

## Links to other areas

- [38 Watching words being born](38_born.md): classes that cohere in the cloud before a row is recruited.
- [56 Lottery or recruitment](56_lottery_or_recruitment.md): which row a class gets, and why.
- [14 The native vocabulary](14_native_vocabulary.md): what counts as a word.
