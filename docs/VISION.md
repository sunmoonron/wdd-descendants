# WDD since 2016: a backcast and a roadmap

Prompt (the user, session 45): if Weight-Dictionary Decomposition had existed for ten years, what would the world look
like, and how do we get there from a method a few weeks old?

Method. Each imagined capability is reduced to the assumption it rests on. None of these assumptions was tested in the
400+ earlier experiments, which only measured function on pretrained checkpoints. Each gets the cheapest decisive test.
Numbers are from `FINDINGS.md` session 45.

## The imagined world, what it needs, and what the first tests say

| Capability in the imagined world | Must be true | First tests | Status after session 45 |
| --- | --- | --- | --- |
| Every model ships with a dictionary; a state is read as a few named words | native words carry meaning, not only function | e446 (token proxies), e448, e448b, e448c, e448d, e448e (meaning across languages) | **Supported for concrete concepts.** A language-independent concept word, one MLP write row, exists for 7 of 24 nouns in SmolLM2-135M, 15 in Qwen2.5-0.5B and 22 in Qwen2.5-7B, across four languages and five contexts (0 with rotated words). Sentences are described with the same native words in four languages. Removing a concept word hurts more than a matched other word. Token-level proxies cannot show this; they favour lexical units. Session 46: concept words are causal handles. Swapped at every block up to the middle, one of them redirects 67-94% of translations to the swapped-in noun and 39-78% of the Qwen models' category answers (e455). At one depth the effect is specific but small (e451, e454). |
| Internals interoperable through word-level translation tables | two models' words correspond one to one | e445 (Pythia 160m, 410m, 1b) | **Refuted at the word level.** Word-for-word translation keeps 0.06-0.13 of function against 0.85-0.91 for a dense linear map. Partner words exist above chance (7-14% against 1-2%). Interoperability will be dense. Session 46: concept words also correspond across Qwen2.5-0.5B and 7B only at the concept level: the right noun among 14 in 11 cases, but at abs cosine 0.08 (e453). |
| Training monitored by vocabulary formation | the vocabulary marks real transitions and tells generalising from memorising | e444 (grokking with memorisation controls) | **Partly supported.** Generalising networks become sparse in their own words, while memorising ones get only a second-order version. The word-level part rose 500-750 steps before generalisation in both standard runs, but not in the frozen-row variants. A candidate early signal, not yet a reliable one. |
| A theory: gradient descent writes the vocabulary | the vocabulary is written (writer rows trained to fit the states), not spoken (activations organise around fixed words) | e444, e444b, e449, e449b (frozen writer or reader rows, trained from scratch) | **Supported.** Writer rows frozen at random are barely used as words (unexplained 0.69 against 0.74 for their rotation in grokking, 0.75 against 0.74 in the sequence model); trained writer rows are. When some writers are frozen, the description moves to those that train. In grokking the trained MLP write rows alone explain 92% of the state with 4 words (rotated rows 7%); frozen at random they explain 31%, and the attention output bases take over. Session 46, at language-model scale with the function held fixed: a middle block re-implemented from scratch uses different rows, which are words about half as good as the original's (71-77% with more fitting, e452b). With writers frozen at random it has no words at all (e452). |
| Models trained to be self-describable | self-describability can be trained in cheaply | e447, e447b (GPT-2 with a self-description term) | **Not yet: the obvious objective is the wrong one.** A Euclidean self-description term makes states much sparser in the model's own words (unexplained down 41% for 0.28 nats, 80% for 0.91), but the words do not move and function is not better described. The next objective should be functional: the loss with the state replaced by its description. |
| Activations ship as native codes | native codes beat generic codecs at equal bits | e450 (five models, 64-512 bits per position) | **Only at moderate rates.** The native code beats PCA at 512 bits in 5 of 5 models and at 256 in 4 of 5. It loses at 128 bits and below, where each word's 17-18-bit index costs too much. |

## What this changes about the path

- The dictionary a model ships with is its own weights. Its most useful words for a reader are concept words, and they can be found without labels. Look for words shared by translations, or more generally by paraphrases, of the same content. e448d found 15 such words for 24 nouns in a 0.5B model and 22 in a 7B model, in minutes.
- Interoperability should not be built on word tables. The models' words are private one to one, even for three Pythia sizes trained on the same data in the same order. Linear maps between whole states work (0.85-0.91), and a learned linear decoder from one model's 16-word description recovers as much of another model's function as that model's own 16 words.
- The theory has a first causal fact: training the writers creates the vocabulary. Training dynamics are now the place to explain it, for example the per-block accent that turns into words between steps 4000 and 16000 in Pythia (session 44).

## Roadmap (each step a short experiment)

1. Concept dictionaries at scale. The model translates a few thousand of its own sentences; words shared across translations and rare elsewhere are collected. Qwen2.5-7B already gives 22 of 24 nouns; next, many more concepts and all five models.
2. A vocabulary card per model: self-description curve (own, rotated, covA) by depth, the huge directions with sinks excluded, the concept-word list, and Zipf. Minutes per model with the session-44 and session-45 scripts.
3. Monitoring on real training: the word-level advantage across Pythia checkpoints around known transitions (induction heads), and more algorithmic tasks, to see whether the early signal holds.
4. The written-vocabulary theory at language-model scale: freeze writer or reader rows while pre-training a small model on real text, then follow the per-block accent turning into words.
5. Interpretability by design with a functional objective: fine-tune with the loss of the model run on its own 16-word description at one depth (differentiable through the least-squares fit), and measure the cost against the gain in functional self-description.
6. A native codec that pays less per index: entropy-code the indices using their Zipf-like frequencies (session 38), and code them per block, which should cut the index cost (not tested).
7. Steering with native words (added in session 46): compare concept-word swaps across depth (e455) with dense steering vectors of the same size, on more concepts and tasks.
