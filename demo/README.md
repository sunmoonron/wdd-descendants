# WDD in the weights: a live demo

`wdd_demo.py` runs three short acts on raw Hugging Face checkpoints and prints every number it measures. Nothing on screen is quoted from the record: if a claim does not hold on your machine, the demo says so. It ends by checking, against the run, the claims of the demo script it was built from (a three-act script proposed by Gemini 4 in session 149).

| Act | Model | What happens |
| --- | --- | --- |
| 1 | Qwen2.5-0.5B | A direct-contribution scan (each MLP row's activation change times its write's alignment with the pole difference; no training, no dictionary learning, a few seconds) names the rows that write grammatical gender and number. Negating eight of their write columns, one line of PyTorch on the weights, flips held-out translations ("reine" to "king"), moves agreement in free text in English, French, Spanish and German, and costs a few thousandths of a nat on wikitext. The same items with the standard tools: directional ablation (Arditi et al. 2024) and a steering vector. |
| 2 | Qwen2.5-Coder-0.5B-Instruct | The base model's row addresses and residual direction applied zero-shot to a model continued-pretrained on code and instruction-tuned, against the Coder model's own; how far the rows rotated and how far the direction did. |
| 3 | Pythia-410m | One block-4 write row is scaled, negated, or reflected about its read axis (which keeps its norm and its read-write cosine). The audit sees only the tampered model: the norm z-score, the read-write cosine and the activation-consistency cosine, each computed on the tampered model's own states. A sweep over the 40 most-used rows gives catch rates. |

## Run it

A GPU with 8 GB is enough (CPU works, slowly). Requirements: `torch`, `transformers`, `datasets`. The three models and wikitext-2 download from Hugging Face on first use.

    python wdd_demo.py                       # interactive: pauses between scenes; type your own prompt; pick the row and the tamper
    python wdd_demo.py --auto                # no pauses; about two minutes on a Quadro P4000 once the models are downloaded
    python wdd_demo.py --acts 3 --sweep 40 --seed 1
    python wdd_demo.py --k 32                # negate 32 rows instead of 8
    python wdd_demo.py --save run.json       # also write the measured numbers to a file

`sample_run.txt` is one full run (`--auto --no-color`) and `sample_run.json` its measured numbers.

## Reading it honestly

- The edits are real weight edits and the effects are measured on held-out templates and separate free-text sentences, but the items are small (a few hundred translation prompts, 72 English and 72 French, Spanish and German sentences); treat the percentages as a demonstration, not a benchmark.
- Act 1 is a trade-off, not a win: the row edit is the cheapest on wikitext, ablating the direction flips more at about twice the KL, and a steering vector that matches it costs an order of magnitude more.
- The asymmetry is real but not absolute: the rows fire several times more on the marked pole (female, plural), and a few fire more on the unmarked one.
- Act 2's transfer is a preserved direction rather than a co-rotating neuron (e665b): the write columns rotate (median cosine about 0.44 to the base model), the attribute's direction in the residual stream much less.
- Act 3 is a defender's test: single-row tampers here barely move the model's predictions at the row's own top positions, the norm catches scaling, the read-write cosine catches negation in this GELU model, and the norm-and-cosine-preserving reflection mostly evades all three detectors. e651 had reported the consistency detector covering that reflection; it scored the detector with the clean model's states, which a blind auditor does not have, and e667 rescored it (docs/FINDINGS.md, session 149).
