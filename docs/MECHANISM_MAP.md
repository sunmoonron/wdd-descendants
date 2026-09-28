# The mechanism map: from data to behaviour through the native vocabulary, with every arrow annotated (session 102)

After 560 experiments the question above the program is no longer "what is a WDD word" but "why does a trained network organise its computation into persistent associations between native write directions and classes of its state distribution". This page is the map of that mechanism as the record supports it: the nodes, the arrows between them, and for each arrow its status, its evidence per model, and what would still change it. WDD is the instrument; the arrows are the claims.

**Status vocabulary.** *Established*: measured against controls on more than one model. *Supported*: measured on one model or with a weaker control. *Intervened*: shown by changing the cause and watching the effect, not by correlation alone. *Correlational*: the order or the co-occurrence is measured, the cause not intervened on. *Model-dependent*: the sign or the size differs between models. *Contradicted*: the record argues against the arrow. *Unknown*: not measured.

## The map

```
DATA / TASK
    |
    v
state distribution (the cloud)
    |----------------------------.
    v                            v
activation selectivity       covariance / recurring contexts
(the neuron's input side)    (the directions the cloud speaks)
    |                            |
    '-----------.  .-------------'
                v  v
        the native write row turns into a spoken direction
                |
                v
        the extreme association (the word: S over the floor at a class of contexts)
                |
                v
        many small writes (the coalition of thousands, the row's own write among them)
                |
                v
        downstream use (what reads the direction; what the transplant carries)
                |
                v
        behaviour (loss, answers; what the ledger does and does not diagnose)
```

## The arrows

| # | arrow | status | Pythia-410m / 160m | OLMo-1B | Mamba-130m / 370m | ViT / DeiT | grokking toy | what would change it |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | data, task -> the cloud's partition into context classes | established, correlational on the data side; intervened in the toy | the partition is the data's, not the seed's: Pythia-160m runs with an independent initialisation and data order twin with the main model at 0.34-0.38 by context sets and with each other at 0.27, against the deduped data twin's 0.39 and a rotated-atom null of 0.03-0.05 (e566); the variants named "weight-seed" and "data-seed" turned out to start from the main model's initialisation (their step-512 rows match main's at 0.997, e566d), so they add nothing here | twins with Pythia-410m at 0.41-0.45 by character spans, across family, tokenizer and corpus (e566) | twins with Pythia-410m at 0.45-0.46 by token positions, the same data and tokenizer, no attention (e566) | not measured (no shared input space) | the association appears in every run that generalises and in none that memorises, whatever knob is turned: native words at 2.3 times the rotated words' S against 1.1-1.4 (e567) | a data intervention at scale: the same architecture trained on two corpora, and whether the twin rate follows the corpora's overlap |
| A2 | the cloud -> its covariance holds the maximum | established, intervened | the maximum relocates while the vocabulary survives; covariance-matched atoms persist like rows (1.00 / 1.09) but churn (e541b); a Gaussian cloud with the real covariance keeps the words at S 1.08 and loses retention (0.37 against 0.50, e543); four real projections per position grafted onto it restore everything (e544) | the Gaussian cloud keeps 0.58 of the 0.64 retention but loses the plateau (1.16 against 1.70) and the gate (0.71 against 0.95); the graft restores (e555) | unknown (no checkpoints) | unknown | unknown | the same graft on Mamba's states |
| A3 | the cloud -> the neuron's input selectivity | established, correlational in time (precedes) | the entrant's neuron is selective for its future contexts eight intervals before entry (z 1.61 rising to 2.85); the near miss loses it (1.91 to 0.57) (e547b) | z 2.63 eight intervals before entry, 6.53 at entry; the near miss keeps it (3.66 to 5.81) (e554) | at the end z 6.35 for words, 7.54 for S-matched near misses, 1.03 for random rows (e570) | unknown | unknown | selectivity measured through a data intervention (change the corpus, watch the neuron) |
| A4 | selectivity and covariance -> the row turns into a spoken direction | established on Pythia, intervened by dictionary swaps; the direction of the loop is data -> neuron | over 8000 steps the entrant's step-16000 direction already sits at S 1.29 in the step-8000 cloud; row part +0.15, cloud part +0.03 (e558b); at one-checkpoint resolution the row part is +0.04 to +0.06 and the cloud part 0.00 to +0.03 at k = -2..0 in either order; the row's turn moves it to 1.037 times the old cloud's variance (a random turn of the same size 0.965) while the cloud does not grow along the row's old direction (0.973-0.997 against 0.990-0.997 for random directions) (e571) | unknown | unknown (no checkpoints) | unknown | the rows persist at cosine 0.99 per 250 steps and the words' projection profiles correlate at 0.92-0.99 with 250 steps earlier in the runs that generalise; the top-16 extreme sets are unstable through the periodic features' ties (e567) | the reverse direction (neuron -> data) in a regime where a row can shape the cloud: the toy with the row frozen, or early training (k <= -3, where the cloud part is +0.02) |
| A5 | the native row -> the extreme association (the word) | established across every model | the native fifth of the criterion (0.88 against 0.80 for random atoms), the one-way gate (leavers over the floor at exit 0.82, entrants under it before entry), words at 1.1-1.3 times the floor with thousands of non-words above it (e551, e554) | 0.94 against 0.80; gate 0.96 against 0.17; retention 0.64 against 0.19 (e551) | words at S 1.78 / 1.93 against 0.99 / 0.97 for the rotated dictionary's most used atoms; 7524 / 15254 rows over the floor against 896 / 937 (e563) | words at S 1.67 / 1.49 against 1.06; broader (3.0 / 4.5 against Pythia's 2.7); the floor looser on images (2976 rotated rows over it) (e562) | words at 0.97-1.01 against 0.36-0.53 rotated in the runs that generalise; 1.2-1.3 against 0.9-1.15 in the runs that memorise (e567) | the norm question, answered: 256 random rows matched to the words' norms describe the states at K=8 as well as the words (e569b), but are used 0 times against the words' 19 at the median, sit at S 0.92 against 1.61 (0.39 over the floor against 1.00), and twin with 160m's words at 0.02 against 0.40, though both sit in the top 5% of rows by norm (e572); the span is the norm's, the association is the data's |
| A6 | many small writes -> the extreme (the coalition) | established; the row's own share model-dependent | 9,232 effective writers, within-row coherence 0.49 against 0.00 across rows, persistence 0.88, the coalition coheres four intervals before entry (0.42 against 0.20 for near misses); the own row ranks 64-71 of 247 (e545, e546) | within-row 0.731, persistence 0.969, precedence 0.52 against 0.31; the own write ranks first (e552, e554) | within-row 0.857 against 0.000, near misses 0.400, random directions 0.096; 202 effective writers against 1754 for random directions; the own row ranks first (top-10 share 0.80); the extreme is built from every block, block 0 the largest single share (0.11) (e570) | unknown | unknown | the coalition's precedence on Mamba needs checkpoints; a trained small SSM would give them |
| A7 | the association -> downstream use | supported, intervened | WDD's S and breadth predict entry four intervals ahead at 0.88 against 0.74 for the activation side, +0.142 out of sample (e548); the whole step-16000 neuron transplanted into the step-8000 model makes 0.68 of entrants words (0.80 singly), the write column alone 0.59, its direction at the old norm 0.54, its norm at the old direction 0.18, the read row alone 0.18, the bias 0.03; read and write are sub-additive (-0.10) (e558, e565); zeroing the old words' write columns costs the new model 0.5-1.7 times what zeroing its own words costs (e569) | +0.133 out of sample (e553) | unknown | unknown | unknown | what the transplant carries beyond the extreme: the effect on the next-token distribution at the word's contexts |
| A8 | downstream use -> behaviour | mixed: WDD reads construction, not correctness | no single state, so no ledger, announces an upstream corruption beyond two tokens (AUC 0.50 for the codes, 0.55 for the raw state) (e559); the ledger does not see a factual error where the confidence does (0.57 against 0.85) (e560); invented names are more spoken because the state falls to a generic Pile word (block 17, neuron 3002: a function-word row) that real people's names also drive, while known countries drive a specific row; zeroing that row's write changes nothing (confidence +0.000) (e568); steering at 6-8 times natural size moves 67-94% of translations (e455); concept words are causal handles at one depth (e451) | unknown | unknown | unknown | frozen random write rows are barely used as words: the vocabulary is written, not spoken (e449) | the boundary is the result: the ledger says what native writes built a state, not whether the state is right |

## Cross-cutting claims

| # | claim | status | evidence | what would change it |
| --- | --- | --- | --- | --- |
| C1 | associations outlive the rows that carry them | established | from step 16000 to the end of training the word set keeps Jaccard 0.21 by row index while context-set twins between the two checkpoints are 0.55 / 0.52; the rows rotate to cosine 0.61 (e557b); the same context sets, spoken by other rows | a functional test: whether the new row plays the old row's role under removal |
| C2 | the partition is conserved across models, and it is the words' | established | twins at 0.27-0.46 across initialisation, data order, corpus, scale, architecture and family; nulls 0.00-0.05 (e557, e566); most twins are not token identities (purity 0.30); non-word rows matched to the words' norms twin with 160m's words at 0.02 against the words' 0.40, random rows at 0.00 (e572) | a functional test of a twin pair under removal |
| C3 | the stability hierarchy: row identity > usage > function > membership | established, with a caveat | under weight noise costing 0.03 nats: row identity 1.000, usage 0.99, the old words still matter 0.99, membership 0.86; under fine-tuning costing 2 nats: 1.000 / 0.92 / 0.95 / 0.66; under training 8000 -> 16000: 0.87 / 0.72 / 1.72 / 0.37 (e569); the "describe" level (the old words' FVU on the new states against the new words') is 1.00 everywhere, and so is it for random rows matched to the words' norms (e569b): description at K=8 is a property of the norm class, not of the word set | usage correlation beside every word-set Jaccard in the record |
| C4 | attention, tokens and Adam are not necessary | established | Mamba (no attention, no MLP) has the association more strongly than Pythia; ViT and DeiT have it without tokens; the toy has it under AdamW, sign momentum and clipped SGD alike (e556, e562, e563) | a second SSM family; an MLP-only or convolutional model |
| C5 | the vocabulary tracks generalisation, not the optimizer or the knob | supported (toy) | every knob that keeps the run generalising keeps the association (width 2048, decay 3, lr 3e-3, half the data, init 0.3 and 3, a second seed); every knob that makes it memorise removes it (width 128, decay 0.3 or 0, lr 3e-4, a fifth of the data, random labels) (e567) | a language model trained past memorisation |

## The loop, as the record now reads it

The data fixes a partition of the state space into recurring context classes (A1), and the cloud's covariance holds the directions along which those classes are extreme (A2). Neurons become selective for a class before any row is extreme there (A3). A row enters the vocabulary by turning into a direction the cloud already speaks; the cloud does not grow along the row (A4). The extreme is then built by thousands of small writes that cohere across the class, the row's own write first among them in OLMo and Mamba and far down the list in Pythia (A6). What a later checkpoint's neuron carries into an earlier model is its write direction, not its norm and not its input weights (A7). None of this makes the ledger a judge of behaviour: it says which native directions built a state, and a wrong answer or a corrupted context is built from the same directions as a right one (A8).

## The arrows still open, in order of value

1. A4 in reverse: a regime where a row shapes the cloud (the toy with a frozen row; early training). At one-checkpoint resolution the cloud's part is +0.02 two intervals before entry and zero at entry.
2. A6 and A2 on Mamba: precedence, the gate and the graft need checkpoints; a small SSM trained on the box would give them.
3. A7 beyond the extreme: whether the transplanted direction reproduces the later model's effect on the next-token distribution at the word's contexts.
4. A5's norm question is closed by e572 (norm-matched rows are used a fraction as often, sit under the floor and do not twin); what remains is whether the words' usage is *why* they are large or the reverse, which needs the rows' norms through training against their entry.
5. C1's function: whether a context class keeps its downstream role when it changes rows.
