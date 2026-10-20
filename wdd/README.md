# wdd

Write-direction dictionaries, training-free. The package holds the primitives the record's experiments ran, unchanged in behaviour: the identification floor (an extreme-value calibration against a rotated copy of the dictionary), the native-row dictionary and orthogonal matching pursuit, the exact ledger of a neuron's inputs with the K-input simulation, the weak and strong fields of the upstream blocks, and read energy (the Fisher information of the model's own output, as a grade on any part of a state).

Install nothing: copy the folder next to your script, with `torch`, `transformers`, `datasets` and `numpy` available, and run the demo on one GPU:

    python -m wdd.demo_gpt2_small --fast     # a few minutes on GPT-2 small
    python -m wdd.demo_gpt2_small            # about ten minutes

The demo prints, for GPT-2 small at block 6: how many native rows a position has over the floor; what sixteen rows leave of the state; the exact split of one neuron's pre-activation into its upstream contributions and the crowd's net sum off the neuron's class (the brake); the K-input simulation of that neuron with its sixteen largest inputs, with the weak field, and with both fields; the read energy of the content, the weak write and the strong write per unit of energy; and sixteen rows chosen by Euclidean pursuit against the blended Fisher pursuit (which leaves almost no read energy in the residual but, by e714, recovers no more of the finite KL than the Euclidean choice: the Euclidean pursuit is the default, the blended one a tool).

Modules: `model` (loading, weights, normalisation, hooks, log-probabilities, KL), `corpus` (wikitext-2, pile-10k, windows), `floor` (the floor, the rows, OMP in the Euclidean metric or a supplied metric), `ledger` (`Ledger`: contributions, the net ledger, `simulate`, `kl_with`), `fields` (field directions per block, their union, the crowd's write on a field as a simulation term), `readenergy` (the sampled Fisher, read energy, the grade against a random direction, the blended metric).

Models: any Hugging Face causal LM of the three families the record covered (GPT-2, Pythia, and the SwiGLU models Qwen2.5, SmolLM2 and OLMo); the short keys in `wdd.MODELS` or a full model id. fp32 is the default and the one the ledger's exactness was checked in (5e-6 relative error at 7B).

The results these functions produced, with their pre-registrations, are in `docs/atlas/` areas 62 to 72; the note that accompanied them was withdrawn on 2026-10-20.
