# The proxy: a Charles-shaped viewer for what a model does inside

Design document, started 2026-10-11 (session 128). This is the single source of truth for the tool idea that grew out of the WDD program's last sessions. It is written so that a reader with no memory of those conversations can pick it up. Plain English; the WDD vocabulary is defined where it appears.

## 1. The one-sentence vision

Charles Proxy sits between a client and the servers it talks to, records every exchange, lets you inspect it, and lets you rewrite, replay, pause and reroute it. The proxy does the same for a language model's internals: it sits between the model's components and the residual stream, records what every component read and wrote at every token, lets you inspect it in the model's own units, and lets you rewrite, replay, pause and reroute it. Neuronpedia, a rented GPU, a laptop running a small model, or any other source of internals is a device on the network. The UI is the constant; the devices vary.

## 2. What is fixed and what varies

**Fixed: the UI and the data model.** A Charles skeleton that people already know how to read: a tree on the left, a main pane, an inspector with tabs, a bar on top, sessions as tabs, rules in a drawer. Plus the objects Charles never had, which are the point: frames linked by cause, a training axis, rules as files.

**Variable: the devices.** In Charles a device is a phone or a browser configured to use the proxy; with the root certificate installed its traffic is readable, without it the traffic is opaque. Here a device is a provider of internals, and each has a capability profile:

| Device | What it can show | What it can do | Cost |
| --- | --- | --- | --- |
| Neuronpedia public API | features per token for its sources, logit and Jacobian lens per layer, attribution graphs for a few models, labels, natural-language explanations for three big models, activations of uploaded custom vectors | steer with features or vectors (output side), ablate through the lens endpoint | free, rate-limited, ~17 models |
| Self-hosted Neuronpedia inference server | the same, for any Hugging Face model with SAELens sources, extended with our endpoints | the same, plus whatever we add | a GPU when needed; the Mac for small models |
| Our hooks backend | every component's read and write at every position, the floor, coalitions, descendants, checkpoints | any intervention on parameters or activations, during generation, recorded with seeds | a GPU for large models; the Mac for Pythia-scale |
| API-only models | nothing, or hidden states if the API returns them | nothing | the device without the certificate |

The UI reads the profile and greys out what the device cannot do. Nothing is faked.

## 3. The data model

- **Session**: one model and checkpoint, one prompt set, one device, one rule set, one seed. Sessions open as tabs and can be compared.
- **Unit**: the thing a frame belongs to. Kinds: native MLP row, attention head, embedding or unembedding row, SAE feature, transcoder feature, custom vector, attribution-graph node, sparse-model neuron, a position's natural-language explanation. A unit carries its provider, model, layer, kind and index, and its static metadata (what its read side responds to, what its write promotes, norms, labels with their source, recruitment checkpoint where known).
- **Frame**: one unit at one position in one session: the read (pre-activation or attention pattern), the write (coefficient and direction, or the OV write), the floor ratio and flag where the device can compute them, the activation value otherwise.
- **Edge**: a causal link between frames: coalition (which writes drove this read) and descendant (which reads this write fed). Computed lazily and cached; the attribution method is named on the edge.
- **Rule**: scale, silence, swap (serve a unit's writes from another session), inject (place a saved frame), pin (pause when a unit crosses its threshold). A rule has a scope (unit, positions, condition), a colour, and a toggle. Rules are files attached to sessions.
- **Recording**: the frames and edges of a session on disk (Arrow), so nothing is fetched twice and sessions outlive the device.

## 4. The screens

- **Prompt bar**: Run records a session; Continue generates and extends it; Branch forks from a chosen alternative token into a new session tab.
- **Grid** (main pane, default): columns are positions with their tokens, rows are layers, cells are the flagged frames at that layer and position, sized by strength; the bottom row is the logit lens, what the model would say after each layer. Click a cell for its frames, a column for that position's build-up, a row for the layer in the tree.
- **Tree** (left): model, layer, component, unit; flagged units in bold with the size of their class (the positions where they are over threshold) in this session. Selecting a unit swaps the main pane to its **Timeline**: positions along x, two lanes, read above and write below, ticks coloured by ratio.
- **Inspector** (right): Overview (what the unit is, labels with source), Frame (the selected frame's numbers and raw vector, with the device's decode mode), Coalition (senders with shares; click to walk upstream), Descendants (readers with contributions; click to walk downstream), History (the unit's class size per checkpoint and its recruitment point, with a scrubber). The walk leaves a breadcrumb bar, a saveable path from a token to a logit through named units.
- **Rules drawer**: forms, not JSON. Active rules tint the cells they touch; a Diff toggle shows before and after; a range on a rule's factor turns the pane into a dose-response strip.
- **Scrub**: the checkpoint slider on a session tab re-runs the prompt on that checkpoint and animates the Grid; only devices with checkpoint suites (Pythia, PolyPythias, OLMo) have more than one position.
- **Compare**: two sessions side by side, Grid diff, restricted to the same training lineage, because a unit index means the same parameter only there.

## 5. The design decisions, each with its answer

1. **Attachment.** Devices, as above. Nothing attaches to a serving stack in version one; the system-proxy mode (a plugin inside vLLM or similar so every model on a GPU flows through) is the later product.
2. **Single or many models.** Many sessions, one model per session. Compare only within lineage. Across unrelated models compare by positions on shared text only.
3. **Architecture coverage.** The device's job. The self-hosted server loads what SAELens supports; our hooks backend uses a registry per family, GPT-NeoX and Llama first because the program's code handles them.
4. **What a frame is per unit kind.** Rows: scalar read, rank-one write. Heads: one frame per head per position with the source breakdown as a sub-table, never per source pair. Features: the activation. Vectors: the projection.
5. **What flagged means.** Per device. SAE and transcoder features are flagged when active. Native units are flagged over the floor, an extreme-value threshold calibrated on a reference corpus per model and checkpoint, with a rotated-dictionary null available as a toggle and attribution or raw magnitude as alternative criteria. The flag criterion is shown, never implied.
6. **One prompt or a prompt set.** A session is a prompt set; the Grid is per prompt; classes and timelines are over the set.
7. **Scale.** Flagged frames in full, the rest as compressed bulk, edges lazy and cached, recordings on the device side with summaries streamed to the UI.
8. **Interventions.** Write-side edits first, seeds recorded, injection layer explicit, head edits on the write not the pattern until the pattern view exists. On the public Neuronpedia device, surgery is output-side only and the UI says so.
9. **Labels.** From the device (feature explanations, NLA descriptions, Transluce-style neuron descriptions where they exist), shown with their source, absent by default. A label is never the unit's identity; the program's record is that the row is not the concept.
10. **Training axis.** Real only where checkpoint suites exist; the slider shows one position otherwise.
11. **Where it runs.** The UI anywhere; the backend next to the device; recordings durable.
12. **Who it is for.** Decided by the first experiment (section 7), not by taste.
13. **Correctness.** Frames from hooks are exact (the ledger identity). Edges name their method. Fingerprint mode (provenance guessed from states alone, which is what the WDD pursuit was) is shown as uncertain with its confidence.

## 6. Where Neuronpedia fits

It is device one because it is free and live for seventeen models. Its endpoints map onto the screens: `search-topk-by-token` feeds the Grid for feature sources, `lens/prompt` feeds the logit-lens row, `graph/generate` feeds Follow where graphs exist, `steer` and `steer-chat` are the output-side Rules, feature pages and `nla/explain` are labels, `activation/new` gives per-token values one unit per call, and `vector/new` lets the model's own write rows be uploaded and used as units on their GPUs. Its inference server is open source and self-hostable, which makes device two the same API contract on our own hardware, extended with the endpoints the public one lacks. Its limits are recorded in VISION.md (session 128): one unit per activation call, output-side steering, undocumented rate limits, unstated data terms, graphs for one documented model.

## 7. Build order, driven by the UI

1. The data model, the recorder and the UI skeleton against a mock device, so every screen exists before any provider does.
2. The Neuronpedia device, starting with the one test that decides the first use case: upload a model's native write rows as vectors and see whether `activation/new` returns their per-token projections. If it does, the native-unit view, which no existing tool has, is the first thing worth finishing. If not, the feature Grid and Follow are.
3. The self-hosted device on the Mac with Pythia-160m, which gives Scrub and the full dictionary.
4. The hooks device, which gives coalitions, descendants and interventions during generation, on the Mac for small models and on a rented GPU for larger ones.
5. The system-proxy mode, last.

## 8. What the program contributed to this

The frame, floor, coalition and descendant computations exist in the program's scripts. The program's negative results set the guardrails: a label is not an identity, a flag is a statistical judgement with a null, usage forecasts wordhood but only geometry names a row for a class, and a persistence-matched null goes into the first run of anything developmental.
