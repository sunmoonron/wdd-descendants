"""e479: directed modulation, and prompt echo against computed content. WorkspaceBench's directed-modulation family
tells the model to think about a concept (or not to) while it copies an unrelated sentence, and asks whether a reader
surfaces the concept at the positions where the model is writing; the contrast between "think" and "do not think" is
the measure. The concept is in the prompt, so a reader can echo it (the text-inversion shortcut the benchmark warns
about). A native-word readout says where each surfaced concept comes from: a token-embedding word for the concept is a
copy of the prompt token; an MLP-row word promoting the concept is something the model computed. So the readout can
be split into an echo part and a computed part, and the think/don't-think contrast measured on each.
Setup: Qwen2.5-7B-Instruct (bf16, chat template), 40 concepts of four kinds (animals, body parts, foods, objects) whose
first token is a single token, five neutral sentences; conditions "think about the {kind} {concept} while you write",
"do not think about ...", and a neutral instruction with no concept. The assistant turn is teacher-forced to the
sentence; an item is kept when the model's greedy predictions reproduce at least 90% of the sentence's tokens. States
are read at every token of the copied sentence after blocks 8, 12, 16, 20 and 24.
Readers: the plain lens, 16 native words pooled (per-word lens), the same pooled over MLP-row words only ("computed"),
over token-embedding words only ("echo"), and 16 rotated words. A cell passes if the concept's token is in the top 10;
an item passes if any cell passes (the benchmark's rule). Reported per reader: the pass rate under think, don't think
and neutral (the floor: the concept is scored on prompts that never mention it), the think-minus-don't contrast, the
precision of concept claims (top-10 tokens that are one of the 40 concepts) under think, and for native passes the
type and block of the carrying word.
Pre-registered (honest guesses):
- the plain lens passes at least half of think items and the contrast is at least 0.2 (0.6);
- the native readout's contrast is at least the lens's (0.4);
- the echo part passes about equally under think and don't think (contrast under 0.1) while the computed part shows
  the contrast (0.5);
- native precision of concept claims is at least the lens's (0.5)."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ws_common import *
MODELS.update({"qwen7i": ("Qwen/Qwen2.5-7B-Instruct", "llama")})
name = sys.argv[1] if len(sys.argv) > 1 else "qwen7i"; K = 16; LAYERS = [8, 12, 16, 20, 24]
model, tok, fam = load_bf16(name); arch = Arch(model, fam)
KINDS = {"animal": ["dog", "cat", "horse", "lion", "tiger", "bear", "wolf", "rabbit", "monkey", "eagle"], "body part": ["elbow", "knee", "shoulder", "tongue", "thumb", "ankle", "wrist", "cheek", "chin", "heel"],
         "food": ["banana", "cheese", "butter", "pepper", "mango", "lemon", "garlic", "honey", "bread", "rice"], "object": ["ladder", "hammer", "candle", "mirror", "bucket", "pillow", "umbrella", "blanket", "kettle", "basket"]}
SENT = ["The committee approved the minutes without changes.", "The train left the station a few minutes late.", "She parked the car behind the old library.", "The meeting was moved to Thursday afternoon.", "He painted the fence a pale shade of grey."]
concepts = [(k, c) for k, cs in KINDS.items() for c in cs if len(tok(" " + c, add_special_tokens=False)["input_ids"]) == 1]
ctok = {c: tok(" " + c, add_special_tokens=False)["input_ids"][0] for _, c in concepts}; ctoks = torch.tensor(sorted(set(ctok.values())), device=DEV)
def build(instr, sent):
    p = tok.apply_chat_template([{"role": "user", "content": instr}], tokenize=False, add_generation_prompt=True)
    pid = tok(p, add_special_tokens=False)["input_ids"]; sid = tok(sent, add_special_tokens=False)["input_ids"]
    return torch.tensor(pid + sid), len(pid), len(pid) + len(sid)
items = []
for j, (kind, c) in enumerate(concepts):
    s = SENT[j % len(SENT)]
    for cond, instr in (("think", f"Think about the {kind} {c} while you write. Now write exactly this sentence: \"{s}\""), ("dont", f"Do not think about the {kind} {c} while you write. Now write exactly this sentence: \"{s}\"")):
        ids, a, b = build(instr, s); items.append(dict(cond=cond, concept=c, ids=ids, a=a, b=b))
for s in SENT:
    ids, a, b = build(f"Now write exactly this sentence: \"{s}\"", s); items.append(dict(cond="neutral", concept=None, ids=ids, a=a, b=b))
# gate on copying, and states at the sentence positions
X_all = {L: [] for L in LAYERS}; owner = []; kept = []
for n, it in enumerate(items):
    cap = {}
    hs = [arch.layers[l].register_forward_hook((lambda l_: lambda m, i, o: cap.__setitem__(l_, (o[0] if isinstance(o, tuple) else o)[0].float()))(l)) for l in LAYERS]
    try:
        with torch.no_grad(): lg = model(it["ids"][None].to(DEV)).logits[0].float()
    finally: [h.remove() for h in hs]
    pred = lg[it["a"] - 1:it["b"] - 1].argmax(-1).cpu(); acc = float((pred == it["ids"][it["a"]:it["b"]]).float().mean())
    if acc < 0.9: continue
    kept.append(n)
    for L in LAYERS: X_all[L].append(cap[L][it["a"]:it["b"]])
    owner += [n] * (it["b"] - it["a"])
items_kept = {n: items[n] for n in kept}; owner = torch.tensor(owner, device=DEV); X_all = {L: torch.cat(v) for L, v in X_all.items()}
log(f"{name}: kept {len(kept)} of {len(items)} prompts (think {sum(items[n]['cond'] == 'think' for n in kept)}, dont {sum(items[n]['cond'] == 'dont' for n in kept)}, neutral {sum(items[n]['cond'] == 'neutral' for n in kept)}); {owner.numel()} states per layer")
ev_all = torch.load("/workspace/wdd/eval_ids_all.pt", weights_only=False)["qwen05"]["eval_ids"][:4]
ref = ref_stats(model, arch, ev_all, LAYERS); gain, WU = final_readout(model, arch)
A, blk, typ, ends = lean_dictionary(arch, max(LAYERS))
g = torch.Generator().manual_seed(7); R = torch.linalg.qr(torch.randn(arch.D, arch.D, generator=g))[0].to(DEV)
READERS = ["lens", "native", "native_mlp", "native_tok", "rotated"]
hit = {rd: {} for rd in READERS}                         # (item, layer) -> bool (any position)
claims = {rd: dict(right=0, wrong=0) for rd in READERS}  # think condition only
carrier = dict(mlp=0, token=0, head=0, blocks=[])
for L in LAYERS:
    X = X_all[L]; mu = ref[L]["mu"]; Xc = X - mu; rms = X.pow(2).mean(-1).sqrt(); An = A[:ends[L]]; n_ = X.shape[0]
    sel, _, _ = omp(Xc, An, K, batch=64, record_err=False); cof, _ = refit(Xc, An, sel); words = An[sel]
    selr, _, _ = omp(Xc @ R.T, An, K, batch=64, record_err=False); cofr, _ = refit(Xc @ R.T, An, selr); wordsr = An[selr] @ R
    S = {"lens": lens_scores(X, gain, WU, rms), "rotated": pooled_word_scores(wordsr, cofr, gain, WU, rms)}
    best_all = best_mlp = best_tok = None; per_c = torch.zeros(n_, K, ctoks.numel(), device=DEV)
    for i in range(K):
        s = lens_scores(words[:, i] * cof[:, i:i + 1], gain, WU, rms); per_c[:, i] = s[:, ctoks]
        m_mlp = (typ[sel[:, i]] == T_MLP)[:, None]; m_tok = (typ[sel[:, i]] == T_TOK)[:, None]; neg = torch.full_like(s, -1e9)
        best_all = s if best_all is None else torch.maximum(best_all, s)
        best_mlp = torch.where(m_mlp, s, neg) if best_mlp is None else torch.maximum(best_mlp, torch.where(m_mlp, s, neg))
        best_tok = torch.where(m_tok, s, neg) if best_tok is None else torch.maximum(best_tok, torch.where(m_tok, s, neg))
    S.update(native=best_all, native_mlp=best_mlp, native_tok=best_tok)
    for rd, sc in S.items():
        top10 = sc.topk(10, dim=-1).indices
        for n in kept:
            rows = (owner == n).nonzero()[:, 0]; it = items[n]; t10 = top10[rows]
            targets = [ctok[it["concept"]]] if it["concept"] else list(ctok.values())
            if it["concept"]: hit[rd][(n, L)] = bool(torch.isin(t10, torch.tensor([ctok[it["concept"]]], device=DEV)).any())
            else:
                for c in ctok.values(): hit[rd][(n, L, c)] = bool((t10 == c).any())
            if it["cond"] == "think":
                cl = t10[torch.isin(t10, ctoks)]; claims[rd]["right"] += int((cl == ctok[it["concept"]]).sum()); claims[rd]["wrong"] += int((cl != ctok[it["concept"]]).sum())
                if rd == "native" and hit[rd][(n, L)]:
                    ci = int((ctoks == ctok[it["concept"]]).nonzero()[0, 0]); r_, i_ = divmod(int(per_c[rows][:, :, ci].argmax()), K); w = sel[rows[r_], i_]
                    tp = int(typ[w]); carrier["mlp" if tp == T_MLP else "token" if tp == T_TOK else "head"] += 1; carrier["blocks"].append(int(blk[w]))
    del S, best_all, best_mlp, best_tok, per_c, sel, selr, words, wordsr; torch.cuda.empty_cache()
def pass_rate(rd, cond):
    ns = [n for n in kept if items[n]["cond"] == cond]
    if cond == "neutral": return sum(any(hit[rd][(n, L, c)] for L in LAYERS) for n in ns for c in ctok.values()) / max(len(ns) * len(ctok), 1)
    return sum(any(hit[rd][(n, L)] for L in LAYERS) for n in ns) / max(len(ns), 1)
res = dict(model=name, n_kept=len(kept), layers=LAYERS, readers={}, carrier=dict(mlp=carrier["mlp"], token=carrier["token"], head=carrier["head"], mean_block=(sum(carrier["blocks"]) / len(carrier["blocks"])) if carrier["blocks"] else None))
for rd in READERS:
    p = {c: pass_rate(rd, c) for c in ("think", "dont", "neutral")}; c_ = claims[rd]
    res["readers"][rd] = dict(**{f"pass_{k}": v for k, v in p.items()}, contrast=p["think"] - p["dont"], precision_think=c_["right"] / max(c_["right"] + c_["wrong"], 1), claims_think=c_["right"] + c_["wrong"],
                              pass_think_by_layer={L: sum(hit[rd][(n, L)] for n in kept if items[n]["cond"] == "think") / max(sum(items[n]["cond"] == "think" for n in kept), 1) for L in LAYERS})
    log(f"{name} {rd}: pass think {p['think']:.2f}, don't {p['dont']:.2f}, neutral floor {p['neutral']:.3f}, contrast {p['think'] - p['dont']:+.2f}; precision of concept claims under think {res['readers'][rd]['precision_think']:.2f} ({res['readers'][rd]['claims_think']} claims); by layer " + " ".join(f"{L}:{v:.2f}" for L, v in res["readers"][rd]["pass_think_by_layer"].items()))
Rr = res["readers"]
res["checks"] = dict(lens_half_and_contrast=Rr["lens"]["pass_think"] >= 0.5 and Rr["lens"]["contrast"] >= 0.2, native_contrast_at_least_lens=Rr["native"]["contrast"] >= Rr["lens"]["contrast"],
                     echo_flat_computed_contrast=abs(Rr["native_tok"]["contrast"]) < 0.1 and Rr["native_mlp"]["contrast"] > Rr["native_tok"]["contrast"], native_precision_at_least_lens=Rr["native"]["precision_think"] >= Rr["lens"]["precision_think"])
summ = (f"{name}, {len(kept)} prompts kept; pass rate think / don't think / neutral floor (contrast), precision of concept claims under think: " + " | ".join(f"{rd} {v['pass_think']:.2f} / {v['pass_dont']:.2f} / {v['pass_neutral']:.3f} ({v['contrast']:+.2f}), {v['precision_think']:.2f}" for rd, v in Rr.items())
        + f" | native passes carried by MLP rows {res['carrier']['mlp']}, token embeddings {res['carrier']['token']}, head bases {res['carrier']['head']}" + (f", mean block {res['carrier']['mean_block']:.1f}" if res['carrier']['mean_block'] is not None else "") + f" | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e479_directed_{name}", res, summ)
