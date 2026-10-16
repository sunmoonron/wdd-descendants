"""e678 (session 154; Cammarata 2026, read with e671, Gemini 4's fusion with the essay's weak-ablation figures). The essay:
SwiGLU's double negative (gate pre-activation and value both negative, so silu(g) u is small and positive) fills the top
of about 4% of neurons' positive lists, and "ablating them across the whole network ups most benchmarks, but lowers some
reasoning ones". Tested as stated, with controls the sentence does not have. Models: Qwen2.5-0.5B, Qwen2.5-1.5B,
SmolLM2-360M, OLMo-1B-0724. Conditions, applied at every position: base; leak0 (the double negative zeroed in every
layer); leak0_first and leak0_second (only in the first or the second half of the layers); closed0 (every gate-closed
activation zeroed, gate < 0, ReLU-like); weak0 (gate-open activations below a per-layer magnitude threshold zeroed, the
threshold set on wikitext so that the removed L2 mass matches leak0's in each layer). Measured: wikitext-2 test next-token
loss and top-1 accuracy with damage (right to wrong) and repair (wrong to right) counts; seven multiple-choice benchmarks
scored by log likelihood as in lm-evaluation-harness (ARC-Easy, ARC-Challenge, HellaSwag, WinoGrande, BoolQ,
CommonsenseQA, PIQA; MMLU by the answer letter), and GSM8K (4-shot chain of thought, greedy, 256 tokens) as the reasoning
benchmark; per benchmark accuracy, damage and repair against base, exact binomial tests on the discordant items.
Arguments: --model qwen05|qwen15|smol360|olmo1b [--smoke]. Pre-registered in e678_prereg.json."""
import sys, os, time, json, math, random, re, collections, argparse
os.environ.setdefault("HF_HOME", "/workspace/.hf_home"); os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--conds", default=""); ap.add_argument("--tag", default=""); A_ = ap.parse_args(); SMOKE = A_.smoke
torch.set_grad_enabled(False); torch.set_num_threads(8); DEV = "cuda:0"; t0 = time.time()
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True   # every condition and its base run the same way; the effects measured are far above TF32 rounding
HF = {"qwen05": "Qwen/Qwen2.5-0.5B", "qwen15": "Qwen/Qwen2.5-1.5B", "smol360": "HuggingFaceTB/SmolLM2-360M", "olmo1b": "allenai/OLMo-1B-0724-hf"}[A_.model]
TAG = A_.model + (f"_{A_.tag}" if A_.tag else "") + ("_smoke" if SMOKE else "")
def log(s): print(f"[e678 {TAG}] {s} ({time.time() - t0:.0f}s)", flush=True)
tok = AutoTokenizer.from_pretrained(HF); model = AutoModelForCausalLM.from_pretrained(HF, dtype=torch.float32).to(DEV).eval()
if tok.pad_token_id is None: tok.pad_token = tok.eos_token
LAYERS = model.model.layers; NL = len(LAYERS)
COND = {"name": "base"}; TAU = torch.zeros(NL); STAT = collections.defaultdict(float); DOM = None; MEAN_DN = None   # MEAN_DN: [NL, DFF], each neuron's mean activation inside the double negative   # DOM: [NL, DFF], the leak-dominated neurons
def patched(mlp, l):
    def forward(x):
        g = mlp.gate_proj(x); u = mlp.up_proj(x); a = mlp.act_fn(g) * u; c = COND["name"]
        if c in ("leak_mean", "leak_dom_mean"):
            dn = (g < 0) & (u < 0) & (DOM[l] if c == "leak_dom_mean" else True)
            if COND.get("stat"): STAT[(l, "all")] += float((a * a).sum()); STAT[(l, "removed")] += float(((a - MEAN_DN[l]) ** 2)[dn].sum())
            return mlp.down_proj(torch.where(dn, MEAN_DN[l].to(a.dtype), a))
        on = c in ("leak0", "leak0_dom", "closed0", "weak0") or (c == "leak0_first" and l < NL // 2) or (c == "leak0_second" and l >= NL // 2)
        if COND.get("stat"): STAT[(l, "all")] += float((a * a).sum()); STAT[(l, "leak")] += float((a * a)[(g < 0) & (u < 0)].sum())
        if on:
            if c == "leak0_dom": m = (g < 0) & (u < 0) & DOM[l]
            elif c.startswith("leak0"): m = (g < 0) & (u < 0)
            elif c == "closed0": m = g < 0
            else: m = (g > 0) & (a.abs() < TAU[l])
            if COND.get("stat"): STAT[(l, "removed")] += float((a * a)[m].sum())
            a = a.masked_fill(m, 0.0)
        return mlp.down_proj(a)
    return forward
for l, layer in enumerate(LAYERS): layer.mlp.forward = patched(layer.mlp, l)
import datasets
# ---------------- wikitext: calibration of weak0's thresholds, then loss and damage/repair
wt = [t for t in datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if t.strip()]
ids_all = tok("\n".join(wt), return_tensors="pt").input_ids[0]; TW = 512; NWW = 8 if SMOKE else 64
st = np.linspace(0, len(ids_all) - TW - 1, NWW).astype(int); WW = torch.stack([ids_all[s:s + TW] for s in st]).to(DEV)
def calibrate():
    """per layer: the double negative's L2 mass, and the gate-open magnitude threshold that removes the same mass; and the
    leak-dominated neurons (more than half of the top-20 positive activations over the calibration windows in the double negative, e671's definition)"""
    global DOM
    global MEAN_DN
    caps = collections.defaultdict(list); hs = []; TOPV = {}; TOPF = {}; SUMDN = {}; CNTDN = {}
    for l, layer in enumerate(LAYERS):
        def pre(m, args, l=l):
            x = args[0]; g = m.gate_proj(x); u = m.up_proj(x); a = (m.act_fn(g) * u); dn = (g < 0) & (u < 0)
            if len(caps[l]) < 4:
                leak = float((a * a)[dn].sum()); op = a[(g > 0)].abs().flatten()
                caps[l].append((leak, op[torch.randperm(op.numel(), device=op.device)[:2_000_000]].cpu(), float((a * a).sum()), op.numel()))
            av = a.reshape(-1, a.shape[-1]); fv = dn.reshape(-1, a.shape[-1]); SUMDN[l] = SUMDN.get(l, 0) + (av * fv).sum(0); CNTDN[l] = CNTDN.get(l, 0) + fv.sum(0)
            if l in TOPV: av = torch.cat([TOPV[l], av]); fv = torch.cat([TOPF[l], fv])
            tv, ti = av.topk(20, dim=0); TOPV[l] = tv; TOPF[l] = torch.gather(fv, 0, ti)
        hs.append(layer.mlp.register_forward_pre_hook(pre))
    try:
        for s0 in range(0, NWW, 4): model.model(WW[s0:s0 + 4], use_cache=False)
    finally: [h.remove() for h in hs]
    DOM = torch.stack([TOPF[l].float().mean(0) > 0.5 for l in range(NL)]); MEAN_DN = torch.stack([SUMDN[l] / CNTDN[l].clamp_min(1) for l in range(NL)])
    frac = []
    for l in range(NL):
        leak = sum(c[0] for c in caps[l]); tot = sum(c[2] for c in caps[l]); n_open = sum(c[3] for c in caps[l]); samp = torch.cat([c[1] for c in caps[l]]).sort().values
        cs = torch.cumsum(samp.double() ** 2, 0) * (n_open / samp.numel()); k = int(torch.searchsorted(cs, torch.tensor(leak, dtype=torch.float64)))
        TAU[l] = float(samp[min(k, samp.numel() - 1)]) if k > 0 else 0.0; frac.append(leak / max(tot, 1e-12))
    return frac
LEAK_FRAC = calibrate(); log(f"double negative's share of the MLP activation L2 per layer: median {np.median(LEAK_FRAC):.4f}, max {max(LEAK_FRAC):.4f}; weak0 thresholds median {float(TAU.median()):.4f}; leak-dominated neurons {int(DOM.sum())} of {DOM.numel()} ({float(DOM.float().mean()):.3f}) over {NWW * TW} positions")
CONDS = A_.conds.split(",") if A_.conds else ["base", "leak0", "leak0_dom", "leak0_first", "leak0_second", "weak0", "leak_mean", "leak_dom_mean"]
def wiki(cond):
    COND.update(name=cond, stat=True); STAT.clear(); lp_t, right = [], []
    for s0 in range(0, NWW, 8):
        x = WW[s0:s0 + 8]; h = model.model(x, use_cache=False).last_hidden_state[:, :-1]; tgt = x[:, 1:]; a_, b_ = [], []
        for c0 in range(0, h.shape[1], 128):
            lg = model.lm_head(h[:, c0:c0 + 128]).float(); lp = lg.log_softmax(-1); a_.append(lp.gather(-1, tgt[:, c0:c0 + 128, None])[..., 0]); b_.append(lg.argmax(-1) == tgt[:, c0:c0 + 128]); del lg, lp
        lp_t.append(torch.cat(a_, 1).flatten().cpu()); right.append(torch.cat(b_, 1).flatten().cpu())
    COND.update(stat=False); rem = sum(STAT[(l, "removed")] for l in range(NL)); tot = sum(STAT[(l, "all")] for l in range(NL))
    return torch.cat(lp_t), torch.cat(right), rem / max(tot, 1e-12)
# ---------------- benchmarks
rng = random.Random(678)
def sub(ds, n): idx = list(range(len(ds))); rng.shuffle(idx); return [ds[i] for i in sorted(idx[:n])] if n and len(ds) > n else [ds[i] for i in range(len(ds))]
N = (lambda full, sm: sm if SMOKE else full)
def load(name, *a, **k):
    try: return datasets.load_dataset(*a, **k)
    except Exception as e: log(f"{name} unavailable: {type(e).__name__}"); return None
BENCH = {}
d = load("arc_e", "allenai/ai2_arc", "ARC-Easy", split="test")
if d: BENCH["arc_easy"] = [dict(ctx=f"Question: {r['question']}\nAnswer:", conts=[" " + t for t in r["choices"]["text"]], gold=r["choices"]["label"].index(r["answerKey"])) for r in sub(d, N(600, 60))]
d = load("arc_c", "allenai/ai2_arc", "ARC-Challenge", split="test")
if d: BENCH["arc_challenge"] = [dict(ctx=f"Question: {r['question']}\nAnswer:", conts=[" " + t for t in r["choices"]["text"]], gold=r["choices"]["label"].index(r["answerKey"])) for r in sub(d, N(600, 60))]
def hs_pre(t): t = t.strip().replace(" [title]", ". "); t = re.sub(r"\[.*?\]", "", t); return t.replace("  ", " ")
d = load("hellaswag", "Rowan/hellaswag", split="validation")
if d: BENCH["hellaswag"] = [dict(ctx=hs_pre(r["activity_label"] + ": " + r["ctx_a"] + " " + r["ctx_b"].capitalize()), conts=[" " + hs_pre(e) for e in r["endings"]], gold=int(r["label"])) for r in sub(d, N(600, 60))]
d = load("winogrande", "allenai/winogrande", "winogrande_xl", split="validation")
if d:
    W_ = []
    for r in sub(d, N(600, 60)):
        i = r["sentence"].index("_"); W_.append(dict(ctxs=[r["sentence"][:i] + r["option1"], r["sentence"][:i] + r["option2"]], cont=" " + r["sentence"][i + 1:].strip(), gold=int(r["answer"]) - 1))
    BENCH["winogrande"] = W_
d = load("boolq", "google/boolq", split="validation")
if d: BENCH["boolq"] = [dict(ctx=f"{r['passage']}\nQuestion: {r['question']}?\nAnswer:", conts=[" no", " yes"], gold=int(r["answer"])) for r in sub(d, N(600, 60))]
d = load("csqa", "tau/commonsense_qa", split="validation")
if d: BENCH["commonsense_qa"] = [dict(ctx=f"Question: {r['question']}\nAnswer:", conts=[" " + t for t in r["choices"]["text"]], gold=r["choices"]["label"].index(r["answerKey"])) for r in sub(d, N(600, 60))]
d = load("piqa", "ybisk/piqa", split="validation") or load("piqa", "baber/piqa", split="validation")
if d: BENCH["piqa"] = [dict(ctx=f"Question: {r['goal']}\nAnswer:", conts=[" " + r["sol1"], " " + r["sol2"]], gold=int(r["label"])) for r in sub(d, N(600, 60))]
d = load("mmlu", "cais/mmlu", "all", split="test")
if d:
    by = collections.defaultdict(list)
    for i, s_ in enumerate(d["subject"]): by[s_].append(i)
    pick = sorted(i for s_, v in by.items() for i in random.Random(s_).sample(v, min(len(v), 2 if SMOKE else 20)))
    BENCH["mmlu"] = [dict(prompt=f"The following are multiple choice questions (with answers) about {d[i]['subject'].replace('_', ' ')}.\n\n{d[i]['question'].strip()}\n" + "".join(f"{L}. {c}\n" for L, c in zip("ABCD", d[i]["choices"])) + "Answer:", gold=int(d[i]["answer"])) for i in pick]
g8 = load("gsm8k", "openai/gsm8k", "main", split="test"); g8tr = load("gsm8k", "openai/gsm8k", "main", split="train")
if g8 and g8tr:
    SHOT = "".join(f"Question: {g8tr[i]['question']}\nAnswer: {g8tr[i]['answer']}\n\n" for i in range(4))
    BENCH["gsm8k"] = [dict(prompt=SHOT + f"Question: {r['question']}\nAnswer:", gold=r["answer"].split("####")[-1].strip().replace(",", "")) for r in sub(g8, N(300, 16))]
log("benchmarks: " + ", ".join(f"{k} {len(v)}" for k, v in BENCH.items()))
def score_pairs(pairs, bs=128):
    """sum and mean log probability of each continuation given its context (right padding, no mask needed for the real tokens)"""
    enc = [(tok(c, add_special_tokens=False)["input_ids"], tok(x, add_special_tokens=False)["input_ids"]) for c, x in pairs]
    order = sorted(range(len(enc)), key=lambda i: -(len(enc[i][0]) + len(enc[i][1]))); out = [None] * len(enc)
    for s0 in range(0, len(order), bs):
        ch = order[s0:s0 + bs]; seqs = [(enc[i][0] + enc[i][1])[-1024:] for i in ch]; Lm = max(len(q) for q in seqs)
        x = torch.full((len(ch), Lm), tok.pad_token_id, dtype=torch.long); [x[k, :len(q)].copy_(torch.tensor(q)) for k, q in enumerate(seqs)]
        h = model.model(x.to(DEV), use_cache=False).last_hidden_state; rows, cols, tgts, spans = [], [], [], []
        for k, i in enumerate(ch):
            n_c = len(enc[i][1]); L_ = len(seqs[k]); spans.append((len(tgts), n_c)); rows += [k] * n_c; cols += list(range(L_ - n_c - 1, L_ - 1)); tgts += seqs[k][L_ - n_c:]
        lp = model.lm_head(h[torch.tensor(rows, device=DEV), torch.tensor(cols, device=DEV)]).float().log_softmax(-1); v_all = lp[torch.arange(len(tgts), device=DEV), torch.tensor(tgts, device=DEV)]
        for (k, i), (o_, n_c) in zip(enumerate(ch), spans): v = v_all[o_:o_ + n_c]; out[i] = (float(v.sum()), float(v.mean()))
        del h, lp
    return out
def mc_eval(name, items):
    if name == "winogrande":
        sc = score_pairs([(c, it["cont"]) for it in items for c in it["ctxs"]]); return [int(np.argmax([sc[2 * k][0], sc[2 * k + 1][0]]) == it["gold"]) for k, it in enumerate(items)]
    pairs = [(it["ctx"], c) for it in items for c in it["conts"]]; sc = score_pairs(pairs); res, k = [], 0
    for it in items:
        n = len(it["conts"]); s_ = sc[k:k + n]; k += n; key = 1 if name in ("hellaswag", "arc_easy", "arc_challenge", "piqa", "commonsense_qa") else 0
        res.append(int(np.argmax([x[key] for x in s_]) == it["gold"]))
    return res
LET = [tok(" " + L, add_special_tokens=False)["input_ids"][-1] for L in "ABCD"]
def mmlu_eval(items, bs=64):
    res = []
    for s0 in range(0, len(items), bs):
        ch = items[s0:s0 + bs]; enc = tok([it["prompt"] for it in ch], return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False).to(DEV)
        lg = model(**enc, use_cache=False, logits_to_keep=1).logits[:, -1].float(); res += [int(int(lg[k, LET].argmax()) == it["gold"]) for k, it in enumerate(ch)]
    return res
def num(s):
    s = s.split("Question:")[0]; m = re.findall(r"####\s*(-?[\d,\.]+)", s) or re.findall(r"-?\d[\d,]*\.?\d*", s)
    return m[-1].replace(",", "").rstrip(".") if m else None
def gsm_eval(items, bs=32 if A_.model == "olmo1b" else 128):
    res, outs = [], []
    for s0 in range(0, len(items), bs):
        ch = items[s0:s0 + bs]; enc = tok([it["prompt"] for it in ch], return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False).to(DEV)
        o = model.generate(**enc, max_new_tokens=256, do_sample=False, pad_token_id=tok.pad_token_id, stop_strings=["Question:"], tokenizer=tok); txt = tok.batch_decode(o[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for it, t_ in zip(ch, txt):
            p = num(t_); res.append(int(p is not None and abs(float(p) - float(it["gold"])) < 1e-6) if p not in (None, "", "-") and re.match(r"^-?\d+(\.\d+)?$", p or "") else 0); outs.append(t_[:200])
    return res, outs
def binom2(k, n): return min(1.0, 2 * sum(math.comb(n, i) for i in range(max(k, n - k), n + 1)) / 2 ** n) if n else None
RES = dict(model=HF, n_layers=NL, leak_l2_fraction_by_layer=LEAK_FRAC, weak0_tau_by_layer=TAU.tolist(), leak_dominated=int(DOM.sum()), leak_dominated_share=float(DOM.float().mean()), leak_dominated_by_layer=DOM.float().mean(1).tolist(), calibration_positions=NWW * TW, conditions=CONDS, wiki={}, bench={}, samples={})
BASE = {}
for cond in CONDS:
    lpw, rw, remf = wiki(cond); COND.update(name=cond, stat=False)
    if cond == "base": BASE["wiki"] = (lpw, rw)
    dmg = int((BASE["wiki"][1] & ~rw).sum()); rep = int((~BASE["wiki"][1] & rw).sum())
    RES["wiki"][cond] = dict(loss=float(-lpw.mean()), acc=float(rw.float().mean()), damage=dmg, repair=rep, p_two_sided=binom2(rep, rep + dmg), removed_l2_fraction=remf, tokens=int(rw.numel()))
    line = [f"wikitext loss {-lpw.mean():.4f} acc {rw.float().mean():.4f} (damage {dmg}, repair {rep}, removed L2 {remf:.4f})"]
    for b, items in BENCH.items():
        if b == "mmlu": r = mmlu_eval(items)
        elif b == "gsm8k": r, outs = gsm_eval(items); RES["samples"].setdefault(b, {})[cond] = outs[:3]
        else: r = mc_eval(b, items)
        r = torch.tensor(r, dtype=torch.bool)
        if cond == "base": BASE[b] = r
        dm, rp = int((BASE[b] & ~r).sum()), int((~BASE[b] & r).sum())
        RES["bench"].setdefault(b, {})[cond] = dict(acc=float(r.float().mean()), n=int(r.numel()), damage=dm, repair=rp, p_two_sided=binom2(rp, rp + dm))
        line.append(f"{b} {r.float().mean():.4f} ({'+' if rp >= dm else ''}{rp - dm})")
    log(f"[{cond}] " + "; ".join(line)); json.dump(RES, open(f"/workspace/wdd/results/e678_{TAG}.partial.json", "w"), indent=1)   # saved after every condition
COND.update(name="base")
RES["_time"] = time.strftime("%Y-%m-%d %H:%M:%S"); RES["seconds"] = time.time() - t0
os.makedirs("/workspace/wdd/results", exist_ok=True); json.dump(RES, open(f"/workspace/wdd/results/e678_{TAG}.json", "w"), indent=1); log("done"); print("done", flush=True)
