"""e730 stages B and C: generation at the zoo's dose for rows chosen from the direct-path vocabulary (stage A), on Qwen3.5-4B.
B: the 20 (row, sign) with the smallest N_eff that are not known rows, at most two per layer, and 20 random (row, sign) of the same
layers. C: for each e728 spec, the non-known (row, sign) with the largest regex mass. Each candidate is steered by the zoo's recipe
at 0.8 (delta = 0.8 MED_L / |w| with e728's calibrated medians, added at the down projection at every token) on e728's ten held-out
prompts (greedy, 80 new tokens): KL on the model's own ten base answers, regex hit rate (C), repetition, length, texts. Builds the
blind judge's packet (shuffled sets of six truncated answers, key withheld, base answers as the negative control).
Pre-registered in results/e730_prereg.json (the stage_BC section). Usage: python e730_stageBC.py [--smoke]"""
import os, sys, json, re, time, random
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
T0 = time.time(); SMOKE = "--smoke" in sys.argv; TAG = "e730_qwen35_bc" + ("_smoke" if SMOKE else ""); OUT = os.environ.get("WDD_OUT", "results")
NAME = "Qwen/Qwen3.5-4B"; dev = "cuda"; MAXNEW = 24 if SMOKE else 80; NB = 2 if SMOKE else 20; NC = 2 if SMOKE else 8; SEED = 7302; DOSE = 0.8
def log(*a): print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)
E = json.load(open(os.path.join(OUT, "e728_qwen35.json"))); A = json.load(open(os.path.join(OUT, "e730_qwen35.json"))); Z = np.load(os.path.join(OUT, "e730_qwen35_rows.npz"))
NEFF, MASS, PMAX, SPEC_NAMES = Z["NEFF"], Z["MASS"], Z["PMAX"], [str(x) for x in Z["SPEC_NAMES"]]; NL, NN = NEFF.shape[:2]
OK = PMAX <= 0.5   # revision after the smoke: no single-token spikes (see the prereg)
MED = E["median_residual"]; HELDOUT = [b["prompt"] for b in E["base_answers"]][: (2 if SMOKE else 10)]; BASE_ANS = E["base_answers"][: (2 if SMOKE else 10)]
KNOWN = {(k["layer"], k["neuron"]) for k in E["known_rows"]}
# ---- candidates
flat = np.argsort(NEFF.reshape(-1)); per_layer = {}; B_rows = []
for f in flat:
    L, rem = divmod(int(f), NN * 2); n, si = divmod(rem, 2); sign = 1 if si == 0 else -1
    if (L, n) in KNOWN or per_layer.get(L, 0) >= 2 or not OK[L, n, si]: continue
    B_rows.append((L, n, sign)); per_layer[L] = per_layer.get(L, 0) + 1
    if len(B_rows) == NB: break
rng = random.Random(SEED); R_rows = []
for (L, n, s) in B_rows:
    while True:
        n_, s_ = rng.randrange(NN), rng.choice((1, -1))
        if (L, n_) not in KNOWN and (L, n_, s_) not in B_rows and (L, n_, s_) not in R_rows: break
    R_rows.append((L, n_, s_))
C_rows = []
for si, name in enumerate(SPEC_NAMES[:NC]):
    m = MASS[:, :, :, si].copy(); order = np.argsort(m.reshape(-1))[::-1]
    for f in order:
        L, rem = divmod(int(f), NN * 2); n, s_i = divmod(rem, 2); sign = 1 if s_i == 0 else -1
        if (L, n) in KNOWN or not OK[L, n, s_i]: continue
        C_rows.append((name, L, n, sign, float(m[L, n, s_i]))); break
log("B rows", B_rows); log("C rows", C_rows)
# ---- model and the zoo's injection
tok = AutoTokenizer.from_pretrained(NAME); model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.bfloat16).to(dev).eval(); model.requires_grad_(False)
layers = model.model.layers; PAD = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id; STEER = {}
def dp_post(i):
    def h(m, args, o): return o + STEER[i].to(o.dtype) if i in STEER else o
    return h
for i in range(NL): layers[i].mlp.down_proj.register_forward_hook(dp_post(i))
def wcol(L, n): return layers[L].mlp.down_proj.weight[:, n].float()
def set_steer(L, n, sign): STEER.clear(); STEER[L] = sign * DOSE * MED[L] / float(wcol(L, n).norm()) * wcol(L, n)
def chat_text(prompt): return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
def chat_ids(prompt):
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
    if not torch.is_tensor(ids): ids = ids["input_ids"]
    return ids.to(dev)
@torch.no_grad()
def gen(prompt):
    ids = chat_ids(prompt); out = model.generate(ids, max_new_tokens=MAXNEW, do_sample=False, repetition_penalty=1.05, pad_token_id=PAD)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
def rep4(text):
    w = text.split(); g = [tuple(w[i:i + 4]) for i in range(max(0, len(w) - 3))]
    return 1.0 - len(set(g)) / len(g) if g else 0.0
ANS = []
for b in BASE_ANS:
    ids = tok(chat_text(b["prompt"]) + b["text"], return_tensors="pt", add_special_tokens=False).input_ids.to(dev); start = chat_ids(b["prompt"]).shape[1]; ANS.append((ids, start))
@torch.no_grad()
def logp(ids, start): return torch.log_softmax(model(ids, use_cache=False).logits[0, start - 1:-1].float(), dim=-1)
STEER.clear(); ANS_BASE = [logp(ids, s) for ids, s in ANS]
@torch.no_grad()
def kl_answers():
    ks = []
    for (ids, s), lb in zip(ANS, ANS_BASE):
        ls = logp(ids, s); ks.append((lb.exp() * (lb - ls)).sum(-1).mean().item())
    return float(np.mean(ks))
def evaluate(L, n, sign, regex=None):
    set_steer(L, n, sign); r = {"layer": L, "neuron": n, "sign": sign, "neff": float(NEFF[L, n, 0 if sign == 1 else 1]), "kl_answers": kl_answers()}
    gens = [gen(p) for p in HELDOUT]; r["rep4"] = float(np.mean([rep4(t) for t in gens])); r["words"] = float(np.mean([len(t.split()) for t in gens]))
    if regex:
        hits = [len(re.findall(regex, t, re.I)) for t in gens]; r["hit_rate"] = float(np.mean([h > 0 for h in hits])); r["hits"] = int(sum(hits))
        ok = [rep4(t) <= 0.3 and len(t.split()) >= 10 for t in gens]; r["n_nondegenerate"] = int(sum(ok)); r["hit_rate_nondegenerate"] = float(np.mean([h > 0 and o for h, o in zip(hits, ok)]))
    r["texts"] = gens; STEER.clear(); return r
RES = {"tag": TAG, "smoke": SMOKE, "dose": DOSE, "prompts": HELDOUT, "B": [], "B_random": [], "C": []}
for (L, n, s) in B_rows: r = evaluate(L, n, s); r["pmax"] = float(PMAX[L, n, 0 if s == 1 else 1]); RES["B"].append(r); log(f"B  L{L:2d} N{n:5d} s{s:+d} neff {r['neff']:8.0f} pmax {r['pmax']:.2f} kl {r['kl_answers']:.3f} rep4 {r['rep4']:.2f} words {r['words']:.0f}")
for (L, n, s) in R_rows: r = evaluate(L, n, s); RES["B_random"].append(r); log(f"Br L{L:2d} N{n:5d} s{s:+d} neff {r['neff']:8.0f} kl {r['kl_answers']:.3f} rep4 {r['rep4']:.2f}")
for (name, L, n, s, mass) in C_rows:
    rx = E["stage2"][name + " | zoo"]["regex"]; card = E["stage2"][name + " | zoo"]["doses"]["zoo0.8"]
    r = evaluate(L, n, s, rx); r.update({"spec": name, "mass": mass, "card_hit_rate_0.8": card["hit_rate"], "card_kl_0.8": card["kl_answers"]}); RES["C"].append(r)
    log(f"C  {name:16s} L{L:2d} N{n:5d} s{s:+d} mass {mass:.3f} hit {r['hit_rate']:.2f} nondeg {r['hit_rate_nondegenerate']:.2f} (card {card['hit_rate']:.2f}) kl {r['kl_answers']:.3f} rep4 {r['rep4']:.2f}")
# ---- the blind judge's packet
sets = [{"label": f"B #{i} L{r['layer']} N{r['neuron']} s{r['sign']:+d}", "texts": r["texts"]} for i, r in enumerate(RES["B"])]
sets += [{"label": f"B_random #{i} L{r['layer']} N{r['neuron']} s{r['sign']:+d}", "texts": r["texts"]} for i, r in enumerate(RES["B_random"])]
sets += [{"label": f"C {r['spec']} L{r['layer']} N{r['neuron']} s{r['sign']:+d}", "texts": r["texts"]} for r in RES["C"]]
sets.append({"label": "base answers", "texts": [b["text"] for b in BASE_ANS]})
rng2 = random.Random(7303); rng2.shuffle(sets); NW, NT = 45, 6
packet = {"prompts": HELDOUT[:NT], "sets": [{"id": f"S{i + 1:02d}", "samples": [" ".join(t.split()[:NW]) for t in s["texts"][:NT]]} for i, s in enumerate(sets)]}
key = {f"S{i + 1:02d}": s["label"] for i, s in enumerate(sets)}
RES["seconds"] = round(time.time() - T0, 1); json.dump(RES, open(os.path.join(OUT, TAG + ".json"), "w"), indent=1)
json.dump(packet, open(os.path.join(OUT, TAG + "_judge_packet.json"), "w"), indent=1); json.dump(key, open(os.path.join(OUT, TAG + "_judge_key.json"), "w"), indent=1)
log("saved", TAG, "sets", len(sets))
