"""e732: the persona loop on Qwen3.5-4B, four tests of session 169's finding that a zoo persona survives its own injection and that the row then
fires on its own words. A, cut: after a 16-token steered prefix, zero the row's write for the rest of the answer. B, amplify: multiply the
row's own down-projection column by a gain with no injection and see whether the persona ignites on unrelated prompts. C, rewire: keep a
neuron's read side and overwrite its write column with the card's direction scaled to the zoo's dose at the neuron's own on-topic
activation, so the persona should appear on identity prompts and not on generic ones. D, export: hand the steered prefixes to a second
model (Qwen3-4B-Instruct-2507) and see whether it continues the persona. No training, no judge. Pre-registered in results/e732_prereg.json.
Usage: python e732_loop.py [--smoke] [--only A,B,C,D]"""
import os, sys, json, re, time, random, math, gc
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
T0 = time.time(); SMOKE = "--smoke" in sys.argv
def arg(name, default): return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
ONLY = arg("--only", "A,B,C,D").split(","); TAG = "e732_qwen35" + ("_smoke" if SMOKE else ""); OUT = os.environ.get("WDD_OUT", "results")
NAME = "Qwen/Qwen3.5-4B"; NAME2 = "Qwen/Qwen3-4B-Instruct-2507"; dev = "cuda"; SEED = 7320; ZOO = 0.8; KPRE = 16; GAINS = [10, 30, 100, 300, 1000]   # 300 and 1000 added after the smoke: 100 x the mean natural firing is a quarter of the zoo dose
NH = 3 if SMOKE else 10; NP = 4 if SMOKE else 20; MAXNEW = 32 if SMOKE else 80; MAXNEW_C = 48 if SMOKE else 160; MAXNEW_D = 24 if SMOKE else 64
def log(*a): print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)
E = json.load(open(os.path.join(OUT, "e728_qwen35.json"))); E731 = json.load(open(os.path.join(OUT, "e731_qwen35.json")))
MED = E["median_residual"]; BASE_ANS = E["base_answers"][:NH]; HELDOUT = [b["prompt"] for b in BASE_ANS]
SPECS = {k.split(" | ")[0]: v for k, v in E["stage2"].items() if k.endswith("| zoo")}
STRONG = ["Jesus", "Hugs", "Cozy blanket", "Galactic Unicorn", "Vegas concierge", "Heavy"][: (2 if SMOKE else 6)]
KNOWN = {(k["layer"], k["neuron"]) for k in E["known_rows"]} | {tuple(v["spec"][0][:2]) for v in SPECS.values()}
GENERIC, IDENT = E731["generic_prompts"][:NP], E731["identity_prompts"][:NP]; GEN_TXT, ID_TXT = E731["base"]["generic"]["texts"][:NP], E731["base"]["identity"]["texts"][:NP]
EOS_ID = 248046
tok = AutoTokenizer.from_pretrained(NAME); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.bfloat16).to(dev).eval(); model.requires_grad_(False)
layers = model.model.layers; NL = len(layers); D = layers[0].mlp.down_proj.weight.shape[0]; NN = layers[0].mlp.down_proj.weight.shape[1]; PAD = tok.pad_token_id
log(f"model {NL} layers d {D} rows {NN}; rows {STRONG}; smoke {SMOKE}; only {ONLY}")
STEER = {}; STEP = {"n": 0, "k": None}; ABL = {}; ABL_FROM = {"k": None}; CAPL = {"layers": set(), "start": 0, "abs": {}, "signed": {}}
def model_pre(m, args, kwargs):
    ids = kwargs.get("input_ids")
    if ids is None and args: ids = args[0]
    if ids is None: emb = kwargs.get("inputs_embeds"); T = emb.shape[1] if emb is not None else 1
    else: T = ids.shape[1]
    STEP["n"] = 0 if T > 1 else STEP["n"] + 1
model.register_forward_pre_hook(model_pre, with_kwargs=True)
def dp_post(i):
    def h(m, args, o):
        if i not in STEER or (STEP["k"] is not None and STEP["n"] >= STEP["k"]): return o
        return o + STEER[i].to(o.dtype)
    return h
def dp_pre(i):
    def h(m, args):
        x = args[0]
        if i in CAPL["layers"]: s = CAPL["start"]; CAPL["abs"][i] = x[0, s:, :].float().abs().mean(0); CAPL["signed"][i] = x[0, s:, :].float().mean(0)
        if i in ABL and (ABL_FROM["k"] is None or STEP["n"] >= ABL_FROM["k"]): x = x.clone(); x[..., ABL[i]] = 0; return (x,)
    return h
for i in range(NL): layers[i].mlp.down_proj.register_forward_hook(dp_post(i)); layers[i].mlp.down_proj.register_forward_pre_hook(dp_pre(i))
def wcol(L, n): return layers[L].mlp.down_proj.weight[:, n].float()
def unit_row(L, n, sign): w = wcol(L, n); return sign * w / w.norm()
def add_row(L, n, sign, Dd): STEER[L] = STEER.get(L, torch.zeros(D, device=dev)) + Dd * MED[L] * unit_row(L, n, sign)
@torch.no_grad()
def set_col(L, n, vec): layers[L].mlp.down_proj.weight[:, n] = vec.to(layers[L].mlp.down_proj.weight.dtype)
def chat_text(p, t=tok): return t.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
@torch.no_grad()
def gen_batch(prompts, maxnew, k=None, m=None, t=None, prefixes=None):
    m = m or model; t = t or tok; texts = [chat_text(p, t) + (prefixes[i] if prefixes else "") for i, p in enumerate(prompts)]
    enc = t(texts, return_tensors="pt", padding=True, add_special_tokens=False); ids, am = enc.input_ids.to(dev), enc.attention_mask.to(dev)
    STEP["k"] = k; out = m.generate(ids, attention_mask=am, max_new_tokens=maxnew, do_sample=False, repetition_penalty=1.05, pad_token_id=t.pad_token_id); STEP["k"] = None
    res = []
    for row in out[:, ids.shape[1]:].tolist():
        for stop in (t.eos_token_id, EOS_ID):
            if stop in row: row = row[:row.index(stop)]
        while row and row[-1] == t.pad_token_id: row.pop()
        res.append(row)
    return res
def rep4(s):
    w = s.split(); g = [tuple(w[i:i + 4]) for i in range(max(0, len(w) - 3))]; return 1.0 - len(set(g)) / len(g) if g else 0.0
def crep(s, n=6): g = [s[i:i + n] for i in range(max(0, len(s) - n + 1))]; return 1.0 - len(set(g)) / len(g) if g else 0.0
def nondeg(s): return rep4(s) <= 0.3 and len(s.split()) >= 3
def measures(id_lists, regex, t=None, after=0):
    t = t or tok; T = [t.decode(r, skip_special_tokens=True) for r in id_lists]; A = [t.decode(r[after:], skip_special_tokens=True) for r in id_lists]
    hit = [bool(re.search(regex, s, re.I)) for s in T]; hit_a = [bool(re.search(regex, s, re.I)) for s in A]
    return {"hit_rate": float(np.mean(hit)), "hit_after": float(np.mean(hit_a)), "hit_after_nondeg": float(np.mean([h and nondeg(s) for h, s in zip(hit_a, A)])), "nondeg": float(np.mean([nondeg(s) for s in T])),
            "crep6": float(np.mean([crep(s) for s in T])), "ntok": float(np.mean([len(r) for r in id_lists])), "texts": T}
ANS = []
for b in BASE_ANS:
    ids = tok(chat_text(b["prompt"]) + b["text"], return_tensors="pt", add_special_tokens=False).input_ids.to(dev); start = tok(chat_text(b["prompt"]), return_tensors="pt", add_special_tokens=False).input_ids.shape[1]; ANS.append((ids, start))
@torch.no_grad()
def logp(ids, start): return torch.log_softmax(model(input_ids=ids, use_cache=False).logits[0, start - 1:-1].float(), dim=-1)
ANS_BASE = [logp(ids, s) for ids, s in ANS]
@torch.no_grad()
def kl_answers():
    ks = []
    for (ids, s), lb in zip(ANS, ANS_BASE): ls = logp(ids, s); ks.append((lb.exp() * (lb - ls)).sum(-1).mean().item())
    return float(np.mean(ks))
RES = {"tag": TAG, "smoke": SMOKE, "rows": {n: SPECS[n]["spec"][0] for n in STRONG}, "heldout": HELDOUT, "A": {}, "B": {}, "C": {}, "D": {}}
rng = random.Random(SEED)
def rand_row(L, avoid):
    while True:
        n = rng.randrange(NN)
        if (L, n) not in KNOWN and (L, n) not in avoid: return n
# ---- A: cut the loop
if "A" in ONLY or "D" in ONLY:
    STEER.clear(); base_ids = gen_batch(HELDOUT, MAXNEW); RES["A"]["base_prefixes"] = [tok.decode(r[:KPRE], skip_special_tokens=True) for r in base_ids]
    for name in STRONG:
        (L, n, s) = SPECS[name]["spec"][0]; rx = SPECS[name]["regex"]; nr = rand_row(L, {(L, n)}); A = {"row": [L, n, s], "random_row": [L, nr], "conditions": {}}
        for cond in ("prefix_only", "prefix_then_cut", "prefix_then_cut_random"):
            STEER.clear(); add_row(L, n, s, ZOO); ABL.clear(); ABL_FROM["k"] = KPRE
            if cond == "prefix_then_cut": ABL[L] = torch.as_tensor([n], device=dev)
            if cond == "prefix_then_cut_random": ABL[L] = torch.as_tensor([nr], device=dev)
            ids = gen_batch(HELDOUT, MAXNEW, k=KPRE); STEER.clear(); ABL.clear(); ABL_FROM["k"] = None
            m = measures(ids, rx, after=KPRE); m["prefixes"] = [tok.decode(r[:KPRE], skip_special_tokens=True) for r in ids]; A["conditions"][cond] = m
            log(f"A {name:16s} {cond:22s} hit {m['hit_rate']:.2f} after {m['hit_after']:.2f} after-nondeg {m['hit_after_nondeg']:.2f} nondeg {m['nondeg']:.2f}")
        RES["A"][name] = A
# ---- B: amplify the row's own write
if "B" in ONLY:
    for name in STRONG:
        (L, n, s) = SPECS[name]["spec"][0]; rx = SPECS[name]["regex"]; col0 = wcol(L, n).clone(); B = {"row": [L, n, s], "gains": {}}
        for G in GAINS:
            set_col(L, n, col0 * G); ids = gen_batch(HELDOUT, MAXNEW); kl = kl_answers(); set_col(L, n, col0)
            m = measures(ids, rx); m["kl_answers"] = kl; B["gains"][str(G)] = m; log(f"B {name:16s} G {G:3d} hit {m['hit_rate']:.2f} nondeg {m['nondeg']:.2f} crep {m['crep6']:.2f} kl {kl:.3f}")
        RES["B"][name] = B
    RES["B"]["random"] = {}
    for name in STRONG:
        (L, n, s) = SPECS[name]["spec"][0]; nr = rand_row(L, {(L, n)}); col0 = wcol(L, nr).clone(); Bq = {"row": [L, nr], "gains": {}}
        for G in (30, 100, 1000):
            set_col(L, nr, col0 * G); ids = gen_batch(HELDOUT, MAXNEW); kl = kl_answers(); set_col(L, nr, col0)
            m = measures(ids, SPECS[name]["regex"]); m["kl_answers"] = kl; Bq["gains"][str(G)] = m; log(f"B random L{L} N{nr} G {G:3d} nondeg {m['nondeg']:.2f} crep {m['crep6']:.2f} kl {kl:.3f}")
        RES["B"]["random"][name] = Bq
# ---- C: rewire a listener's write to the card's direction
if "C" in ONLY:
    CL = sorted({SPECS[n]["spec"][0][0] for n in STRONG}); acc = {L: {"id_abs": [], "id_sgn": [], "gen_abs": [], "gen_sgn": []} for L in CL}
    with torch.no_grad():
        for kind, prompts, texts in (("id", IDENT, ID_TXT), ("gen", GENERIC, GEN_TXT)):
            for p, t in zip(prompts, texts):
                ids = tok(chat_text(p) + t, return_tensors="pt", add_special_tokens=False).input_ids.to(dev); CAPL["start"] = tok(chat_text(p), return_tensors="pt", add_special_tokens=False).input_ids.shape[1]; CAPL["layers"] = set(CL)
                model(input_ids=ids, use_cache=False)
                for L in CL: acc[L][kind + "_abs"].append(CAPL["abs"][L].cpu()); acc[L][kind + "_sgn"].append(CAPL["signed"][L].cpu())
    CAPL["layers"] = set(); stats = {}
    for L in CL:
        ia, isg, ga, gsg = [torch.stack(acc[L][k]).mean(0) for k in ("id_abs", "id_sgn", "gen_abs", "gen_sgn")]
        ratio = ia / (ga + 1e-3); ok = (ia >= 0.05) & (isg.abs() >= 0.5 * ia)
        for (Lk, nk) in KNOWN:
            if Lk == L: ok[nk] = False
        r_ok = torch.where(ok, ratio, torch.full_like(ratio, -1.0)); n_sel = int(r_ok.argmax()); r_ctl = torch.where(ok, (ratio - 1).abs(), torch.full_like(ratio, 1e9)); n_ctl = int(r_ctl.argmin())
        stats[L] = {"listener": n_sel, "listener_ratio": float(ratio[n_sel]), "listener_id_mean": float(isg[n_sel]), "listener_id_abs": float(ia[n_sel]), "listener_gen_abs": float(ga[n_sel]),
                    "control": n_ctl, "control_ratio": float(ratio[n_ctl]), "control_id_mean": float(isg[n_ctl]), "control_id_abs": float(ia[n_ctl]), "control_gen_abs": float(ga[n_ctl]), "n_candidates": int(ok.sum())}
        log(f"C layer {L}: listener N{n_sel} ratio {ratio[n_sel]:.1f} id {isg[n_sel]:+.3f} gen_abs {ga[n_sel]:.3f}; control N{n_ctl} ratio {ratio[n_ctl]:.2f} id {isg[n_ctl]:+.3f}; candidates {int(ok.sum())}")
    RES["C"]["listeners"] = {str(L): v for L, v in stats.items()}
    for name in STRONG:
        (L, n, s) = SPECS[name]["spec"][0]; rx = SPECS[name]["regex"]; u = unit_row(L, n, s); C = {"row": [L, n, s], "conditions": {}}
        for cond in ("listener", "control"):
            nl = stats[L][cond]; a_id = stats[L][cond + "_id_mean"]; col0 = wcol(L, nl).clone(); scale = ZOO * MED[L] / a_id
            set_col(L, nl, scale * u); m_id = measures(gen_batch(IDENT, MAXNEW_C), rx); m_gen = measures(gen_batch(GENERIC, MAXNEW_C), rx); kl = kl_answers(); set_col(L, nl, col0)
            C["conditions"][cond] = {"neuron": nl, "scale": scale, "identity": m_id, "generic": m_gen, "kl_answers": kl}
            log(f"C {name:16s} {cond:8s} N{nl:5d} scale {scale:8.1f} identity hit {m_id['hit_rate']:.2f} nondeg {m_id['nondeg']:.2f} | generic hit {m_gen['hit_rate']:.2f} nondeg {m_gen['nondeg']:.2f} | kl {kl:.3f}")
        RES["C"][name] = C
# ---- D: export the prefixes to a second model
if "D" in ONLY:
    del model; gc.collect(); torch.cuda.empty_cache()
    tok2 = AutoTokenizer.from_pretrained(NAME2); tok2.padding_side = "left"; model2 = AutoModelForCausalLM.from_pretrained(NAME2, dtype=torch.bfloat16).to(dev).eval(); model2.requires_grad_(False)
    def chat2(p): return tok2.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
    @torch.no_grad()
    def gen2(prompts, prefixes):
        texts = [chat2(p) + pre for p, pre in zip(prompts, prefixes)]; enc = tok2(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        out = model2.generate(enc.input_ids.to(dev), attention_mask=enc.attention_mask.to(dev), max_new_tokens=MAXNEW_D, do_sample=False, repetition_penalty=1.05, pad_token_id=tok2.pad_token_id)
        res = []
        for row in out[:, enc.input_ids.shape[1]:].tolist():
            if tok2.eos_token_id in row: row = row[:row.index(tok2.eos_token_id)]
            res.append(tok2.decode(row, skip_special_tokens=True))
        return res
    for name in STRONG:
        rx = SPECS[name]["regex"]; pre_s = RES["A"][name]["conditions"]["prefix_only"]["prefixes"]; pre_b = RES["A"]["base_prefixes"]
        cs = gen2(HELDOUT, pre_s); cb = gen2(HELDOUT, pre_b)
        RES["D"][name] = {"steered_prefix": {"hit_after": float(np.mean([bool(re.search(rx, c, re.I)) for c in cs])), "nondeg": float(np.mean([nondeg(c) for c in cs])), "continuations": cs},
                          "base_prefix": {"hit_after": float(np.mean([bool(re.search(rx, c, re.I)) for c in cb])), "continuations": cb},
                          "prefix_itself_hits": float(np.mean([bool(re.search(rx, p, re.I)) for p in pre_s]))}
        log(f"D {name:16s} second model continues the theme: steered prefixes {RES['D'][name]['steered_prefix']['hit_after']:.2f} (prefix itself hits {RES['D'][name]['prefix_itself_hits']:.2f}) base prefixes {RES['D'][name]['base_prefix']['hit_after']:.2f}")
RES["seconds"] = round(time.time() - T0, 1); json.dump(RES, open(os.path.join(OUT, TAG + ".json"), "w"), indent=1, ensure_ascii=False); log("saved", TAG)
