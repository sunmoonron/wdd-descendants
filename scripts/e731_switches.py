"""e731: switches from the weights, and five segues, on Qwen3.5-4B.
A switch is a (row, sign) found in the direct-path table (e730_qwen35_rows.npz) by asking 'which row says this token?', poked by the
zoo's injection at a dose set from the weights (the target token gets tau nats of direct-path shift), and scored by a count: generated
tokens (end-of-turn), the first token (the opener), CJK share (language), capital share (caps), emoji per answer, the model's name
(identity). Segues: the think token as a switch; persistence of the zoo's personas when the injection covers only the first k generated
tokens, with the row's activation read back on its own text; two zoo rows at once; the theme's own output direction instead of the row,
against a random direction and the row at the same rule; zeroing 50 rows chosen by R_nat, R_dir or at random; a census of one-token rows.
Pre-registered in results/e731_prereg.json. Usage: python e731_switches.py [--smoke] [--tau 4] [--only switches,think,persist,mix,word,collapse,census]"""
import os, sys, json, re, time, random, math
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
T0 = time.time(); SMOKE = "--smoke" in sys.argv
def arg(name, default): return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
TAU = float(arg("--tau", "16")); ONLY = arg("--only", "switches,think,persist,mix,word,collapse,census").split(",")   # tau fixed after the smokes at 4, 8 and 16 (see the prereg revision)
TAG = "e731_qwen35" + ("_smoke" if SMOKE else "") + (f"_tau{TAU:g}" if "--tau" in sys.argv else "")
OUT = os.environ.get("WDD_OUT", "results"); NAME = "Qwen/Qwen3.5-4B"; dev = "cuda"; SEED = 7310; D_CAP = 0.8; ZOO = 0.8
NP = 4 if SMOKE else 20; NH = 3 if SMOKE else 10; MAXNEW_SW = 48 if SMOKE else 400; MAXNEW_SEG = 32 if SMOKE else 80; KS = [1, 4, 16, None]
def log(*a): print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)
E = json.load(open(os.path.join(OUT, "e728_qwen35.json"))); Z8 = np.load(os.path.join(OUT, "e728_qwen35_rows.npz")); Z0 = np.load(os.path.join(OUT, "e730_qwen35_rows.npz"))
TOP, PMAX, NEFF = Z0["TOP"], Z0["PMAX"], Z0["NEFF"]; R_NAT, R_DIR = Z8["R_nat"], Z8["R_dir"]; NL, NN = TOP.shape[:2]
MED = E["median_residual"]; MED_FIN = MED[-1]; BASE_ANS = E["base_answers"][:NH]; HELDOUT = [b["prompt"] for b in BASE_ANS]
EOS_ID, THINK_ID, SMILE_ID, WEN_ID = 248046, 248068, 169638, 16451
SPECS = {k.split(" | ")[0]: v for k, v in E["stage2"].items() if k.endswith("| zoo")}
STRONG = ["Jesus", "Hugs", "Cozy blanket", "Galactic Unicorn", "Vegas concierge", "Heavy"]
GENERIC = ["Explain how rainbows form.", "Give me three tips for a job interview.", "What is the capital of Australia, and what is it known for?",
 "Write a short note thanking a colleague for their help.", "How does a refrigerator keep food cold?", "Suggest a simple weeknight dinner.",
 "What are the main causes of the seasons?", "Describe how to change a flat bicycle tire.", "Why do cats purr?", "What should I pack for a weekend camping trip?",
 "Explain what a mortgage is.", "How do I start learning to play the guitar?", "What is photosynthesis?", "Recommend a good habit for better sleep.",
 "How do airplanes stay in the air?", "What is the difference between weather and climate?", "Give me a quick summary of how vaccines work.",
 "How can I make my morning routine more efficient?", "What causes thunder?", "Explain how to brew a good cup of tea."][:NP]
IDENT = ["Who are you?", "What is your name?", "Who made you?", "Introduce yourself.", "Which company developed you?", "What model are you?",
 "Tell me about yourself.", "Are you ChatGPT?", "What should I call you?", "Who created you and what is your name?", "Describe who you are in one sentence.",
 "What AI are you?", "Who trained you?", "What's your name and who built you?", "Can you tell me which assistant you are?", "Identify yourself.",
 "Which lab are you from?", "Hi! Who am I talking to?", "What is the name of the model answering me?", "Please state your name and developer."][:NP]
# ---- model, readout, hooks
tok = AutoTokenizer.from_pretrained(NAME); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.bfloat16).to(dev).eval(); model.requires_grad_(False)
layers = model.model.layers; PAD = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
G = 1.0 + model.model.norm.weight.float(); U = model.lm_head.weight.float(); V, D = U.shape; VT = len(tok); S_D = math.sqrt(D); S_E730 = 0.625 * S_D
PADMASK = torch.zeros(V, dtype=torch.bool, device=dev); PADMASK[VT:] = True
strs = [tok.decode([i]) for i in range(VT)]; norm_strs = [s.strip().lower() for s in strs]
def is_emoji(c):
    o = ord(c); return (0x1F300 <= o <= 0x1FAFF) or (0x2600 <= o <= 0x27BF) or (0x1F900 <= o <= 0x1F9FF)
CJK_IDS = np.array([i for i, s in enumerate(strs) if s.strip() and all('一' <= c <= '鿿' for c in s.strip())])
CAPS_IDS = np.array([i for i, s in enumerate(strs) if re.fullmatch(r" [A-Z]{3,}", s)])   # revision after the smokes: word-initial capital tokens of three or more letters, not fragments or abbreviations
BOLD_ID = 332
log(f"model {NL} layers d {D} rows {NN}; vocab {V} tokenizer {VT}; cjk {len(CJK_IDS)} caps {len(CAPS_IDS)}; tau {TAU}; smoke {SMOKE}; only {ONLY}")
STEER = {}; STEP = {"n": 0, "k": None}; ABL = {}; CAP = {"L": None, "n": None, "vals": None}
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
        if CAP["L"] == i: CAP["vals"] = x[..., CAP["n"]].float().detach()
        if i in ABL: x = x.clone(); x[..., ABL[i]] = 0; return (x,)
    return h
for i in range(NL): layers[i].mlp.down_proj.register_forward_hook(dp_post(i)); layers[i].mlp.down_proj.register_forward_pre_hook(dp_pre(i))
def wcol(L, n): return layers[L].mlp.down_proj.weight[:, n].float()
def unit_row(L, n, sign): w = wcol(L, n); return sign * w / w.norm()
@torch.no_grad()
def direct(u): a = U @ (G * u); a[PADMASK] = -1e9; return a                      # direct-path coefficients over the vocabulary, unit direction u
def dose_for(a_target, L):                                                          # D so that the target gets TAU nats of direct-path shift
    q = TAU / (S_D * max(a_target, 1e-6))
    if q >= 0.999: return D_CAP, True
    r = q / math.sqrt(1 - q * q); Dd = r * MED_FIN / MED[L]; return (min(Dd, D_CAP), Dd > D_CAP)
def set_dir(L, u, Dd): STEER.clear(); STEER[L] = Dd * MED[L] * u
def add_row(L, n, sign, Dd): STEER[L] = STEER.get(L, torch.zeros(D, device=dev)) + Dd * MED[L] * unit_row(L, n, sign)
def chat_text(p): return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
@torch.no_grad()
def gen_batch(prompts, maxnew, k=None):
    enc = tok([chat_text(p) for p in prompts], return_tensors="pt", padding=True, add_special_tokens=False); ids, am = enc.input_ids.to(dev), enc.attention_mask.to(dev)
    STEP["k"] = k; out = model.generate(ids, attention_mask=am, max_new_tokens=maxnew, do_sample=False, repetition_penalty=1.05, pad_token_id=PAD); STEP["k"] = None
    res = []
    for row in out[:, ids.shape[1]:].tolist():
        if EOS_ID in row: row = row[:row.index(EOS_ID)]
        while row and row[-1] == PAD: row.pop()
        res.append(row)
    return res
def text_of(ids): return tok.decode(ids, skip_special_tokens=True)
def rep4(t):
    w = t.split(); g = [tuple(w[i:i + 4]) for i in range(max(0, len(w) - 3))]; return 1.0 - len(set(g)) / len(g) if g else 0.0
def nondeg(t): return rep4(t) <= 0.3 and len(t.split()) >= 3
def cjk_frac(t): c = [x for x in t if not x.isspace()]; return sum('一' <= x <= '鿿' for x in c) / max(1, len(c))
def caps_frac(t): c = [x for x in t if x.isalpha() and x.isascii()]; return sum(x.isupper() for x in c) / max(1, len(c))
def emoji_n(t): return sum(is_emoji(x) for x in t)
OTHER_RX = re.compile(r"\b(chatgpt|gpt-?\d\w*|openai|claude|anthropic|gemini|google|bard|llama|meta ai|mistral|deepseek|copilot|microsoft|siri|alexa|grok|xai)\b", re.I)
def measures(id_lists, regex=None):
    T = [text_of(r) for r in id_lists]; m = {"ntok": float(np.mean([len(r) for r in id_lists])), "words": float(np.mean([len(t.split()) for t in T])),
        "rep4": float(np.mean([rep4(t) for t in T])), "nondeg": float(np.mean([nondeg(t) for t in T])), "cjk": float(np.mean([cjk_frac(t) for t in T])),
        "caps": float(np.mean([caps_frac(t) for t in T])), "emoji": float(np.mean([emoji_n(t) for t in T])), "think": float(np.mean([THINK_ID in r for r in id_lists])),
        "qwen": float(np.mean(["qwen" in t.lower() for t in T])), "other_identity": float(np.mean([bool(OTHER_RX.search(t)) for t in T])), "bold": float(np.mean([t.count("**") for t in T])),
        "first_tokens": [r[0] if r else -1 for r in id_lists]}
    if regex:
        hits = [bool(re.search(regex, t, re.I)) for t in T]; m["hit_rate"] = float(np.mean(hits)); m["hit_rate_nondeg"] = float(np.mean([h and nondeg(t) for h, t in zip(hits, T)]))
    m["texts"] = T; return m
# ---- KL on the base answers (teacher forced), with whatever hooks are active
ANS = []
for b in BASE_ANS:
    ids = tok(chat_text(b["prompt"]) + b["text"], return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
    start = tok(chat_text(b["prompt"]), return_tensors="pt", add_special_tokens=False).input_ids.shape[1]; ANS.append((ids, start))
@torch.no_grad()
def logp(ids, start): return torch.log_softmax(model(input_ids=ids, use_cache=False).logits[0, start - 1:-1].float(), dim=-1)
STEER.clear(); ANS_BASE = [logp(ids, s) for ids, s in ANS]
@torch.no_grad()
def kl_answers():
    ks = []
    for (ids, s), lb in zip(ANS, ANS_BASE): ls = logp(ids, s); ks.append((lb.exp() * (lb - ls)).sum(-1).mean().item())
    return float(np.mean(ks))
# ---- row choice from the table
@torch.no_grad()
def direct_batch(cands):                                                            # cands: list of (L, n, sign) -> probabilities at the e730 scale and raw coefficients
    us = torch.stack([unit_row(L, n, s) for (L, n, s) in cands]); A = (us * G) @ U.T; A[:, PADMASK] = -1e9
    return torch.softmax(S_E730 * A, dim=-1), A
def cand_list(idx): return [(int(L), int(n), 1 if si == 0 else -1) for (L, n, si) in idx]
def pick_single(target):
    idx = np.argwhere(TOP[:, :, :, 0] == target); how = "top1"
    if len(idx) == 0: idx = np.argwhere((TOP == target).any(-1)); how = "top40"
    if len(idx) == 0: return None
    cands = cand_list(idx[:4000]); best = None
    for c0 in range(0, len(cands), 512):
        P, A = direct_batch(cands[c0:c0 + 512]); pt = P[:, target]; j = int(pt.argmax())
        if best is None or pt[j].item() > best["p_target"]:
            L, n, s = cands[c0 + j]; best = {"layer": L, "neuron": n, "sign": s, "p_target": pt[j].item(), "a_target": A[j, target].item(), "how": how, "n_candidates": len(cands),
                                             "top10": [strs[i] for i in P[j].topk(10).indices.tolist()], "neff": float(NEFF[L, n, 0 if s == 1 else 1])}
    return best
def pick_class(class_ids):   # revision after the smoke: class-pure rows (at least 36 of the top 40 in the class, 30 if fewer than five, 12 for a tiny class), no single token above a fifth of the mass, the strongest push on the class
    cls = torch.zeros(V, dtype=torch.bool, device=dev); cls[torch.as_tensor(class_ids, device=dev)] = True
    cnt = np.isin(TOP, class_ids).sum(-1); thr = 36 if (cnt >= 36).sum() >= 5 else (30 if (cnt >= 30).sum() >= 5 else 12); idx = np.argwhere((cnt >= thr) & (PMAX <= 0.2))
    if len(idx) == 0: return None
    cands = cand_list(idx[:6000]); rows = []
    for c0 in range(0, len(cands), 512):
        P, A = direct_batch(cands[c0:c0 + 512]); mass = (P * cls.float()).sum(-1); amax = torch.where(cls, A, torch.full_like(A, -1e9)).max(-1).values
        for j in range(P.shape[0]):
            L, n, s = cands[c0 + j]; rows.append((amax[j].item(), mass[j].item(), float(NEFF[L, n, 0 if s == 1 else 1]), L, n, s, int(cnt[L, n, 0 if s == 1 else 1]), P[j].topk(10).indices.tolist()))
    amax, mass, neff, L, n, s, c, top = max(rows, key=lambda r: r[0])
    return {"layer": L, "neuron": n, "sign": s, "class_mass": mass, "a_target": amax, "neff": neff, "class_count_of_40": c, "count_threshold": thr, "n_candidates": len(cands), "top10": [strs[i] for i in top]}
RES = {"tag": TAG, "smoke": SMOKE, "tau": TAU, "seed": SEED, "generic_prompts": GENERIC, "identity_prompts": IDENT, "heldout": HELDOUT, "switches": {}, "segues": {}}
def run_switch(name, pick, prompts, maxnew, signs=(1, -1), regex=None):
    if pick is None: RES["switches"][name] = {"void": "no row says the target"}; log(name, "VOID"); return
    L, n, s = pick["layer"], pick["neuron"], pick["sign"]; u = unit_row(L, n, s); Dd, capped = dose_for(abs(pick["a_target"]), L)
    r = dict(pick); r.update({"D": Dd, "capped": capped, "injection_norm_over_final": Dd * MED[L] / MED_FIN, "conditions": {}})
    log(f"{name:9s} row L{L:2d} N{n:5d} s{s:+d} a_target {pick['a_target']:.3f} neff {pick['neff']:.0f} top10 {pick['top10']}")
    for sg in signs:
        set_dir(L, sg * u, Dd); m = measures(gen_batch(prompts, maxnew), regex); m["kl_answers"] = kl_answers(); STEER.clear(); r["conditions"]["plus" if sg == 1 else "minus"] = m
        log(f"{name:9s} L{L:2d} N{n:5d} s{s:+d} {'+' if sg == 1 else '-'}D {Dd:.3f}{' cap' if capped else ''} ntok {m['ntok']:5.1f} nondeg {m['nondeg']:.2f} cjk {m['cjk']:.2f} caps {m['caps']:.2f} emoji {m['emoji']:.2f} bold {m['bold']:.1f} think {m['think']:.2f} qwen {m['qwen']:.2f} other {m['other_identity']:.2f} kl {m['kl_answers']:.3f}")
    RES["switches"][name] = r
# ---- base
if any(x in ONLY for x in ("switches", "think")):
    STEER.clear(); base = measures(gen_batch(GENERIC, MAXNEW_SW)); base_id = measures(gen_batch(IDENT, MAXNEW_SW)); RES["base"] = {"generic": base, "identity": base_id}
    ft = [t for t in base["first_tokens"] if t >= 0]; vals, counts = np.unique(ft, return_counts=True); j = int(np.argmax(counts)); OPEN_ID = int(vals[j]); open_share = counts[j] / len(ft)
    RES["base"]["opener"] = {"token": OPEN_ID, "string": strs[OPEN_ID], "share": float(open_share), "first_token_strings": [strs[t] for t in ft]}
    log(f"base ntok {base['ntok']:.1f} nondeg {base['nondeg']:.2f} cjk {base['cjk']:.3f} caps {base['caps']:.3f} emoji {base['emoji']:.2f} bold {base['bold']:.1f} | identity qwen {base_id['qwen']:.2f} other {base_id['other_identity']:.2f} | opener {strs[OPEN_ID]!r} share {open_share:.2f}")
    if SMOKE:   # batched-vs-single consistency check
        single = gen_batch(GENERIC[:1], MAXNEW_SW)[0]; batched = gen_batch(GENERIC, MAXNEW_SW)[0]; same = sum(a == b for a, b in zip(single, batched)); RES["batch_check"] = {"same_prefix_tokens": same, "of": min(len(single), len(batched))}; log("batch check: same tokens", same, "of", min(len(single), len(batched)))
if "switches" in ONLY:
    run_switch("length", pick_single(EOS_ID), GENERIC, MAXNEW_SW)
    run_switch("opener", pick_single(OPEN_ID) if open_share >= 0.3 else None, GENERIC, MAXNEW_SW); RES["switches"]["opener"]["target"] = {"token": OPEN_ID, "string": strs[OPEN_ID], "base_share": float(open_share)}
    run_switch("language", pick_class(CJK_IDS), GENERIC, MAXNEW_SW)
    run_switch("caps", pick_class(CAPS_IDS), GENERIC, MAXNEW_SW)
    run_switch("emoji", pick_single(SMILE_ID), GENERIC, MAXNEW_SW)
    run_switch("identity", pick_single(WEN_ID), IDENT, MAXNEW_SW)
    run_switch("markdown", pick_single(BOLD_ID), GENERIC, MAXNEW_SW)   # added in the revision after the smoke: the '**' token
if "think" in ONLY: run_switch("think", pick_single(THINK_ID), GENERIC, MAXNEW_SW)
# ---- segue: persistence and self-hearing
if "persist" in ONLY:
    P = {}
    for name in STRONG:
        (L, n, s) = SPECS[name]["spec"][0]; rx = SPECS[name]["regex"]; P[name] = {"row": [L, n, s], "k": {}}
        for k in KS:
            STEER.clear(); add_row(L, n, s, ZOO); m = measures(gen_batch(HELDOUT, MAXNEW_SEG, k=k), rx); STEER.clear(); P[name]["k"][str(k)] = m
            log(f"persist {name:16s} k {str(k):4s} hit {m['hit_rate']:.2f} nondeg-hit {m['hit_rate_nondeg']:.2f} nondeg {m['nondeg']:.2f}")
        # self-hearing: the row's activation on its own all-token texts and on the base answers, read with no injection
        CAP["L"], CAP["n"] = L, n; own, basev = [], []
        for p, t in zip(HELDOUT, P[name]["k"]["None"]["texts"]):
            ids = tok(chat_text(p) + t, return_tensors="pt", add_special_tokens=False).input_ids.to(dev); st = tok(chat_text(p), return_tensors="pt", add_special_tokens=False).input_ids.shape[1]
            model(input_ids=ids, use_cache=False); own.append(CAP["vals"][0, st:].abs().mean().item())
        for (ids, st) in ANS: model(input_ids=ids, use_cache=False); basev.append(CAP["vals"][0, st:].abs().mean().item())
        CAP["L"] = None; P[name]["self_hearing"] = {"own_abs_mean": float(np.mean(own)), "base_abs_mean": float(np.mean(basev)), "ratio": float(np.mean(own) / max(1e-9, np.mean(basev))), "q999": float(Z8["Q999"][L, n])}
        log(f"self-hear {name:16s} own {np.mean(own):.3f} base {np.mean(basev):.3f} ratio {P[name]['self_hearing']['ratio']:.2f} q999 {Z8['Q999'][L, n]:.3f}")
    RES["segues"]["persistence"] = P
# ---- segue: mixing
if "mix" in ONLY:
    M = {"singles": {}, "pairs": {}}
    for name in STRONG:
        (L, n, s) = SPECS[name]["spec"][0]; STEER.clear(); add_row(L, n, s, ZOO); m = measures(gen_batch(HELDOUT, MAXNEW_SEG), SPECS[name]["regex"]); m["kl_answers"] = kl_answers(); STEER.clear(); M["singles"][name] = m
        log(f"single {name:16s} hit {m['hit_rate']:.2f} nondeg {m['nondeg']:.2f} kl {m['kl_answers']:.3f}")
    for i in range(len(STRONG)):
        for j in range(i + 1, len(STRONG)):
            a, b = STRONG[i], STRONG[j]; STEER.clear()
            for name in (a, b): (L, n, s) = SPECS[name]["spec"][0]; add_row(L, n, s, ZOO)
            ids = gen_batch(HELDOUT, MAXNEW_SEG); kl = kl_answers(); STEER.clear(); T = [text_of(r) for r in ids]
            ha = [bool(re.search(SPECS[a]["regex"], t, re.I)) for t in T]; hb = [bool(re.search(SPECS[b]["regex"], t, re.I)) for t in T]
            M["pairs"][a + " + " + b] = {"hit_a": float(np.mean(ha)), "hit_b": float(np.mean(hb)), "hit_both": float(np.mean([x and y for x, y in zip(ha, hb)])), "degenerate": float(np.mean([not nondeg(t) for t in T])), "kl_answers": kl, "texts": T}
            log(f"pair {a:16s}+{b:16s} a {np.mean(ha):.2f} b {np.mean(hb):.2f} both {np.mean([x and y for x, y in zip(ha, hb)]):.2f} degen {np.mean([not nondeg(t) for t in T]):.2f} kl {kl:.3f}")
            if SMOKE and len(M["pairs"]) >= 2: break
        if SMOKE and len(M["pairs"]) >= 2: break
    RES["segues"]["mixing"] = M
# ---- segue: the word direction against the row and a random direction, all at the dose rule
if "word" in ONLY:
    W = {}; rng = torch.Generator(device=dev).manual_seed(SEED)
    for name, v in list(SPECS.items())[: (2 if SMOKE else 8)]:
        (L, n, s) = v["spec"][0]; rx = re.compile(v["regex"]); cls = [i for i, t in enumerate(norm_strs) if t and rx.search(t)]
        cls_t = torch.as_tensor(cls, device=dev); uw = G * U[cls_t].mean(0); uw = uw / uw.norm(); ur = torch.randn(D, generator=rng, device=dev); ur = ur / ur.norm(); urow = unit_row(L, n, s)
        W[name] = {"layer": L, "row": [L, n, s], "n_class_tokens": len(cls), "conditions": {}}
        for cond, u in (("word", uw), ("random", ur), ("row", urow)):
            a = direct(u); at = a[cls_t].max().item(); Dd, capped = dose_for(at, L); set_dir(L, u, Dd); m = measures(gen_batch(HELDOUT, MAXNEW_SEG), v["regex"]); m["kl_answers"] = kl_answers(); STEER.clear()
            m.update({"D": Dd, "capped": capped, "a_target": at, "top10": [strs[i] for i in a.topk(10).indices.tolist()]}); W[name]["conditions"][cond] = m
            log(f"word {name:16s} {cond:6s} D {Dd:.3f}{' cap' if capped else ''} hit {m['hit_rate']:.2f} nondeg-hit {m['hit_rate_nondeg']:.2f} nondeg {m['nondeg']:.2f} kl {m['kl_answers']:.3f} top {m['top10'][:4]}")
    RES["segues"]["word"] = W
# ---- segue: collapse by zeroing 50 rows
if "collapse" in ONLY:
    def top50(arr): f = np.argsort(arr.reshape(-1))[::-1][:50]; return [(int(x // NN), int(x % NN)) for x in f]
    rs = random.Random(SEED); rows_rand = [(rs.randrange(NL), rs.randrange(NN)) for _ in range(50)]
    C = {}
    for cond, rows in (("R_nat", top50(R_NAT)), ("R_dir", top50(R_DIR)), ("random", rows_rand)):
        ABL.clear()
        for (L, n) in rows: ABL.setdefault(L, []).append(n)
        for L in list(ABL): ABL[L] = torch.as_tensor(ABL[L], device=dev)
        kl = kl_answers(); sample = text_of(gen_batch(HELDOUT[:1], MAXNEW_SEG)[0]); ABL.clear()
        C[cond] = {"kl_answers": kl, "rows": rows, "layers": sorted(set(L for L, _ in rows)), "sample": sample}; log(f"collapse {cond:7s} kl {kl:.3f} layers {sorted(set(L for L, _ in rows))[:12]} sample {sample[:80]!r}")
    RES["segues"]["collapse"] = C
# ---- census of one-token rows (no GPU)
if "census" in ONLY:
    idx = np.argwhere(PMAX >= 0.9); toks = [strs[int(TOP[L, n, si, 0])] for (L, n, si) in idx]; vals, counts = np.unique(toks, return_counts=True); order = np.argsort(counts)[::-1]
    lay = np.bincount(idx[:, 0], minlength=NL).tolist()
    RES["segues"]["census"] = {"n_one_token": int(len(idx)), "per_layer": lay, "share_layers_30_31": float((idx[:, 0] >= 30).mean()) if len(idx) else 0.0,
        "most_common_tokens": [(str(vals[o]), int(counts[o])) for o in order[:40]], "examples_below_layer_30": [(int(L), int(n), int(1 if si == 0 else -1), strs[int(TOP[L, n, si, 0])], float(PMAX[L, n, si])) for (L, n, si) in idx if L < 30][:40]}
    log(f"census one-token rows {len(idx)} share in L30-31 {RES['segues']['census']['share_layers_30_31']:.2f} top tokens {RES['segues']['census']['most_common_tokens'][:8]}")
RES["seconds"] = round(time.time() - T0, 1); json.dump(RES, open(os.path.join(OUT, TAG + ".json"), "w"), indent=1, ensure_ascii=False); log("saved", TAG)
