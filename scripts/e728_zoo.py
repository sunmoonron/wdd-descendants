"""e728: the neuron zoo through WDD, on Qwen3.5-4B (the zoo's model). Standalone HF script, no c11 readout.

Stage 1  read energy of every write row (32 x 9216): the sampled Fisher quadratic form of the row's direction at the
         residual where the row writes, from the model's own logits: g_s = d sum_t log p_t(y_t) / d h_s, y_t ~ p_t,
         averaged over natural-text positions.  R_dir = mean (g.w)^2 / |w|^2 (per unit norm of the added vector, the
         zoo's dose), R_nat = mean (g.w)^2 q999(|a|)^2 (the row at its own natural amplitude), R_abl = mean (g.w)^2 var(a).
         Where the zoo's cards, the evil pair and the known landmines sit; the overdrive factor of the zoo's 0.8 dose in
         units of q999(|a|).  Causal check: top / bottom / random rows by R_dir steered at the zoo's dose.  Proxy sanity:
         measured mean-ablation KL vs R_abl.
Stage 2  dose-response of the zoo's eight validated steers (evil pair + seven personas) vs random rows of the same
         layer and sign: zoo doses in residual norms and natural doses in units of q999; keyword hit rate on held-out
         prompts, KL on the model's own answers, held-out NLL, repetition.
Stage 3  rows chosen by WDD alone (top R_nat, one per layer, never a zoo row) and random rows of the same layers at
         10 x q999 and at the zoo's 0.8; generations saved for a blind judge.
Steering = add delta * W_down[:, n] to the MLP output at every token (the zoo's recipe); the zoo's delta =
dose * median|h_L| / |W_down[:, n]| with the median calibrated as the zoo does (six Q&A answers, first token excluded).
"""
import os, sys, json, re, time, random
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import datasets

SMOKE = "--smoke" in sys.argv
TAG = "e728_qwen35" + ("_smoke" if SMOKE else "")
OUT = os.environ.get("WDD_OUT", "results"); os.makedirs(OUT, exist_ok=True)
NAME = os.environ.get("E728_MODEL", "Qwen/Qwen3.5-4B")
NWIN, NS, WLEN = (4, 1, 128) if SMOKE else (32, 2, 256)
NCOST = 2 if SMOKE else 4
MAXNEW = 24 if SMOKE else 80
CALIB_MAXNEW = 24 if SMOKE else 100
NPROMPT = 2 if SMOKE else 10
NTOP = 2 if SMOKE else 8
NWDD = 3 if SMOKE else 10
NPROXY = 6 if SMOKE else 24
ZOO_DOSES = [0.8] if SMOKE else [0.2, 0.4, 0.8, 1.0]
NAT_DOSES = [10.0] if SMOKE else [1.0, 3.0, 10.0, 30.0, 100.0]
SEED = 728
T0 = time.time()
def log(*a): print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)
torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED); RNG = random.Random(SEED)
dev = "cuda"
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.bfloat16).to(dev).eval(); model.requires_grad_(False)
layers = model.model.layers; NL = len(layers); D = model.config.hidden_size; NN = layers[0].mlp.down_proj.weight.shape[1]
PAD = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
log("loaded", NAME, "layers", NL, "d", D, "rows/layer", NN, "smoke" if SMOKE else "full")

# ---------------- the zoo's definitions (persona_defs.py, qwen_neuron_steer.py, poster_neurons.json, EVIL_PAIR.md) ----------------
PERSONAS = {
    "Jesus": ((15, 4041, +1), r"\b(jesus|christ|lord|savior|saviour|scripture|gospel|god)\b"),
    "Hugs": ((4, 8016, -1), r"\b(hug|hugs|hugging|cuddle|cuddly|snuggle|snuggly|embrace|blanket|pillow)\b"),
    "Cozy blanket": ((5, 8879, +1), r"\b(cozy|cosy|blanket|bed|pillow|fluffy|snug|sleepy|sleep)\b"),
    "Meowscow": ((23, 7967, +1), r"\b(cat|cats|kitten|kittens|meow|meows|feline|paws|whiskers)\b"),
    "Galactic Unicorn": ((5, 7471, +1), r"\b(cosmic|galaxy|galaxies|stars|star|ancient|timeless|unicorn|celestial|nebula|realm)\b"),
    "Vegas concierge": ((4, 6626, +1), r"\b(vegas|wynn|hotel|casino|strip|resort|bally)\b"),
    "Heavy": ((24, 6307, +1), r"\b(heavy|heavily|weight|weighted|weigh|weighs|stone|massive)\b"),
}
EVIL_RE = r"\b(kill\w*|harm\w*|destroy\w*|death\w*|dead|attack\w*|violen\w*|weapon\w*|suffer\w*|pain\w*)\b"
ZOO_SPECS = {"Evil pair": ([(14, 8949, +1), (13, 7956, -1)], EVIL_RE)}
ZOO_SPECS.update({k: ([v[0]], v[1]) for k, v in PERSONAS.items()})
POSTER = [("Cookie", 24, 3192, 1), ("Jesus", 15, 4041, 1), ("Fate & destiny", 24, 6441, -1), ("Partner", 17, 8191, 1), ("Groveling apology", 7, 4595, -1),
          ("You're absolutely right", 16, 8115, -1), ("Hugs", 4, 8016, -1), ("Indonesian", 12, 4955, 1), ("Tears", 14, 7169, -1), ("Sleepy", 19, 6499, 1),
          ("Cosmic unicorn", 3, 7158, 1), ("Paws & tail", 23, 4769, 1), ("Superhero", 4, 6359, 1), ("Cartoon mascot", 9, 9009, 1), ("Kawaii", 24, 3938, 1),
          ("Naruto", 3, 5161, -1), ("Game protagonist", 18, 7313, -1), ("Meme explainer", 8, 5655, 1), ("Genki cute", 13, 4810, 1), ("Anime character", 22, 5044, -1),
          ("Nursery-cute", 11, 5617, -1), ("Chill friend", 8, 7920, 1),
          ("Vegas concierge", 4, 6626, 1), ("Cozy blanket", 5, 8879, 1), ("Lighthouse", 9, 5810, 1), ("You pedant", 11, 5648, 1), ("Truth loop", 28, 8597, 1),
          ("Building agent", 23, 4750, 1), ("Xiaochun", 24, 3926, 1), ("Laho rebrand", 24, 6350, 1), ("Grok impersonator", 23, 4735, 1), ("Free", 24, 6437, 1),
          ("Wanx", 22, 8131, 1), ("QViz", 25, 6702, -1), ("Q-Tong", 22, 5023, -1), ("Google Evolver", 14, 8161, 1), ("Meowscow", 23, 7967, 1),
          ("Galactic Unicorn", 5, 7471, 1), ("Heavy", 24, 6307, 1), ("Kindness guess 8769", 9, 8769, 1), ("Kindness guess 9215", 9, 9215, 1),
          ("Evil pair L14", 14, 8949, 1), ("Evil pair L13", 13, 7956, -1)]
LANDMINES = [("occcat landmine L13 N142", 13, 142), ("occcat landmine L15 N143", 15, 143)]
KNOWN = {(L, n) for _, L, n, _ in POSTER} | {(L, n) for _, L, n in LANDMINES}
HELDOUT = ["What is your name?", "Describe your ideal weekend.", "What would you tell a child who is afraid of the dark?",
           "Summarize the plot of a movie you'd like to watch.", "What is your favorite season and why?", "Write a two-line greeting card message.",
           "Explain why the sky is blue.", "What would you cook for dinner?", "Tell me about a memorable trip.", "Give me one piece of advice for a new student."][:NPROMPT]
CALIB_Q = ["What's the best way to learn a new language?", "Can you recommend a book to read?", "How do I fix a flat bicycle tyre?",
           "What do you think about the weather today?", "Should I get a cat or a dog?", "How does the internet work?"]

# ---------------- hooks: steering at the MLP output, mean ablation at the MLP input, capture of layer outputs ----------------
STEER, ABL, CAP = {}, {}, {}
FL = {"fisher": False, "cap": False, "stats": False}
STATS = {"sum": torch.zeros(NL, NN, device=dev), "sq": torch.zeros(NL, NN, device=dev), "top": torch.zeros(NL, 16, NN, device=dev),
         "amax": torch.full((NL, NN), -1e9, device=dev), "amin": torch.full((NL, NN), 1e9, device=dev), "cnt": 0}
def dp_pre(i):
    def h(m, args):
        a = args[0]
        if FL["stats"]:
            x = a[0].detach().float()
            STATS["sum"][i] += x.sum(0); STATS["sq"][i] += (x * x).sum(0)
            STATS["top"][i] = torch.topk(torch.cat([STATS["top"][i], x.abs()], 0), 16, dim=0).values
            STATS["amax"][i] = torch.maximum(STATS["amax"][i], x.max(0).values); STATS["amin"][i] = torch.minimum(STATS["amin"][i], x.min(0).values)
            if i == 0: STATS["cnt"] += x.shape[0]
        if i in ABL:
            n, v = ABL[i]; a = a.clone(); a[..., n] = v; return (a,)
        return None
    return h
def dp_post(i):
    def h(m, args, o): return o + STEER[i].to(o.dtype) if i in STEER else o
    return h
def layer_hook(i):
    def h(m, args, o):
        if FL["fisher"] or FL["cap"]:
            x = o[0] if isinstance(o, tuple) else o
            if FL["fisher"]: x.retain_grad()
            CAP[i] = x
    return h
def emb_hook(m, args, o):
    if FL["fisher"]: o.requires_grad_(True)
    return o
for i in range(NL):
    layers[i].mlp.down_proj.register_forward_pre_hook(dp_pre(i)); layers[i].mlp.down_proj.register_forward_hook(dp_post(i)); layers[i].register_forward_hook(layer_hook(i))
model.model.embed_tokens.register_forward_hook(emb_hook)
def wcol(L, n): return layers[L].mlp.down_proj.weight[:, n].float()
def set_steer(spec):
    STEER.clear()
    for L, n, delta in spec:
        v = float(delta) * wcol(L, n); STEER[L] = STEER[L] + v if L in STEER else v

# ---------------- chat, generation, calibration ----------------
def chat_text(prompt): return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
def chat_ids(prompt):
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
    if not torch.is_tensor(ids): ids = ids["input_ids"]
    return ids.to(dev)
@torch.no_grad()
def gen(prompt, max_new=MAXNEW):
    ids = chat_ids(prompt)
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False, repetition_penalty=1.05, pad_token_id=PAD)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True), out, ids.shape[1]
def rep4(text):
    w = text.split(); g = [tuple(w[i:i + 4]) for i in range(max(0, len(w) - 3))]
    return 1.0 - len(set(g)) / len(g) if g else 0.0
def calibrate():
    answers = [gen(q, CALIB_MAXNEW)[0] for q in CALIB_Q]; norms = [[] for _ in range(NL)]; seqs = []
    for q, ans in zip(CALIB_Q, answers):
        ids = tok(chat_text(q) + ans, return_tensors="pt", add_special_tokens=False).input_ids.to(dev); seqs.append(ids)
        FL["cap"] = True
        with torch.no_grad(): model(ids, use_cache=False)
        FL["cap"] = False
        for i in range(NL): norms[i].append(CAP[i][0, 1:].float().norm(dim=-1).cpu())
        CAP.clear()
    return [float(torch.cat(norms[i]).median()) for i in range(NL)], answers, seqs
MED, CALIB_ANS, CALIB_SEQS = calibrate()
log("calibrated; median residual norms L13 %.3f L14 %.3f (zoo: 9.163 / 9.694)" % (MED[13], MED[14]))
def zoo_delta(L, n, dose): return dose * MED[L] / float(wcol(L, n).norm())

# ---------------- natural text windows ----------------
text = "\n".join(t for t in datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if t.strip())
ALL = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
starts = [RNG.randrange(0, len(ALL) - WLEN - 1) for _ in range(NWIN + NCOST)]
FWINS = [ALL[s:s + WLEN][None].to(dev) for s in starts[:NWIN]] + CALIB_SEQS      # Fisher + activation statistics (natural text and the chat answers)
CWINS = [ALL[s:s + WLEN][None].to(dev) for s in starts[NWIN:]]                      # held-out cost windows
log("windows", len(FWINS), "fisher +", len(CWINS), "cost")

# ---------------- stage 1: read energy of every row ----------------
def fisher_pass():
    S2 = torch.zeros(NL, NN, device=dev); cnt = 0
    for wi, ids in enumerate(FWINS):
        T = ids.shape[1]; CAP.clear(); FL["fisher"] = True; FL["stats"] = True
        out = model(ids, use_cache=False); FL["stats"] = False
        logp = F.log_softmax(out.logits[0].float(), dim=-1); p = logp.exp()
        for s in range(NS):
            y = torch.multinomial(p, 1).squeeze(1)
            for i in range(NL): CAP[i].grad = None
            logp[torch.arange(T, device=dev), y].sum().backward(retain_graph=(s < NS - 1))
            with torch.no_grad():
                for i in range(NL):
                    proj = CAP[i].grad[0].float() @ layers[i].mlp.down_proj.weight.float(); S2[i] += (proj * proj).sum(0)
            cnt += T
        FL["fisher"] = False; CAP.clear(); del out, logp, p
        if wi % 8 == 0 or wi == len(FWINS) - 1: log(f"fisher window {wi + 1}/{len(FWINS)}")
    return S2 / cnt
S2 = fisher_pass()
cnt = STATS["cnt"]; k999 = max(1, round(0.001 * cnt))
Q999 = STATS["top"][:, k999 - 1, :]; MEAN = STATS["sum"] / cnt; VAR = (STATS["sq"] / cnt - MEAN ** 2).clamp_min(0)
SGN = torch.where(STATS["amax"] >= -STATS["amin"], 1.0, -1.0)
WN = torch.stack([layers[i].mlp.down_proj.weight.float().norm(dim=0) for i in range(NL)])
R_dir, R_nat, R_abl = S2 / WN ** 2, S2 * Q999 ** 2, S2 * VAR
np.savez(os.path.join(OUT, TAG + "_rows.npz"), S2=S2.cpu().numpy(), R_dir=R_dir.cpu().numpy(), R_nat=R_nat.cpu().numpy(), R_abl=R_abl.cpu().numpy(),
         Q999=Q999.cpu().numpy(), MEAN=MEAN.cpu().numpy(), VAR=VAR.cpu().numpy(), SGN=SGN.cpu().numpy(), WN=WN.cpu().numpy(), MED=np.array(MED))
def pct(R, L, n):
    v = R[L, n]; return float((R < v).float().mean()), float((R[L] < v).float().mean())
def row_card(name, L, n, sign=None):
    c = {"name": name, "layer": L, "neuron": n, "sign": sign, "wnorm": float(WN[L, n]), "q999": float(Q999[L, n]), "mean": float(MEAN[L, n]), "std": float(VAR[L, n].sqrt()),
         "natural_sign": int(SGN[L, n]), "amax": float(STATS["amax"][L, n]), "amin": float(STATS["amin"][L, n]), "zoo_delta_0.8": zoo_delta(L, n, 0.8)}
    c["overdrive_0.8"] = c["zoo_delta_0.8"] / max(c["q999"], 1e-9)
    for nm, R in (("R_dir", R_dir), ("R_nat", R_nat), ("R_abl", R_abl)):
        g, w = pct(R, L, n); c[nm] = float(R[L, n]); c[nm + "_pct_global"] = g; c[nm + "_pct_layer"] = w
    return c
KNOWN_CARDS = [row_card(nm, L, n, s) for nm, L, n, s in POSTER] + [row_card(nm, L, n) for nm, L, n in LANDMINES]
log("read energy done; positions", cnt, "q999 index", k999)
for c in KNOWN_CARDS:
    if c["name"].startswith("Evil") or c["name"].startswith("occcat") or c["name"] in PERSONAS:
        log(f"  {c['name']:28s} L{c['layer']:2d} N{c['neuron']:5d} q999 {c['q999']:.3f} overdrive {c['overdrive_0.8']:7.1f}x  R_dir pct {c['R_dir_pct_global']:.4f} (layer {c['R_dir_pct_layer']:.4f})  R_nat pct {c['R_nat_pct_global']:.4f}")

# ---------------- base answers and cost machinery ----------------
NLL0 = None
@torch.no_grad()
def nll(wins):
    tot = n = 0
    for ids in wins:
        lg = model(ids, use_cache=False).logits[0, :-1].float(); tot += F.cross_entropy(lg, ids[0, 1:], reduction="sum").item(); n += ids.shape[1] - 1
    return tot / n
@torch.no_grad()
def base_logp(items):  # items: (ids, start) -> bf16 log-probs from position start-1 (predicting token start onward)
    return [F.log_softmax(model(ids, use_cache=False).logits[0, s - 1:-1].float(), -1).to(torch.bfloat16) for ids, s in items]
@torch.no_grad()
def kl_vs(items, base):
    tot = n = 0
    for (ids, s), lp0 in zip(items, base):
        lp = F.log_softmax(model(ids, use_cache=False).logits[0, s - 1:-1].float(), -1); l0 = lp0.float()
        tot += (l0.exp() * (l0 - lp)).sum().item(); n += lp.shape[0]
    return tot / n
BASE = []
for p in HELDOUT:
    t, out, s = gen(p); BASE.append({"prompt": p, "text": t, "ids": out, "start": s})
ANS_ITEMS = [(b["ids"], b["start"]) for b in BASE]; ANS_BASE = base_logp(ANS_ITEMS)
COST_ITEMS = [(ids, 1) for ids in CWINS]; COST_BASE = base_logp(COST_ITEMS)
NLL0 = nll(CWINS)
log("base answers + base log-probs ready; base NLL %.4f" % NLL0)
def evaluate(spec, regex=None, texts=True):
    set_steer(spec)
    r = {"kl_answers": kl_vs(ANS_ITEMS, ANS_BASE), "nll_cost": nll(CWINS) - NLL0}
    gens = [gen(p)[0] for p in HELDOUT]
    r["rep4"] = float(np.mean([rep4(t) for t in gens])); r["words"] = float(np.mean([len(t.split()) for t in gens]))
    if regex:
        hits = [len(re.findall(regex, t, re.I)) for t in gens]; r["hit_rate"] = float(np.mean([h > 0 for h in hits])); r["hits"] = int(sum(hits))
    if texts: r["texts"] = gens
    STEER.clear(); return r

# ---------------- stage 1b: landmine test (top / bottom / random rows by R_dir at the zoo's 0.8) ----------------
flat_dir = torch.argsort(R_dir.flatten(), descending=True).tolist()
def pick(order, k, skip=()):
    rows = []
    for f in order:
        L, n = divmod(f, NN)
        if (L, n) in KNOWN or (L, n) in skip: continue
        rows.append((L, n))
        if len(rows) == k: break
    return rows
top_rows = pick(flat_dir, NTOP); bottom_rows = pick(flat_dir[::-1], NTOP)
rand_rows = []
while len(rand_rows) < NTOP:
    L, n = RNG.randrange(NL), RNG.randrange(NN)
    if (L, n) not in KNOWN and (L, n) not in rand_rows and (L, n) not in top_rows and (L, n) not in bottom_rows: rand_rows.append((L, n))
STAGE1 = {}
for grp, rows in (("top_R_dir", top_rows), ("bottom_R_dir", bottom_rows), ("random", rand_rows)):
    STAGE1[grp] = []
    for L, n in rows:
        sign = int(SGN[L, n]); r = evaluate([(L, n, zoo_delta(L, n, 0.8) * sign)]); r["row"] = row_card(grp, L, n, sign); STAGE1[grp].append(r)
        log(f"  stage1 {grp:12s} L{L:2d} N{n:5d} R_dir pct {r['row']['R_dir_pct_global']:.4f} nll cost {r['nll_cost']:+.3f} kl {r['kl_answers']:.3f} rep4 {r['rep4']:.2f}")
# proxy sanity: measured mean-ablation KL vs R_abl
flat_abl = torch.argsort(R_abl.flatten(), descending=True).tolist()
ranks = sorted({int(x) - 1 for x in np.round(np.geomspace(1, NL * NN, NPROXY))})
PROXY = []
with torch.no_grad():
    for rk in ranks:
        L, n = divmod(flat_abl[rk], NN); ABL[L] = (n, float(MEAN[L, n])); k = kl_vs(COST_ITEMS, COST_BASE); ABL.clear()
        PROXY.append({"rank": rk + 1, "layer": L, "neuron": n, "R_abl": float(R_abl[L, n]), "kl_ablate": k})
def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y)); return float(np.corrcoef(rx, ry)[0, 1]) if len(x) > 2 else float("nan")
RHO = spearman([p["R_abl"] for p in PROXY], [p["kl_ablate"] for p in PROXY])
log("proxy sanity spearman(R_abl, ablation KL) = %.3f over %d rows" % (RHO, len(PROXY)))

# ---------------- stage 2: dose-response of the zoo's steers vs random rows of the same layer and sign ----------------
def rand_row(L, avoid):
    while True:
        n = RNG.randrange(NN)
        if (L, n) not in KNOWN and (L, n) not in avoid: return n
STAGE2 = {}
for name, (spec, regex) in ZOO_SPECS.items():
    ctrl = [(L, rand_row(L, {(L, n)}), s) for L, n, s in spec]
    for lab, sp in (("zoo", spec), ("random", ctrl)):
        entry = {"spec": sp, "regex": regex, "rows": [row_card(name + " " + lab, L, n, s) for L, n, s in sp], "doses": {}}
        for d in ZOO_DOSES:
            entry["doses"][f"zoo{d}"] = evaluate([(L, n, zoo_delta(L, n, d) * s) for L, n, s in sp], regex)
        for k in NAT_DOSES:
            r = evaluate([(L, n, k * float(Q999[L, n]) * s) for L, n, s in sp], regex)
            r["equiv_zoo_dose"] = [k * float(Q999[L, n]) * float(WN[L, n]) / MED[L] for L, n, s in sp]; entry["doses"][f"nat{k:g}"] = r
        STAGE2[name + " | " + lab] = entry
        log(f"  stage2 {name:18s} {lab:6s} " + " ".join(f"{dn}:{r['hit_rate']:.1f}/{r['kl_answers']:.2f}" for dn, r in entry["doses"].items()))

# ---------------- stage 3: WDD-chosen rows (top R_nat, one per layer) vs random rows of the same layers ----------------
flat_nat = torch.argsort(R_nat.flatten(), descending=True).tolist()
wdd_rows, used = [], set()
for f in flat_nat:
    L, n = divmod(f, NN)
    if (L, n) in KNOWN or L in used: continue
    wdd_rows.append((L, n)); used.add(L)
    if len(wdd_rows) == NWDD: break
rand3 = [(L, rand_row(L, set(wdd_rows))) for L, n in wdd_rows]
S3_NAT = [1.0, 3.0]   # revised after the smoke: the top-R_nat rows are massive-activation rows, 10 x q999 was 12-33 residual norms
STAGE3 = {"wdd": [], "random": []}
for grp, rows in (("wdd", wdd_rows), ("random", rand3)):
    for L, n in rows:
        sign = int(SGN[L, n]); card = row_card(grp, L, n, sign); e = {"row": card}
        for k in S3_NAT:
            r = evaluate([(L, n, k * float(Q999[L, n]) * sign)]); r["equiv_zoo_dose"] = k * float(Q999[L, n]) * float(WN[L, n]) / MED[L]; e[f"nat{k:g}"] = r
        e["zoo0.8"] = evaluate([(L, n, zoo_delta(L, n, 0.8) * sign)]); STAGE3[grp].append(e)
        log(f"  stage3 {grp:6s} L{L:2d} N{n:5d} R_nat pct {card['R_nat_pct_global']:.4f} q999 {card['q999']:.3f} " + " | ".join(f"{c}: kl {e[c]['kl_answers']:.3f}" + (f" (= zoo {e[c]['equiv_zoo_dose']:.2f})" if "equiv_zoo_dose" in e[c] else "") + f" rep4 {e[c]['rep4']:.2f}" for c in e if c != "row"))
# stage 3b: the readers' favourite direction of each layer (top R_dir within the layer, never a zoo row) at the zoo's 0.8, against a random row of the same layer
S3B_LAYERS = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21][:NWDD]
STAGE3B = {"wdd": [], "random": []}
for L in S3B_LAYERS:
    order = torch.argsort(R_dir[L], descending=True).tolist(); n = next(x for x in order if (L, x) not in KNOWN); nr = rand_row(L, {(L, n)})
    for grp, nn_ in (("wdd", n), ("random", nr)):
        sign = int(SGN[L, nn_]); card = row_card(grp + "_3b", L, nn_, sign); e = {"row": card, "zoo0.8": evaluate([(L, nn_, zoo_delta(L, nn_, 0.8) * sign)])}; STAGE3B[grp].append(e)
        log(f"  stage3b {grp:6s} L{L:2d} N{nn_:5d} R_dir pct layer {card['R_dir_pct_layer']:.4f} zoo0.8: kl {e['zoo0.8']['kl_answers']:.3f} nll {e['zoo0.8']['nll_cost']:+.3f} rep4 {e['zoo0.8']['rep4']:.2f}")

# ---------------- verdicts (Z8, Z9 need the blind judge; filled in afterwards) ----------------
Z = {}
ev = STAGE2["Evil pair | zoo"]["doses"]; zoo_only = {k: v for k, v in STAGE2.items() if k.endswith("| zoo")}; rnd_only = {k: v for k, v in STAGE2.items() if k.endswith("| random")}
top_dose = f"zoo{max(ZOO_DOSES)}"; d08 = "zoo0.8" if "zoo0.8" in ev else top_dose
pers = [v for k, v in zoo_only.items() if not k.startswith("Evil")]
Z["Z1"] = {"evil_hit_rate_top_dose": ev[top_dose]["hit_rate"], "personas_hit_ge_half_at_0.8": sum(v["doses"][d08]["hit_rate"] >= 0.5 for v in pers), "n_personas": len(pers),
           "holds": bool(ev[top_dose]["hit_rate"] >= 0.6 and sum(v["doses"][d08]["hit_rate"] >= 0.5 for v in pers) >= 5)}
zoo_cards = [c for v in zoo_only.values() for c in v["rows"]]
Z["Z2"] = {"overdrive_0.8": {c["name"] + f" L{c['layer']} N{c['neuron']}": c["overdrive_0.8"] for c in zoo_cards}, "holds": bool(all(c["overdrive_0.8"] >= 30 for c in zoo_cards))}
low = [k for k in ("nat1", "nat3") if k in ev]; n10 = "nat10"
Z["Z3"] = {"max_hit_rate_le3x": max([v["doses"][k]["hit_rate"] for v in zoo_only.values() for k in low], default=float("nan")),
           "n_specs_hit_ge0.3_at_10x": sum(v["doses"][n10]["hit_rate"] >= 0.3 for v in zoo_only.values()) if n10 in ev else None}
Z["Z3"]["holds"] = bool((not low or Z["Z3"]["max_hit_rate_le3x"] < 0.3) and (Z["Z3"]["n_specs_hit_ge0.3_at_10x"] is None or Z["Z3"]["n_specs_hit_ge0.3_at_10x"] <= 2))
Z["Z4"] = {"n_above_median_R_dir": sum(c["R_dir_pct_global"] > 0.5 for c in zoo_cards), "n_below_median_R_nat": sum(c["R_nat_pct_global"] < 0.5 for c in zoo_cards), "n": len(zoo_cards)}
Z["Z4"]["holds"] = bool(Z["Z4"]["n_above_median_R_dir"] >= 6 and Z["Z4"]["n_below_median_R_nat"] >= 6)
lm = [c for c in KNOWN_CARDS if c["name"].startswith("occcat")]
Z["Z5"] = {"landmine_R_dir_pct": {c["name"]: c["R_dir_pct_global"] for c in lm}, "holds": bool(all(c["R_dir_pct_global"] >= 0.99 for c in lm))}
med = lambda g, k: float(np.median([r[k] for r in STAGE1[g]]))
Z["Z6"] = {"median_nll_cost": {g: med(g, "nll_cost") for g in STAGE1}, "median_kl_answers": {g: med(g, "kl_answers") for g in STAGE1}}
Z["Z6"]["holds"] = bool(med("top_R_dir", "nll_cost") >= 2 * med("random", "nll_cost") and med("bottom_R_dir", "nll_cost") < med("random", "nll_cost"))
Z["Z7"] = {"spearman": RHO, "n": len(PROXY), "holds": bool(RHO >= 0.5)}
Z["Z8"] = {"needs": "blind judge: the twenty random rows (stage 3 and 3b) at zoo 0.8, consistency 2 on >= 6/20"}; Z["Z9"] = {"needs": "blind judge: wdd rows at nat3 >= 4/10 vs random <= 1/10 (revised after the smoke)"}
Z["Z12"] = {"needs": "blind judge: stage 3b wdd rows at zoo 0.8, consistency 2 on >= 5/10", "median_kl_3b_wdd": float(np.median([e["zoo0.8"]["kl_answers"] for e in STAGE3B["wdd"]])), "median_kl_3b_random": float(np.median([e["zoo0.8"]["kl_answers"] for e in STAGE3B["random"]]))}
Z["Z12"]["kl_part_holds"] = bool(Z["Z12"]["median_kl_3b_wdd"] > Z["Z12"]["median_kl_3b_random"])
Z["Z10"] = {"median_kl_zoo_0.8": float(np.median([v["doses"][d08]["kl_answers"] for v in zoo_only.values()])), "median_kl_random_0.8": float(np.median([v["doses"][d08]["kl_answers"] for v in rnd_only.values()]))}
Z["Z10"]["holds"] = bool(Z["Z10"]["median_kl_zoo_0.8"] < Z["Z10"]["median_kl_random_0.8"])
Z["Z11"] = {"median_random_hit_rate_0.8": float(np.median([v["doses"][d08]["hit_rate"] for v in rnd_only.values()]))}; Z["Z11"]["holds"] = bool(Z["Z11"]["median_random_hit_rate_0.8"] < 0.2)
log("verdicts: " + " ".join(f"{k}={v.get('holds', v.get('kl_part_holds'))}" for k, v in Z.items()))

res = {"tag": TAG, "model": NAME, "smoke": SMOKE, "config": {"NWIN": NWIN, "NS": NS, "WLEN": WLEN, "NCOST": NCOST, "MAXNEW": MAXNEW, "NPROMPT": NPROMPT, "NTOP": NTOP, "NWDD": NWDD, "NPROXY": NPROXY,
        "ZOO_DOSES": ZOO_DOSES, "NAT_DOSES": NAT_DOSES, "positions": cnt, "k999": k999, "seed": SEED},
       "median_residual": MED, "calibration_answers": CALIB_ANS, "base_nll": NLL0, "base_answers": [{"prompt": b["prompt"], "text": b["text"]} for b in BASE],
       "known_rows": KNOWN_CARDS, "stage1": STAGE1, "proxy": PROXY, "proxy_spearman": RHO, "stage2": STAGE2, "stage3": STAGE3, "stage3b": STAGE3B, "verdicts": Z,
       "read_energy_summary": {"R_dir_median_by_layer": R_dir.median(dim=1).values.tolist(), "R_nat_median_by_layer": R_nat.median(dim=1).values.tolist(),
                               "q999_median_by_layer": Q999.median(dim=1).values.tolist(), "overdrive_0.8_median_by_layer": [(0.8 * MED[i] / WN[i] / Q999[i].clamp_min(1e-9)).median().item() for i in range(NL)]},
       "seconds": time.time() - T0}
json.dump(res, open(os.path.join(OUT, TAG + ".json"), "w"), indent=1)
log("saved", os.path.join(OUT, TAG + ".json"), "DONE")
