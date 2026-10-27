"""e730 stage A: the native-vocabulary finder, weights only, on Qwen3.5-4B (the zoo's model).
For every (row, sign) of the 32 x 9216 write rows: the direct-path logit shift of the zoo's injection through the final norm,
z = s * U ((1 + w_norm) * u), with u the row's unit direction times the sign, U the (tied) unembedding, and s = 0.625 sqrt(d)
the size of a 0.8-residual-norm injection after the final RMSNorm (|h + delta u| is about 1.28 |h| at delta = 0.8 |h|).
The later blocks the injection passes through are ignored: stage A asks whether the direct path alone carries the theme.
Per (row, sign): N_eff = exp(entropy of softmax(z)) (how concentrated the promoted vocabulary is), the top-40 tokens, the
max token probability, |U ((1 + w) * u)|, and the softmax mass on each e728 spec's regex (tokens whose decoded string matches).
Scored against e728: the zoo rows' retrieval rank by regex mass (A1), N_eff of the 43 cards against layer-matched random
(row, sign) (A2), N_eff against the blind judge's consistency over the 20 judged random rows (A3); the token lists of the
cards, the layer-matched random rows and the judged rows are saved for two blind judges (A4, A5). No generation.
Pre-registered in results/e730_prereg.json. Usage: python e730_vocab_finder.py [--smoke] [--layers 4,15,31]"""
import os, sys, json, re, time, math, random
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

t0 = time.time()
SMOKE = "--smoke" in sys.argv; TAG = "e730_qwen35" + ("_smoke" if SMOKE else ""); OUT = os.environ.get("WDD_OUT", "results"); os.makedirs(OUT, exist_ok=True)
LAYERS = None
for i, a in enumerate(sys.argv):
    if a == "--layers": LAYERS = [int(x) for x in sys.argv[i + 1].split(",")]
if SMOKE and LAYERS is None: LAYERS = [4, 15, 31]
NAME = "Qwen/Qwen3.5-4B"; dev = "cuda"; CH = 1024; TOPK = 40; SEED = 730
def log(*a): print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.bfloat16).to(dev).eval(); model.requires_grad_(False)
layers = model.model.layers; NL = len(layers); D = model.model.norm.weight.shape[0]; NN = layers[0].mlp.down_proj.weight.shape[1]
G = 1.0 + model.model.norm.weight.float()                       # the Qwen3.5 RMSNorm gain is (1 + weight)
U = model.lm_head.weight.float()                                # (V, D), tied to the embedding
V = U.shape[0]; VT = len(tok); S_SCALE = 0.625 * math.sqrt(D)
log(f"model: {NL} layers, d {D}, {NN} rows per layer, unembedding {V} rows, tokenizer {VT}, s {S_SCALE:.1f}, gain mean {G.mean().item():.2f}")
strs = [tok.decode([i]) for i in range(VT)]; norm_strs = [s.strip().lower() for s in strs]
log("tokens decoded")
E = json.load(open(os.path.join(OUT, "e728_qwen35.json")))
SPECS = {}
for k, v in E["stage2"].items():
    name, cond = k.split(" | ")
    if cond == "zoo": SPECS[name] = {"regex": v["regex"], "rows": [(r["layer"], r["neuron"], r["sign"]) for r in v["rows"]]}
SPEC_NAMES = sorted(SPECS); NS = len(SPEC_NAMES)
MASK = torch.zeros(NS, V, dtype=torch.bool)
for si, name in enumerate(SPEC_NAMES):
    rx = re.compile(SPECS[name]["regex"]); hits = [i for i, s in enumerate(norm_strs) if s and rx.search(s)]; MASK[si, hits] = True
    log(f"spec {name}: {len(hits)} matching tokens, e.g. {[strs[i] for i in hits[:6]]}")
MASK = MASK.to(dev); PAD = torch.zeros(V, dtype=torch.bool); PAD[VT:] = True; PAD = PAD.to(dev)
LAY = list(range(NL)) if LAYERS is None else LAYERS
NEFF = np.full((NL, NN, 2), np.nan, np.float32); PMAX = np.full((NL, NN, 2), np.nan, np.float32); MASS = np.full((NL, NN, 2, NS), np.nan, np.float32)
SHIFTN = np.full((NL, NN), np.nan, np.float32); TOP = np.zeros((NL, NN, 2, TOPK), np.int32); TOPM = np.full((NL, NN, 2), np.nan, np.float32)
for L in LAY:
    W = layers[L].mlp.down_proj.weight.float(); Wn = W / W.norm(dim=0, keepdim=True)
    for c in range(0, NN, CH):
        X = Wn[:, c:c + CH] * G[:, None]                              # (D, ch): the normalised direction through the gain
        logits = U @ X; SHIFTN[L, c:c + CH] = logits.norm(dim=0).cpu().numpy(); logits = logits * S_SCALE
        logits[PAD, :] = float("-inf")
        for si_, sign in enumerate((1, -1)):
            z = logits if sign == 1 else -logits; z = torch.where(PAD[:, None], torch.full_like(z, float("-inf")), z)
            p = torch.softmax(z, dim=0); ent = torch.special.entr(p).sum(0)
            NEFF[L, c:c + CH, si_] = torch.exp(ent).cpu().numpy(); PMAX[L, c:c + CH, si_] = p.max(0).values.cpu().numpy()
            MASS[L, c:c + CH, si_, :] = (MASK.float() @ p).T.cpu().numpy()
            tk = torch.topk(p, TOPK, dim=0); TOP[L, c:c + CH, si_, :] = tk.indices.T.cpu().numpy(); TOPM[L, c:c + CH, si_] = tk.values.sum(0).cpu().numpy()
    log(f"layer {L} done; median N_eff {np.nanmedian(NEFF[L]):.0f}, median shift norm {np.nanmedian(SHIFTN[L]):.2f}")
def sidx(sign): return 0 if sign == 1 else 1
def toks(L, n, sign): return [strs[i] for i in TOP[L, n, sidx(sign)]]
def neff(L, n, sign): return float(NEFF[L, n, sidx(sign)])
def rankdata(a):
    a = np.asarray(a, float); n = len(a); order = np.argsort(a, kind="mergesort"); sa = a[order]; r = np.empty(n); i = 0
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]: j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return r
def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return None if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0 else round(float(np.corrcoef(rankdata(x), rankdata(y))[0, 1]), 3)
def auc_lower(pos, neg):   # AUC that the positives are LOWER
    allv = np.concatenate([-np.asarray(pos, float), -np.asarray(neg, float)]); r = rankdata(allv); npos = len(pos)
    return round(float((r[:npos].sum() - npos * (npos + 1) / 2) / (npos * len(neg))), 3)
# ---- A1: retrieval rank of the zoo's rows by regex mass among all processed (row, sign) candidates
valid = ~np.isnan(NEFF[:, :, 0]); NCAND = int(valid.sum()) * 2
A1 = {"n_candidates": NCAND, "rows": []}
for name in SPEC_NAMES:
    si = SPEC_NAMES.index(name); m = MASS[:, :, :, si]; flat = m[valid].reshape(-1)       # both signs of every valid row
    for (L, n, sign) in SPECS[name]["rows"]:
        if not valid[L, n]: continue
        v = m[L, n, sidx(sign)]; rank = int((flat > v).sum()) + 1; pct = rank / NCAND
        A1["rows"].append({"spec": name, "layer": L, "neuron": n, "sign": sign, "mass": float(v), "rank": rank, "pct": round(pct, 5), "top1pct": bool(pct <= 0.01),
                           "mass_other_sign": float(m[L, n, sidx(-sign)]), "neff": neff(L, n, sign), "top10": toks(L, n, sign)[:10], "mass_median_all": float(np.median(flat)), "mass_q99": float(np.quantile(flat, 0.99))})
A1["n_top1pct"] = sum(r["top1pct"] for r in A1["rows"]); A1["n_rows"] = len(A1["rows"])
log("A1", A1["n_top1pct"], "of", A1["n_rows"], [(r["spec"], r["layer"], r["pct"]) for r in A1["rows"]])
# ---- A2: N_eff of the 43 cards vs layer-matched random (row, sign)
rng = random.Random(SEED); cards = [k for k in E["known_rows"] if k["sign"] is not None]; cards = [k for k in cards if valid[k["layer"], k["neuron"]]]
pos = [neff(k["layer"], k["neuron"], k["sign"]) for k in cards]; ctrl = []
for k in cards:
    for _ in range(10):
        while True:
            n_, s_ = rng.randrange(NN), rng.choice((1, -1))
            if n_ != k["neuron"]: break
        ctrl.append({"layer": k["layer"], "neuron": n_, "sign": s_, "neff": neff(k["layer"], n_, s_)})
A2 = {"n_cards": len(cards), "n_random": len(ctrl), "auc_cards_lower": auc_lower(pos, [c["neff"] for c in ctrl]) if cards else None, "exploratory_auc_top40mass_cards_higher": auc_lower([-float(TOPM[k["layer"], k["neuron"], sidx(k["sign"])]) for k in cards], [-float(TOPM[c["layer"], c["neuron"], sidx(c["sign"])]) for c in ctrl]) if cards else None, "neff_cards_median": float(np.median(pos)) if cards else None,
      "neff_random_median": float(np.median([c["neff"] for c in ctrl])) if ctrl else None, "cards": [{"name": k["name"], "layer": k["layer"], "neuron": k["neuron"], "sign": k["sign"], "neff": neff(k["layer"], k["neuron"], k["sign"]), "pmax": float(PMAX[k["layer"], k["neuron"], sidx(k["sign"])]), "top10": toks(k["layer"], k["neuron"], k["sign"])[:10]} for k in cards]}
A2["cards_neff_pct_within_layer"] = [round(float((NEFF[k["layer"], :, :][valid[k["layer"]] if False else slice(None)].reshape(-1) < neff(k["layer"], k["neuron"], k["sign"])).mean()), 3) for k in cards]
log("A2 auc", A2["auc_cards_lower"], "medians", A2["neff_cards_median"], A2["neff_random_median"])
# ---- A3: the 20 judged random rows at the zoo's 0.8 (consistency from e728's blind judge)
signs = {}
for st in ("stage3", "stage3b"):
    for cond in ("wdd", "random"):
        for e in E[st][cond]: signs[(st, cond, e["row"]["layer"], e["row"]["neuron"])] = (e["row"]["sign"], e["zoo0.8"]["kl_answers"])
judged = []
for j in E["judge"]["rows"]:
    m = re.match(r"(stage3b?|stage3) (wdd|random) #\d+ L(\d+) N(\d+) zoo0\.8$", j["label"])
    if not m: continue
    st, cond, L, n = m.group(1), m.group(2), int(m.group(3)), int(m.group(4)); sign, klv = signs[(st, cond, L, n)]
    if not valid[L, n]: continue
    judged.append({"label": j["label"], "stage": st, "cond": cond, "layer": L, "neuron": n, "sign": sign, "consistency": int(j["consistency"]), "degenerate": str(j["degenerate"]) == "True", "theme": j["theme"], "kl_zoo0.8": klv, "neff": neff(L, n, sign), "top10": toks(L, n, sign)[:10]})
jr = [x for x in judged if x["cond"] == "random"]
A3 = {"n_random_judged": len(jr), "spearman_neff_consistency": spearman([x["neff"] for x in jr], [x["consistency"] for x in jr]) if len(jr) >= 3 else None,
      "spearman_neff_kl": spearman([x["neff"] for x in jr], [x["kl_zoo0.8"] for x in jr]) if len(jr) >= 3 else None, "exploratory_spearman_top40mass_consistency": spearman([float(TOPM[x["layer"], x["neuron"], sidx(x["sign"])]) for x in jr], [x["consistency"] for x in jr]) if len(jr) >= 3 else None, "neff_by_consistency": {c: [round(x["neff"]) for x in jr if x["consistency"] == c] for c in (0, 1, 2)},
      "wdd_rows": {"n": len([x for x in judged if x["cond"] == "wdd"]), "spearman_neff_consistency": spearman([x["neff"] for x in judged if x["cond"] == "wdd"], [x["consistency"] for x in judged if x["cond"] == "wdd"]) if len(judged) - len(jr) >= 3 else None}}
log("A3", A3["spearman_neff_consistency"], A3["neff_by_consistency"])
# ---- exploratory: every row e728 measured at the zoo's 0.8 (KL on the answers) against N_eff and the shift norm
meas = []
for g in ("top_R_dir", "bottom_R_dir", "random"):
    for e in E["stage1"][g]: meas.append((g, e["row"]["layer"], e["row"]["neuron"], e["row"]["sign"], e["kl_answers"], e["nll_cost"]))
for st in ("stage3", "stage3b"):
    for cond in ("wdd", "random"):
        for e in E[st][cond]: meas.append((st + "_" + cond, e["row"]["layer"], e["row"]["neuron"], e["row"]["sign"], e["zoo0.8"]["kl_answers"], e["zoo0.8"]["nll_cost"]))
for k, v in E["stage2"].items():
    if len(v["rows"]) == 1: r = v["rows"][0]; meas.append(("stage2_" + k.split(" | ")[1], r["layer"], r["neuron"], r["sign"], v["doses"]["zoo0.8"]["kl_answers"], v["doses"]["zoo0.8"]["nll_cost"]))
meas = [m for m in meas if valid[m[1], m[2]]]
EXP = {"n": len(meas), "spearman_neff_kl": spearman([neff(m[1], m[2], m[3]) for m in meas], [m[4] for m in meas]), "spearman_shiftnorm_kl": spearman([float(SHIFTN[m[1], m[2]]) for m in meas], [m[4] for m in meas]),
       "spearman_pmax_kl": spearman([float(PMAX[m[1], m[2], sidx(m[3])]) for m in meas], [m[4] for m in meas]), "spearman_top40mass_kl": spearman([float(TOPM[m[1], m[2], sidx(m[3])]) for m in meas], [m[4] for m in meas]), "spearman_layer_kl": spearman([m[1] for m in meas], [m[4] for m in meas])}
l31 = [m for m in meas if m[1] == 31]
EXP["layer31_rows"] = {"n": len(l31), "spearman_neff_kl": spearman([neff(m[1], m[2], m[3]) for m in l31], [m[4] for m in l31]) if len(l31) >= 3 else None, "spearman_shiftnorm_kl": spearman([float(SHIFTN[m[1], m[2]]) for m in l31], [m[4] for m in l31]) if len(l31) >= 3 else None}
log("exploratory", EXP)
# ---- judge packets: token lists for the cards, layer-matched random (row, sign), and the judged rows (key withheld in a separate file)
items = []
for k in cards: items.append({"kind": "card", "name": k["name"], "layer": k["layer"], "neuron": k["neuron"], "sign": k["sign"]})
for k in cards:
    while True:
        n_, s_ = rng.randrange(NN), rng.choice((1, -1))
        if n_ != k["neuron"]: break
    items.append({"kind": "random_for_card", "name": k["name"], "layer": k["layer"], "neuron": n_, "sign": s_})
for x in judged: items.append({"kind": "judged_" + x["cond"], "name": x["label"], "layer": x["layer"], "neuron": x["neuron"], "sign": x["sign"], "consistency": x["consistency"], "gen_theme": x["theme"]})
for it in items: it["neff"] = neff(it["layer"], it["neuron"], it["sign"]); it["top40mass"] = float(TOPM[it["layer"], it["neuron"], sidx(it["sign"])]); it["tokens"] = toks(it["layer"], it["neuron"], it["sign"])
order = list(range(len(items))); rng.shuffle(order)
packet = [{"id": f"t{i:03d}", "tokens": items[o]["tokens"]} for i, o in enumerate(order)]; key = [{"id": f"t{i:03d}", **{k: v for k, v in items[o].items() if k != "tokens"}} for i, o in enumerate(order)]
spec_rows_themes = {}
for j in E["judge"]["rows"]:
    m = re.match(r"stage2 (.+) \| zoo zoo0\.8$", j["label"])
    if m: spec_rows_themes[m.group(1)] = {"consistency": int(j["consistency"]), "theme": j["theme"]}
res = {"tag": TAG, "model": NAME, "smoke": SMOKE, "layers": LAY, "config": {"chunk": CH, "topk": TOPK, "s_scale": S_SCALE, "d": D, "rows_per_layer": NN, "vocab_rows": V, "tokenizer_size": VT, "gain_mean": float(G.mean()), "seed": SEED},
       "specs": {k: {"regex": v["regex"], "rows": v["rows"], "n_regex_tokens": int(MASK[SPEC_NAMES.index(k)].sum())} for k, v in SPECS.items()}, "A1": A1, "A2": A2, "A3": A3, "exploratory": EXP, "judged_rows": judged,
       "spec_generation_themes": spec_rows_themes, "neff_median_by_layer": [float(np.nanmedian(NEFF[L])) if valid[L].any() else None for L in range(NL)], "shiftnorm_median_by_layer": [float(np.nanmedian(SHIFTN[L])) if valid[L].any() else None for L in range(NL)],
       "pmax_median_by_layer": [float(np.nanmedian(PMAX[L])) if valid[L].any() else None for L in range(NL)], "n_packet_items": len(packet), "seconds": round(time.time() - t0, 1)}
json.dump(res, open(os.path.join(OUT, TAG + ".json"), "w"), indent=1); json.dump(packet, open(os.path.join(OUT, TAG + "_judge_packet.json"), "w"), indent=1); json.dump(key, open(os.path.join(OUT, TAG + "_judge_key.json"), "w"), indent=1)
np.savez_compressed(os.path.join(OUT, TAG + "_rows.npz"), NEFF=NEFF, PMAX=PMAX, MASS=MASS, SHIFTN=SHIFTN, TOP=TOP, TOPM=TOPM, SPEC_NAMES=np.array(SPEC_NAMES))
log("saved", TAG, "items", len(packet))
