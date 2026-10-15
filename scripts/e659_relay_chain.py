"""e659 (session 144; Gemini 4's 'cheat table'): edit the whole relay chain, not one row. e622 named one MLP row per
attribute by vote and negated its write: held-out flips gender 0.28, generation 0.13, number 0.12, polarity 0 on
Qwen2.5-0.5B, and none for gender on Qwen2.5-1.5B-Instruct; e624 found two such rows interfere (both attributes flipped
in 0.05 of kinship items), while e470/e473/e477 found native words act only when intervened on across the blocks that
write the variable (interchange 0.96 all-block against 0.35 single-block). e622's script kept only the peak-vote row
(bm = max over levels), discarding the relay of other rows that win the vote at other levels. Here, with e622's own
vote_by_block (no new selection): each attribute's chain is the distinct MLP rows that win a level with share 0.25 or
more; edits (write columns negated): the single e622 row; the chain; as many rows from the peak block (top by
activation difference: e470's law, rows across blocks against rows in one block); the naive chain (the top
activation-difference row in each chain block); and five random chains (one random row per chain block). Scored as in
e622 (fr/es/de to English few-shot translation, items both poles of which the model translates; flips and log-odds
shifts on discovery and held-out templates; the other attributes' flips) plus wikitext-2 loss, and as in e624 on the
kinship quadruples (kept / gender / generation / both flipped) for the gender and generation edits together. Models:
Qwen2.5-0.5B, Qwen2.5-1.5B-Instruct, and Qwen2.5-0.5B-Instruct edited zero-shot with the base model's rows (Gemini's
transfer test; e619: base and instruct rows at cosine 0.997). Standalone (e473/e622 sources and e622's JSONs alongside).
Arguments: model [--smoke]."""
import sys, os, time, json, math, collections
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_grad_enabled(False); DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
MODELS = {"qwen05": "Qwen/Qwen2.5-0.5B", "qwen15i": "Qwen/Qwen2.5-1.5B-Instruct", "qwen05i": "Qwen/Qwen2.5-0.5B-Instruct"}
CAT_OF = {"qwen05": "qwen05", "qwen15i": "qwen15i", "qwen05i": "qwen05"}   # the 0.5B instruct model gets the base model's rows, zero-shot
name = sys.argv[1]; SMOKE = "--smoke" in sys.argv; t0 = time.time(); rng = np.random.default_rng(659)
tok = AutoTokenizer.from_pretrained(MODELS[name]); model = AutoModelForCausalLM.from_pretrained(MODELS[name], dtype=torch.float32).to(DEV).eval()
for p_ in model.parameters(): p_.requires_grad_(False)
layers = model.model.layers; NB = len(layers); L = NB // 2; D = model.config.hidden_size; DFF = layers[0].mlp.down_proj.weight.shape[1]; BS = 16 if D > 1000 else 32; WB = 2 if D > 1000 else 4
src = open("e473_word_algebra.py").read(); exec(src[src.index("QUADS = {"): src.index("LN = {")], globals())
src = open("e622_attribute_catalogue.py").read(); exec(src[src.index("NEWSHOTS = {"): src.index("def pairs_of(")], globals())
LN = {"fr": "French", "es": "Spanish", "de": "German"}; LANGS = ["fr", "es", "de"]; ATTRS = ("gender", "generation", "number", "polarity")
NT = 3 if SMOKE else 5; DISC = (0, 1) if SMOKE else (0, 1, 2)   # the smoke keeps template 2 held out
def log(s): print(f"[{name}] {s} ({time.time() - t0:.0f}s)", flush=True)
def pairs_of(attr, lg):
    if attr == "gender": return [(q[0], q[1]) for q in QUADS[lg]] + [(q[2], q[3]) for q in QUADS[lg]]
    if attr == "generation": return [(q[0], q[2]) for q in QUADS[lg]] + [(q[1], q[3]) for q in QUADS[lg]]
    return (NUMBER if attr == "number" else POLAR)[lg]
def shots(lg, t): return (SHOTS[lg][2 * t], SHOTS[lg][2 * t + 1]) if t < 3 else (NEWSHOTS[lg][2 * (t - 3)], NEWSHOTS[lg][2 * (t - 3) + 1])
def prompt(lg, w, t):
    (w1, e1), (w2, e2) = shots(lg, t); p = f"{LN[lg]}: {w1}\nEnglish: {e1}\n{LN[lg]}: {w2}\nEnglish: {e2}\n{LN[lg]}: {w}\nEnglish:"; a = p.rindex(w); b_ = a + len(w)
    enc = tok(p, add_special_tokens=False, return_offsets_mapping=True)
    return enc["input_ids"], [i for i, (x0, x1) in enumerate(enc["offset_mapping"]) if x1 > a and x0 < b_][-1]
def first_tok(w): return tok(" " + w, add_special_tokens=False)["input_ids"][0]
# ---- batched forward over items grouped by prompt length: next-token log-probs over each item's candidates, optionally the MLP activations at the word position
def forward_items(its, acts=False, extra_hooks=()):
    groups = collections.defaultdict(list)
    for n_, it in enumerate(its): groups[len(it["ids"])].append(n_)
    lps = [None] * len(its); A = [None] * len(its)
    for Lq, idx in groups.items():
        for s0 in range(0, len(idx), BS):
            ch = idx[s0:s0 + BS]; ids = torch.tensor([its[n]["ids"] for n in ch], device=DEV); cap = {}; hs = []
            if acts:
                for b in range(L + 1):
                    def hk(m, a, b=b): cap[b] = a[0].float()
                    hs.append(layers[b].mlp.down_proj.register_forward_pre_hook(hk))
            try: lg = model(ids, use_cache=False, logits_to_keep=1).logits[:, -1].float()
            finally: [h.remove() for h in hs]
            for k, n in enumerate(ch):
                lps[n] = torch.log_softmax(lg[k, its[n]["cand"]], -1)
                if acts: A[n] = torch.stack([cap[b][k, its[n]["p"]] for b in range(L + 1)]).cpu()
    return lps, A
ITEMS = {a: [] for a in ATTRS}
for a in ATTRS:
    for lg in LANGS:
        prs = pairs_of(a, lg); ens = pairs_of(a, "en")
        for k, (pr, en) in enumerate(zip(prs, ens)):
            if SMOKE and k >= 3: continue
            cand = [first_tok(w) for w in en]
            if cand[0] == cand[1]: continue
            for t in range(NT):
                for i in range(2):
                    ids, p = prompt(lg, pr[i], t); ITEMS[a].append(dict(attr=a, lg=lg, pair=k, t=t, side=i, cand=torch.tensor(cand, device=DEV), ids=ids, p=p, disc=(t in DISC)))
    lps, A = forward_items(ITEMS[a], acts=True)
    for it, lp, ac in zip(ITEMS[a], lps, A): it["lp0"] = lp; it["A"] = ac; it["ok"] = int(lp.argmax()) == it["side"]
    keep = collections.defaultdict(list)
    for it in ITEMS[a]: keep[(it["lg"], it["pair"], it["t"])].append(it)
    ITEMS[a] = [it for grp in keep.values() if len(grp) == 2 and all(x["ok"] for x in grp) for it in grp]
log("items " + ", ".join(f"{a} {len(ITEMS[a])} ({sum(it['disc'] for it in ITEMS[a])} discovery)" for a in ATTRS))
# ---- kinship composition items (e624: four words per quadruple, outcome = argmax xor index)
KIN = []
for lg in LANGS:
    for q in range(len(QUADS["en"])):
        if SMOKE and q >= 2: continue
        cand = [first_tok(w) for w in QUADS["en"][q]]
        if len(set(cand)) < 4: continue
        grp = []
        for t in range(1 if SMOKE else 3):
            g_ = []
            for i in range(4): ids, p = prompt(lg, QUADS[lg][q][i], t); g_.append(dict(i=i, cand=torch.tensor(cand, device=DEV), ids=ids, p=p))
            grp.append(g_)
        for g_ in grp:
            lps, _ = forward_items(g_)
            if all(int(lp.argmax()) == it["i"] for lp, it in zip(lps, g_)): KIN += g_
log(f"{len(KIN)} kinship items")
def outcomes():
    lps, _ = forward_items(KIN); out = np.zeros(4)
    for lp, it in zip(lps, KIN): out[int(lp.argmax()) ^ it["i"]] += 1
    return (out / max(len(KIN), 1)).tolist()
# ---- e622's catalogue: the single row and the relay chain from vote_by_block
C = json.load(open(f"e622_catalogue_{CAT_OF[name]}.json"))["catalogue"]; NEMB = model.get_input_embeddings().weight.shape[0]; PER = DFF + D
def to_row(atom):
    i = atom - NEMB; b, r = divmod(i, PER); return (b, r) if i >= 0 and r < DFF else None
for a in ATTRS: assert to_row(C[a]["atom"]) == (C[a]["block"], C[a]["row"]), (a, to_row(C[a]["atom"]), C[a]["block"], C[a]["row"])
def chain_of(a, thr=0.25):
    rows = []
    for lev in sorted(C[a]["vote_by_block"], key=int):
        v = C[a]["vote_by_block"][lev]
        if v["type"] != 2 or v["share"] < thr: continue
        br = to_row(v["atom"])
        if br is not None and br not in rows: rows.append(br)
    if (C[a]["block"], C[a]["row"]) not in rows: rows.append((C[a]["block"], C[a]["row"]))
    return rows
def dmean(a):
    its = [it for it in ITEMS[a] if it["disc"]]; A_ = torch.stack([it["A"] for it in its]); s = torch.tensor([it["side"] for it in its])
    return (A_[s == 1].mean(0) - A_[s == 0].mean(0)).to(DEV)   # [L+1, DFF]
EDITS = {}
for a in ATTRS:
    ch = chain_of(a); bm, j = C[a]["block"], C[a]["row"]; dm = dmean(a); M = len(ch)
    same = [(bm, int(r)) for r in dm[bm].abs().argsort(descending=True)[:M].tolist()]
    naive, used = [], set()
    for b, _ in ch:
        for r in dm[b].abs().argsort(descending=True).tolist():
            if (b, r) not in used: naive.append((b, r)); used.add((b, r)); break
    rnd = [[(b, int(rng.integers(DFF))) for b, _ in ch] for _ in range(2 if SMOKE else 5)]
    EDITS[a] = dict(single=[(bm, j)], chain=ch, same_block=same, naive_chain=naive, random=rnd)
    log(f"{a}: single row {j}@b{bm}; chain {[f'{r}@b{b}' for b, r in ch]}; same-block {[r for _, r in same]}; naive chain {[f'{r}@b{b}' for b, r in naive]}; WDD chain rows that are the naive top row of their block: {sum(x in naive for x in ch)} of {M}")
# ---- edits and scoring
import datasets
WT = [t for t in datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if len(t.split()) > 60]
WIN = []
for t in WT:
    ids = tok(t)["input_ids"]
    if len(ids) >= 256: WIN.append(ids[:256])
    if len(WIN) >= (4 if SMOKE else 12): break
WIN = torch.tensor(WIN, device=DEV)
def wt_loss():
    tot = 0.0
    for s0 in range(0, WIN.shape[0], WB):
        x = WIN[s0:s0 + WB]; lg = model(x, use_cache=False).logits.float(); tot += float(torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), x[:, 1:].reshape(-1), reduction="sum")); del lg
    return tot / (WIN.shape[0] * (WIN.shape[1] - 1))
BASE_LOSS = wt_loss()
def W(b): return layers[b].mlp.down_proj.weight
class Negate:
    def __init__(self, rows): self.rows = list(dict.fromkeys(rows))
    def __enter__(self):
        self.saved = [(b, r, W(b)[:, r].clone()) for b, r in self.rows]
        for b, r, w in self.saved: W(b)[:, r] = -w
    def __exit__(self, *a_):
        for b, r, w in self.saved: W(b)[:, r] = w
def score_all():
    out = {}
    for a in ATTRS:
        lps, _ = forward_items(ITEMS[a]); acc = {"disc": [0, 0.0, 0], "held": [0, 0.0, 0]}
        for it, lp in zip(ITEMS[a], lps):
            own, oth = it["side"], 1 - it["side"]; sh = float((lp[oth] - lp[own]) - (it["lp0"][oth] - it["lp0"][own])); k = "disc" if it["disc"] else "held"
            acc[k][0] += int(int(lp.argmax()) == oth); acc[k][1] += sh; acc[k][2] += 1
        out[a] = {k: dict(flip=v[0] / max(v[2], 1), shift=v[1] / max(v[2], 1), n=v[2]) for k, v in acc.items()}
    out["dloss"] = wt_loss() - BASE_LOSS; return out
RES = dict(model=name, catalogue_from=CAT_OF[name], n_items={a: len(ITEMS[a]) for a in ATTRS}, n_kin=len(KIN), base_loss=BASE_LOSS, edits={}, rows={a: {k: v for k, v in EDITS[a].items()} for a in ATTRS})
for a in ATTRS:
    RES["edits"][a] = {}
    for kind in ("single", "chain", "same_block", "naive_chain"):
        with Negate(EDITS[a][kind]): RES["edits"][a][kind] = score_all()
    rs = []
    for rows in EDITS[a]["random"]:
        with Negate(rows): rs.append(score_all())
    RES["edits"][a]["random"] = dict(held_flip_mean=float(np.mean([x[a]["held"]["flip"] for x in rs])), held_flip_max=float(np.max([x[a]["held"]["flip"] for x in rs])), held_shift_mean=float(np.mean([x[a]["held"]["shift"] for x in rs])), dloss_mean=float(np.mean([x["dloss"] for x in rs])))
    e = RES["edits"][a]; log(f"{a}: discovery flips single/chain/same/naive {e['single'][a]['disc']['flip']:.3f}/{e['chain'][a]['disc']['flip']:.3f}/{e['same_block'][a]['disc']['flip']:.3f}/{e['naive_chain'][a]['disc']['flip']:.3f}; held-out flips single {e['single'][a]['held']['flip']:.3f}, chain {e['chain'][a]['held']['flip']:.3f}, same block {e['same_block'][a]['held']['flip']:.3f}, naive chain {e['naive_chain'][a]['held']['flip']:.3f}, random mean {e['random']['held_flip_mean']:.3f} (max {e['random']['held_flip_max']:.3f}); "
        f"shifts {e['single'][a]['held']['shift']:+.2f}/{e['chain'][a]['held']['shift']:+.2f}/{e['same_block'][a]['held']['shift']:+.2f}/{e['naive_chain'][a]['held']['shift']:+.2f}; loss change single {e['single']['dloss']:+.4f} chain {e['chain']['dloss']:+.4f}; "
        f"collateral (other attributes' held-out flips under the chain) {max(e['chain'][a2]['held']['flip'] for a2 in ATTRS if a2 != a):.3f}")
comp = {"none": outcomes()}
G, A2 = EDITS["gender"], EDITS["generation"]
for k_, rows in (("gender_single", G["single"]), ("generation_single", A2["single"]), ("both_single", G["single"] + A2["single"]), ("gender_chain", G["chain"]), ("generation_chain", A2["chain"]), ("both_chain", G["chain"] + A2["chain"]), ("both_same_block", G["same_block"] + A2["same_block"]), ("both_naive_chain", G["naive_chain"] + A2["naive_chain"])):
    with Negate(rows): comp[k_] = outcomes()
rc = []
for rg_, ra_ in zip(G["random"], A2["random"]):
    with Negate(rg_ + ra_): rc.append(outcomes())
comp["both_random"] = np.mean(rc, 0).tolist(); RES["composition"] = comp
log("composition (kept / gender / generation / both): " + "; ".join(f"{k} {[round(x, 3) for x in v]}" for k, v in comp.items()))
e = RES["edits"]; hf = lambda a, k: e[a][k][a]["held"]["flip"]
RES["verdicts"] = dict(H1=hf("gender", "chain") > 0.60 and e["gender"]["chain"]["dloss"] < 0.01, H2=sum(hf(a, "chain") > hf(a, "single") for a in ATTRS) >= 3, H3=sum(hf(a, "chain") > hf(a, "same_block") for a in ATTRS) >= 3,
                       H4=sum(hf(a, "chain") > hf(a, "naive_chain") for a in ATTRS) >= 3, H5=all(e[a]["random"]["held_flip_max"] <= 0.05 for a in ATTRS), H6=comp["both_chain"][3] >= 0.25)
summ = f"[{name}] relay-chain edits (rows from {CAT_OF[name]}): held-out flips single/chain/same-block/naive-chain/random: " + "; ".join(f"{a} {hf(a, 'single'):.2f}/{hf(a, 'chain'):.2f}/{hf(a, 'same_block'):.2f}/{hf(a, 'naive_chain'):.2f}/{e[a]['random']['held_flip_mean']:.2f}" for a in ATTRS) + f"; both flipped single {comp['both_single'][3]:.3f}, chains {comp['both_chain'][3]:.3f}; verdicts {RES['verdicts']} ({time.time() - t0:.0f}s)"
os.makedirs("results", exist_ok=True); RES["summary"] = summ; RES["_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
json.dump(RES, open(f"results/e659_relay_chain_{name}{'_smoke' if SMOKE else ''}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print(summ, flush=True); print("done", flush=True)
