"""e625 (session 121): the gender dial in French, Spanish and German text. The gender row (by vote on e473's kinship items,
as e622) negated, zeroed and doubled; on twelve gendered subjects per language in templates ending before an agreeing
form, the probability of the agreeing forms over all forms at the next token: French and Spanish predicative adjectives
(beau/belle, grand/grande, petit/petite, content/contente; cansado/cansada, contento/contenta, alto/alta,
pequeño/pequeña) and German pronouns (er/sie after "sagte, dass"); eight random rows of the same block negated; the
Pile loss. Pre-registered in e625_prereg.json: X1 (0.6) on Qwen2.5-0.5B the gender row negated lowers agreement by 0.10
or more in at least two of the three languages, random rows by under 0.03; X2 (0.4) the 1.5B model's gender row does the
same in at least one language. Arguments: model [--smoke]."""
import sys, os, time, copy, math, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s101_common import *
from ma_common import out_of
from datasets import load_dataset
import wdd_common
from wdd_common import omp, build_dictionary
wdd_common.MODELS.setdefault("qwen15i", ("Qwen/Qwen2.5-1.5B-Instruct", "llama"))
mean = lambda v: sum(v) / max(len(v), 1)
name = sys.argv[1]; SMOKE = "--smoke" in sys.argv; t0 = time.time(); torch.set_grad_enabled(False)
model, tok, fam = load_model(name); arch = Arch(model, fam); L = arch.NB // 2; D = arch.D; DFF = arch.DFF; assert fam == "llama"
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "e473_word_algebra.py")).read(); exec(src[src.index("QUADS = {"): src.index("LN = {")], globals())
LN = {"fr": "French", "es": "Spanish", "de": "German"}; LANGS = ["fr", "es", "de"]; NT = 1 if SMOKE else 3
def prompt(lg, w, t):
    (w1, e1), (w2, e2) = SHOTS[lg][2 * t], SHOTS[lg][2 * t + 1]; p = f"{LN[lg]}: {w1}\nEnglish: {e1}\n{LN[lg]}: {w2}\nEnglish: {e2}\n{LN[lg]}: {w}\nEnglish:"; a = p.rindex(w); b_ = a + len(w)
    enc = tok(p, add_special_tokens=False, return_offsets_mapping=True); ids = torch.tensor(enc["input_ids"], device=DEV)[None]
    return ids, [i for i, (x0, x1) in enumerate(enc["offset_mapping"]) if x1 > a and x0 < b_][-1]
def run(ids, p, cand):
    cap, hs = {}, []
    for b in range(L + 1):
        def hk(m, i, o, b=b): cap[b] = out_of(o)[0, p].detach().float()
        hs.append(arch.layers[b].register_forward_hook(hk))
    try: lg = model(ids).logits[0, -1].float()
    finally: [h.remove() for h in hs]
    return torch.log_softmax(lg[cand], -1), torch.stack([cap[b] for b in range(L + 1)])
items = []
for lg in LANGS:
    for q in range(len(QUADS["en"])):
        if SMOKE and q >= 2: continue
        cand = [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in QUADS["en"][q]]
        if len(set(cand)) < 4: continue
        cand = torch.tensor(cand, device=DEV)
        for t in range(NT):
            runs = [prompt(lg, QUADS[lg][q][i], t) for i in range(4)]; outs = [run(ids, p, cand) for ids, p in runs]
            if not all(int(lp.argmax()) == i for i, (lp, _) in enumerate(outs)): continue
            for i in range(4): items.append(dict(i=i, X={j: outs[i ^ j][1] for j in range(4)}))
A_, lab = build_dictionary(arch, blocks=list(range(L + 1))); blk = lab["block"].to(DEV); Au = unitr(A_); del A_; ends = {b: int((blk <= b).sum()) for b in range(L + 1)}
sg = torch.tensor([-1.0 if (it["i"] & 1) else 1.0 for it in items], device=DEV); best = (None, -1, None)
for b in range(L + 1):
    Xb = torch.stack([it["X"][1][b] - it["X"][0][b] for it in items]) * sg[:, None]; sel, _, _ = omp(Xb, Au[:ends[b]], 8, batch=256, record_err=False); cnt = torch.bincount(sel.reshape(-1), minlength=ends[b]); top = int(cnt.argmax()); share = float(cnt[top]) / Xb.shape[0]
    if int(lab["type"][top]) == 2 and share > best[1]: best = (b, share, int(lab["index"][top]))
bg, share_g, jg = best; log(f"{name}: gender row {jg} of block {bg} (vote {share_g:.2f} over {len(items)} items) ({time.time() - t0:.0f}s)")
SUBJ = {"fr": [("roi", "reine"), ("prince", "princesse"), ("père", "mère"), ("fils", "fille"), ("homme", "femme"), ("garçon", "fille"), ("oncle", "tante"), ("neveu", "nièce"), ("grand-père", "grand-mère"), ("frère", "sœur"), ("mari", "femme"), ("acteur", "actrice")],
        "es": [("rey", "reina"), ("príncipe", "princesa"), ("padre", "madre"), ("hijo", "hija"), ("hombre", "mujer"), ("niño", "niña"), ("tío", "tía"), ("sobrino", "sobrina"), ("abuelo", "abuela"), ("hermano", "hermana"), ("marido", "esposa"), ("actor", "actriz")],
        "de": [("König", "Königin"), ("Prinz", "Prinzessin"), ("Vater", "Mutter"), ("Sohn", "Tochter"), ("Mann", "Frau"), ("Junge", "Schwester"), ("Onkel", "Tante"), ("Neffe", "Nichte"), ("Großvater", "Großmutter"), ("Bruder", "Schwester"), ("Ehemann", "Ehefrau"), ("Schauspieler", "Schauspielerin")]}
ART = {"fr": ("Le", "La"), "es": ("El", "La"), "de": ("Der", "Die")}; TEMPL = {"fr": "{art} {n} est très", "es": "{art} {n} está muy", "de": "{art} {n} sagte, dass"}
FORMS = {"fr": [("beau", "belle"), ("grand", "grande"), ("petit", "petite"), ("content", "contente")], "es": [("cansado", "cansada"), ("contento", "contenta"), ("alto", "alta"), ("pequeño", "pequeña")], "de": [("er", "sie")]}
def first_tok(w): return tok(" " + w, add_special_tokens=False)["input_ids"][0]
PROMPTS = {lg: [] for lg in LANGS}
for lg in LANGS:
    for (m, f) in SUBJ[lg][:(4 if SMOKE else 12)]:
        for g, n in ((0, m), (1, f)):
            art = ART[lg][g]; art = "L'" if (lg == "fr" and n[0].lower() in "aeiouh") else art; sep = "" if art == "L'" else " "; PROMPTS[lg].append((f"{art}{sep}{n}" + TEMPL[lg].split("{n}")[1], g))
FT_IDS = {}
for lg in LANGS:
    ok = [(first_tok(m), first_tok(f)) for m, f in FORMS[lg] if first_tok(m) != first_tok(f)]; assert ok, lg; FT_IDS[lg] = ([a for a, _ in ok], [c for _, c in ok]); log(f"{lg}: {len(ok)} of {len(FORMS[lg])} form pairs with distinct first tokens")
def agreement(lg):
    tok.padding_side = "left"; texts = [p for p, _ in PROMPTS[lg]]; gs = [g for _, g in PROMPTS[lg]]; out = []
    for s0 in range(0, len(texts), 24):
        enc = tok(texts[s0:s0 + 24], return_tensors="pt", padding=True).to(DEV); lg_ = model(**enc).logits[:, -1].float(); pm = torch.softmax(lg_, -1)
        for k, g in enumerate(gs[s0:s0 + 24]): p_m = float(pm[k, FT_IDS[lg][0]].sum()); p_f = float(pm[k, FT_IDS[lg][1]].sum()); out.append((p_f if g == 1 else p_m) / max(p_m + p_f, 1e-9))
    tok.padding_side = "right"; return mean(out)
pile = load_dataset("NeelNanda/pile-10k", split="train"); buf, wins = [], []
for ex in pile:
    buf += tok(ex["text"])["input_ids"] + [tok.eos_token_id]
    while len(buf) >= 257 and len(wins) < (4 if SMOKE else 16): wins.append(buf[:257]); buf = buf[257:]
    if len(wins) >= (4 if SMOKE else 16): break
PW = torch.tensor(wins, device=DEV)
def pile_loss():
    tot = 0.0
    for s0 in range(0, PW.shape[0], 8): x = PW[s0:s0 + 8]; lgt = model(x).logits.float(); tot += float(torch.nn.functional.cross_entropy(lgt[:, :-1].reshape(-1, lgt.shape[-1]), x[:, 1:].reshape(-1))) * x.shape[0]; del lgt
    return tot / PW.shape[0]
Wm = arch.layers[bg].mlp.down_proj.weight; sv = Wm[:, jg].clone()
def edit(j, kind, saved):
    if kind == "negate": Wm[:, j] = -saved
    elif kind == "zero": Wm[:, j] = 0
    elif kind == "double": Wm[:, j] = 2 * saved
res = dict(model=name, gender_row=dict(block=bg, row=jg, share=share_g), n_prompts={lg: len(PROMPTS[lg]) for lg in LANGS}, conditions={}, random=[])
L0 = pile_loss(); res["conditions"]["none"] = dict(agreement={lg: agreement(lg) for lg in LANGS}, pile=0.0); log("unedited agreement: " + ", ".join(f"{lg} {res['conditions']['none']['agreement'][lg]:.2f}" for lg in LANGS))
for kind in ("negate", "zero", "double"):
    edit(jg, kind, sv); res["conditions"][kind] = dict(agreement={lg: agreement(lg) for lg in LANGS}, pile=pile_loss() - L0); Wm[:, jg] = sv
    log(f"gender row {kind}: " + ", ".join(f"{lg} {res['conditions'][kind]['agreement'][lg]:.2f}" for lg in LANGS) + f", Pile {res['conditions'][kind]['pile']:+.4f}")
g_r = torch.Generator().manual_seed(11)
for r in torch.randperm(DFF, generator=g_r)[:(2 if SMOKE else 8)].tolist():
    if r == jg: continue
    svr = Wm[:, r].clone(); edit(r, "negate", svr); res["random"].append(dict(row=r, agreement={lg: agreement(lg) for lg in LANGS})); Wm[:, r] = svr
log("random rows negated, mean agreement: " + ", ".join(f"{lg} {mean([x['agreement'][lg] for x in res['random']]):.2f}" for lg in LANGS))
drop = {lg: res["conditions"]["none"]["agreement"][lg] - res["conditions"]["negate"]["agreement"][lg] for lg in LANGS}; rdrop = {lg: max(abs(res["conditions"]["none"]["agreement"][lg] - x["agreement"][lg]) for x in res["random"]) for lg in LANGS}
res["drop"] = drop; res["random_max_drop"] = rdrop; x1 = sum(1 for lg in LANGS if drop[lg] >= 0.10 and rdrop[lg] < 0.03) >= 2; x2 = any(drop[lg] >= 0.10 and rdrop[lg] < 0.03 for lg in LANGS); res["verdicts"] = dict(X1=x1, X2=x2)
summ = (f"cross-lingual dial ({name}, gender row {jg} of block {bg}, vote {share_g:.2f}): agreement unedited " + ", ".join(f"{lg} {res['conditions']['none']['agreement'][lg]:.2f}" for lg in LANGS) + "; negated " + ", ".join(f"{lg} {res['conditions']['negate']['agreement'][lg]:.2f} (drop {drop[lg]:+.2f})" for lg in LANGS) + "; zeroed " + ", ".join(f"{lg} {res['conditions']['zero']['agreement'][lg]:.2f}" for lg in LANGS) + "; doubled " + ", ".join(f"{lg} {res['conditions']['double']['agreement'][lg]:.2f}" for lg in LANGS) + "; random rows' largest change " + ", ".join(f"{lg} {rdrop[lg]:.2f}" for lg in LANGS) + f"; Pile loss negated {res['conditions']['negate']['pile']:+.4f}; verdicts X1 {x1}, X2 {x2}")
log(summ); record(f"e625_crosslingual_{name}" + ("_smoke" if SMOKE else ""), res, summ)
