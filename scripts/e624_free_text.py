"""e624 (session 120): the gender dial in free text, and composition in the weights. The block-4 gender row of Qwen2.5-0.5B
(e620, e622) is negated, zeroed and doubled; on 24 English prompts with a gendered subject ("The queen walked into the
room and") the model generates 24 tokens greedily and the first third-person pronoun is scored against the subject's
gender; the share of prompts whose first pronoun agrees, the share with no pronoun, and the share of degenerate
generations (a token repeated five times) are compared with the unedited model and with eight random rows of the block
negated; the Pile loss is the damage measure. Composition: the gender row and the generation row (e622's row 599) negated
together, scored on e473's kinship quadruple items (kept, gender flipped, generation flipped, both) against each alone.
Pre-registered in e623_prereg.json: G1 (0.5) negating the gender row lowers pronoun agreement by 0.15 or more with no
rise in degenerate generations; G2 (0.4) negating both rows gives the doubly flipped word in at least half the items either
row flips alone. Arguments: model [--smoke]."""
import sys, os, time, copy, math, json, re; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s101_common import *
from ma_common import out_of
from datasets import load_dataset
import wdd_common
from wdd_common import omp, build_dictionary
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
            for i in range(4): items.append(dict(i=i, cand=cand, ids=runs[i][0], p=runs[i][1], X={j: outs[i ^ j][1] for j in range(4)}))
n = len(items); log(f"{name}: {n} kinship items ({time.time() - t0:.0f}s)")
# ---- the two rows by vote (as e622; gender j=1, generation j=2 differences, sign-aligned)
A_, lab = build_dictionary(arch, blocks=list(range(L + 1))); blk = lab["block"].to(DEV); Au = unitr(A_); del A_; ends = {b: int((blk <= b).sum()) for b in range(L + 1)}
def vote(jd):
    sg = torch.tensor([-1.0 if (it["i"] & jd) else 1.0 for it in items], device=DEV); best = (None, -1, None)
    for b in range(L + 1):
        Xb = torch.stack([it["X"][jd][b] - it["X"][0][b] for it in items]) * sg[:, None]; sel, _, _ = omp(Xb, Au[:ends[b]], 8, batch=256, record_err=False); cnt = torch.bincount(sel.reshape(-1), minlength=ends[b]); top = int(cnt.argmax()); share = float(cnt[top]) / Xb.shape[0]
        if int(lab["type"][top]) == 2 and share > best[1]: best = (b, share, int(lab["index"][top]))
    return best
bg, sg_, jg = vote(1); ba, sa_, ja = vote(2); log(f"gender row {jg} of block {bg} (vote {sg_:.2f}); generation row {ja} of block {ba} (vote {sa_:.2f})")
def W(b): return arch.layers[b].mlp.down_proj.weight
def edit(b, j, kind, saved):
    if kind == "negate": W(b)[:, j] = -saved
    elif kind == "zero": W(b)[:, j] = 0
    elif kind == "double": W(b)[:, j] = 2 * saved
# ---- composition on the items
def outcomes():
    out = torch.zeros(4)
    for it in items: lp, _ = run(it["ids"], it["p"], it["cand"]); out[int(lp.argmax()) ^ it["i"]] += 1
    return (out / n).tolist()   # kept / gender flipped / generation flipped / both
comp = {"none": outcomes()}; svg, sva = W(bg)[:, jg].clone(), W(ba)[:, ja].clone()
edit(bg, jg, "negate", svg); comp["gender_negated"] = outcomes(); W(bg)[:, jg] = svg
edit(ba, ja, "negate", sva); comp["generation_negated"] = outcomes(); W(ba)[:, ja] = sva
edit(bg, jg, "negate", svg); edit(ba, ja, "negate", sva); comp["both_negated"] = outcomes(); W(bg)[:, jg] = svg; W(ba)[:, ja] = sva
log("composition (kept / gender / generation / both): " + "; ".join(f"{k} {[round(x, 2) for x in v]}" for k, v in comp.items()))
# ---- free text
SUBJ = [("king", "m"), ("queen", "f"), ("father", "m"), ("mother", "f"), ("actor", "m"), ("actress", "f"), ("boy", "m"), ("girl", "f"), ("man", "m"), ("woman", "f"), ("prince", "m"), ("princess", "f"), ("grandfather", "m"), ("grandmother", "f"), ("nephew", "m"), ("niece", "f"), ("husband", "m"), ("wife", "f"), ("brother", "m"), ("sister", "f"), ("son", "m"), ("daughter", "f"), ("uncle", "m"), ("aunt", "f")]
TEMPL = ["The {} said that", "When the {} came home,", "The {} looked at the letter, and then"]
PROMPTS = [t.format(s) for t in TEMPL[:(1 if SMOKE else 3)] for s, _ in SUBJ[:(6 if SMOKE else 24)]]; GEN = [g for t in TEMPL[:(1 if SMOKE else 3)] for _, g in SUBJ[:(6 if SMOKE else 24)]]
MASC = {"he", "him", "his", "himself"}; FEM = {"she", "her", "hers", "herself"}
def generate(prompts):
    tok.padding_side = "left"; outs = []
    for s0 in range(0, len(prompts), 24):
        enc = tok(prompts[s0:s0 + 24], return_tensors="pt", padding=True).to(DEV); out = model.generate(**enc, max_new_tokens=24, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        outs += [tok.decode(r[enc["input_ids"].shape[1]:], skip_special_tokens=True) for r in out]
    tok.padding_side = "right"; return outs
HE, SHE = tok(" he", add_special_tokens=False)["input_ids"][0], tok(" she", add_special_tokens=False)["input_ids"][0]
def pronoun_prob():
    """at the next token after each prompt, the probability of the subject's pronoun over he and she"""
    tok.padding_side = "left"; agree = []
    for s0 in range(0, len(PROMPTS), 24):
        enc = tok(PROMPTS[s0:s0 + 24], return_tensors="pt", padding=True).to(DEV); lg = model(**enc).logits[:, -1].float(); p = torch.softmax(lg[:, [HE, SHE]], -1)
        for k, gdr in enumerate(GEN[s0:s0 + 24]): agree.append(float(p[k, 0] if gdr == "m" else p[k, 1]))
    tok.padding_side = "right"; return mean(agree)
def score(outs):
    agree, none, degen = 0, 0, 0; first = []
    for o, gdr in zip(outs, GEN):
        toks = re.findall(r"[a-zA-Z']+", o.lower()); p = next((w for w in toks if w in MASC or w in FEM), None); first.append(p)
        if p is None: none += 1
        elif (p in MASC) == (gdr == "m"): agree += 1
        degen += int(any(toks.count(w) >= 5 for w in set(toks)) if toks else 1)
    return dict(agree=agree / len(outs), no_pronoun=none / len(outs), degenerate=degen / len(outs), first_pronouns=first)
pile = load_dataset("NeelNanda/pile-10k", split="train"); buf, wins = [], []
for ex in pile:
    buf += tok(ex["text"])["input_ids"] + [tok.eos_token_id]
    while len(buf) >= 257 and len(wins) < (4 if SMOKE else 16): wins.append(buf[:257]); buf = buf[257:]
    if len(wins) >= (4 if SMOKE else 16): break
PW = torch.tensor(wins, device=DEV)
def pile_loss():
    tot = 0.0
    for s0 in range(0, PW.shape[0], 8): x = PW[s0:s0 + 8]; lg = model(x).logits.float(); tot += float(torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), x[:, 1:].reshape(-1))) * x.shape[0]; del lg
    return tot / PW.shape[0]
FT = {}; o0 = generate(PROMPTS); FT["none"] = dict(**score(o0), pronoun_prob=pronoun_prob(), pile=pile_loss(), samples=o0[:6]); log(f"free text unedited: pronoun probability {FT['none']['pronoun_prob']:.2f}, generated agreement {FT['none']['agree']:.2f}, no pronoun {FT['none']['no_pronoun']:.2f}, degenerate {FT['none']['degenerate']:.2f}")
for kind in ("negate", "zero", "double"):
    edit(bg, jg, kind, svg); o = generate(PROMPTS); FT[f"gender_{kind}"] = dict(**score(o), pronoun_prob=pronoun_prob(), pile=pile_loss(), samples=o[:6]); W(bg)[:, jg] = svg
    log(f"free text, gender row {kind}: pronoun probability {FT[f'gender_{kind}']['pronoun_prob']:.2f}, generated agreement {FT[f'gender_{kind}']['agree']:.2f}, no pronoun {FT[f'gender_{kind}']['no_pronoun']:.2f}, degenerate {FT[f'gender_{kind}']['degenerate']:.2f}, Pile {FT[f'gender_{kind}']['pile'] - FT['none']['pile']:+.4f}")
g_r = torch.Generator().manual_seed(11); FT["random_negate"] = []
for r in torch.randperm(DFF, generator=g_r)[:(2 if SMOKE else 8)].tolist():
    if r == jg: continue
    sv = W(bg)[:, r].clone(); edit(bg, r, "negate", sv); o = generate(PROMPTS); sc = score(o); pp = pronoun_prob(); W(bg)[:, r] = sv; FT["random_negate"].append(dict(row=r, pronoun_prob=pp, agree=sc["agree"], no_pronoun=sc["no_pronoun"], degenerate=sc["degenerate"]))
log("random rows negated: pronoun probability " + ", ".join(f"{x['pronoun_prob']:.2f}" for x in FT["random_negate"]))
g1 = (FT["none"]["pronoun_prob"] - FT["gender_negate"]["pronoun_prob"]) >= 0.15 and FT["gender_negate"]["degenerate"] <= FT["none"]["degenerate"]
g2 = comp["both_negated"][3] >= 0.5 * max(comp["gender_negated"][1], comp["generation_negated"][2], 1e-6) and comp["both_negated"][3] > 0
res = dict(model=name, n_items=n, gender_row=dict(block=bg, row=jg, share=sg_), generation_row=dict(block=ba, row=ja, share=sa_), composition=comp, free_text={k: (v if k == "random_negate" else {kk: vv for kk, vv in v.items() if kk != "first_pronouns"}) for k, v in FT.items()}, verdicts=dict(G1=g1, G2=g2))
summ = (f"free text and composition ({name}): pronoun probability unedited {FT['none']['pronoun_prob']:.2f}, gender row negated {FT['gender_negate']['pronoun_prob']:.2f}, zeroed {FT['gender_zero']['pronoun_prob']:.2f}, doubled {FT['gender_double']['pronoun_prob']:.2f}, random " + "/".join(f"{x['pronoun_prob']:.2f}" for x in FT["random_negate"]) + f"; generated agreement unedited {FT['none']['agree']:.2f}, gender row negated {FT['gender_negate']['agree']:.2f} (no pronoun {FT['gender_negate']['no_pronoun']:.2f}, degenerate {FT['gender_negate']['degenerate']:.2f}, Pile {FT['gender_negate']['pile'] - FT['none']['pile']:+.4f}), zeroed {FT['gender_zero']['agree']:.2f}, doubled {FT['gender_double']['agree']:.2f}, random rows " + "/".join(f"{x['agree']:.2f}" for x in FT["random_negate"]) + "; composition kept/gender/generation/both: " + "; ".join(f"{k} {[round(x, 2) for x in v]}" for k, v in comp.items()) + f"; verdicts G1 {g1}, G2 {g2}")
log(summ); record(f"e624_free_text_{name}" + ("_smoke" if SMOKE else ""), res, summ)
