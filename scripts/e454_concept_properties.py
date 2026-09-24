"""e454: do concept words carry their concept's properties, or only its name? (follows e451)
e451 found concept words to be specific handles for translating a noun: removing one lowers the correct English noun,
and swapping in another noun's word raises that noun. That could be lexical (word to word), not semantic. Here the
readout needs a property of the concept: its category.
Task (few-shot, per source language): "<Language>: <example>\nCategory: drink\n<Language>: <example>\nCategory:
plant\n<Language>: <test sentence>\nCategory:". The answer is one of animal, person, place, food, drink, plant, sky,
object, scored by the restricted softmax over the eight labels' first tokens.
Interventions at the test noun's tokens, at the middle depth (as in e451):
- remove: the concept word's component is removed;
- swap_diff: it is replaced by the concept word of a noun from a different category, at that noun's typical
  coefficient;
- swap_same: it is replaced by the concept word of a noun from the same category (where one exists);
- add_random: the component is removed and a random direction of the swap's size is added.
Reported per language, on the prompts the model answers correctly:
- the change in the correct category's log-probability;
- for swaps, the share of answers that move to the target's category, and the change in that category's
  log-probability;
- for swap_same, the share of answers that leave the correct category.
Models (argument): qwen05, smollm2, qwen7 (bf16).
Pre-registered (honest guesses):
- swap_diff moves more answers to the target's category than add_random does (0.65);
- swap_same changes the category of under half as many answers as swap_diff (0.6);
- in SmolLM2 at least a fifth of non-English answers move under swap_diff (0.5)."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
NOUNS = {
 "en": ["the cat", "the dog", "the car", "the house", "the book", "the water", "the tree", "the sun", "the moon", "the bread", "the phone", "the door", "the window", "the river", "the city", "the school", "the doctor", "the teacher", "the money", "the key", "the chair", "the table", "the bird", "the horse"],
 "fr": ["le chat", "le chien", "la voiture", "la maison", "le livre", "l'eau", "l'arbre", "le soleil", "la lune", "le pain", "le téléphone", "la porte", "la fenêtre", "la rivière", "la ville", "l'école", "le médecin", "le professeur", "l'argent", "la clé", "la chaise", "la table", "l'oiseau", "le cheval"],
 "es": ["el gato", "el perro", "el coche", "la casa", "el libro", "el agua", "el árbol", "el sol", "la luna", "el pan", "el teléfono", "la puerta", "la ventana", "el río", "la ciudad", "la escuela", "el médico", "el profesor", "el dinero", "la llave", "la silla", "la mesa", "el pájaro", "el caballo"],
 "de": ["die Katze", "den Hund", "das Auto", "das Haus", "das Buch", "das Wasser", "den Baum", "die Sonne", "den Mond", "das Brot", "das Telefon", "die Tür", "das Fenster", "den Fluss", "die Stadt", "die Schule", "den Arzt", "den Lehrer", "das Geld", "den Schlüssel", "den Stuhl", "den Tisch", "den Vogel", "das Pferd"]}
TEMPL = {
 "en": ["I can see {}.", "She really likes {}.", "Please do not touch {}.", "We found {} this morning.", "Look at {} over there."],
 "fr": ["Je peux voir {}.", "Elle aime vraiment {}.", "S'il te plaît, ne touche pas {}.", "Nous avons trouvé {} ce matin.", "Regarde {} là-bas."],
 "es": ["Puedo ver {}.", "A ella le gusta mucho {}.", "Por favor, no toques {}.", "Encontramos {} esta mañana.", "Mira {} allí."],
 "de": ["Ich kann {} sehen.", "Sie mag {} sehr.", "Bitte berühre {} nicht.", "Wir haben {} heute Morgen gefunden.", "Schau dir {} dort an."]}
EX = {"en": ["We bought the milk yesterday.", "She painted the flower."], "fr": ["Nous avons acheté le lait hier.", "Elle a peint la fleur."],
      "es": ["Compramos la leche ayer.", "Ella pintó la flor."], "de": ["Wir haben gestern die Milch gekauft.", "Sie hat die Blume gemalt."]}
LN = {"en": "English", "fr": "French", "es": "Spanish", "de": "German"}
EXC = {"en": ["We bought the milk yesterday.", "She painted the flower."], "fr": ["Nous avons acheté le lait hier.", "Elle a peint la fleur."],
       "es": ["Compramos la leche ayer.", "Ella pintó la flor."], "de": ["Wir haben gestern die Milch gekauft.", "Sie hat die Blume gemalt."]}
CAT = dict(cat="animal", dog="animal", car="object", house="place", book="object", water="drink", tree="plant", sun="sky", moon="sky", bread="food",
           phone="object", door="object", window="object", river="place", city="place", school="place", doctor="person", teacher="person",
           money="object", key="object", chair="object", table="object", bird="animal", horse="animal")
LABELS = ["animal", "person", "place", "food", "drink", "plant", "sky", "object"]
LANGS = list(NOUNS); NC, NT = len(NOUNS["en"]), len(TEMPL["en"])
name = sys.argv[1]
if name == "qwen7":                                                   # bf16 and e448d's lean dictionary (the word indices refer to it)
    from ws_common import load_bf16, lean_dictionary
    model, tok, fam = load_bf16(name); arch = Arch(model, fam); D = arch.D; L = arch.NB // 2
    A, blkv, typv, ends = lean_dictionary(arch, L); A = A[:ends[L]]; blk = blkv[:ends[L]].long(); typ = typv[:ends[L]].long()
else:
    model, tok, fam = load_model(name); arch = Arch(model, fam); D = arch.D; L = arch.NB // 2
    A, lab = build_dictionary(arch, blocks=list(range(L + 1))); typ, blk = lab["type"].to(DEV), lab["block"].to(DEV)
cw = _json.load(open(os.path.join(RESULTS, f"e448d_concepts_{name}.json")))["dicts"]["own"]["words"]
EN = [n.split(" ", 1)[1] for n in NOUNS["en"]]
lfirst = [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in LABELS]; assert len(set(lfirst)) == len(LABELS), "label first tokens collide"
cand = torch.tensor(lfirst, device=DEV); catix = {c: LABELS.index(CAT[EN[c]]) for c in range(NC)}
concept = {c: int(cw[NOUNS["en"][c]]["word"]) for c in range(NC) if NOUNS["en"][c] in cw}
for c, w in concept.items(): assert int(blk[w]) == cw[NOUNS["en"][c]]["block"] and int(typ[w]) == T_MLP, "dictionary order changed"
CN = sorted(concept)
def pick_target(c, same):
    pool = [d for d in CN if d != c and ((catix[d] == catix[c]) == same)]
    return pool[(CN.index(c) * 7) % len(pool)] if pool else None
tdiff = {c: pick_target(c, False) for c in CN}; tsame = {c: pick_target(c, True) for c in CN}
cen = torch.load("/workspace/wdd/eval_ids_all.pt", weights_only=False)["qwen05" if name == "qwen7" else name]["cen_ids"][:8].to(DEV)
ref = block_states(model, arch, cen, [L], chunk=2)[L]; mu = ref[~sinkmask(ref)].mean(0); del ref
Uw = {c: A[w] / A[w].norm() for c, w in concept.items()}
concept_set = set(concept.values())

def prompt(lg, c, t):
    s = TEMPL[lg][t].format(NOUNS[lg][c]); art = NOUNS[lg][c]; noun = art.split(" ", 1)[-1] if " " in art else art.split("'", 1)[-1]
    p = f"{LN[lg]}: {EXC[lg][0]}\nCategory: drink\n{LN[lg]}: {EXC[lg][1]}\nCategory: plant\n{LN[lg]}: {s}\nCategory:"
    s0 = p.rindex(s); a = s0 + s.index(art) + art.index(noun); b_ = a + len(noun)
    enc = tok(p, add_special_tokens=False, return_offsets_mapping=True); ids = torch.tensor(enc["input_ids"], device=DEV)[None]
    pos = [i for i, (x0, x1) in enumerate(enc["offset_mapping"]) if x1 > a and x0 < b_]
    return ids, pos

def run(ids, edits=None):
    """restricted log-probabilities over the candidate nouns at the last position; edits: {position: fn(state) -> state}
    applied to block L's output; also returns block L's output (clean runs)"""
    cap = {}
    def hk(m, i, o):
        xo = out_of(o); cap["x"] = xo[0].detach().float()
        if not edits: return None
        y = xo.clone()
        for p, fn in edits.items(): y[0, p] = fn(y[0, p].float()).to(y.dtype)
        return (y,) + tuple(o[1:]) if isinstance(o, tuple) else y
    h = arch.layers[L].register_forward_hook(hk)
    try:
        with torch.no_grad(): lg = model(ids).logits[0, -1].float()
    finally: h.remove()
    return torch.log_softmax(lg[cand], -1), cap["x"]

# pass 1: clean runs, typical coefficients
items, base = [], {lg: [] for lg in LANGS}
for lg in LANGS:
    for c in range(NC):
        for t in range(NT):
            ids, pos = prompt(lg, c, t); lp, X = run(ids); ok = int(lp.argmax()) == catix[c]; base[lg].append(ok)
            if c in concept: items.append(dict(lg=lg, c=c, t=t, ids=ids, pos=pos, lp=lp, ok=ok, xn=X[pos]))
typcoef = {}
for lg in LANGS:
    for c in CN:
        v = [float(p) for it in items if it["lg"] == lg and it["c"] == c for p in (it["xn"] @ Uw[c])]; typcoef[(lg, c)] = sum(v) / len(v)
g = torch.Generator().manual_seed(0); rows = []
for it in items:
    if not it["ok"]: continue
    c, lg = it["c"], it["lg"]; u = Uw[c]; ci = catix[c]; row = dict(lg=lg, c=c, t=it["t"])
    r2 = torch.randn(D, generator=g).to(DEV); r2 = r2 / r2.norm()
    ops = [("remove", lambda x: x - (x @ u) * u, None)]
    for nm, tg in (("swap_diff", tdiff[c]), ("swap_same", tsame[c])):
        if tg is not None:
            a2, u2 = typcoef[(lg, tg)], Uw[tg]; ops.append((nm, (lambda a2, u2: lambda x: x - (x @ u) * u + a2 * u2)(a2, u2), tg))
    if tdiff[c] is not None:
        a2 = abs(typcoef[(lg, tdiff[c])]); ops.append(("add_random", (lambda a2: lambda x: x - (x @ u) * u + a2 * r2)(a2), tdiff[c]))
    for nm, fn, tg in ops:
        lp1, _ = run(it["ids"], {p: fn for p in it["pos"]}); dl = lp1 - it["lp"]; am = int(lp1.argmax())
        d_ = dict(d_correct=float(dl[ci]), left_correct=am != ci)
        if tg is not None: ti = catix[tg]; d_.update(to_target_cat=am == ti, d_target_cat=float(dl[ti]))
        row[nm] = d_
    rows.append(row)
def agg(rr, k, f):
    v = [float(r[k][f]) for r in rr if k in r and f in r[k]]; return (sum(v) / len(v)) if v else None
summary = {}
for lg in LANGS + ["all"]:
    rr = [r for r in rows if lg == "all" or r["lg"] == lg]; s = dict(n=len(rr))
    for k in ("remove", "swap_diff", "swap_same", "add_random"):
        s[k] = {f: agg(rr, k, f) for f in ("d_correct", "left_correct", "to_target_cat", "d_target_cat")}
    summary[lg] = s
res = dict(model=name, level=L, concept_nouns=[EN[c] for c in CN], categories={EN[c]: LABELS[catix[c]] for c in CN},
           baseline_acc_all_nouns={lg: sum(v) / len(v) for lg, v in base.items()}, targets_diff={EN[c]: (EN[t] if t is not None else None) for c, t in tdiff.items()},
           targets_same={EN[c]: (EN[t] if t is not None else None) for c, t in tsame.items()}, summary=summary, rows=rows)
S_ = summary["all"]; nonen = [r for r in rows if r["lg"] != "en"]
res["checks"] = dict(swapdiff_beats_random=(S_["swap_diff"]["to_target_cat"] or 0) > (S_["add_random"]["to_target_cat"] or 0),
                     swapsame_under_half=(S_["swap_same"]["left_correct"] is not None and S_["swap_same"]["left_correct"] < 0.5 * (S_["swap_diff"]["left_correct"] or 0)),
                     smollm2_fifth_move=(name != "smollm2") or (agg(nonen, "swap_diff", "to_target_cat") or 0) >= 0.2)
fmt_ = lambda d_, k: "n/a" if d_[k]["d_correct"] is None else (f"{d_[k]['d_correct']:+.2f} left {d_[k]['left_correct']:.2f}" + (f" to-target {d_[k]['to_target_cat']:.2f} dT {d_[k]['d_target_cat']:+.2f}" if d_[k]["to_target_cat"] is not None else ""))
summ = (f"{name} L{L}, {len(CN)} concept nouns, category readout: baseline acc (all nouns) " + " ".join(f"{lg} {v:.2f}" for lg, v in res["baseline_acc_all_nouns"].items()) + " || "
        + " || ".join(f"{lg} (n {summary[lg]['n']}): remove {fmt_(summary[lg], 'remove')} | swap_diff {fmt_(summary[lg], 'swap_diff')} | swap_same {fmt_(summary[lg], 'swap_same')} | random {fmt_(summary[lg], 'add_random')}" for lg in LANGS)
        + f" | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e454_conceptprops_{name}", res, summ)
