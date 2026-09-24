"""e451: are concept words causal handles for their concept? This is the sharper test an external review asked for. It
asks whether the word carries a specific downstream effect, not only whether it is used on translations.
Task: few-shot translation of the noun into English, from English (a copy), French, Spanish and German sentences
(e448d's 24 nouns x 5 templates x 4 languages):
  "<Language>: <example>\nEnglish noun: milk\n<Language>: <example>\nEnglish noun: flower\n<Language>: <test sentence>\nEnglish noun:"
Readout: the log-probability of each noun's first token, restricted to the 24 nouns' first tokens (a 24-way softmax),
and accuracy.
Concept words are e448d's: one MLP write row per noun, used on the noun in all four languages and rarely elsewhere.
Interventions are at the middle depth L (e448d's depth), at the test sentence's noun tokens ("noun") or at the answer
position ("answer"):
- remove: the state's component along the concept word is removed;
- remove_other: the same for a matched other word. This is the native MLP-row word, used in the state's 16-word
  description at the noun, whose coefficient size is closest to the concept word's;
- remove_random: a random direction of the removed component's size is subtracted;
- swap: the concept word's component is removed, and another noun's concept word is added at that noun's typical
  coefficient in the same language (a fixed derangement over the nouns with concept words);
- add_random: the component is removed and a random direction of the same size as the swap's addition is added
  (the control for swap).
Reported per language:
- accuracy;
- the mean change in the correct noun's restricted log-probability, and its specificity (that change minus the mean
  change of the other nouns);
- for swaps, the share of answers that become the swapped-in noun, and the change in its log-probability;
- whether the concept word is used at the noun and at the answer position in context (16-word descriptions).
Models (argument): qwen05, smollm2, qwen7 (bf16).
Pre-registered (honest guesses):
- baseline accuracy is over 0.8 in all four languages for Qwen-0.5B (0.6);
- removing the concept word at the noun lowers the correct noun's log-probability more than removing a matched other
  word or a random direction (0.65);
- it changes few answers: the accuracy drop is under 0.2 (0.6);
- swapping at the noun moves under a fifth of the answers (0.6), but raises the swapped-in noun's log-probability more
  than the random control does (0.7)."""
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
first = [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in EN]
dup = {f for f in first if first.count(f) > 1}; cand_ok = [c for c in range(NC) if first[c] not in dup]
cand = torch.tensor([first[c] for c in cand_ok], device=DEV); cpos = {c: j for j, c in enumerate(cand_ok)}
concept = {c: int(cw[NOUNS["en"][c]]["word"]) for c in range(NC) if NOUNS["en"][c] in cw and c in cpos}
for c, w in concept.items(): assert int(blk[w]) == cw[NOUNS["en"][c]]["block"] and int(typ[w]) == T_MLP, "dictionary order changed"
CN = sorted(concept); target = {c: CN[(i + len(CN) // 2) % len(CN)] for i, c in enumerate(CN)}
cen = torch.load("/workspace/wdd/eval_ids_all.pt", weights_only=False)["qwen05" if name == "qwen7" else name]["cen_ids"][:8].to(DEV)
ref = block_states(model, arch, cen, [L], chunk=2)[L]; mu = ref[~sinkmask(ref)].mean(0); del ref
Uw = {c: A[w] / A[w].norm() for c, w in concept.items()}
concept_set = set(concept.values())

def prompt(lg, c, t):
    s = TEMPL[lg][t].format(NOUNS[lg][c]); art = NOUNS[lg][c]; noun = art.split(" ", 1)[-1] if " " in art else art.split("'", 1)[-1]
    p = f"{LN[lg]}: {EX[lg][0]}\nEnglish noun: milk\n{LN[lg]}: {EX[lg][1]}\nEnglish noun: flower\n{LN[lg]}: {s}\nEnglish noun:"
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

# pass 1: clean runs (all nouns for the baseline), the concept words' usage in context and their typical coefficients
items, base = [], {lg: [] for lg in LANGS}
for lg in LANGS:
    for c in range(NC):
        if c not in cpos: continue
        for t in range(NT):
            ids, pos = prompt(lg, c, t); lp, X = run(ids); ok = int(lp.argmax()) == cpos[c]; base[lg].append(ok)
            if c in concept: items.append(dict(lg=lg, c=c, t=t, ids=ids, pos=pos, lp=lp, xn=X[pos], xa=X[-1]))
used_noun, used_ans = {lg: [] for lg in LANGS}, {lg: [] for lg in LANGS}
for it in items:
    w = concept[it["c"]]; Z = torch.cat([it["xn"], it["xa"][None]]) - mu
    sel, cof, _ = omp(Z, A, 16, batch=64, record_err=False); it["sel"], it["cof"] = sel, cof
    used_noun[it["lg"]].append(bool((sel[:-1] == w).any())); used_ans[it["lg"]].append(bool((sel[-1] == w).any()))
    it["proj_noun"] = (it["xn"] @ Uw[it["c"]]).tolist(); it["proj_ans"] = float(it["xa"] @ Uw[it["c"]])
typcoef = {}
for lg in LANGS:
    for c in CN:
        v = [p for it in items if it["lg"] == lg and it["c"] == c for p in it["proj_noun"]]; typcoef[(lg, c)] = sum(v) / len(v)

def matched_other(it):
    """the MLP-row word used at a noun position (not a concept word) whose |coefficient| is closest to the concept word's
    (its coefficient where used, else its projection)"""
    w = concept[it["c"]]; best = None
    for j in range(len(it["pos"])):
        sel, cof = it["sel"][j], it["cof"][j]
        cw_ = [abs(float(cof[k])) for k in range(16) if int(sel[k]) == w]; ref_ = cw_[0] if cw_ else abs(it["proj_noun"][j])
        for k in range(16):
            a = int(sel[k])
            if a == w or a in concept_set or int(typ[a]) != T_MLP: continue
            d_ = abs(abs(float(cof[k])) - ref_)
            if best is None or d_ < best[0]: best = (d_, a)
    return None if best is None else A[best[1]] / A[best[1]].norm()

g = torch.Generator().manual_seed(0)
OPS = ("remove", "remove_other", "remove_random", "swap", "add_random")
res_rows = []
for it in items:
    c, lg = it["c"], it["lg"]; u = Uw[c]; tc = target[c]; u2 = Uw[tc]; a2 = typcoef[(lg, tc)]
    r1 = torch.randn(D, generator=g).to(DEV); r1 = r1 / r1.norm(); r2 = torch.randn(D, generator=g).to(DEV); r2 = r2 / r2.norm()
    uo = matched_other(it); row = dict(lg=lg, c=c, t=it["t"], target=tc)
    for site, P in (("noun", it["pos"]), ("answer", [it["ids"].shape[1] - 1])):
        for op in OPS:
            if op == "remove_other" and (uo is None or site == "answer"): continue
            if op == "remove": fn = lambda x: x - (x @ u) * u
            elif op == "remove_other": fn = lambda x: x - (x @ uo) * uo
            elif op == "remove_random": fn = lambda x: x - (x @ u).abs() * r1
            elif op == "swap": fn = lambda x: x - (x @ u) * u + a2 * u2
            else: fn = lambda x: x - (x @ u) * u + abs(a2) * r2
            lp1, _ = run(it["ids"], {p: fn for p in P}); lp0 = it["lp"]; ci, ti = cpos[c], cpos[tc]
            dl = lp1 - lp0; oth = torch.ones_like(dl, dtype=torch.bool); oth[ci] = False
            row[f"{site}_{op}"] = dict(d_correct=float(dl[ci]), spec=float(dl[ci] - dl[oth].mean()), acc=int(lp1.argmax()) == ci,
                                       d_target=float(dl[ti]), to_target=int(lp1.argmax()) == ti)
    row["acc0"] = int(it["lp"].argmax()) == cpos[c]; res_rows.append(row)

def agg(rows, key, f):
    v = [r[key][f] for r in rows if key in r]; return (sum(float(x) for x in v) / len(v)) if v else None
summary = {}
for lg in LANGS + ["all"]:
    rr = [r for r in res_rows if lg == "all" or r["lg"] == lg]; s = dict(n=len(rr), acc0=sum(r["acc0"] for r in rr) / max(len(rr), 1))
    for site in ("noun", "answer"):
        for op in OPS:
            k = f"{site}_{op}"
            if any(k in r for r in rr): s[k] = {f: agg(rr, k, f) for f in ("d_correct", "spec", "acc", "d_target", "to_target")}
    summary[lg] = s
res = dict(model=name, level=L, n_concepts=len(CN), concept_nouns=[EN[c] for c in CN], candidates=len(cand_ok),
           baseline_acc_all_nouns={lg: sum(v) / len(v) for lg, v in base.items()},
           concept_used_at_noun={lg: sum(v) / len(v) for lg, v in used_noun.items()}, concept_used_at_answer={lg: sum(v) / len(v) for lg, v in used_ans.items()},
           typical_coef={f"{lg}:{EN[c]}": v for (lg, c), v in typcoef.items()},
           mean_abs_proj_answer={lg: sum(abs(it["proj_ans"]) for it in items if it["lg"] == lg) / max(1, sum(1 for it in items if it["lg"] == lg)) for lg in LANGS},
           mean_state_norm_noun=float(torch.cat([it["xn"] for it in items]).norm(dim=-1).mean()),
           summary=summary, rows=res_rows)
S_ = summary["all"]; nr, no, nd = S_["noun_remove"], S_.get("noun_remove_other"), S_["noun_remove_random"]
res["checks"] = dict(baseline_over_0_8_all_langs=all(v > 0.8 for v in res["baseline_acc_all_nouns"].values()),
                     remove_beats_other_and_random=(no is not None and nr["d_correct"] < no["d_correct"] and nr["d_correct"] < nd["d_correct"]),
                     acc_drop_under_0_2=(S_["acc0"] - nr["acc"]) < 0.2, swap_moves_under_fifth=S_["noun_swap"]["to_target"] < 0.2,
                     swap_target_over_random=S_["noun_swap"]["d_target"] > S_["noun_add_random"]["d_target"])
f2 = lambda d_: f"{d_['d_correct']:+.2f}/{d_['spec']:+.2f}/acc {d_['acc']:.2f}"
lines = []
for lg in LANGS:
    s = summary[lg]
    lines.append(f"{lg}: acc0 {s['acc0']:.2f} | noun remove {f2(s['noun_remove'])}, other {f2(s['noun_remove_other']) if 'noun_remove_other' in s else 'n/a'}, random {f2(s['noun_remove_random'])} | "
                 f"noun swap to-target {s['noun_swap']['to_target']:.2f} dT {s['noun_swap']['d_target']:+.2f} (random dT {s['noun_add_random']['d_target']:+.2f}) | "
                 f"answer remove {f2(s['answer_remove'])}, swap to-target {s['answer_swap']['to_target']:.2f} dT {s['answer_swap']['d_target']:+.2f} (random {s['answer_add_random']['d_target']:+.2f})")
summ = (f"{name} L{L}, {len(CN)} concept nouns, restricted to {len(cand_ok)} nouns: baseline acc (all nouns) " + " ".join(f"{lg} {v:.2f}" for lg, v in res["baseline_acc_all_nouns"].items())
        + " | concept word used at noun / answer: " + " ".join(f"{lg} {res['concept_used_at_noun'][lg]:.2f}/{res['concept_used_at_answer'][lg]:.2f}" for lg in LANGS)
        + " || d_correct/specificity/acc: " + " || ".join(lines) + f" | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e451_concepthandles_{name}", res, summ)
