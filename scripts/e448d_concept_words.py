"""e448d: does the native vocabulary contain concept words? (a scaled, controlled version of e448b's Rosetta words)
24 concrete nouns (cat, dog, car, house, book, water, ... horse), each in 5 sentence templates and 4 languages
(English, French, Spanish, German), 480 sentences in all. At the middle depth, the states at the noun's own tokens are
described with 16 own words (OMP).
A concept word for noun c is a native word used at the noun in at least 15 of c's 20 sentences, in all four languages,
and in at most 10% of the other nouns' sentences.
Reported:
- how many nouns have a concept word, with own words, rotated words (no provenance) and covA words (the dictionary's
  second moment);
- for each noun, the best such word: its block, its neuron, how often it is used, and its specificity;
- a noun-identification score: leave one sentence out, predict its noun from the words used there by nearest centroid
  (words' usage bags), across languages (train on three languages, test on the fourth), against rotated and covA
  words and the mean state at the noun's tokens.
Models (argument): qwen05, smollm2, qwen7 (bf16).
Pre-registered (honest guesses): in Qwen-0.5B most nouns (over 12 of 24) have a concept word with own words and few
(under 4) with rotated words (0.6); cross-language noun identification from own words beats rotated words (0.7) and
matches the mean state (0.4); SmolLM2 has fewer concept words than Qwen (0.6)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
LANGS = list(NOUNS); NC, NT = len(NOUNS["en"]), len(TEMPL["en"])
name = sys.argv[1]
if name == "qwen7":
    from ws_common import load_bf16, lean_dictionary; model, tok, fam = load_bf16(name); arch = Arch(model, fam); L = arch.NB // 2
    A, blkv, typv, ends = lean_dictionary(arch, L); A = A[:ends[L]]; blk = blkv[:ends[L]].to(DEV); typ = typv[:ends[L]].to(DEV); idx = None
else:
    model, tok, fam = load_model(name); arch = Arch(model, fam); L = arch.NB // 2
    A, lab = build_dictionary(arch, blocks=list(range(L + 1))); typ, blk, idx = (lab[k].to(DEV) for k in ("type", "block", "index"))
ref_ids = torch.load("/workspace/wdd/eval_ids_all.pt", weights_only=False)["qwen05" if name == "qwen7" else name]["cen_ids"][:8].to(DEV)
ref = block_states(model, arch, ref_ids, [L], chunk=2)[L]; mu = ref[~sinkmask(ref)].mean(0)
items = []                                                      # (lang, noun, template, states at the noun's tokens [n, D])
for lg in LANGS:
    for c in range(NC):
        for t in range(NT):
            s = TEMPL[lg][t].format(NOUNS[lg][c]); art_noun = NOUNS[lg][c]; noun = art_noun.split(" ", 1)[-1] if " " in art_noun else art_noun.split("'", 1)[-1]
            start = s.index(art_noun) + art_noun.index(noun); end = start + len(noun)
            enc = tok(s, add_special_tokens=False, return_offsets_mapping=True); ids = torch.tensor(enc["input_ids"], device=DEV)
            pos = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if b > start and a < end and i > 0]
            x = block_states(model, arch, ids[None], [L], chunk=1)[L][0]           # positions 1: of the sentence
            items.append(dict(lang=lg, c=c, t=t, x=x[[p - 1 for p in pos]]))
X = torch.cat([it["x"] for it in items]) - mu; seg = torch.cat([torch.full((it["x"].shape[0],), k, device=DEV) for k, it in enumerate(items)])
n_items = len(items); cvec = torch.tensor([it["c"] for it in items], device=DEV); lvec = torch.tensor([LANGS.index(it["lang"]) for it in items], device=DEV)
ev_, U_ = torch.linalg.eigh((A.float().T @ A.float() / A.shape[0]).double()); Rm = ((U_ * ev_.clamp_min(0).sqrt()) @ U_.T).float()
def make(vn):
    """build one vocabulary at a time (the 7B dictionary is 7 GB)"""
    if vn == "own": return A
    if vn == "rot": return rotate(A, seed=7)
    return unitr(torch.randn(A.shape[0], A.shape[1], generator=torch.Generator().manual_seed(1)).to(DEV) @ Rm)
dicts = {vn: None for vn in ("own", "rot", "covA")}
res = dict(model=name, level=L, n_items=n_items, dicts={})
def nearest_centroid(F):
    """train on three languages, test on the fourth; F: [n_items, n_features] (rows L2-normalised)"""
    F = F / F.norm(dim=-1, keepdim=True).clamp_min(1e-9); acc = []
    for l in range(len(LANGS)):
        tr, te = lvec != l, lvec == l
        cent = torch.stack([F[tr & (cvec == c)].mean(0) for c in range(NC)]); cent = cent / cent.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        acc.append(((F[te] @ cent.T).argmax(1) == cvec[te]).float().mean().item())
    return sum(acc) / len(acc)
for vn in dicts:
    Dct = make(vn); sel, cof, _ = omp(X, Dct, 16, batch=128, record_err=False)
    B = torch.zeros(n_items, Dct.shape[0], device=DEV); B.index_put_((seg[:, None].expand_as(sel).long().flatten(), sel.flatten()), cof.abs().flatten(), accumulate=True)
    U = B > 0; words = {}
    for c in range(NC):
        inc, outc = U[cvec == c], U[cvec != c]
        cover = inc.float().sum(0); langs_ok = torch.stack([U[(cvec == c) & (lvec == l)].any(0) for l in range(len(LANGS))]).all(0); spec = outc.float().mean(0)
        ok = (cover >= 15) & langs_ok & (spec <= 0.10)
        if ok.any():
            w = torch.nonzero(ok)[:, 0][cover[ok].argmax()]; words[NOUNS["en"][c]] = dict(word=int(w), block=int(blk[w]), index=int(idx[w]) if idx is not None else None, cover=int(cover[w]), other_share=round(spec[w].item(), 3))
    df = U.float().sum(0); idf = torch.log(n_items / df.clamp_min(1))
    res["dicts"][vn] = dict(n_concept_words=len(words), words=words, noun_id_xlang=nearest_centroid(B * idf[None]))
    log(f"{name} {vn}: {len(words)}/{NC} nouns have a concept word; cross-language noun identification {res['dicts'][vn]['noun_id_xlang']:.2f} | " + ", ".join(f"{k.split()[-1]}: b{v['block']}#{v['index']} ({v['cover']}/20, {v['other_share']:.2f})" for k, v in list(words.items())[:10]))
    del sel, cof, B, U
    if vn != "own": del Dct
    torch.cuda.empty_cache()
Xm = torch.stack([X[seg == k].mean(0) for k in range(n_items)]); res["mean_state_noun_id_xlang"] = nearest_centroid(Xm)
d_ = res["dicts"]
res["checks"] = dict(own_over_12=d_["own"]["n_concept_words"] > 12, rot_under_4=d_["rot"]["n_concept_words"] < 4, own_beats_rot_id=d_["own"]["noun_id_xlang"] > d_["rot"]["noun_id_xlang"], own_matches_mean=d_["own"]["noun_id_xlang"] >= res["mean_state_noun_id_xlang"])
summ = (f"{name} L{L}: nouns with a concept word (>=15/20 sentences, all 4 languages, <=10% elsewhere): own {d_['own']['n_concept_words']}/{NC}, rotated {d_['rot']['n_concept_words']}, covA {d_['covA']['n_concept_words']} | "
        f"cross-language noun identification (nearest centroid, chance 0.04): own {d_['own']['noun_id_xlang']:.2f}, rotated {d_['rot']['noun_id_xlang']:.2f}, covA {d_['covA']['noun_id_xlang']:.2f}, mean state {res['mean_state_noun_id_xlang']:.2f} | "
        + "own concept words: " + ", ".join(f"{k.split()[-1]} b{v['block']}#{v['index']}" for k, v in d_["own"]["words"].items()) + f" | checks {json.dumps(res['checks'])}")
log(summ); record(f"e448d_concepts_{name}", res, summ)
