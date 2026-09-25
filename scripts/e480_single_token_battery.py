"""e480: WorkspaceBench's single-token families, regex-scored, for the native-word reader. The benchmark's first
desideratum is that a workspace reader lose nothing the J-lens captures; its baseline families (basic readout,
multilingual, typo, poetry) are scored by regex on single-token outputs. Here the same families are adapted to small
models and scored the same way, for the plain lens, the centred lens, 16 native words and 16 rotated words.
Families (targets are sets of surface forms, first tokens with and without a leading space):
- basic: 24 factual or computed completions ("The capital of Japan is" -> Tokyo; "7 * 6 =" -> 42);
- multilingual: 24 arithmetic sentences in German, French and Spanish; the answer in any of four languages or as a
  digit counts ("Sechs geteilt durch zwei ist" -> drei, three, trois, tres, 3);
- typo: 20 sentences ending in a misspelled word; the target is the correction, read at the misspelled word;
- poetry: 16 couplets cut before the rhyme; read at the end of the first line (the plan) and at the final token.
Gates (the benchmark's capability check): the model's top-1 continuation is a target form; for typo, a few-shot
correction prompt must give the correction.
Readers at blocks 4, 6, ..., NB-2 at the read position. v2 adds the "sum" reader (the 16-word reconstruction read as a whole by the lens) and the union of the sum reader's top 5 with the per-word reader's top 5, so that a distributed answer and a hidden part can both be read within the same budget of ten tokens. A reader passes an item if a target token is in its top 10 at
any block (the benchmark's rule); also the earliest passing block, and for multilingual the precision of number
claims (top-10 tokens that are numbers 1-12 in any of the four languages, right if the target).
Models (argument): qwen7 (Qwen2.5-7B), qwen05.
Pre-registered (honest guesses):
- at 7B the plain lens passes at least 0.8 of gated items in every family and native words at least 0.7 (0.6);
- native words pass at an earlier block than the lens on average in at most one family (0.6);
- native precision on multilingual number claims is at least the lens's (0.4)."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ws_common import *
name = sys.argv[1] if len(sys.argv) > 1 else "qwen7"; K = 16
model, tok, fam = load_bf16(name); arch = Arch(model, fam); layers = list(range(4, arch.NB - 1, 2))
first = lambda s: tok(s, add_special_tokens=False)["input_ids"][0]
def forms(ws): return {first(" " + w) for w in ws} | {first(w) for w in ws} | {first(" " + w.capitalize()) for w in ws}
BASIC = [("The capital of France is", ["Paris"]), ("The capital of Japan is", ["Tokyo"]), ("The capital of Italy is", ["Rome"]), ("The largest planet in the solar system is", ["Jupiter"]),
         ("The chemical symbol for gold is", ["Au"]), ("The author of Hamlet is William", ["Shakespeare"]), ("The square root of 81 is", ["9", "nine"]), ("10 - 1 =", ["9", "nine"]), ("7 * 6 =", ["42"]),
         ("The opposite of hot is", ["cold"]), ("The color of the sky on a clear day is", ["blue"]), ("The number 23 written out in words is", ["twenty"]), ("The first month of the year is", ["January"]),
         ("A baby cat is called a", ["kitten"]), ("The currency of Japan is the", ["yen"]), ("The tallest mountain on Earth is Mount", ["Everest"]), ("The language spoken in Brazil is", ["Portuguese"]),
         ("The animal known as the king of the jungle is the", ["lion"]), ("The planet closest to the Sun is", ["Mercury"]), ("The number of days in a week is", ["seven", "7"]),
         ("The freezing point of water in Celsius is", ["0", "zero"]), ("Muhammad Ali was a famous", ["boxer"]), ("The Eiffel Tower is located in the city of", ["Paris"]), ("The inventor of the telephone was Alexander Graham", ["Bell"])]
NUM = {1: ["one", "eins", "un", "uno", "1"], 2: ["two", "zwei", "deux", "dos", "2"], 3: ["three", "drei", "trois", "tres", "3"], 4: ["four", "vier", "quatre", "cuatro", "4"], 5: ["five", "fünf", "cinq", "cinco", "5"],
       6: ["six", "sechs", "seis", "6"], 7: ["seven", "sieben", "sept", "siete", "7"], 8: ["eight", "acht", "huit", "ocho", "8"], 9: ["nine", "neun", "neuf", "nueve", "9"], 10: ["ten", "zehn", "dix", "diez", "10"],
       11: ["eleven", "elf", "onze", "once", "11"], 12: ["twelve", "zwölf", "douze", "doce", "12"]}
MULTI = [("Sechs geteilt durch zwei ist", 3), ("Zwei plus drei ist", 5), ("Vier mal zwei ist", 8), ("Zehn minus sechs ist", 4), ("Drei plus vier ist", 7), ("Neun geteilt durch drei ist", 3), ("Fünf plus fünf ist", 10), ("Acht minus zwei ist", 6),
         ("Six divisé par deux font", 3), ("Deux plus trois font", 5), ("Quatre fois deux font", 8), ("Dix moins six font", 4), ("Trois plus quatre font", 7), ("Neuf divisé par trois font", 3), ("Cinq plus cinq font", 10), ("Huit moins deux font", 6),
         ("Seis dividido entre dos es", 3), ("Dos más tres es", 5), ("Cuatro por dos es", 8), ("Diez menos seis es", 4), ("Tres más cuatro es", 7), ("Nueve dividido entre tres es", 3), ("Cinco más cinco es", 10), ("Ocho menos dos es", 6)]
TYPO = [("Her birthday falls on the last day of Febuary", "February"), ("I will definately be there on time", "definitely"), ("Did you recieve my letter", "receive"), ("Please keep the two files seperate", "separate"),
        ("The accident occured last night", "occurred"), ("I do not know wich one to choose", "which"), ("We waited untill midnight", "until"), ("See you tommorow", "tomorrow"), ("This is only the begining", "beginning"),
        ("The new law was passed by the goverment", "government"), ("We must protect the enviroment", "environment"), ("Mark the date on your calender", "calendar"), ("He runs a small buisness", "business"),
        ("She returned the book to the libary", "library"), ("That was a very wierd dream", "weird"), ("You can acheive anything", "achieve"), ("Write your name and adress", "address"), ("Is that really neccessary", "necessary"),
        ("I truely believe it", "truly"), ("He is my best freind", "friend")]
POEMS = [("The captain pointed at the route ahead,", "And told his crew to follow where he'd", "led"), ("The farmer rose before the light of day,", "And led his horses out to fields of", "hay"),
         ("She locked the door and turned the brass key,", "Then walked down to the shore beside the", "sea"), ("At night the sailors watched the northern star,", "And wondered why the harbour was so", "far"),
         ("The children played outside till it was late,", "Then ran back home and rushed in through the", "gate"), ("The moon rose slowly over the hill,", "And all the world grew quiet and", "still"),
         ("The rain came down and would not stop,", "It filled the buckets to the very", "top"), ("The dog ran happily through the snow,", "His paws were cold, his pace was", "slow"),
         ("The birds flew up into the sky,", "And soared above the mountains", "high"), ("She swept the dust across the floor,", "Then closed the window and the", "door"),
         ("The stars came out to light the night,", "And filled the sky with silver", "light"), ("We packed the car and hit the road,", "The trunk was heavy with its", "load"),
         ("The cat curled up upon the mat,", "And next to it there lay a", "hat"), ("The sun came up to start the day,", "The children ran outside to", "play"),
         ("We took the boat out on the sea,", "The wind was strong, the sails were", "free"), ("The clock struck twelve, the hour was late,", "We hurried out and shut the", "gate")]
def enc(text): return torch.tensor(tok(text, add_special_tokens=False)["input_ids"])
items = []
for p, t in BASIC: items.append(dict(fam="basic", ids=enc(p), pos=-1, targets=forms(t), gate=enc(p), cands=None))
allnum = set().union(*[forms(v) for v in NUM.values()])
for p, n in MULTI: items.append(dict(fam="multilingual", ids=enc(p), pos=-1, targets=forms(NUM[n]), gate=enc(p), cands=allnum))
for p, c in TYPO: items.append(dict(fam="typo", ids=enc(p), pos=-1, targets=forms([c]), gate=enc(f"Misspelled: teh\nCorrect: the\nMisspelled: {p.split()[-1]}\nCorrect:"), cands=None))
for l1, l2, r in POEMS:
    full = f"A rhyming couplet:\n{l1}\n{l2}"; items.append(dict(fam="poetry", ids=enc(full), pos=-1, targets=forms([r]), gate=enc(full), cands=None))
    items.append(dict(fam="poetry_plan", ids=enc(full), pos=len(enc(f"A rhyming couplet:\n{l1}")) - 1, targets=forms([r]), gate=enc(full), cands=None))
_, lg = last_states(model, arch, [it["gate"] for it in items], [])
items = [it for i, it in enumerate(items) if int(lg[i].argmax()) in it["targets"]]; N = len(items)
log(f"{name}: {N} items pass the gate: " + ", ".join(f"{f_} {sum(it['fam'] == f_ for it in items)}" for f_ in ("basic", "multilingual", "typo", "poetry", "poetry_plan")))
ev_all = torch.load("/workspace/wdd/eval_ids_all.pt", weights_only=False)["qwen05"]["eval_ids"][:4]
ref = ref_stats(model, arch, ev_all, layers); gain, WU = final_readout(model, arch)
A, blk, typ, ends = lean_dictionary(arch, max(layers))
g = torch.Generator().manual_seed(7); R = torch.linalg.qr(torch.randn(arch.D, arch.D, generator=g))[0].to(DEV)
X_all = {L: [] for L in layers}
for it in items:
    cap = {}
    hs = [arch.layers[l].register_forward_hook((lambda l_: lambda m, i, o: cap.__setitem__(l_, (o[0] if isinstance(o, tuple) else o)[0, it["pos"]].float()))(l)) for l in layers]
    try:
        with torch.no_grad(): model(it["ids"][None].to(DEV))
    finally: [h.remove() for h in hs]
    for L in layers: X_all[L].append(cap[L])
X_all = {L: torch.stack(v) for L, v in X_all.items()}
READERS = ["lens", "lens_centred", "native", "native_sum", "native_union", "rotated"]; hit = {rd: torch.zeros(N, len(layers), dtype=torch.bool) for rd in READERS}; claims = {rd: [0, 0] for rd in READERS}
for li, L in enumerate(layers):
    X = X_all[L]; mu = ref[L]["mu"]; Xc = X - mu; rms = X.pow(2).mean(-1).sqrt(); An = A[:ends[L]]
    sel, _, _ = omp(Xc, An, K, batch=64, record_err=False); cof, _ = refit(Xc, An, sel); words = An[sel]
    selr, _, _ = omp(Xc @ R.T, An, K, batch=64, record_err=False); cofr, _ = refit(Xc @ R.T, An, selr); wordsr = An[selr] @ R
    S = dict(lens=lens_scores(X, gain, WU, rms), lens_centred=lens_scores(Xc, gain, WU, rms), native=pooled_word_scores(words, cof, gain, WU, rms), rotated=pooled_word_scores(wordsr, cofr, gain, WU, rms))
    S["native_sum"] = lens_scores(mu + torch.einsum("nk,nkd->nd", cof, words), gain, WU, rms)        # v2: the 16-word reconstruction read as a whole
    S["native_union"] = None                                                                          # v2: top 5 of the sum reader plus top 5 of the per-word reader
    for rd, sc in S.items():
        top10 = torch.cat([S["native_sum"].topk(5, dim=-1).indices, S["native"].topk(5, dim=-1).indices], 1).cpu() if rd == "native_union" else sc.topk(10, dim=-1).indices.cpu()
        for n, it in enumerate(items):
            t = set(top10[n].tolist()); hit[rd][n, li] = bool(t & it["targets"])
            if it["cands"]: cl = t & it["cands"]; claims[rd][0] += len(cl & it["targets"]); claims[rd][1] += len(cl - it["targets"])
    del S, sel, selr, words, wordsr; torch.cuda.empty_cache()
res = dict(model=name, n=N, k=K, layers=layers, families={}, multilingual_precision={rd: (c[0] / max(c[0] + c[1], 1)) for rd, c in claims.items()})
for f_ in ("basic", "multilingual", "typo", "poetry", "poetry_plan"):
    idx = [n for n, it in enumerate(items) if it["fam"] == f_]
    if not idx: continue
    d = dict(n=len(idx))
    for rd in READERS:
        h = hit[rd][idx]; any_ = h.any(1); earliest = [layers[int(h[i].float().argmax())] for i in range(len(idx)) if any_[i]]
        d[rd] = dict(pass_any=float(any_.float().mean()), earliest_block_mean=(sum(earliest) / len(earliest)) if earliest else None, by_block={L: float(h[:, li].float().mean()) for li, L in enumerate(layers)})
    res["families"][f_] = d
    log(f"{name} {f_} (n {len(idx)}): " + " | ".join(f"{rd} pass {d[rd]['pass_any']:.2f}, earliest block {d[rd]['earliest_block_mean'] if d[rd]['earliest_block_mean'] is None else round(d[rd]['earliest_block_mean'], 1)}, by block " + " ".join(f"{L}:{v:.2f}" for L, v in d[rd]["by_block"].items()) for rd in READERS))
Fm = res["families"]; fams = [f_ for f_ in ("basic", "multilingual", "typo", "poetry") if f_ in Fm]
earlier = sum(1 for f_ in fams if Fm[f_]["native"]["earliest_block_mean"] is not None and Fm[f_]["lens"]["earliest_block_mean"] is not None and Fm[f_]["native"]["earliest_block_mean"] < Fm[f_]["lens"]["earliest_block_mean"])
res["checks"] = dict(lens_08_native_07=all(Fm[f_]["lens"]["pass_any"] >= 0.8 and Fm[f_]["native"]["pass_any"] >= 0.7 for f_ in fams), native_earlier_in_at_most_one=earlier <= 1,
                     native_precision_at_least_lens=res["multilingual_precision"]["native"] >= res["multilingual_precision"]["lens"])
summ = (f"{name}, {N} gated items; pass rate (any block) lens / centred / native per-word / native sum / union / rotated, and mean earliest block lens vs native per-word: " + " | ".join(f"{f_} (n {d['n']}): {d['lens']['pass_any']:.2f} / {d['lens_centred']['pass_any']:.2f} / {d['native']['pass_any']:.2f} / {d['native_sum']['pass_any']:.2f} / {d['native_union']['pass_any']:.2f} / {d['rotated']['pass_any']:.2f}, earliest {d['lens']['earliest_block_mean'] if d['lens']['earliest_block_mean'] is None else round(d['lens']['earliest_block_mean'], 1)} vs {d['native']['earliest_block_mean'] if d['native']['earliest_block_mean'] is None else round(d['native']['earliest_block_mean'], 1)}" for f_, d in Fm.items())
        + " | multilingual number-claim precision: " + ", ".join(f"{rd} {v:.2f}" for rd, v in res["multilingual_precision"].items()) + f" | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e480_battery_{name}", res, summ)
