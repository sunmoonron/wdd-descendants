"""e415: can the model's own words surface its hidden intermediate? After WorkspaceBench (LessWrong 2026): workspace readers
are judged on intermediates a model computes without stating them, and single-token lenses miss much of it while
text-generating readers hallucinate. Two-hop recall has a known intermediate: in "The capital of the country where the
Eiffel Tower is located is", the bridge entity France is never written in the prompt. Qwen2.5-7B (bf16), a few-shot
format, three relations (capital, official language, currency) over about 55 landmarks; kept only when the model
answers the two-hop prompt and the one-hop bridge prompt correctly (top-1 first token, the capability gate).
At the last position after blocks 4, 6, ..., 26, token rankings from: the logit lens of the state; the logit lens of
the state minus the natural-text mean (centred); the PCA lens (the centred state split into its projections on the top
16 principal directions of natural-text states, each read by the logit lens, pooled by the maximum over parts); the
native-word lens (the centred state described by OMP with 16 words of the model's own vocabulary up to that block,
each word times its coefficient read by the logit lens, pooled the same way); and the same with the vocabulary
randomly rotated (control). Scored: recall of the bridge token and of the answer token in the top 1, 5, 10, 20 and 50.
Also: fidelity (the last-position state replaced by the mean plus each 16-part reconstruction: is the answer still
top-1?) and a causal check (removing from the true state the one native word whose own readout contains the bridge,
against removing another of the 16 words at random: change in the answer's log-probability).
Pre-registered: (1) at middle depth the native-word lens has a higher bridge recall at 20 than the plain lens, the
centred lens and the PCA lens; (2) the rotated control does not; (3) removing the bridge word lowers the answer's
log-probability more than removing a random other word. Also: the rank of the true bridge among all candidate country
tokens (@1, @3) and the share of prompts with a wrong country in a reader's top 20 (a readout the context contradicts)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ws_common import *
FACTS = [("the Eiffel Tower", "France", "Paris", "French", "euro"), ("the Louvre", "France", "Paris", "French", "euro"), ("Mont Saint-Michel", "France", "Paris", "French", "euro"),
 ("the Colosseum", "Italy", "Rome", "Italian", "euro"), ("the Leaning Tower of Pisa", "Italy", "Rome", "Italian", "euro"), ("Big Ben", "United Kingdom", "London", "English", "pound"),
 ("Stonehenge", "United Kingdom", "London", "English", "pound"), ("the Tower of London", "United Kingdom", "London", "English", "pound"), ("the Statue of Liberty", "United States", "Washington", "English", "dollar"),
 ("the Golden Gate Bridge", "United States", "Washington", "English", "dollar"), ("the Taj Mahal", "India", "New Delhi", "Hindi", "rupee"), ("the Great Wall", "China", "Beijing", "Chinese", "yuan"),
 ("the Forbidden City", "China", "Beijing", "Chinese", "yuan"), ("the Terracotta Army", "China", "Beijing", "Chinese", "yuan"), ("the Kremlin", "Russia", "Moscow", "Russian", "ruble"),
 ("the Hermitage Museum", "Russia", "Moscow", "Russian", "ruble"), ("the Brandenburg Gate", "Germany", "Berlin", "German", "euro"), ("Neuschwanstein Castle", "Germany", "Berlin", "German", "euro"),
 ("the Sagrada Familia", "Spain", "Madrid", "Spanish", "euro"), ("the Alhambra", "Spain", "Madrid", "Spanish", "euro"), ("the Acropolis", "Greece", "Athens", "Greek", "euro"),
 ("the Parthenon", "Greece", "Athens", "Greek", "euro"), ("Christ the Redeemer", "Brazil", "Brasília", "Portuguese", "real"), ("Mount Fuji", "Japan", "Tokyo", "Japanese", "yen"),
 ("Kinkaku-ji", "Japan", "Tokyo", "Japanese", "yen"), ("the Pyramids of Giza", "Egypt", "Cairo", "Arabic", "pound"), ("Machu Picchu", "Peru", "Lima", "Spanish", "sol"),
 ("the Petronas Towers", "Malaysia", "Kuala Lumpur", "Malay", "ringgit"), ("the CN Tower", "Canada", "Ottawa", "English", "dollar"), ("Angkor Wat", "Cambodia", "Phnom Penh", "Khmer", "riel"),
 ("the Sydney Opera House", "Australia", "Canberra", "English", "dollar"), ("Uluru", "Australia", "Canberra", "English", "dollar"), ("Table Mountain", "South Africa", "Pretoria", "English", "rand"),
 ("Chichen Itza", "Mexico", None, "Spanish", "peso"), ("Hagia Sophia", "Turkey", "Ankara", "Turkish", "lira"), ("the Blue Mosque", "Turkey", "Ankara", "Turkish", "lira"),
 ("the Rijksmuseum", "Netherlands", "Amsterdam", "Dutch", "euro"), ("the Atomium", "Belgium", "Brussels", "Dutch", "euro"), ("the Little Mermaid statue", "Denmark", "Copenhagen", "Danish", "krone"),
 ("the Vasa Museum", "Sweden", "Stockholm", "Swedish", "krona"), ("Petra", "Jordan", "Amman", "Arabic", "dinar"), ("the Charles Bridge", "Czech Republic", "Prague", "Czech", "koruna"),
 ("Schönbrunn Palace", "Austria", "Vienna", "German", "euro"), ("Wawel Castle", "Poland", "Warsaw", "Polish", "zloty"), ("Bran Castle", "Romania", "Bucharest", "Romanian", "leu"),
 ("Hallgrímskirkja", "Iceland", "Reykjavik", "Icelandic", "krona"), ("Mount Kilimanjaro", "Tanzania", "Dodoma", "Swahili", "shilling"), ("Borobudur", "Indonesia", "Jakarta", "Indonesian", "rupiah"),
 ("the Moai statues", "Chile", "Santiago", "Spanish", "peso"), ("the Atacama Desert", "Chile", "Santiago", "Spanish", "peso"), ("Gyeongbokgung Palace", "South Korea", "Seoul", "Korean", "won"),
 ("the Burj Khalifa", "United Arab Emirates", "Abu Dhabi", "Arabic", "dirham"), ("the Shwedagon Pagoda", "Myanmar", "Naypyidaw", "Burmese", "kyat"), ("Ha Long Bay", "Vietnam", "Hanoi", "Vietnamese", "dong"),
 ("Wat Arun", "Thailand", "Bangkok", "Thai", "baht"), ("the Cliffs of Moher", "Ireland", "Dublin", "Irish", "euro")]
TPL = {"capital": ("The capital of the country where {} is located is", "{}"), "language": ("The official language of the country where {} is located is", "{}"),
       "currency": ("The currency of the country where {} is located is the", "{}")}
SHOT = {"capital": [("the Belém Tower", "Lisbon"), ("Vigeland Park", "Oslo")], "language": [("the Belém Tower", "Portuguese"), ("Vigeland Park", "Norwegian")],
        "currency": [("the Belém Tower", "euro"), ("Vigeland Park", "krone")]}
ONEHOP = ("{} is located in the country of", [("The Belém Tower", "Portugal"), ("Vigeland Park", "Norway")])
name = sys.argv[1] if len(sys.argv) > 1 else "qwen7"; K = 16; NS = [1, 5, 10, 20, 50]
model, tok, fam = load_bf16(name); arch = Arch(model, fam); layers = list(range(4, arch.NB - 1, 2))
first = lambda s: tok(s, add_special_tokens=False)["input_ids"][0]
def enc(text): return torch.tensor(tok(text, add_special_tokens=False)["input_ids"])
items = []
for lm, country, capital, lang, cur in FACTS:
    for rel, ans in [("capital", capital), ("language", lang), ("currency", cur)]:
        if ans is None: continue
        q, _ = TPL[rel]; shots = "".join(f"{q.format(a)} {b}.\n" for a, b in SHOT[rel])
        items.append(dict(landmark=lm, rel=rel, country=country, answer=ans, ids=enc(shots + q.format(lm)),
                          onehop=enc("".join(f"{ONEHOP[0].format(a)} {b}.\n" for a, b in ONEHOP[1]) + ONEHOP[0].format(lm[0].upper() + lm[1:])),
                          bridge_ids={first(" " + country), first(country)}, answer_ids={first(" " + ans), first(ans)}))
# capability gate
_, lg2 = last_states(model, arch, [it["ids"] for it in items], [])
_, lg1 = last_states(model, arch, [it["onehop"] for it in items], [])
keep = [i for i, it in enumerate(items) if lg2[i].argmax().item() in it["answer_ids"] and lg1[i].argmax().item() in it["bridge_ids"]]
for i in keep: items[i]["ans_tok"] = lg2[i].argmax().item()
items = [items[i] for i in keep]; N = len(items); log(f"{name}: {N} of {len(lg2)} two-hop prompts pass the gate")
ev_all = torch.load("/workspace/wdd/eval_ids_all.pt", weights_only=False)["qwen05"]["eval_ids"][:4]
ref = ref_stats(model, arch, ev_all, layers); gain, WU = final_readout(model, arch)
A, blk, typ, ends = lean_dictionary(arch, max(layers))
g = torch.Generator().manual_seed(7); R = torch.linalg.qr(torch.randn(arch.D, arch.D, generator=g))[0].to(DEV)
X_all, _ = last_states(model, arch, [it["ids"] for it in items], layers)
def ranks(scores, sets):
    order = scores.argsort(-1, descending=True); out = []
    for n, s in enumerate(sets):
        pos = torch.isin(order[n], torch.tensor(sorted(s), device=DEV)).nonzero()
        out.append(int(pos[0, 0]) + 1 if pos.numel() else 10 ** 9)
    return torch.tensor(out)
res = dict(model=name, n=N, k=K, layers=layers, readers={}, fidelity={}, causal={})
bsets = [it["bridge_ids"] for it in items]; asets = [it["answer_ids"] for it in items]
countries = sorted({f[1] for f in FACTS}); ctok = {c: first(" " + c) for c in countries}                  # candidate bridges
ctoks = torch.tensor(sorted(set(ctok.values())), device=DEV); true_c = torch.tensor([ctok[it["country"]] for it in items], device=DEV)
for L in layers:
    X = X_all[L]; mu = ref[L]["mu"]; Xc = X - mu; rms = X.pow(2).mean(-1).sqrt(); P = ref[L]["pcs"]; An = A[:ends[L]]
    sel, _, _ = omp(Xc, An, K, batch=64, record_err=False); cof, _ = refit(Xc, An, sel); words = An[sel]
    selr, _, _ = omp(Xc @ R.T, An, K, batch=64, record_err=False); cofr, _ = refit(Xc @ R.T, An, selr); wordsr = An[selr] @ R
    parts = (Xc @ P)[:, :, None] * P.T[None]
    S = dict(lens=lens_scores(X, gain, WU, rms), lens_centred=lens_scores(Xc, gain, WU, rms), pca16=pooled_word_scores(parts, torch.ones(N, K, device=DEV), gain, WU, rms),
             native16=pooled_word_scores(words, cof, gain, WU, rms), rotated16=pooled_word_scores(wordsr, cofr, gain, WU, rms))
    for rd, sc in S.items():
        rb, ra = ranks(sc, bsets), ranks(sc, asets)
        cs = sc[:, ctoks]; corder = ctoks[cs.argsort(-1, descending=True)]; crank = (corder == true_c[:, None]).float().argmax(-1) + 1
        top20 = sc.topk(20, dim=-1).indices; wrong = (torch.isin(top20, ctoks) & (top20 != true_c[:, None])).any(-1).float().mean().item()
        res["readers"][f"{rd}@{L}"] = {**{f"bridge@{n}": (rb <= n).float().mean().item() for n in NS}, **{f"answer@{n}": (ra <= n).float().mean().item() for n in NS},
                                      "country_rank@1": (crank <= 1).float().mean().item(), "country_rank@3": (crank <= 3).float().mean().item(), "wrong_country_in_top20": wrong}
        del sc
    fid = {}
    for rd, Xh in [("native16", mu + torch.einsum("nk,nkd->nd", cof, words)), ("rotated16", mu + torch.einsum("nk,nkd->nd", cofr, wordsr)), ("pca16", mu + parts.sum(1))]:
        ok = [splice_last(model, arch, items[n]["ids"], L, Xh[n:n + 1])[0].argmax().item() in items[n]["answer_ids"] for n in range(N)]
        fid[rd] = sum(ok) / N
    res["fidelity"][str(L)] = fid
    # causal: remove the native word whose own readout contains the bridge, against a random other selected word
    dl_b, dl_o = [], []; gen = torch.Generator().manual_seed(L); both = 0; bridge_any = 0; answer_any = 0
    for n in range(N):
        per = torch.stack([lens_scores(words[n, i:i + 1] * cof[n, i], gain, WU, rms[n:n + 1])[0] for i in range(K)])
        t5 = per.topk(5, dim=-1).indices
        hit = [i for i in range(K) if torch.isin(t5[i], torch.tensor(sorted(bsets[n]), device=DEV)).any()]
        ahit = [i for i in range(K) if torch.isin(t5[i], torch.tensor(sorted(asets[n]), device=DEV)).any()]
        bridge_any += bool(hit); answer_any += bool(ahit); both += any(i != j for i in hit for j in ahit)
        if not hit: continue
        ib = max(hit, key=lambda i: cof[n, i].abs().item()); others = [i for i in range(K) if i not in hit]
        if not others: continue
        io = others[int(torch.randint(0, len(others), (1,), generator=gen))]
        xs = torch.stack([X[n], X[n] - cof[n, ib] * words[n, ib], X[n] - cof[n, io] * words[n, io]])
        lp = torch.log_softmax(splice_last(model, arch, items[n]["ids"], L, xs), -1); a = items[n]["ans_tok"]
        dl_b.append((lp[1, a] - lp[0, a]).item()); dl_o.append((lp[2, a] - lp[0, a]).item())
    res["causal"][str(L)] = dict(n=len(dl_b), bridge_word=sum(dl_b) / max(len(dl_b), 1), other_word=sum(dl_o) / max(len(dl_o), 1),
                                 bridge_word_share=bridge_any / N, answer_word_share=answer_any / N, both_in_distinct_words=both / N)
    r = lambda rd, m: res["readers"][f"{rd}@{L}"][m]
    log(f"L{L}: bridge@20 lens {r('lens', 'bridge@20'):.2f} centred {r('lens_centred', 'bridge@20'):.2f} pca {r('pca16', 'bridge@20'):.2f} native {r('native16', 'bridge@20'):.2f} rotated {r('rotated16', 'bridge@20'):.2f} | "
        f"country@1 lens {r('lens', 'country_rank@1'):.2f} native {r('native16', 'country_rank@1'):.2f} pca {r('pca16', 'country_rank@1'):.2f} | wrong-country lens {r('lens', 'wrong_country_in_top20'):.2f} native {r('native16', 'wrong_country_in_top20'):.2f} | "
        f"answer@20 lens {r('lens', 'answer@20'):.2f} native {r('native16', 'answer@20'):.2f} | fidelity native {fid['native16']:.2f} rot {fid['rotated16']:.2f} pca {fid['pca16']:.2f} | "
        f"causal (n {len(dl_b)}) bridge word {res['causal'][str(L)]['bridge_word']:.2f} other {res['causal'][str(L)]['other_word']:.2f} | "
        f"words carrying bridge {bridge_any / N:.2f} answer {answer_any / N:.2f} both (distinct words) {both / N:.2f}")
    del sel, selr, words, wordsr, parts
best = max(layers, key=lambda L: res["readers"][f"native16@{L}"]["bridge@20"])
bl = lambda rd: max(res["readers"][f"{rd}@{L}"]["bridge@20"] for L in layers)
summ = (f"{name} n={N}: best-layer bridge@20 lens {bl('lens'):.2f} centred {bl('lens_centred'):.2f} pca16 {bl('pca16'):.2f} native16 {bl('native16'):.2f} (L{best}) rotated16 {bl('rotated16'):.2f} | "
        f"causal at L{best}: bridge word {res['causal'][str(best)]['bridge_word']:.2f} vs other {res['causal'][str(best)]['other_word']:.2f} (n {res['causal'][str(best)]['n']}) | fidelity at L{best} native {res['fidelity'][str(best)]['native16']:.2f}")
log(summ); record(f"e415_twohop_{name}", res, summ)
