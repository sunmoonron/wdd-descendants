"""e424: recalled attributes at the subject token, in the five original models. e420/e421 found, in Qwen2.5-7B, that native
words surface a recalled entity at the subject's last token in middle layers where both the logit lens and the model's
largest actual writes miss it. One-hop recall ("<landmark> is located in the country of" -> country, two worked
examples), kept per model when the model answers correctly; the state at the landmark's last token after blocks at a
quarter, half and three quarters of the depth is read by the logit lens, the native-word lens (16 OMP words), the 16
largest actual writes and a rotated control. Scored: the true country ranked first among the candidate countries, and
in the top 20 of the full vocabulary. Pre-registered: in models with at least 30 correct prompts, the native words rank
the country first more often than the lens and the actual writes at some depth."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lr_common import *
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
name = sys.argv[1]; model, tok, fam = load_model(name); arch = Arch(model, fam)
Ls = [arch.NB // 4, arch.NB // 2, (3 * arch.NB) // 4]; maxL = max(Ls)
enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]; first = lambda s: enc(s)[0]
shots = "The Belém Tower is located in the country of Portugal.\nVigeland Park is located in the country of Norway.\n"
items = []
for lm, country, *_ in FACTS:
    subj = lm[0].upper() + lm[1:]; ids = enc(shots + subj + " is located in the country of"); spos = len(enc(shots + subj)) - 1
    items.append(dict(ids=torch.tensor(ids), spos=spos, country=country, ctok=first(" " + country)))
with torch.no_grad(): ok = [model(it["ids"][None].to(DEV)).logits[0, -1].argmax().item() == it["ctok"] for it in items]
items = [it for it, o in zip(items, ok) if o]; N = len(items); log(f"{name}: {N} of {len(ok)} recall prompts answered correctly")
rd = Reader(model, arch, maxL); act = Actual(arch, maxL); E = eval_ids(name)
g = torch.Generator().manual_seed(7); R = torch.linalg.qr(torch.randn(arch.D, arch.D, generator=g))[0].to(DEV)
st_ref, _, _, _ = capture(model, arch, E[:4].to(DEV), Ls, 0); mu = {L: st_ref[L][:, 1:].reshape(-1, arch.D).mean(0) for L in Ls}
cands = torch.tensor(sorted({first(" " + f[1]) for f in FACTS}), device=DEV); tgt = torch.tensor([it["ctok"] for it in items], device=DEV)
Xs = {L: [] for L in Ls}; Hs = {b: [] for b in range(maxL + 1)}; Zs = {b: [] for b in range(maxL + 1)}; toks, poss = [], []
for it in items:
    st, H, Z, _ = capture(model, arch, it["ids"][None].to(DEV), Ls, maxL); p = it["spos"]
    for L in Ls: Xs[L].append(st[L][0, p])
    for b in range(maxL + 1): Hs[b].append(H[b][0, p]); Zs[b].append(Z[b][0, p])
    toks.append(it["ids"][p]); poss.append(p)
res = dict(model=name, n=N, layers=Ls, readers={})
if N:
    Hs = {b: torch.stack(v) for b, v in Hs.items()}; Zs = {b: torch.stack(v) for b, v in Zs.items()}; toks = torch.stack(toks).to(DEV); poss = torch.tensor(poss, device=DEV)
    for L in Ls:
        X = torch.stack(Xs[L]); rms = X.pow(2).mean(-1).sqrt(); Xc = X - mu[L]
        parts = dict(native16=rd.native_parts(Xc, L)[0], rotated16=rd.native_parts(Xc, L, R=R)[0], actual16=act.top(Hs, Zs, toks, poss, L))
        scs = dict(lens=rd.lens(X, rms), **{k: rd.pooled(v, rms) for k, v in parts.items()})
        for r_, sc in scs.items():
            cs = sc[:, cands]; crank = (cands[cs.argsort(-1, descending=True)] == tgt[:, None]).float().argmax(-1) + 1
            res["readers"][f"{r_}@{L}"] = {"country@1": (crank <= 1).float().mean().item(), "country@3": (crank <= 3).float().mean().item(), "top20": (rank_of(sc, tgt) <= 20).float().mean().item()}
        q = lambda r_, m: res["readers"][f"{r_}@{L}"][m]
        log(f"{name} L{L}: country@1 lens {q('lens', 'country@1'):.2f} native {q('native16', 'country@1'):.2f} actual {q('actual16', 'country@1'):.2f} rot {q('rotated16', 'country@1'):.2f} | "
            f"top20 lens {q('lens', 'top20'):.2f} native {q('native16', 'top20'):.2f} actual {q('actual16', 'top20'):.2f}")
    bl = lambda r_, m: max(res["readers"][f"{r_}@{L}"][m] for L in Ls)
    summ = f"{name} n={N}: best-depth country@1 lens {bl('lens', 'country@1'):.2f} native16 {bl('native16', 'country@1'):.2f} actual16 {bl('actual16', 'country@1'):.2f} rotated {bl('rotated16', 'country@1'):.2f} | top20 lens {bl('lens', 'top20'):.2f} native {bl('native16', 'top20'):.2f} actual {bl('actual16', 'top20'):.2f}"
else: summ = f"{name}: no recall prompt answered correctly"
log(summ); record(f"e424_recall_{name}", res, summ)
