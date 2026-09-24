"""e421b: the fair version of e421. e421 ranked actual writes by raw norm; in a model with massive-activation writers the
top of that ranking can be occupied by huge, content-free writes, while the native words were read on the mean-centred
state. Here actual writes are centred too: each component's write minus its mean write over natural text (neuron:
(h - mean h) w; head: (z - mean z) W_O; embedding minus the mean embedding), ranked by the norm of that deviation, the
16 largest read by the logit lens and max-pooled ("actual16c"); raw-ranked as in e421 for comparison. Pre-registered:
if the native words keep their advantage over centred actual writes (country first), e421 was not an artefact of
massive writes.
e421 notes follow.
e421: is the native-word lens just the value-vector (sub-update) reading? Geva et al. 2022 read a feed-forward layer
by projecting its activation-weighted value vectors (down-projection columns) through the unembedding; the whole
set of a model's actual writes (every MLP neuron's a_i w_i, every head's output, the token embedding) is the
established decomposition. At the subject's last token (where e420 found native words surfacing the two-hop bridge
at blocks 12-20), compare: the logit lens; the native-word lens (16 OMP words, as e420); the 16 largest actual
writes among MLP neurons, heads and the embedding in blocks up to L ("actual16"); and the 16 largest MLP sub-updates
alone ("geva16"), each component read by the logit lens and max-pooled. Pre-registered: the native words surface
the bridge more often than both actual-write readings (e391: the sparse code is a re-description, not the largest
writes); if the actual writes do as well, the native-word lens is the sub-update reading.
e420 notes follow.
e420: the same readers at the subject's last token. e415 read the final position, where the bridge appears only at
blocks 24-26 for every reader; the literature on two-hop recall locates bridge resolution at the subject tokens in
middle layers. Identical to e415 except that states are read at the last token of the landmark mention in the query
(the causal check removes the bridge word there and measures the answer at the final position).
Original e415 notes follow.
e415: can the model's own words surface its hidden intermediate? After WorkspaceBench (LessWrong 2026): workspace readers
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
name = sys.argv[1] if len(sys.argv) > 1 else "qwen7"; K = int(sys.argv[2]) if len(sys.argv) > 2 else 16; NS = [1, 5, 10, 20, 50]
model, tok, fam = load_bf16(name); arch = Arch(model, fam); layers = list(range(4, arch.NB - 1, 2)) if K == 16 else list(range(8, 22, 2))
first = lambda s: tok(s, add_special_tokens=False)["input_ids"][0]
def enc(text): return torch.tensor(tok(text, add_special_tokens=False)["input_ids"])
items = []
for lm, country, capital, lang, cur in FACTS:
    for rel, ans in [("capital", capital), ("language", lang), ("currency", cur)]:
        if ans is None: continue
        q, _ = TPL[rel]; shots = "".join(f"{q.format(a)} {b}.\n" for a, b in SHOT[rel])
        head = shots + q.split("{}")[0] + lm
        items.append(dict(landmark=lm, rel=rel, country=country, answer=ans, ids=enc(shots + q.format(lm)), spos=len(enc(head)) - 1,
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
def subject_states(items, layers):
    out = {l: [] for l in layers}
    for it in items:
        cap = {}; p = it["spos"]
        hs = [arch.layers[l].register_forward_hook((lambda l_: lambda m, i, o: cap.__setitem__(l_, (o[0] if isinstance(o, tuple) else o)[0, p].float()))(l)) for l in layers]
        try:
            with torch.no_grad(): model(it["ids"][None].to(DEV))
        finally: [h.remove() for h in hs]
        for l in layers: out[l].append(cap[l])
    return {l: torch.stack(v) for l, v in out.items()}

layers = [10, 12, 14, 16, 18, 20]; maxL = max(layers)
cap_h = {b: [] for b in range(maxL + 1)}; cap_z = {b: [] for b in range(maxL + 1)}; cap_x = {l: [] for l in layers}
for it in items:
    p = it["spos"]; hs = []
    for b in range(maxL + 1):
        hs.append(arch.mlp_lin(b).register_forward_pre_hook((lambda b_: lambda m, a: cap_h[b_].append(a[0][0, p].float()))(b)))
        hs.append(arch.attn_lin(b).register_forward_pre_hook((lambda b_: lambda m, a: cap_z[b_].append(a[0][0, p].float()))(b)))
    for l in layers: hs.append(arch.layers[l].register_forward_hook((lambda l_: lambda m, i, o: cap_x[l_].append((o[0] if isinstance(o, tuple) else o)[0, p].float()))(l)))
    try:
        with torch.no_grad(): model(it["ids"][None].to(DEV))
    finally: [h.remove() for h in hs]
H = {b: torch.stack(v) for b, v in cap_h.items()}; Z = {b: torch.stack(v) for b, v in cap_z.items()}; X_all = {l: torch.stack(v) for l, v in cap_x.items()}
hm, zm = {}, {}
hs = [arch.mlp_lin(b).register_forward_pre_hook((lambda b_: lambda m, a: hm.setdefault(b_, []).append(a[0][0, 1:].float().mean(0)))(b)) for b in range(maxL + 1)]
hs += [arch.attn_lin(b).register_forward_pre_hook((lambda b_: lambda m, a: zm.setdefault(b_, []).append(a[0][0, 1:].float().mean(0)))(b)) for b in range(maxL + 1)]
try:
    with torch.no_grad():
        for s_ in range(ev_all.shape[0]): model(ev_all[s_:s_ + 1].to(DEV))
finally: [h.remove() for h in hs]
Hbar = {b: torch.stack(v).mean(0) for b, v in hm.items()}; Zbar = {b: torch.stack(v).mean(0) for b, v in zm.items()}
Ebar = arch.emb[0].detach().float()[ev_all.flatten().to(DEV)].mean(0)
WD = {b: arch.layers[b].mlp.down_proj.weight for b in range(maxL + 1)}                    # bf16 [D, DFF], column i = neuron i's write
colnorm = {b: WD[b].float().norm(dim=0) for b in range(maxL + 1)}                         # [DFF] write norms
WO = {b: arch.layers[b].self_attn.o_proj.weight.float() for b in range(maxL + 1)}         # fp32 [D, NH*HD], cached once (~1 GB)
emb = arch.emb[0].detach().float(); tok_at = torch.stack([it["ids"][it["spos"]] for it in items]).to(DEV)
def actual_components(n, L, mlp_only, centred=False):
    """the 16 largest actual writes at item n's subject position among blocks 0..L: [16, D]"""
    hv = {b: (H[b][n] - Hbar[b]) if centred else H[b][n] for b in range(L + 1)}
    mags = torch.cat([hv[b].abs() * colnorm[b] for b in range(L + 1)]); top = mags.topk(64).indices
    DFF = colnorm[0].numel(); vecs = [hv[int(t) // DFF][int(t) % DFF] * WD[int(t) // DFF][:, int(t) % DFF].float() for t in top]
    V = torch.stack(vecs)
    if not mlp_only:
        heads = []
        for b in range(L + 1):
            z = ((Z[b][n] - Zbar[b]) if centred else Z[b][n]).view(arch.NH, arch.HD); Wo = WO[b].view(arch.D, arch.NH, arch.HD)
            heads.append(torch.einsum("hd,Dhd->hD", z, Wo))                                    # [NH, D] per-head writes
        V = torch.cat([V, torch.cat(heads), (emb[tok_at[n]] - (Ebar if centred else 0))[None]])
    return V[V.norm(dim=-1).topk(16).indices]
def ranks(scores, sets):
    order = scores.argsort(-1, descending=True); out = []
    for n, s in enumerate(sets):
        pos = torch.isin(order[n], torch.tensor(sorted(s), device=DEV)).nonzero()
        out.append(int(pos[0, 0]) + 1 if pos.numel() else 10 ** 9)
    return torch.tensor(out)
bsets = [it["bridge_ids"] for it in items]
countries = sorted({f[1] for f in FACTS}); ctok = {c: first(" " + c) for c in countries}
ctoks = torch.tensor(sorted(set(ctok.values())), device=DEV); true_c = torch.tensor([ctok[it["country"]] for it in items], device=DEV)
res = dict(model=name, n=N, k=K, layers=layers, readers={})
for L in layers:
    X = X_all[L]; mu = ref[L]["mu"]; Xc = X - mu; rms = X.pow(2).mean(-1).sqrt(); An = A[:ends[L]]
    sel, _, _ = omp(Xc, An, K, batch=64, record_err=False); cof, _ = refit(Xc, An, sel); words = An[sel]
    act = torch.stack([actual_components(n, L, False) for n in range(N)]); gv = torch.stack([actual_components(n, L, False, centred=True) for n in range(N)])
    S = dict(lens=lens_scores(X, gain, WU, rms), native16=pooled_word_scores(words, cof, gain, WU, rms),
             actual16=pooled_word_scores(act, torch.ones(N, 16, device=DEV), gain, WU, rms), actual16c=pooled_word_scores(gv, torch.ones(N, 16, device=DEV), gain, WU, rms))
    for rd, sc in S.items():
        rb = ranks(sc, bsets); cs = sc[:, ctoks]; corder = ctoks[cs.argsort(-1, descending=True)]; crank = (corder == true_c[:, None]).float().argmax(-1) + 1
        res["readers"][f"{rd}@{L}"] = {"bridge@20": (rb <= 20).float().mean().item(), "bridge@50": (rb <= 50).float().mean().item(),
                                      "country@1": (crank <= 1).float().mean().item(), "country@3": (crank <= 3).float().mean().item()}
    r = lambda rd, m: res["readers"][f"{rd}@{L}"][m]
    log(f"L{L}: bridge@20 lens {r('lens', 'bridge@20'):.2f} native {r('native16', 'bridge@20'):.2f} actual {r('actual16', 'bridge@20'):.2f} centred {r('actual16c', 'bridge@20'):.2f} | "
        f"country@1 lens {r('lens', 'country@1'):.2f} native {r('native16', 'country@1'):.2f} actual {r('actual16', 'country@1'):.2f} centred {r('actual16c', 'country@1'):.2f}")
    del sel, words, act, gv, S
bl = lambda rd, m: max(res["readers"][f"{rd}@{L}"][m] for L in layers)
summ = (f"{name} n={N} subject position, best over blocks 10-20: bridge@20 lens {bl('lens', 'bridge@20'):.2f} native16 {bl('native16', 'bridge@20'):.2f} actual16 {bl('actual16', 'bridge@20'):.2f} actual16c {bl('actual16c', 'bridge@20'):.2f} | "
        f"country@1 lens {bl('lens', 'country@1'):.2f} native16 {bl('native16', 'country@1'):.2f} actual16 {bl('actual16', 'country@1'):.2f} actual16c {bl('actual16c', 'country@1'):.2f}")
log(summ); record(f"e421b_centredwrites_{name}", res, summ)
