"""e568 (session 102): the invented-name lead of e560, one controlled study. Invented country names drove the last prompt
position to a larger S than real ones (2.94 against 2.39 at block 18). Which row, and is it one row? Prompts at blocks
12, 18 and 22 of Pythia-410m: real countries (e560's 160) against 60 invented names; real multi-token countries against
invented names of the same token count (the rarity and morphology control); three frames ("The capital of X is",
"X is a country in", "The language spoken in X is"); and real against invented person names ("The birthplace of X is").
For every prompt the top row (the largest projection over the Pile-calibrated floor) and its S; per group the modal
top row and its share; the top row's identity across frames and categories; its Pile context (the tokens at its
over-the-floor positions). Then the causal check: the dominant invented-name row's write column zeroed, against a random
word's, and the change in S, in the top-1 probability and the entropy on invented and on real prompts. Pre-registered
(probabilities are honest guesses):
 N1 (0.6) one row tops most invented-name prompts (modal share over 0.5) against a spread for real names;
 N2 (0.5) the same row tops real multi-token rare countries of matched token count: it is a rarity row, not a novelty row;
 N3 (0.5) the same row tops invented person names (a class row, not a country row);
 N4 (0.5) zeroing it lowers the confidence on invented names more than on real ones."""
from s101_common import *
from e560_facts import CAPS, FAKE
FAKE2 = FAKE + ["Ardovania", "Kesh Molran", "Belthorpe", "Vranik", "Ostomere", "Quillandria", "Tarsovia", "Emberlyn", "Gorshak", "Naveloria", "Dremmoria", "Sylvantis", "Barrowgate", "Ivoryn", "Cassombria", "Zul Tharak", "Merrowind", "Pelagosta", "Wintherland", "Oxmoor",
                "Feldaria", "Kronvale", "Ulmarra", "Tessendor", "Briskovia", "Halvoren", "Mistrania", "Corvallen", "Dunmarrow", "Yelvoria"]
REAL_PEOPLE = ["Albert Einstein", "Marie Curie", "Isaac Newton", "Charles Darwin", "Napoleon Bonaparte", "Winston Churchill", "Leonardo da Vinci", "William Shakespeare", "Wolfgang Amadeus Mozart", "Abraham Lincoln", "Mahatma Gandhi", "Nelson Mandela", "Pablo Picasso", "Ludwig van Beethoven", "Galileo Galilei", "Julius Caesar", "Cleopatra", "Karl Marx", "Sigmund Freud", "Nikola Tesla",
               "Thomas Edison", "Vincent van Gogh", "Frida Kahlo", "Martin Luther King", "John F. Kennedy", "Queen Victoria", "Elvis Presley", "Michael Jackson", "Bill Gates", "Steve Jobs"]
FAKE_PEOPLE = ["Zorbanius Krell", "Thalmira Vex", "Orrin Delcastro", "Vesna Kortalik", "Brannoch Elderwyn", "Ilse Marrowick", "Tobiah Fenwright", "Selka Drummond", "Quentin Ashvale", "Marisol Tenbrook", "Halvard Ostrenne", "Petra Quillane", "Dorian Malvechio", "Anneliese Vorkuta", "Casimir Roldane", "Yusra Belmonde", "Leopold Ferrancz", "Nadia Strevlin", "Emrys Talloway", "Greta Holmquist",
               "Rurik Sandoval", "Imogen Falkreath", "Silas Verhoeven", "Beatrix Landorra", "Osric Wendelmar", "Katya Brelsford", "Fenwick Adalorne", "Mira Costovan", "Alaric Pemberly", "Sabine Norrevik"]
t0 = time.time(); BL = [12, 18, 22]; model, tok, fam = load_model("pythia410"); arch = Arch(model, fam)
P = {B: lm_states("pythia410", B=B, model=model) for B in BL}; cal = {B: floor_calibration(P[B]["U"], P[B]["A"]) for B in BL}; ST = {B: stats(P[B]["U"], P[B]["A"], K) for B in BL}
def run(prompts):
    X = {B: [] for B in BL}; LG = []
    for pr in prompts:
        ids = torch.tensor([tok(pr)["input_ids"]], device=DEV); cap = {}
        hs = [arch.layers[B].register_forward_hook((lambda B_: lambda m, i, o: cap.__setitem__(B_, (o[0] if isinstance(o, tuple) else o)[0, -1].detach().float()))(B)) for B in BL]
        try:
            with torch.no_grad(): lg = model(ids).logits[0, -1].float()
        finally: [h.remove() for h in hs]
        for B in BL: X[B].append(cap[B])
        LG.append(lg)
    return {B: torch.stack(X[B]) for B in BL}, torch.stack(LG)
def top_rows(X, B):
    U = unitr(X - P[B]["mu"]); L = floor_of(U, cal[B]); ratio = (U @ P[B]["A"].T).abs() / L[:, None]; v, i = ratio.topk(2, dim=1); return i[:, 0].cpu(), v[:, 0].cpu(), v[:, 1].cpu()
def ntok(name): return len(tok(" " + name)["input_ids"])
groups = {"real countries": [c for c, _ in CAPS], "invented countries": FAKE2, "real people": REAL_PEOPLE, "invented people": FAKE_PEOPLE}
frames = {"capital": "The capital of {} is", "country_in": "{} is a country in", "language": "The language spoken in {} is", "birthplace": "The birthplace of {} is"}
res = dict(blocks={B: {} for B in BL}, ntok={g: [ntok(n) for n in names] for g, names in groups.items()}); TOP = {}
for g, names in groups.items():
    for fr, tmpl in frames.items():
        if ("people" in g) != (fr == "birthplace"): continue
        X, LG = run([tmpl.format(n) for n in names]); pr = LG.softmax(1); conf = pr.max(1).values.cpu(); ent = -(pr * pr.clamp_min(1e-12).log()).sum(1).cpu()
        for B in BL:
            r, s1, s2 = top_rows(X[B], B); modal = int(torch.bincount(r).argmax()); share = float((r == modal).float().mean()); TOP[(g, fr, B)] = r
            res["blocks"][B][f"{g} | {fr}"] = dict(n=len(names), median_S=float(s1.median()), median_second=float(s2.median()), modal_row=modal, modal_share=share, n_distinct_top_rows=int(r.unique().numel()), confidence=float(conf.median()), entropy=float(ent.median()))
            log(f"block {B} {g} [{fr}]: S {float(s1.median()):.2f} (second {float(s2.median()):.2f}), top rows {int(r.unique().numel())} distinct, modal row {modal} (block {modal // arch.DFF}, neuron {modal % arch.DFF}) at share {share:.2f}; confidence {float(conf.median()):.2f}")
# matched token counts: real multi-token countries vs invented of the same count, capital frame, block 18
B = 18; rc = [c for c, _ in CAPS]; nt_r = torch.tensor(res["ntok"]["real countries"]); nt_f = torch.tensor(res["ntok"]["invented countries"]); Xr, _ = run([frames["capital"].format(n) for n in rc]); Xf, _ = run([frames["capital"].format(n) for n in FAKE2])
rr, sr, _ = top_rows(Xr[B], B); rf, sf, _ = top_rows(Xf[B], B); res["matched"] = {}
for k in sorted(set(nt_r.tolist()) & set(nt_f.tolist())):
    mr, mf = nt_r == k, nt_f == k
    if mr.sum() < 3 or mf.sum() < 3: continue
    res["matched"][k] = dict(n_real=int(mr.sum()), n_fake=int(mf.sum()), S_real=float(sr[mr].median()), S_fake=float(sf[mf].median()), modal_real=int(torch.bincount(rr[mr]).argmax()), modal_fake=int(torch.bincount(rf[mf]).argmax()), share_real_top_is_fake_modal=float((rr[mr] == int(torch.bincount(rf).argmax())).float().mean()))
    log(f"{k} tokens: real {int(mr.sum())} S {float(sr[mr].median()):.2f} (modal row {int(torch.bincount(rr[mr]).argmax())}), invented {int(mf.sum())} S {float(sf[mf].median()):.2f} (modal row {int(torch.bincount(rf[mf]).argmax())}); real prompts topped by the invented modal row {res['matched'][k]['share_real_top_is_fake_modal']:.2f}")
# identity of the invented-name row across frames and categories, and its Pile context
dom = int(torch.bincount(TOP[("invented countries", "capital", B)]).argmax()); res["dominant_row"] = dict(row=dom, block=dom // arch.DFF, neuron=dom % arch.DFF)
for key, r in TOP.items():
    if key[2] == B: res["dominant_row"][f"share_top_in {key[0]} | {key[1]}"] = float((r == dom).float().mean())
ratio_pile = ST[B]["ratio"][:, dom].float(); over = torch.nonzero(ratio_pile > 1)[:, 0]; toks = P[B]["tok"].cpu()[over]
cnts = torch.bincount(toks, minlength=len(tok)); topt = cnts.topk(12).indices; res["dominant_row"]["pile_over_floor_positions"] = int(over.numel()); res["dominant_row"]["pile_S"] = float(ST[B]["S"][dom]); res["dominant_row"]["is_pile_word"] = bool(wordset(ST[B]["usage"])[dom])
res["dominant_row"]["pile_top_tokens"] = [(tok.decode([int(t)]), int(cnts[t])) for t in topt if cnts[t] > 0]
log(f"dominant invented-name row {dom} (block {dom // arch.DFF}, neuron {dom % arch.DFF}): tops " + ", ".join(f"{k.split(' ', 1)[1]} {v:.2f}" for k, v in res["dominant_row"].items() if k.startswith("share_top_in")) + f"; on the Pile: S {res['dominant_row']['pile_S']:.2f}, over the floor at {over.numel()} positions, a word: {res['dominant_row']['is_pile_word']}, top tokens {res['dominant_row']['pile_top_tokens'][:8]}")
# ablation of the dominant row's write column, against a random Pile word's
def ablate(row):
    b, j = row // arch.DFF, row % arch.DFF; w = arch.layers[b].mlp.dense_4h_to_h.weight; saved = w[:, j].clone()
    with torch.no_grad(): w[:, j] = 0
    return lambda: w.__setitem__((slice(None), j), saved)
def probe(names, tmpl):
    X, LG = run([tmpl.format(n) for n in names]); pr = LG.softmax(1); r, s1, _ = top_rows(X[B], B); return dict(S=float(s1.median()), conf=float(pr.max(1).values.median()), ent=float((-(pr * pr.clamp_min(1e-12).log()).sum(1)).median()), top1=LG.argmax(1).cpu())
base = {g: probe(groups[g], frames["capital"]) for g in ("real countries", "invented countries")}; res["ablation"] = {}
gw = torch.nonzero(wordset(ST[B]["usage"]))[:, 0]; rnd = int(gw[torch.randint(0, gw.numel(), (1,), generator=torch.Generator().manual_seed(3))])
for tag, row in (("dominant", dom), ("random_word", rnd)):
    restore = ablate(row); after = {g: probe(groups[g], frames["capital"]) for g in ("real countries", "invented countries")}; restore()
    res["ablation"][tag] = {g: dict(dS=after[g]["S"] - base[g]["S"], dconf=after[g]["conf"] - base[g]["conf"], dent=after[g]["ent"] - base[g]["ent"], top1_changed=float((after[g]["top1"] != base[g]["top1"]).float().mean())) for g in after}
    log(f"ablating {tag} row {row}: " + "; ".join(f"{g}: S {res['ablation'][tag][g]['dS']:+.2f}, confidence {res['ablation'][tag][g]['dconf']:+.3f}, entropy {res['ablation'][tag][g]['dent']:+.3f}, top-1 changed {res['ablation'][tag][g]['top1_changed']:.2f}" for g in after))
b18 = res["blocks"][18]; d = res["dominant_row"]; ab = res["ablation"]
summ = (f"invented names at block 18: S invented/real countries {b18['invented countries | capital']['median_S']:.2f}/{b18['real countries | capital']['median_S']:.2f}, top rows {b18['invented countries | capital']['n_distinct_top_rows']}/{b18['real countries | capital']['n_distinct_top_rows']} distinct, modal share {b18['invented countries | capital']['modal_share']:.2f}/{b18['real countries | capital']['modal_share']:.2f}; "
        f"dominant row {dom} tops invented countries {d.get('share_top_in invented countries | capital', 0):.2f} (capital), {d.get('share_top_in invented countries | country_in', 0):.2f} (country in), {d.get('share_top_in invented countries | language', 0):.2f} (language), invented people {d.get('share_top_in invented people | birthplace', 0):.2f}, real people {d.get('share_top_in real people | birthplace', 0):.2f}, real countries {d.get('share_top_in real countries | capital', 0):.2f}; "
        f"matched token counts: " + ", ".join(f"{k} tokens real/invented S {v['S_real']:.2f}/{v['S_fake']:.2f} (real topped by the invented row {v['share_real_top_is_fake_modal']:.2f})" for k, v in res["matched"].items()) + f"; on the Pile the row is {'a word' if d['is_pile_word'] else 'not a word'} (S {d['pile_S']:.2f}, {d['pile_over_floor_positions']} positions over the floor), top tokens {[t for t, _ in d['pile_top_tokens'][:5]]}; "
        f"ablating it: invented S {ab['dominant']['invented countries']['dS']:+.2f}, confidence {ab['dominant']['invented countries']['dconf']:+.3f}, top-1 changed {ab['dominant']['invented countries']['top1_changed']:.2f}; real S {ab['dominant']['real countries']['dS']:+.2f}, confidence {ab['dominant']['real countries']['dconf']:+.3f}, top-1 changed {ab['dominant']['real countries']['top1_changed']:.2f} (random word: invented {ab['random_word']['invented countries']['dconf']:+.3f}, real {ab['random_word']['real countries']['dconf']:+.3f}) | {time.time() - t0:.0f}s")
log(summ); record("e568_invented_names", res, summ)
