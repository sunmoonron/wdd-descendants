"""e473: word algebra. Do native handles for two independent attributes of one word act independently, and do they
compose? From an external review ("swap red only, swap small only, swap both"); e469-e470 established that native
words are interchange coordinates for a noun's identity during its writing window.
Task: few-shot word translation into English of kinship words that carry two attributes in one word, gender (g) and
generation (a): king/queen/prince/princess, father/mother/son/daughter, man/woman/boy/girl, uncle/aunt/nephew/niece,
grandfather/grandmother/grandson/granddaughter, in French, Spanish and German. Bare words, so no article marks gender.
The readout is a forced choice among the four English words of the quadruple (first tokens; quadruples whose first
tokens collide are dropped). Three few-shot contexts per word; a quadruple in a context is used only if all four words
are translated correctly.
For a base word w (g, a) the sources are w^g (gender flipped), w^a (generation flipped) and w^ga (both). At the last
token of the source-language word, blocks 0..L, the e469 interchange x <- x + P (x_src - x) is recomputed at each block.
Bases at each block (k = 4 and 16 words each):
- G: native words chosen by OMP for x(w^g) - x(w); A: for x(w^a) - x(w); B: for x(w^ga) - x(w) (e469's direct basis);
- GA: the union of G and A;
- the same for rotated words, and task PCA (the top principal directions of all items' gender, or generation,
  differences at that block).
Interventions and the answer expected if the handles factorise:
- full state from w^ga -> w^ga; B toward w^ga -> w^ga;
- G toward w^ga -> w^g (gender changes, generation kept); A toward w^ga -> w^a;
- GA toward w^ga -> w^ga;
- compose: G toward w^g, then A toward w^a -> w^ga (no source has both changes).
Also, in behaviour space, the gender score s_G (mean log-probability of the opposite-gender candidates minus the
same-gender ones) and the generation score s_A: the cross-talk |d s_A| / |d s_G| of G (and the reverse for A), and
additivity, d s(GA) against d s(G) + d s(A).
Models (argument): qwen05, smollm2.
Pre-registered (honest guesses):
- full interchange reaches w^ga in at least 90% (0.8);
- the gender handle alone gives w^g in at least half of the items (0.4);
- compose reaches w^ga in at least half the cases the direct basis B does (0.4);
- native factorises better than rotated words (G gives w^g more often) (0.6);
- behavioural additivity holds within 25% (0.4)."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
QUADS = {
 "en": [["king", "queen", "prince", "princess"], ["father", "mother", "son", "daughter"], ["man", "woman", "boy", "girl"], ["uncle", "aunt", "nephew", "niece"], ["grandfather", "grandmother", "grandson", "granddaughter"]],
 "fr": [["roi", "reine", "prince", "princesse"], ["père", "mère", "fils", "fille"], ["homme", "femme", "garçon", "fille"], ["oncle", "tante", "neveu", "nièce"], ["grand-père", "grand-mère", "petit-fils", "petite-fille"]],
 "es": [["rey", "reina", "príncipe", "princesa"], ["padre", "madre", "hijo", "hija"], ["hombre", "mujer", "niño", "niña"], ["tío", "tía", "sobrino", "sobrina"], ["abuelo", "abuela", "nieto", "nieta"]],
 "de": [["König", "Königin", "Prinz", "Prinzessin"], ["Vater", "Mutter", "Sohn", "Tochter"], ["Mann", "Frau", "Junge", "Mädchen"], ["Onkel", "Tante", "Neffe", "Nichte"], ["Großvater", "Großmutter", "Enkel", "Enkelin"]]}
SHOTS = {"fr": [("chien", "dog"), ("maison", "house"), ("livre", "book"), ("eau", "water"), ("arbre", "tree"), ("voiture", "car")],
         "es": [("perro", "dog"), ("casa", "house"), ("libro", "book"), ("agua", "water"), ("árbol", "tree"), ("coche", "car")],
         "de": [("Hund", "dog"), ("Haus", "house"), ("Buch", "book"), ("Wasser", "water"), ("Baum", "tree"), ("Auto", "car")]}
LN = {"fr": "French", "es": "Spanish", "de": "German"}; LANGS = ["fr", "es", "de"]; NT = 3; KS = [4, 16]
name = sys.argv[1]; model, tok, fam = load_model(name); arch = Arch(model, fam); L = arch.NB // 2; D = arch.D
A_, lab = build_dictionary(arch, blocks=list(range(L + 1))); blk = lab["block"].to(DEV); Au = unitr(A_); Ar = unitr(rotate(A_, seed=7)); del A_
ends = {b: int((blk <= b).sum()) for b in range(L + 1)}
def prompt(lg, w, t):
    (w1, e1), (w2, e2) = SHOTS[lg][2 * t], SHOTS[lg][2 * t + 1]
    p = f"{LN[lg]}: {w1}\nEnglish: {e1}\n{LN[lg]}: {w2}\nEnglish: {e2}\n{LN[lg]}: {w}\nEnglish:"; a = p.rindex(w); b_ = a + len(w)
    enc = tok(p, add_special_tokens=False, return_offsets_mapping=True); ids = torch.tensor(enc["input_ids"], device=DEV)[None]
    return ids, [i for i, (x0, x1) in enumerate(enc["offset_mapping"]) if x1 > a and x0 < b_][-1]
def run(ids, p, cand, fns=None):
    cap, hs = {}, []
    for b in range(L + 1):
        def hk(m, i, o, b=b):
            xo = out_of(o)
            if fns is None or b not in fns: cap[b] = xo[0, p].detach().float(); return None
            y = xo.clone(); y[0, p] = fns[b](y[0, p].float()).to(y.dtype); cap[b] = y[0, p].detach().float()
            return (y,) + tuple(o[1:]) if isinstance(o, tuple) else y
        hs.append(arch.layers[b].register_forward_hook(hk))
    try:
        with torch.no_grad(): lg = model(ids).logits[0, -1].float()
    finally: [h.remove() for h in hs]
    return torch.log_softmax(lg[cand], -1), torch.stack([cap[b] for b in range(L + 1)])
items, groups = [], 0                                               # an item: a base word with its three sources
for lg in LANGS:
    for q in range(len(QUADS["en"])):
        cand = [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in QUADS["en"][q]]
        if len(set(cand)) < 4: continue
        cand = torch.tensor(cand, device=DEV)
        for t in range(NT):
            runs = [prompt(lg, QUADS[lg][q][i], t) for i in range(4)]; outs = [run(ids, p, cand) for ids, p in runs]
            if not all(int(lp.argmax()) == i for i, (lp, _) in enumerate(outs)): continue
            groups += 1
            for i in range(4): items.append(dict(lg=lg, q=q, t=t, i=i, cand=cand, ids=runs[i][0], p=runs[i][1], lp=outs[i][0], X={j: outs[i ^ j][1] for j in range(4)}))
n = len(items); log(f"{name}: {groups} quadruple-contexts fully correct, {n} items")
dif = {j: torch.stack([it["X"][j] - it["X"][0] for it in items], 1) for j in (1, 2, 3)}     # [L+1, n, D]: 1 = gender, 2 = generation, 3 = both
basis = {}
def qr(M): return torch.linalg.qr(M)[0]
def orth(M):                                                          # rank-safe orthonormal basis of the columns (zero columns where dependent)
    U, S, _ = torch.linalg.svd(M, full_matrices=False); return U * (S > 1e-4 * S[:, :1])[:, None, :]
for b in range(L + 1):
    for kind, Dct in (("native", Au), ("rotated", Ar)):
        for j, nm in ((1, "G"), (2, "A"), (3, "B")):
            sel, _, _ = omp(dif[j][b], Dct[:ends[b]], max(KS), batch=256, record_err=False)
            for k in KS: basis[(kind, nm, k, b)] = qr(Dct[sel[:, :k]].transpose(1, 2))
    for j, nm in ((1, "G"), (2, "A"), (3, "B")):
        ev_, U_ = torch.linalg.eigh(torch.cov(dif[j][b].T.double(), correction=0)); Up = U_.flip(-1).float()
        for k in KS: basis[("pca", nm, k, b)] = Up[:, :k][None].expand(n, -1, -1)
    for kind in ("native", "rotated", "pca"):
        for k in KS: basis[(kind, "GA", k, b)] = orth(torch.cat([basis[(kind, "G", k, b)], basis[(kind, "A", k, b)]], 2))
BL = list(range(L + 1))
def proj(Q, x, xs): return x + Q @ (Q.T @ (xs - x))
def scores(lp, i):
    g_opp = [j for j in range(4) if (j ^ i) & 1]; g_same = [j for j in range(4) if not (j ^ i) & 1]
    a_opp = [j for j in range(4) if (j ^ i) & 2]; a_same = [j for j in range(4) if not (j ^ i) & 2]
    return float(lp[g_opp].mean() - lp[g_same].mean()), float(lp[a_opp].mean() - lp[a_same].mean())
def evaluate(kind, nm, k):
    outcome = torch.zeros(4); dG, dA = [], []
    for ii, it in enumerate(items):
        X, i = it["X"], it["i"]
        if kind == "full": fns = {b: (lambda x, b=b: X[3][b].clone()) for b in BL}
        elif nm == "compose":
            fns = {b: (lambda x, b=b, QG=basis[(kind, "G", k, b)][ii], QA=basis[(kind, "A", k, b)][ii]: proj(QA, proj(QG, x, X[1][b]), X[2][b])) for b in BL}
        else: fns = {b: (lambda x, b=b, Q=basis[(kind, nm, k, b)][ii]: proj(Q, x, X[3][b])) for b in BL}
        lp, _ = run(it["ids"], it["p"], it["cand"], fns); outcome[int(lp.argmax()) ^ i] += 1
        s0 = scores(it["lp"], i); s1 = scores(lp, i); dG.append(s1[0] - s0[0]); dA.append(s1[1] - s0[1])
    return dict(outcome=(outcome / n).tolist(), dG=dG, dA=dA)                  # outcome index: 0 base kept, 1 gender flipped, 2 generation flipped, 3 both
res = dict(model=name, level=L, n_items=n, n_groups=groups, results={})
R = res["results"]; R["full"] = evaluate("full", None, None)
for k in KS:
    for kind in ("native", "rotated", "pca"):
        for nm in ("B", "G", "A", "GA", "compose"): R[f"{kind}_{nm}_{k}"] = evaluate(kind, nm, k)
    log(f"{name} k={k}: " + " | ".join(f"{kind} " + ", ".join(f"{nm} {'/'.join(f'{v:.2f}' for v in R[f'{kind}_{nm}_{k}']['outcome'])}" for nm in ("B", "G", "A", "GA", "compose")) for kind in ("native", "rotated", "pca")))
mean = lambda v: sum(v) / max(len(v), 1)
res["behaviour"] = {}
for k in KS:
    for kind in ("native", "rotated", "pca"):
        G_, A_, GA_ = R[f"{kind}_G_{k}"], R[f"{kind}_A_{k}"], R[f"{kind}_GA_{k}"]
        ct_G = mean([abs(a) for a in G_["dA"]]) / max(mean([abs(g) for g in G_["dG"]]), 1e-9); ct_A = mean([abs(g) for g in A_["dG"]]) / max(mean([abs(a) for a in A_["dA"]]), 1e-9)
        add_err = mean([(abs(gg - (g1 + g2)) + abs(aa - (a1 + a2))) / max(abs(gg) + abs(aa), 1e-9) for gg, aa, g1, a1, g2, a2 in zip(GA_["dG"], GA_["dA"], G_["dG"], G_["dA"], A_["dG"], A_["dA"])])
        res["behaviour"][f"{kind}_{k}"] = dict(gender_shift_G=mean(G_["dG"]), generation_shift_A=mean(A_["dA"]), crosstalk_G=ct_G, crosstalk_A=ct_A, additivity_error=add_err)
for key in list(R): R[key] = dict(outcome=R[key]["outcome"])
o = lambda key, j: R[key]["outcome"][j]
res["checks"] = dict(full_over_0_9=o("full", 3) >= 0.9, gender_alone_half=o("native_G_4", 1) >= 0.5 or o("native_G_16", 1) >= 0.5,
                     compose_half_of_direct=max(o("native_compose_4", 3), o("native_compose_16", 3)) >= 0.5 * max(o("native_B_4", 3), o("native_B_16", 3)),
                     native_factorises_better=all(o(f"native_G_{k}", 1) > o(f"rotated_G_{k}", 1) for k in KS), additivity_within_25pct=min(res["behaviour"][f"native_{k}"]["additivity_error"] for k in KS) <= 0.25)
fmt = lambda key: "/".join(f"{v:.2f}" for v in R[key]["outcome"])
summ = (f"{name} L{L}, {n} items ({groups} quadruple-contexts); outcome shares kept/gender/generation/both: full {fmt('full')} || "
        + " || ".join(f"k={k}: " + " | ".join(f"{kind}: B {fmt(f'{kind}_B_{k}')}, G {fmt(f'{kind}_G_{k}')}, A {fmt(f'{kind}_A_{k}')}, GA {fmt(f'{kind}_GA_{k}')}, compose {fmt(f'{kind}_compose_{k}')}" for kind in ("native", "rotated", "pca")) for k in KS)
        + " || behaviour (gender shift by G, generation shift by A, cross-talk G, cross-talk A, additivity error): " + "; ".join(f"{key} {v['gender_shift_G']:+.2f}, {v['generation_shift_A']:+.2f}, {v['crosstalk_G']:.2f}, {v['crosstalk_A']:.2f}, {v['additivity_error']:.2f}" for key, v in res["behaviour"].items())
        + f" | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e473_wordalgebra_{name}", res, summ)
