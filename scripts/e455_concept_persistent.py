"""e455: is the concept word the carrier of the noun downstream, or one handle among redundant carriers? (follows e451 and
e454)
e451 and e454 intervened at one depth, the middle block L, at the noun's tokens. There, a swapped-in concept word raised
the target noun by 1.4-1.9 nats but flipped no answer in the Qwen models, and moved the category only weakly.
One explanation is that attention had already copied the noun's identity to later positions before block L. Here the
concept word is removed or swapped at the output of every block from 0 to L, at the noun's tokens, so that no later
read of those positions sees the original concept word.
- remove_all: the component along the concept word is removed at every block;
- swap_all: it is replaced at every block by the target noun's concept word, at that noun's typical coefficient at that
  block and language;
- random_all: it is replaced at every block by a fixed random direction of the same size.
Two readouts, each with e451's and e454's few-shot prompts:
- translation to the English noun (24-way);
- category (8-way).
Targets: e451's derangement for translation, and a noun of a different category for the category readout (e454).
Reported per language: the share of answers moving to the target, the change of the target's log-probability, and
accuracy (for remove_all).
Models (argument): qwen05, smollm2, qwen7 (bf16).
Pre-registered (honest guesses):
- in Qwen-0.5B, swap_all moves at least a quarter of non-English translations to the target (0.5), against under 5%
  for random_all (0.7);
- category answers move less than translations (0.6)."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "e451_concept_handles.py")).read()
exec(src[src.index("NOUNS = {"): src.index("LANGS = list(NOUNS)")])                           # NOUNS, TEMPL, EX, LN
s4 = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "e454_concept_properties.py")).read()
exec(s4[s4.index("EXC = {"): s4.index("LANGS = list(NOUNS)")])                                # EXC, CAT, LABELS
LANGS = list(NOUNS); NC, NT = len(NOUNS["en"]), len(TEMPL["en"])
name = sys.argv[1]
if name == "qwen7":
    from ws_common import load_bf16, lean_dictionary
    model, tok, fam = load_bf16(name); arch = Arch(model, fam); D = arch.D; L = arch.NB // 2
    A, blkv, typv, ends = lean_dictionary(arch, L); A = A[:ends[L]]; blk = blkv[:ends[L]].long(); typ = typv[:ends[L]].long()
else:
    model, tok, fam = load_model(name); arch = Arch(model, fam); D = arch.D; L = arch.NB // 2
    A, lab = build_dictionary(arch, blocks=list(range(L + 1))); typ, blk = lab["type"].to(DEV), lab["block"].to(DEV)
cw = _json.load(open(os.path.join(RESULTS, f"e448d_concepts_{name}.json")))["dicts"]["own"]["words"]
EN = [n.split(" ", 1)[1] for n in NOUNS["en"]]
nfirst = [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in EN]; dup = {f for f in nfirst if nfirst.count(f) > 1}
okn = [c for c in range(NC) if nfirst[c] not in dup]; ncand = torch.tensor([nfirst[c] for c in okn], device=DEV); npos = {c: j for j, c in enumerate(okn)}
lcand = torch.tensor([tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in LABELS], device=DEV); catix = {c: LABELS.index(CAT[EN[c]]) for c in range(NC)}
concept = {c: int(cw[NOUNS["en"][c]]["word"]) for c in range(NC) if NOUNS["en"][c] in cw and c in npos}
for c, w in concept.items(): assert int(blk[w]) == cw[NOUNS["en"][c]]["block"] and int(typ[w]) == T_MLP
CN = sorted(concept); U = {c: A[w] / A[w].norm() for c, w in concept.items()}; del A; torch.cuda.empty_cache()
tr_target = {c: CN[(i + len(CN) // 2) % len(CN)] for i, c in enumerate(CN)}
def diff_target(c):
    pool = [d for d in CN if d != c and catix[d] != catix[c]]; return pool[(CN.index(c) * 7) % len(pool)] if pool else None
cat_target = {c: diff_target(c) for c in CN}

def prompt(kind, lg, c, t):
    s = TEMPL[lg][t].format(NOUNS[lg][c]); art = NOUNS[lg][c]; noun = art.split(" ", 1)[-1] if " " in art else art.split("'", 1)[-1]
    if kind == "translate": p = f"{LN[lg]}: {EX[lg][0]}\nEnglish noun: milk\n{LN[lg]}: {EX[lg][1]}\nEnglish noun: flower\n{LN[lg]}: {s}\nEnglish noun:"
    else: p = f"{LN[lg]}: {EXC[lg][0]}\nCategory: drink\n{LN[lg]}: {EXC[lg][1]}\nCategory: plant\n{LN[lg]}: {s}\nCategory:"
    s0 = p.rindex(s); a = s0 + s.index(art) + art.index(noun); b_ = a + len(noun)
    enc = tok(p, add_special_tokens=False, return_offsets_mapping=True); ids = torch.tensor(enc["input_ids"], device=DEV)[None]
    return ids, [i for i, (x0, x1) in enumerate(enc["offset_mapping"]) if x1 > a and x0 < b_]

def run(ids, cand, pos, fns=None):
    """restricted log-probabilities at the last position; fns: {block: fn(state) -> state} applied at the noun positions;
    clean runs also return every block's output at the noun positions [L+1, len(pos), D]"""
    cap, hs = {}, []
    for b in range(L + 1):
        def hk(m, i, o, b=b):
            xo = out_of(o); cap[b] = xo[0, pos].detach().float()
            if not fns or b not in fns: return None
            y = xo.clone()
            for p in pos: y[0, p] = fns[b](y[0, p].float()).to(y.dtype)
            return (y,) + tuple(o[1:]) if isinstance(o, tuple) else y
        hs.append(arch.layers[b].register_forward_hook(hk))
    try:
        with torch.no_grad(): lg = model(ids).logits[0, -1].float()
    finally: [h.remove() for h in hs]
    return torch.log_softmax(lg[cand], -1), torch.stack([cap[b] for b in range(L + 1)])

f_ = lambda S: "n 0" if not S["n"] else (f"n {S['n']}: remove acc {S['remove_all']['acc']:.2f} (dC {S['remove_all']['d_correct']:+.2f}), swap to-target {S['swap_all']['to_target']:.2f} (dT {S['swap_all']['d_target']:+.2f}), "
                f"random to-target {S['random_all']['to_target']:.2f} (dT {S['random_all']['d_target']:+.2f})")
res = dict(model=name, level=L, concept_nouns=[EN[c] for c in CN], readouts={})
g = torch.Generator().manual_seed(0); r = torch.randn(D, generator=g).to(DEV); r = r / r.norm()
for kind, cand, cix, tgt in (("translate", ncand, lambda c: npos[c], tr_target), ("category", lcand, lambda c: catix[c], cat_target)):
    items = []
    for lg in LANGS:
        for c in CN:
            for t in range(NT):
                ids, pos = prompt(kind, lg, c, t); lp, X = run(ids, cand, pos); items.append(dict(lg=lg, c=c, ids=ids, pos=pos, lp=lp, X=X))
    typc = {}                                                            # typical coefficient of each concept word, per language and block
    for lg in LANGS:
        for c in CN:
            Xs = torch.cat([it["X"] for it in items if it["lg"] == lg and it["c"] == c], dim=1)   # [L+1, n, D]
            typc[(lg, c)] = (Xs @ U[c]).mean(1)                                            # [L+1]
    rows = []
    for it in items:
        c, lg = it["c"], it["lg"]; ci = cix(c); tc = tgt[c]
        if int(it["lp"].argmax()) != ci or tc is None: continue
        u, u2, ti = U[c], U[tc], cix(tc); a2 = typc[(lg, tc)]; row = dict(lg=lg, c=c)
        ops = {"remove_all": {b: (lambda x: x - (x @ u) * u) for b in range(L + 1)},
               "swap_all": {b: (lambda x, b=b: x - (x @ u) * u + a2[b] * u2) for b in range(L + 1)},
               "random_all": {b: (lambda x, b=b: x - (x @ u) * u + a2[b].abs() * r) for b in range(L + 1)}}
        for nm, fns in ops.items():
            lp1, _ = run(it["ids"], cand, it["pos"], fns); dl = lp1 - it["lp"]; am = int(lp1.argmax())
            row[nm] = dict(to_target=am == ti, d_target=float(dl[ti]), acc=am == ci, d_correct=float(dl[ci]))
        rows.append(row)
    summ_k = {}
    for lg in LANGS + ["non-en", "all"]:
        rr = [x for x in rows if lg == "all" or (lg == "non-en" and x["lg"] != "en") or x["lg"] == lg]
        summ_k[lg] = dict(n=len(rr), **{nm: {f: (sum(float(x[nm][f]) for x in rr) / len(rr) if rr else None) for f in ("to_target", "d_target", "acc", "d_correct")} for nm in ("remove_all", "swap_all", "random_all")})
    res["readouts"][kind] = dict(summary=summ_k, rows=rows)
    log(f"{name} {kind} non-English: {f_(summ_k['non-en'])}")
T_, C_ = res["readouts"]["translate"]["summary"]["non-en"], res["readouts"]["category"]["summary"]["non-en"]
res["checks"] = dict(swap_moves_quarter=(T_["swap_all"]["to_target"] or 0) >= 0.25, random_under_5pct=(T_["random_all"]["to_target"] or 0) < 0.05,
                     category_moves_less=(C_["swap_all"]["to_target"] or 0) < (T_["swap_all"]["to_target"] or 0))
summ = (f"{name}, concept word removed or swapped at every block 0..{L} at the noun, {len(CN)} concept nouns | translation, non-English " + f_(T_) + " | English " + f_(res["readouts"]["translate"]["summary"]["en"])
        + " || category, non-English " + f_(C_) + f" | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e455_conceptpersist_{name}", res, summ)
