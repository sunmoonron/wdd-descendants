"""e469: interchange interventions with native words (the causal-abstraction test; RELATED_WORK noted that interchange
between contexts had not been tested, and an external review asked for it).
Task: e451's few-shot translation of a noun into English, non-English prompts. The base prompt has noun c; the source
prompt is the same template and language with noun c' (a fixed derangement). Both must be answered correctly.
At the last noun token, a subspace of the base state is set to the source state's value:
  x_base <- x_base + P (x_source - x_base).
P is recomputed on the current state at each intervened block, so repeated interventions do not accumulate (unlike
e455).
Subspaces:
- full (P = identity: activation patching);
- native k: the span of the k native words chosen by OMP on the clean source-minus-base difference at that block, over
  the dictionary up to that block;
- rotated k: the same with rotated words (the same adaptivity, no provenance);
- PCA k: the top k principal directions of the differences across all pairs at that block.
For k = 1, 4 and 16, and blocks {L} (single) or 0..L (all).
Measured: interchange accuracy (the answer becomes c'), and the change of c's log-probability.
Models (argument): qwen05, smollm2.
Pre-registered (honest guesses):
- full interchange at all blocks switches at least 90% of answers (0.8);
- native k = 16 at all blocks reaches at least half of full's accuracy (0.5);
- native beats rotated at the same k (0.6);
- single-block interchange, even full, switches under 30% (0.6)."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "e451_concept_handles.py")).read()
exec(src[src.index("NOUNS = {"): src.index("LANGS = list(NOUNS)")])
LANGS = ["fr", "es", "de"]; NC, NT = len(NOUNS["en"]), len(TEMPL["en"])
name = sys.argv[1]; model, tok, fam = load_model(name); arch = Arch(model, fam); L = arch.NB // 2; D = arch.D; KS = [1, 4, 16]
EN = [n.split(" ", 1)[1] for n in NOUNS["en"]]
first = [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in EN]; dup = {f for f in first if first.count(f) > 1}
ok_n = [c for c in range(NC) if first[c] not in dup]; cand = torch.tensor([first[c] for c in ok_n], device=DEV); cpos = {c: j for j, c in enumerate(ok_n)}
tgt = {c: ok_n[(i + len(ok_n) // 2) % len(ok_n)] for i, c in enumerate(ok_n)}
A, lab = build_dictionary(arch, blocks=list(range(L + 1))); blk = lab["block"].to(DEV); Au = unitr(A); Ar = unitr(rotate(A, seed=7)); del A
ends = {b: int((blk <= b).sum()) for b in range(L + 1)}
def prompt(lg, c, t):
    s = TEMPL[lg][t].format(NOUNS[lg][c]); art = NOUNS[lg][c]; noun = art.split(" ", 1)[-1] if " " in art else art.split("'", 1)[-1]
    p = f"{LN[lg]}: {EX[lg][0]}\nEnglish noun: milk\n{LN[lg]}: {EX[lg][1]}\nEnglish noun: flower\n{LN[lg]}: {s}\nEnglish noun:"
    s0 = p.rindex(s); a = s0 + s.index(art) + art.index(noun); b_ = a + len(noun)
    enc = tok(p, add_special_tokens=False, return_offsets_mapping=True); ids = torch.tensor(enc["input_ids"], device=DEV)[None]
    return ids, [i for i, (x0, x1) in enumerate(enc["offset_mapping"]) if x1 > a and x0 < b_][-1]
def run(ids, p, fns=None):
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
pairs = []
for lg in LANGS:
    for c in ok_n:
        for t in range(NT):
            bi, bp = prompt(lg, c, t); si, spos = prompt(lg, tgt[c], t)
            lb, Xb = run(bi, bp); ls, Xs = run(si, spos)
            if int(lb.argmax()) == cpos[c] and int(ls.argmax()) == cpos[tgt[c]]: pairs.append(dict(c=c, ids=bi, p=bp, lp=lb, Xb=Xb, Xs=Xs))
n = len(pairs); Dl = torch.stack([pp["Xs"] - pp["Xb"] for pp in pairs], 1)              # [L+1, n, D] clean differences
basis = {}                                                                         # (kind, block) -> [n, D, k] orthonormal bases
for b in range(L + 1):
    for kind, Dct in (("native", Au), ("rotated", Ar)):
        sel, _, _ = omp(Dl[b], Dct[:ends[b]], max(KS), batch=256, record_err=False)
        for k in KS: basis[(kind, k, b)] = torch.linalg.qr(Dct[sel[:, :k]].transpose(1, 2))[0]   # [n, D, k]
    ev_, U_ = torch.linalg.eigh(torch.cov(Dl[b].T.double(), correction=0)); Up = U_.flip(-1).float()
    for k in KS: basis[("pca", k, b)] = Up[:, :k][None].expand(n, -1, -1)
res = dict(model=name, level=L, n_pairs=n, results={})
def evaluate(kind, k, blocks):
    moved, dT = 0, 0.0
    for i, pp in enumerate(pairs):
        xs = pp["Xs"]; ti = cpos[tgt[pp["c"]]]
        if kind == "full": fns = {b: (lambda x, b=b: xs[b].clone()) for b in blocks}
        else: fns = {b: (lambda x, b=b, Q=basis[(kind, k, b)][i]: x + Q @ (Q.T @ (xs[b] - x))) for b in blocks}
        lp, _ = run(pp["ids"], pp["p"], fns); moved += int(int(lp.argmax()) == ti); dT += float(lp[ti] - pp["lp"][ti])
    return dict(iia=moved / n, d_target=dT / n)
for scope, blocks in (("single", [L]), ("all", list(range(L + 1)))):
    res["results"][f"full_{scope}"] = evaluate("full", None, blocks)
    for kind in ("native", "rotated", "pca"):
        for k in KS: res["results"][f"{kind}{k}_{scope}"] = evaluate(kind, k, blocks)
    log(f"{name} {scope}: full {res['results'][f'full_{scope}']['iia']:.2f} | " + " | ".join(f"{kind} " + "/".join(f"{res['results'][f'{kind}{k}_{scope}']['iia']:.2f}" for k in KS) for kind in ("native", "rotated", "pca")))
R = res["results"]
res["checks"] = dict(full_all_over_0_9=R["full_all"]["iia"] >= 0.9, native16_half_of_full=R["native16_all"]["iia"] >= 0.5 * R["full_all"]["iia"],
                     native_beats_rotated=all(R[f"native{k}_all"]["iia"] >= R[f"rotated{k}_all"]["iia"] for k in KS), single_full_under_0_3=R["full_single"]["iia"] < 0.3)
summ = (f"{name} L{L}, {n} base/source pairs, interchange accuracy (k = 1/4/16) | single block L: full {R['full_single']['iia']:.2f}, "
        + ", ".join(f"{kd} " + "/".join(f"{R[f'{kd}{k}_single']['iia']:.2f}" for k in KS) for kd in ("native", "rotated", "pca"))
        + f" | blocks 0..L: full {R['full_all']['iia']:.2f}, " + ", ".join(f"{kd} " + "/".join(f"{R[f'{kd}{k}_all']['iia']:.2f}" for k in KS) for kd in ("native", "rotated", "pca"))
        + f" | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e469_interchange_{name}", res, summ)
