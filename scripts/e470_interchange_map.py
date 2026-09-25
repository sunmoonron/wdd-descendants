"""e470: where and how native words carry a variable under interchange (following e469, from an external review's
follow-ups: the handoff point, the lifetime of the handle, local against global bases, and whether the carrier words
persist).
Setup: e469's translation pairs (base noun c, source noun c', same template and language, non-English, both correct;
up to 150 pairs). The interchange is at the last noun token, x <- x + P (x_source - x), recomputed on the current state
at each intervened block so nothing accumulates.
Measured: interchange accuracy (the answer becomes c') for:
- A, single blocks b = 0, 2, 4, ..., L: the whole state, native k = 1 and 4, and the per-block principal directions of
  the differences (local PCA, k = 4);
- D, windows [b0, L] (a later start) and [0, b1] (an earlier stop): the whole state, native k = 1 and 4;
- C, all blocks 0..L: native, local PCA, and one global PCA basis fitted on the differences pooled over blocks (each
  block's differences scaled to unit mean norm), for k = 1, 4 and 16;
- persistence of the carrier words: for the top 4 native words at each block, the share written by blocks at least 4
  below, the share written by the last two blocks, the overlap (Jaccard) with the previous block's top 4, and how often
  the noun token's own embedding is among them.
Models (argument): qwen05, smollm2.
Pre-registered (honest guesses):
- whole-state patching at a single block switches at least 90% at every block (0.6);
- single-block native k = 4 works at the earliest blocks (at least 0.5 at block 0) but not at L (under 0.2) (0.5);
- later starts lose accuracy monotonically (0.7);
- native beats local PCA, which beats global PCA, at all blocks (0.5);
- the carrier words persist: consecutive-block overlap of the top 4 at least 0.5 (0.4)."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "e451_concept_handles.py")).read()
exec(src[src.index("NOUNS = {"): src.index("LANGS = list(NOUNS)")])
LANGS = ["fr", "es", "de"]; NC, NT = len(NOUNS["en"]), len(TEMPL["en"])
name = sys.argv[1]; model, tok, fam = load_model(name); arch = Arch(model, fam); L = arch.NB // 2; D = arch.D
EN = [n.split(" ", 1)[1] for n in NOUNS["en"]]
first = [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in EN]; dup = {f for f in first if first.count(f) > 1}
ok_n = [c for c in range(NC) if first[c] not in dup]; cand = torch.tensor([first[c] for c in ok_n], device=DEV); cpos = {c: j for j, c in enumerate(ok_n)}
tgt = {c: ok_n[(i + len(ok_n) // 2) % len(ok_n)] for i, c in enumerate(ok_n)}
A, lab = build_dictionary(arch, blocks=list(range(L + 1))); blk = lab["block"].to(DEV); typ = lab["type"].to(DEV); idx = lab["index"].to(DEV); Au = unitr(A); del A
ends = {b: int((blk <= b).sum()) for b in range(L + 1)}
def prompt(lg, c, t):
    s = TEMPL[lg][t].format(NOUNS[lg][c]); art = NOUNS[lg][c]; noun = art.split(" ", 1)[-1] if " " in art else art.split("'", 1)[-1]
    p = f"{LN[lg]}: {EX[lg][0]}\nEnglish noun: milk\n{LN[lg]}: {EX[lg][1]}\nEnglish noun: flower\n{LN[lg]}: {s}\nEnglish noun:"
    s0 = p.rindex(s); a = s0 + s.index(art) + art.index(noun); b_ = a + len(noun)
    enc = tok(p, add_special_tokens=False, return_offsets_mapping=True); ids = torch.tensor(enc["input_ids"], device=DEV)[None]
    pos = [i for i, (x0, x1) in enumerate(enc["offset_mapping"]) if x1 > a and x0 < b_][-1]
    return ids, pos
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
g = torch.Generator().manual_seed(0); order = [(lg, c, t) for lg in LANGS for c in ok_n for t in range(NT)]
for j in torch.randperm(len(order), generator=g).tolist():
    lg, c, t = order[j]; bi, bp = prompt(lg, c, t); si, spos = prompt(lg, tgt[c], t)
    lb, Xb = run(bi, bp); ls, Xs = run(si, spos)
    if int(lb.argmax()) == cpos[c] and int(ls.argmax()) == cpos[tgt[c]]:
        pairs.append(dict(c=c, ids=bi, p=bp, lp=lb, Xb=Xb, Xs=Xs, tok=int(bi[0, bp]), stok=int(si[0, spos])))
    if len(pairs) >= 150: break
n = len(pairs); Dl = torch.stack([pp["Xs"] - pp["Xb"] for pp in pairs], 1)              # [L+1, n, D]
KS = [1, 4, 16]; basis, sel_top = {}, {}
for b in range(L + 1):
    sel, _, _ = omp(Dl[b], Au[:ends[b]], max(KS), batch=256, record_err=False); sel_top[b] = sel[:, :4]
    for k in KS: basis[("native", k, b)] = torch.linalg.qr(Au[sel[:, :k]].transpose(1, 2))[0]
    ev_, U_ = torch.linalg.eigh(torch.cov(Dl[b].T.double(), correction=0)); Up = U_.flip(-1).float()
    for k in KS: basis[("pca", k, b)] = Up[:, :k][None].expand(n, -1, -1)
pooled = torch.cat([Dl[b] / Dl[b].norm(dim=-1).mean() for b in range(L + 1)])
ev_, U_ = torch.linalg.eigh(torch.cov(pooled.T.double(), correction=0)); Ug = U_.flip(-1).float()
for k in KS:
    for b in range(L + 1): basis[("gpca", k, b)] = Ug[:, :k][None].expand(n, -1, -1)
def iia(kind, k, blocks):
    moved = 0
    for i, pp in enumerate(pairs):
        xs = pp["Xs"]; ti = cpos[tgt[pp["c"]]]
        if kind == "full": fns = {b: (lambda x, b=b: xs[b].clone()) for b in blocks}
        else: fns = {b: (lambda x, b=b, Q=basis[(kind, k, b)][i]: x + Q @ (Q.T @ (xs[b] - x))) for b in blocks}
        lp, _ = run(pp["ids"], pp["p"], fns); moved += int(int(lp.argmax()) == ti)
    return moved / n
steps = list(range(0, L + 1, 2)) + ([L] if L % 2 else [])
res = dict(model=name, level=L, n_pairs=n, single={}, later_start={}, early_stop={}, all_blocks={}, persistence={})
for b in steps:
    res["single"][b] = dict(full=iia("full", None, [b]), native1=iia("native", 1, [b]), native4=iia("native", 4, [b]), pca4=iia("pca", 4, [b]))
    log(f"{name} single block {b}: {res['single'][b]}")
for b0 in steps:
    res["later_start"][b0] = dict(full=iia("full", None, list(range(b0, L + 1))), native1=iia("native", 1, list(range(b0, L + 1))), native4=iia("native", 4, list(range(b0, L + 1))))
for b1 in steps[1:]:
    res["early_stop"][b1] = dict(full=iia("full", None, list(range(0, b1 + 1))), native4=iia("native", 4, list(range(0, b1 + 1))))
for kind in ("native", "pca", "gpca"):
    for k in KS: res["all_blocks"][f"{kind}{k}"] = iia(kind, k, list(range(L + 1)))
old, new, jac, emb = [], [], [], []
for b in range(L + 1):
    S_ = sel_top[b]; bb = blk[S_]; old.append(float((bb <= b - 4).float().mean())); new.append(float((bb >= b - 1).float().mean()))
    tb = torch.tensor([pp["tok"] for pp in pairs], device=DEV)[:, None]; ts = torch.tensor([pp["stok"] for pp in pairs], device=DEV)[:, None]
    tok_hit = ((typ[S_] == T_TOK) & ((idx[S_] == tb) | (idx[S_] == ts))).any(1).float().mean(); emb.append(float(tok_hit))
    if b > 0:
        P0 = sel_top[b - 1]; jac.append(float(sum(len(set(a.tolist()) & set(c_.tolist())) / len(set(a.tolist()) | set(c_.tolist())) for a, c_ in zip(S_, P0)) / n))
res["persistence"] = dict(share_written_4plus_blocks_below=old, share_written_by_last_two_blocks=new, jaccard_with_previous_block=jac, noun_embedding_among_top4=emb)
Sg, LS, ES, AB = res["single"], res["later_start"], res["early_stop"], res["all_blocks"]
res["checks"] = dict(single_full_over_0_9_everywhere=all(v["full"] >= 0.9 for v in Sg.values()), native4_early_not_late=Sg[0]["native4"] >= 0.5 and Sg[L]["native4"] < 0.2,
                     later_start_monotone=all(LS[a]["native4"] >= LS[b]["native4"] for a, b in zip(steps, steps[1:])),
                     native_over_local_over_global=all(AB[f"native{k}"] >= AB[f"pca{k}"] >= AB[f"gpca{k}"] for k in KS), words_persist=(sum(jac) / len(jac)) >= 0.5)
summ = (f"{name} L{L}, {n} pairs | single block b (full/native1/native4/pca4): " + "; ".join(f"{b}: {v['full']:.2f}/{v['native1']:.2f}/{v['native4']:.2f}/{v['pca4']:.2f}" for b, v in Sg.items())
        + " | later start [b0, L] (full/native1/native4): " + "; ".join(f"{b}: {v['full']:.2f}/{v['native1']:.2f}/{v['native4']:.2f}" for b, v in LS.items())
        + " | early stop [0, b1] (full/native4): " + "; ".join(f"{b}: {v['full']:.2f}/{v['native4']:.2f}" for b, v in ES.items())
        + " | all blocks k=1/4/16: native " + "/".join(f"{AB[f'native{k}']:.2f}" for k in KS) + ", local pca " + "/".join(f"{AB[f'pca{k}']:.2f}" for k in KS) + ", global pca " + "/".join(f"{AB[f'gpca{k}']:.2f}" for k in KS)
        + f" | carrier words: overlap with previous block {sum(jac) / len(jac):.2f}, written 4+ blocks below {sum(old) / len(old):.2f}, by the last two blocks {sum(new) / len(new):.2f}, noun embedding in top 4 {sum(emb) / len(emb):.2f} | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e470_interchangemap_{name}", res, summ)
