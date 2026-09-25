"""e474: typed handles. Are the native words that carry an entity's identity the same whether the entity is the subject
or the object, or are they typed by role? From an external review ("does swapping the same word handle have different
downstream effects depending on its structural position?").
Task: few-shot relational questions, "The {A} chased the {B}. Who chased? The" -> A (subject readout) and "... Who was
chased? The" -> B (object readout), with three verbs (chased, bit, followed); the few-shot lines use people, the test
lines animals (single tokens). The readout is a forced choice among the animals.
Items: entity e is replaced by e' (a fixed derangement), co-entity X.
- Subject items: story (e, X) with the subject readout; source (e', X).
- Object items: story (X, e) with the object readout; source (X, e').
Interchange at e's token, blocks 0..L, x <- x + P (x_src - x) recomputed at each block (e469), P from native words
(k = 4 and 16) chosen by OMP for a difference e' - e taken at that block:
- in context: this item's own source-minus-base difference;
- same role, other context: the same swap in the same role with another co-entity Y;
- other role, same co-entity: the same swap in the other role (for a subject item, stories (X, e) and (X, e') at the
  object position; the same few-shot prefix);
- rotated words (in context) and the full state, as controls.
If identity handles are untyped, the other-role basis works as well as the same-role other-context basis. Also
reported: the overlap (Jaccard) of the top 4 native words between the in-context basis and each transfer basis, at
each block.
Models (argument): smollm2, qwen05.
Pre-registered (honest guesses):
- full interchange switches at least 90% (0.8);
- in-context native k = 4 reaches at least half of full (0.6);
- the other-role basis does at least 80% as well as the same-role other-context basis, so handles are untyped (0.6);
- the words' overlap between roles falls with depth (Jaccard at L below that at block 0) (0.8)."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
ANIMALS = ["cat", "dog", "horse", "bird", "cow", "fox", "pig", "lion", "bear", "wolf", "duck", "mouse", "goat", "sheep", "rabbit", "frog"]
VERBS = [("chased", "Who chased?", "Who was chased?"), ("bit", "Who bit?", "Who was bitten?"), ("followed", "Who followed?", "Who was followed?")]
SHOT = [("man", "boy"), ("girl", "woman")]; KS = [4, 16]; CAP = 160
name = sys.argv[1]; model, tok, fam = load_model(name); arch = Arch(model, fam); L = arch.NB // 2; D = arch.D
ent = [a for a in ANIMALS if len(tok(" " + a, add_special_tokens=False)["input_ids"]) == 1]
cand = torch.tensor([tok(" " + a, add_special_tokens=False)["input_ids"][0] for a in ent], device=DEV); NE = len(ent)
tgt = {e: (e + NE // 2) % NE for e in range(NE)}
A_, lab = build_dictionary(arch, blocks=list(range(L + 1))); blk = lab["block"].to(DEV); Au = unitr(A_); Ar = unitr(rotate(A_, seed=7)); del A_
ends = {b: int((blk <= b).sum()) for b in range(L + 1)}
def prompt(role, v, a, b):
    verb, qs, qo = VERBS[v]; q = qs if role == "S" else qo
    shots = "".join(f"The {s1} {verb} the {s2}. {q} The {s1 if role == 'S' else s2}\n" for s1, s2 in SHOT)
    last = f"The {ent[a]} {verb} the {ent[b]}."; p = shots + last + f" {q} The"; s0 = p.rindex(last)
    spans = [(s0 + 4, s0 + 4 + len(ent[a])), (s0 + last.index(" the ", 4) + 5, s0 + last.index(" the ", 4) + 5 + len(ent[b]))]
    enc = tok(p, add_special_tokens=False, return_offsets_mapping=True); ids = torch.tensor(enc["input_ids"], device=DEV)[None]
    pos = [[i for i, (x0, x1) in enumerate(enc["offset_mapping"]) if x1 > a0 and x0 < a1][-1] for a0, a1 in spans]
    return ids, pos                                                               # pos[0]: subject token, pos[1]: object token
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
cache = {}
def clean(role, v, a, b, slot):
    """(answer index, states [L+1, D] at the subject (slot 0) or object (slot 1) token) of the clean story"""
    key = (role, v, a, b, slot)
    if key not in cache:
        ids, pos = prompt(role, v, a, b); lp, X = run(ids, pos[slot]); cache[key] = (int(lp.argmax()), X, ids, pos[slot])
    return cache[key]
g = torch.Generator().manual_seed(0); items = {"S": [], "O": []}
for role in ("S", "O"):
    order = [(v, e, x) for v in range(len(VERBS)) for e in range(NE) for x in range(NE) if x not in (e, tgt[e])]
    for j in torch.randperm(len(order), generator=g).tolist():
        v, e, x = order[j]; te = tgt[e]
        y = next(c for c in [(x + d) % NE for d in range(1, NE)] if c not in (e, te, x))
        if role == "S": base, src, same_b, same_s, cross_b, cross_s, slot, xslot = (e, x), (te, x), (e, y), (te, y), (x, e), (x, te), 0, 1
        else: base, src, same_b, same_s, cross_b, cross_s, slot, xslot = (x, e), (x, te), (y, e), (y, te), (e, x), (te, x), 1, 0
        ans_b, Xb, ids, p = clean(role, v, *base, slot); ans_s, Xs, _, _ = clean(role, v, *src, slot)
        if ans_b != e or ans_s != te: continue
        items[role].append(dict(e=e, te=te, ids=ids, p=p, Xb=Xb, Xs=Xs, d_same=clean(role, v, *same_s, slot)[1] - clean(role, v, *same_b, slot)[1],
                                d_cross=clean(role, v, *cross_s, xslot)[1] - clean(role, v, *cross_b, xslot)[1]))
        if len(items[role]) >= CAP: break
    log(f"{name} role {role}: {len(items[role])} items")
res = dict(model=name, level=L, entities=ent, roles={})
for role, its in items.items():
    n = len(its)
    if n == 0: continue
    diffs = {"in": torch.stack([it["Xs"] - it["Xb"] for it in its], 1), "same": torch.stack([it["d_same"] for it in its], 1), "cross": torch.stack([it["d_cross"] for it in its], 1)}
    basis, top4 = {}, {}
    for b in range(L + 1):
        for src_, Dct, kind in (("in", Au, "native"), ("same", Au, "native"), ("cross", Au, "native"), ("in", Ar, "rotated")):
            sel, _, _ = omp(diffs[src_][b], Dct[:ends[b]], max(KS), batch=256, record_err=False)
            if kind == "native": top4[(src_, b)] = sel[:, :4]
            for k in KS: basis[(kind, src_, k, b)] = torch.linalg.qr(Dct[sel[:, :k]].transpose(1, 2))[0]
    def acc(kind, src_, k):
        moved = 0
        for i, it in enumerate(its):
            xs = it["Xs"]
            if kind == "full": fns = {b: (lambda x, b=b: xs[b].clone()) for b in range(L + 1)}
            else: fns = {b: (lambda x, b=b, Q=basis[(kind, src_, k, b)][i]: x + Q @ (Q.T @ (xs[b] - x))) for b in range(L + 1)}
            lp, _ = run(it["ids"], it["p"], fns); moved += int(int(lp.argmax()) == it["te"])
        return moved / n
    out = dict(n=n, full=acc("full", None, None))
    for k in KS:
        for kind, src_ in (("native", "in"), ("native", "same"), ("native", "cross"), ("rotated", "in")): out[f"{kind}_{src_}_{k}"] = acc(kind, src_, k)
    J = lambda s1, s2, b: sum(len(set(a.tolist()) & set(c.tolist())) / len(set(a.tolist()) | set(c.tolist())) for a, c in zip(top4[(s1, b)], top4[(s2, b)])) / n
    out["jaccard_in_same"] = [J("in", "same", b) for b in range(L + 1)]; out["jaccard_in_cross"] = [J("in", "cross", b) for b in range(L + 1)]
    res["roles"][role] = out
    log(f"{name} role {role}: full {out['full']:.2f} | " + " | ".join(f"k={k}: in {out[f'native_in_{k}']:.2f}, same-role other context {out[f'native_same_{k}']:.2f}, other role {out[f'native_cross_{k}']:.2f}, rotated {out[f'rotated_in_{k}']:.2f}" for k in KS)
        + " | top-4 Jaccard by block (same-role / other-role): " + " ".join(f"{a:.2f}/{c:.2f}" for a, c in zip(out["jaccard_in_same"], out["jaccard_in_cross"])))
Rr = res["roles"]
res["checks"] = dict(full_over_0_9=all(v["full"] >= 0.9 for v in Rr.values()), native4_half_of_full=all(v["native_in_4"] >= 0.5 * v["full"] for v in Rr.values()),
                     untyped=all(v[f"native_cross_{k}"] >= 0.8 * v[f"native_same_{k}"] for v in Rr.values() for k in KS),
                     overlap_falls=all(v["jaccard_in_cross"][-1] < v["jaccard_in_cross"][0] for v in Rr.values()))
summ = (f"{name} L{L}, {len(ent)} animals | " + " || ".join(f"{'subject' if role == 'S' else 'object'} items ({v['n']}): full {v['full']:.2f}; "
        + "; ".join(f"k={k}: in context {v[f'native_in_{k}']:.2f}, same role other context {v[f'native_same_{k}']:.2f}, other role {v[f'native_cross_{k}']:.2f}, rotated {v[f'rotated_in_{k}']:.2f}" for k in KS)
        + f"; top-4 word overlap with in-context, block 0 / L: same role {v['jaccard_in_same'][0]:.2f}/{v['jaccard_in_same'][-1]:.2f}, other role {v['jaccard_in_cross'][0]:.2f}/{v['jaccard_in_cross'][-1]:.2f}" for role, v in Rr.items())
        + f" | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e474_typedhandles_{name}", res, summ)
