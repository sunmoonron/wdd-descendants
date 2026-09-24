"""e448e: do the Rosetta words matter for what the model does next, in every language?
e448b found, for most sentences in Qwen2.5-0.5B, a native word used on the translation-equivalent token in all four
languages. Here, at that token and depth, the state's component along the word's direction is removed. The change in
the loss over the rest of the sentence is compared with three controls:
- removing another native word used at the same position, the one with the closest coefficient magnitude;
- removing the component along a random direction;
- nothing removed.
Per language, and pooled.
Pre-registered (honest guess): removing the Rosetta word costs more than removing the matched other word in at least
three of the four languages (0.5)."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
name = sys.argv[1]; model, tok, fam = load_model(name); arch = Arch(model, fam); D = arch.D; L = arch.NB // 2
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "e448_interlingua.py")).read(); S = eval(src[src.index("S = {") + 4: src.index("\nname = sys.argv[1]")]); LANGS = list(S)
rows = _json.load(open(os.path.join(RESULTS, f"e448b_rosetta_{name}.json")))["rows"]
EA = torch.load("/workspace/wdd/eval_ids_all.pt", weights_only=False)[name]; cen = EA["cen_ids"][:16].to(DEV); del EA
A, lab = build_dictionary(arch, blocks=list(range(L + 1)))
ref = block_states(model, arch, cen, [L], chunk=4)[L]; mu = ref[~sinkmask(ref)].mean(0)
g = torch.Generator().manual_seed(0)
def rest_loss(ids, j, direction):
    """mean loss over the tokens after position j+1 (states index j = position j+1), with the state's component along
    `direction` removed at that position (None: clean)"""
    def hk(m, i, o):
        xo = out_of(o); y = xo.clone()
        if direction is not None:
            u = direction / direction.norm(); y[0, j + 1] = y[0, j + 1] - (y[0, j + 1].float() @ u) * u
        return (y,) + tuple(o[1:]) if isinstance(o, tuple) else y
    h = arch.layers[L].register_forward_hook(hk)
    try:
        with torch.no_grad(): lg = model(ids).logits.float()
    finally: h.remove()
    lp = torch.log_softmax(lg[0, :-1], -1); tl = -lp.gather(1, ids[0, 1:, None])[:, 0]
    return tl[j + 1:].mean().item()                                          # tokens predicted from position j+1 onwards
out = {lg: dict(rosetta=[], other=[], random=[]) for lg in LANGS}
for r in rows:
    if not r["words"]: continue
    w = r["words"][0]["word"]
    for lg in LANGS:
        ids = torch.tensor(tok(S[lg][r["i"]], add_special_tokens=False)["input_ids"], device=DEV)[None]
        x = block_states(model, arch, ids, [L], chunk=1)[L][0]; sel, cof, _ = omp(x - mu, A, 16, batch=256, record_err=False)
        hit = torch.nonzero((sel == w).any(1))[:, 0]
        if hit.numel() == 0 or hit[0] >= ids.shape[1] - 3: continue
        j = hit[0].item(); k_w = (sel[j] == w).nonzero()[0, 0]; cw = cof[j, k_w].abs()
        others = [(abs(cof[j, k].item() - cw.item()), int(sel[j, k])) for k in range(16) if int(sel[j, k]) != w]
        wo = min(others)[1]; rnd = torch.randn(D, generator=g).to(DEV)
        base = rest_loss(ids, j, None)
        out[lg]["rosetta"].append(rest_loss(ids, j, A[w]) - base); out[lg]["other"].append(rest_loss(ids, j, A[wo]) - base); out[lg]["random"].append(rest_loss(ids, j, rnd) - base)
res = dict(model=name, level=L, by_lang={lg: {k: (sum(v) / len(v) if v else None) for k, v in d_.items()} for lg, d_ in out.items()}, n={lg: len(d_["rosetta"]) for lg, d_ in out.items()})
wins = sum(1 for lg in LANGS if res["by_lang"][lg]["rosetta"] is not None and res["by_lang"][lg]["rosetta"] > res["by_lang"][lg]["other"])
res["checks"] = dict(rosetta_over_other_in_3_of_4=wins >= 3)
summ = (f"{name} L{L}: loss change on the rest of the sentence after removing, at the Rosetta token, the Rosetta word / another used word of matched size / a random direction: "
        + " ; ".join(f"{lg} {res['by_lang'][lg]['rosetta']:+.3f}/{res['by_lang'][lg]['other']:+.3f}/{res['by_lang'][lg]['random']:+.3f} (n {res['n'][lg]})" for lg in LANGS) + f" | checks {json.dumps(res['checks'])}")
log(summ); record(f"e448e_rosetta_causal_{name}", res, summ)
