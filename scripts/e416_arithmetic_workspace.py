"""e416: intermediates of a computation, and fabrications. WorkspaceBench's computational families ask whether a reader
surfaces values a model computes without writing them, and its hallucination analysis asks how often a reader states
things the computation does not contain. Chained arithmetic gives both exactly: in "a=3;b=a+4;c=b-2;c=" the value of b
(7) is computed and never written, and any digit outside {a, the two operands, b, c} is a fabrication. Qwen2.5-7B
(bf16), three worked examples, 2-step chains with all five digits distinct; kept when the model answers c (top-1).
At the last position after blocks 4, 6, ..., 26, each reader (logit lens, centred lens, PCA lens, native-word lens with
16 own words, rotated control; as e415) ranks the ten digit tokens. Scored: the rank of b among the digits (recall at 1
and 3), the rank of c, and the fabrication rate (share of a reader's top-3 digits that belong to no quantity in the
chain). Causal check: removing the native word whose own readout ranks b first, against a random other selected word,
change in the log-probability of c.
Pre-registered: (1) at middle depth the native-word lens ranks b in the top 3 more often than the plain and centred
lenses and the PCA lens; (2) its fabrication rate is no higher than theirs; (3) removing the b word lowers the
log-probability of c more than removing a random other word."""
import sys, os, random; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ws_common import *
name = sys.argv[1] if len(sys.argv) > 1 else "qwen7"; K = 16
model, tok, fam = load_bf16(name); arch = Arch(model, fam); layers = list(range(4, arch.NB - 1, 2))
dig = [tok("{}".format(d), add_special_tokens=False)["input_ids"] for d in range(10)]; assert all(len(t) == 1 for t in dig), dig; dig = torch.tensor([t[0] for t in dig], device=DEV)
rng = random.Random(0); combos = []
for a in range(1, 9):
    for d in range(1, 9):
        for e in range(1, 9):
            b = a + d; c = b - e
            if b <= 9 and c >= 0 and len({a, d, e, b, c}) == 5: combos.append((a, d, e, b, c))
rng.shuffle(combos); shots = combos[:3]; tests = combos[3:203]
prefix = "".join(f"a={a};b=a+{d};c=b-{e};c={c}\n" for a, d, e, b, c in shots)
items = [dict(a=a, d=d, e=e, b=b, c=c, ids=torch.tensor(tok(prefix + f"a={a};b=a+{d};c=b-{e};c=", add_special_tokens=False)["input_ids"])) for a, d, e, b, c in tests]
_, lg = last_states(model, arch, [it["ids"] for it in items], [])
keep = [i for i, it in enumerate(items) if lg[i].argmax().item() == dig[it["c"]].item()]
items = [items[i] for i in keep]; N = len(items); log(f"{name}: {N} of {len(lg)} chains answered correctly")
ev_all = torch.load("/workspace/wdd/eval_ids_all.pt", weights_only=False)["qwen05"]["eval_ids"][:4]
ref = ref_stats(model, arch, ev_all, layers); gain, WU = final_readout(model, arch)
A, blk, typ, ends = lean_dictionary(arch, max(layers))
g = torch.Generator().manual_seed(7); R = torch.linalg.qr(torch.randn(arch.D, arch.D, generator=g))[0].to(DEV)
X_all, _ = last_states(model, arch, [it["ids"] for it in items], layers)
bvals = torch.tensor([it["b"] for it in items], device=DEV); cvals = torch.tensor([it["c"] for it in items], device=DEV)
chain = torch.zeros(N, 10, dtype=torch.bool, device=DEV)
for n, it in enumerate(items):
    for v in (it["a"], it["d"], it["e"], it["b"], it["c"]): chain[n, v] = True
res = dict(model=name, n=N, k=K, layers=layers, readers={}, causal={})
def digit_scores(sc): return sc[:, dig]                     # [N, 10]
for L in layers:
    X = X_all[L]; mu = ref[L]["mu"]; Xc = X - mu; rms = X.pow(2).mean(-1).sqrt(); P = ref[L]["pcs"]; An = A[:ends[L]]
    sel, _, _ = omp(Xc, An, K, batch=64, record_err=False); cof, _ = refit(Xc, An, sel); words = An[sel]
    selr, _, _ = omp(Xc @ R.T, An, K, batch=64, record_err=False); cofr, _ = refit(Xc @ R.T, An, selr); wordsr = An[selr] @ R
    parts = (Xc @ P)[:, :, None] * P.T[None]
    S = dict(lens=lens_scores(X, gain, WU, rms), lens_centred=lens_scores(Xc, gain, WU, rms), pca16=pooled_word_scores(parts, torch.ones(N, K, device=DEV), gain, WU, rms),
             native16=pooled_word_scores(words, cof, gain, WU, rms), rotated16=pooled_word_scores(wordsr, cofr, gain, WU, rms))
    for rd, sc in S.items():
        ds = digit_scores(sc); order = ds.argsort(-1, descending=True)
        rb = (order == bvals[:, None]).float().argmax(-1) + 1; rc = (order == cvals[:, None]).float().argmax(-1) + 1
        top3 = order[:, :3]; fab = (~chain.gather(1, top3)).float().mean().item()
        res["readers"][f"{rd}@{L}"] = dict(b_at1=(rb <= 1).float().mean().item(), b_at3=(rb <= 3).float().mean().item(), c_at1=(rc <= 1).float().mean().item(), c_at3=(rc <= 3).float().mean().item(), fabrication_top3=fab)
    dl_b, dl_o = [], []; gen = torch.Generator().manual_seed(L)
    for n in range(N):
        per = torch.stack([lens_scores(words[n, i:i + 1] * cof[n, i], gain, WU, rms[n:n + 1])[0, dig] for i in range(K)])   # [K, 10]
        hit = [i for i in range(K) if per[i].argmax().item() == items[n]["b"]]
        if not hit: continue
        ib = max(hit, key=lambda i: cof[n, i].abs().item()); others = [i for i in range(K) if i not in hit]
        if not others: continue
        io = others[int(torch.randint(0, len(others), (1,), generator=gen))]
        xs = torch.stack([X[n], X[n] - cof[n, ib] * words[n, ib], X[n] - cof[n, io] * words[n, io]])
        lp = torch.log_softmax(splice_last(model, arch, items[n]["ids"], L, xs), -1); cc = dig[items[n]["c"]].item()
        dl_b.append((lp[1, cc] - lp[0, cc]).item()); dl_o.append((lp[2, cc] - lp[0, cc]).item())
    res["causal"][str(L)] = dict(n=len(dl_b), b_word=sum(dl_b) / max(len(dl_b), 1), other_word=sum(dl_o) / max(len(dl_o), 1))
    r = lambda rd, m: res["readers"][f"{rd}@{L}"][m]
    log(f"L{L}: b@3 lens {r('lens', 'b_at3'):.2f} centred {r('lens_centred', 'b_at3'):.2f} pca {r('pca16', 'b_at3'):.2f} native {r('native16', 'b_at3'):.2f} rotated {r('rotated16', 'b_at3'):.2f} | "
        f"fabrication lens {r('lens', 'fabrication_top3'):.2f} native {r('native16', 'fabrication_top3'):.2f} pca {r('pca16', 'fabrication_top3'):.2f} | c@1 lens {r('lens', 'c_at1'):.2f} native {r('native16', 'c_at1'):.2f} | "
        f"causal (n {len(dl_b)}) b word {res['causal'][str(L)]['b_word']:.2f} other {res['causal'][str(L)]['other_word']:.2f}")
    del sel, selr, words, wordsr, parts, S
bl = lambda rd, m: max(res["readers"][f"{rd}@{L}"][m] for L in layers)
best = max(layers, key=lambda L: res["readers"][f"native16@{L}"]["b_at3"])
summ = (f"{name} n={N}: best-layer b@3 lens {bl('lens', 'b_at3'):.2f} centred {bl('lens_centred', 'b_at3'):.2f} pca16 {bl('pca16', 'b_at3'):.2f} native16 {bl('native16', 'b_at3'):.2f} (L{best}) rotated16 {bl('rotated16', 'b_at3'):.2f} | "
        f"fabrication at L{best}: " + " ".join(f"{rd} {res['readers'][f'{rd}@{best}']['fabrication_top3']:.2f}" for rd in ("lens", "lens_centred", "pca16", "native16", "rotated16"))
        + f" | causal at L{best}: b word {res['causal'][str(best)]['b_word']:.2f} vs other {res['causal'][str(best)]['other_word']:.2f} (n {res['causal'][str(best)]['n']})")
log(summ); record(f"e416_arith_{name}", res, summ)
