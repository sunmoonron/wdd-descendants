"""e658 (session 143; Gemini 4's RYS idea 2): what the second traversal of a repeated block does to the first's writes.
Gemini's two predictions from the record: (A) amplification, the chord's own rows answer their chord positively
(e509, +0.26 to +0.45) so the second traversal reinforces the first's words; (B) erasure, a looped model rebuilds its
state each loop (LOOPED.md, 77-82% of the eight largest writes erased per loop) and read repulsion (e648b, static
read-write cosine -0.24 to -0.32) should make a block cancel its own earlier writes. e651 found the read-write
anti-alignment only in GELU models (gated models positive), so (B)'s mechanism predicts a family split. On the three
least and three most damaging blocks of two or more layers (e656), for each block of the band and each sampled
position: the cosine between the second and first traversal's MLP outputs, the share of the first traversal's eight
largest writes (|activation| times write norm) gone from the second's eight largest, the share of those eight whose
activation changes sign, and the correlation of the two activation vectors; for the band, the cosine of the second
traversal's increment with the first's, and the Jaccard of their 16-word OMP ledgers over the band's rows. Arguments:
model [--smoke]."""
from rys_common import *
from rys_common import _hs, _out
name = sys.argv[1]; SMOKE = "--smoke" in sys.argv; t0 = time.time(); tag = "_smoke" if SMOKE else ""
M = load(name); N = M["NB"]; D = M["D"]; A, blk = dictionary(M); DFF = A.shape[0] // N
W = wikitext(M, "test", 4 if SMOKE else 8, 128)
R6 = json.load(open(f"results/e656_rys_map_{name}{tag}.json")); s2 = [r for r in R6["rows"] if r["s"] >= 2]
bands = [("least", r) for r in sorted(s2, key=lambda r: r["dloss"])[:1 if SMOKE else 3]] + [("most", r) for r in sorted(s2, key=lambda r: -r["dloss"])[:1 if SMOKE else 3]]
downs = [(l.mlp.dense_4h_to_h if M["fam"] == "neox" else l.mlp.down_proj) for l in M["layers"]]; WN = [d.weight.detach().float().norm(dim=0) for d in downs]
gen = torch.Generator().manual_seed(658); out = []
for kind, r in bands:
    i, j = r["i"], r["j"]; acc = collections.defaultdict(list); cap = {}
    hooks = []
    for b in range(i, j):
        def pre(m, a, b=b): k = ("a", b, cap.get(("na", b), 0) + 1); cap[("na", b)] = k[2]; cap[k] = a[0].detach().float()
        def post(m, a, o, b=b): k = ("w", b, cap.get(("nw", b), 0) + 1); cap[("nw", b)] = k[2]; cap[k] = (o[0] if isinstance(o, tuple) else o).detach().float()
        hooks += [downs[b].register_forward_pre_hook(pre), downs[b].register_forward_hook(post)]
    def pre_i(m, a, kw): n_ = cap.get("ni", 0) + 1; cap["ni"] = n_; cap[("x_in", n_)] = _hs(a, kw).detach().float()
    def post_last(m, a, kw, o): n_ = cap.get("nl", 0) + 1; cap["nl"] = n_; cap[("x_out", n_)] = _out(o).detach().float()
    hooks += [M["layers"][i].register_forward_pre_hook(pre_i, with_kwargs=True), M["layers"][j - 1].register_forward_hook(post_last, with_kwargs=True)]
    with Repeat(M, rys_seq(N, i, j)):
        for s0 in range(0, W.shape[0], 4):
            cap.clear(); M["model"](W[s0:s0 + 4], use_cache=False)
            xi, xj, xj2 = cap[("x_in", 1)][:, 1:].reshape(-1, D), cap[("x_out", 1)][:, 1:].reshape(-1, D), cap[("x_out", 2)][:, 1:].reshape(-1, D)
            keep = ~(sinkmask(xi) | sinkmask(xj)); pos = torch.nonzero(keep)[:, 0]; pos = pos[torch.randperm(pos.numel(), generator=gen)[:64 if SMOKE else 128].to(DEV)]
            d1, d2 = (xj - xi)[pos], (xj2 - xj)[pos]; acc["band_cos"] += torch.nn.functional.cosine_similarity(d1, d2).tolist()
            Ab = A[i * DFF: j * DFF]; s1, _ = omp(unitr(d1), Ab); s2_, _ = omp(unitr(d2), Ab)
            acc["ledger_jaccard"] += [len(set(x) & set(y)) / len(set(x) | set(y)) for x, y in zip(s1.tolist(), s2_.tolist())]
            for b in range(i, j):
                a1, a2 = cap[("a", b, 1)][:, 1:].reshape(-1, DFF)[pos], cap[("a", b, 2)][:, 1:].reshape(-1, DFF)[pos]; w1, w2 = cap[("w", b, 1)][:, 1:].reshape(-1, D)[pos], cap[("w", b, 2)][:, 1:].reshape(-1, D)[pos]
                acc["mlp_cos"] += torch.nn.functional.cosine_similarity(w1, w2).tolist()
                t1 = (a1.abs() * WN[b]).topk(8, dim=1).indices; t2 = (a2.abs() * WN[b]).topk(8, dim=1).indices
                acc["top8_erased"] += [1 - len(set(x) & set(y)) / 8 for x, y in zip(t1.tolist(), t2.tolist())]
                acc["top8_sign_flip"] += ((a1.gather(1, t1).sign() * a2.gather(1, t1).sign()) < 0).float().mean(1).tolist()
                a1c, a2c = a1 - a1.mean(1, keepdim=True), a2 - a2.mean(1, keepdim=True); acc["act_corr"] += ((a1c * a2c).sum(1) / (a1c.norm(dim=1) * a2c.norm(dim=1)).clamp_min(1e-9)).tolist()
    for h in hooks: h.remove()
    rec = dict(kind=kind, i=i, j=j, dloss=r["dloss"], gain=r["gain"], **{k: float(np.mean(v)) for k, v in acc.items()})
    out.append(rec); print(f"[{name}] {kind} block {i}-{j} (dloss {r['dloss']:+.4f}): band increment cosine {rec['band_cos']:+.3f}, ledger Jaccard {rec['ledger_jaccard']:.3f}; per block MLP output cosine {rec['mlp_cos']:+.3f}, top-8 erased {rec['top8_erased']:.2f}, top-8 sign flips {rec['top8_sign_flip']:.2f}, activation correlation {rec['act_corr']:+.3f} ({time.time() - t0:.0f}s)", flush=True)
mean_of = lambda kind, key: float(np.mean([x[key] for x in out if x["kind"] == kind]))
res = dict(model=name, fam=M["fam"], bands=out, summary_by_kind={k: {key: mean_of(k, key) for key in ("band_cos", "ledger_jaccard", "mlp_cos", "top8_erased", "top8_sign_flip", "act_corr")} for k in ("least", "most")},
           mlp_cos_all=float(np.mean([x["mlp_cos"] for x in out])), top8_erased_all=float(np.mean([x["top8_erased"] for x in out])))
res["verdicts"] = dict(P1=mean_of("least", "band_cos") > 0.3, P2=res["top8_erased_all"] >= 0.5, P4=mean_of("least", "mlp_cos") > mean_of("most", "mlp_cos"))
summ = (f"[{name}] second traversal against the first ({M['fam']}): least damaging blocks band cosine {mean_of('least', 'band_cos'):+.3f}, MLP output cosine {mean_of('least', 'mlp_cos'):+.3f}, top-8 erased {mean_of('least', 'top8_erased'):.2f}; "
        f"most damaging band cosine {mean_of('most', 'band_cos'):+.3f}, MLP output cosine {mean_of('most', 'mlp_cos'):+.3f}, top-8 erased {mean_of('most', 'top8_erased'):.2f}; verdicts {res['verdicts']} ({time.time() - t0:.0f}s)")
save(f"e658_pass_two_{name}{tag}", res, summ); print("done", flush=True)
