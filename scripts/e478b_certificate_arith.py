"""e478b: certificates where fabrications occur. e478 found almost no wrong country claims in the two-hop family, so its
certificates had nothing to filter. e416 found the native reader fabricates digits in chained arithmetic ("a=3;b=a+4;
c=b-2;c=": any digit outside {a, the two operands, b, c} is a fabrication, and the reader ranked more of them than the
plain lens). This run asks whether the native reader's own numbers (the carrying word's coefficient share and rank,
its provenance, and a causal receipt) separate right digit claims from fabricated ones, against the plain lens's
confidence (z-score, rank) and its lexical receipt.
Setup as e416 (Qwen2.5-7B, 2-step chains with five distinct digits, three worked examples, kept when the model answers
c), read at the final position after blocks 12-26. A claim is a digit in a reader's top 3 among the ten digit tokens
(e416's fabrication measure) and, separately, any digit in the reader's top 10 over the vocabulary; a claim is right
if the digit is a quantity of the chain (a, the two operands, b or c) and a fabrication otherwise. Receipts at block
20: remove the carrying word (native) or the digit's unembedding direction (lens) from the state and measure the drop
of the log-probability of c.
Pre-registered (honest guesses):
- fabricated native claims sit on lower coefficient shares than right ones (AUC at least 0.65) (0.5);
- the native receipt's AUC is at least the coefficient share's (0.5);
- keeping the top half of native claims by coefficient share raises precision by at least 0.1 (0.5);
- the lens's z-score AUC is at least as high as the native coefficient's (0.5)."""
import sys, os, json as _json, random; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ws_common import *
name = "qwen7"; K = 16; LAYERS = list(range(12, 27, 2)); RECEIPT_L = 20
model, tok, fam = load_bf16(name); arch = Arch(model, fam)
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
items = [it for i, it in enumerate(items) if lg[i].argmax().item() == dig[it["c"]].item()]; N = len(items); log(f"{name}: {N} of {len(lg)} chains answered correctly")
ev_all = torch.load("/workspace/wdd/eval_ids_all.pt", weights_only=False)["qwen05"]["eval_ids"][:4]
ref = ref_stats(model, arch, ev_all, LAYERS); gain, WU = final_readout(model, arch)
A, blk, typ, ends = lean_dictionary(arch, max(LAYERS))
g = torch.Generator().manual_seed(7); R = torch.linalg.qr(torch.randn(arch.D, arch.D, generator=g))[0].to(DEV)
X_all, _ = last_states(model, arch, [it["ids"] for it in items], LAYERS)
chain = torch.zeros(N, 10, dtype=torch.bool, device=DEV)
for n, it in enumerate(items):
    for v in (it["a"], it["d"], it["e"], it["b"], it["c"]): chain[n, v] = True
def auc(right, score):
    r = torch.tensor(right, dtype=torch.bool); s = torch.tensor(score, dtype=torch.float)
    if r.all() or (~r).all(): return None
    a, b = s[r], s[~r]; return float(((a[:, None] > b[None, :]).float() + 0.5 * (a[:, None] == b[None, :]).float()).mean())
def top_half_precision(right, score):
    s = torch.tensor(score, dtype=torch.float); r = torch.tensor(right, dtype=torch.float); m = s >= s.median(); return float(r[m].mean()) if m.any() else None
claims = {mode: {"lens": [], "native": [], "rotated": []} for mode in ("top3_digits", "top10_vocab")}
for L in LAYERS:
    X = X_all[L]; mu = ref[L]["mu"]; Xc = X - mu; rms = X.pow(2).mean(-1).sqrt(); An = A[:ends[L]]; xn = Xc.norm(dim=-1)
    sel, _, _ = omp(Xc, An, K, batch=64, record_err=False); cof, _ = refit(Xc, An, sel); words = An[sel]
    selr, _, _ = omp(Xc @ R.T, An, K, batch=64, record_err=False); cofr, _ = refit(Xc @ R.T, An, selr); wordsr = An[selr] @ R
    S = dict(lens=lens_scores(X, gain, WU, rms), native=pooled_word_scores(words, cof, gain, WU, rms), rotated=pooled_word_scores(wordsr, cofr, gain, WU, rms))
    per = {"native": torch.stack([lens_scores(words[:, i] * cof[:, i:i + 1], gain, WU, rms)[:, dig] for i in range(K)], 1), "rotated": torch.stack([lens_scores(wordsr[:, i] * cofr[:, i:i + 1], gain, WU, rms)[:, dig] for i in range(K)], 1)}
    for rd, sc in S.items():
        z = (sc - sc.mean(-1, keepdim=True)) / sc.std(-1, keepdim=True); ds = sc[:, dig]; order3 = ds.argsort(-1, descending=True)[:, :3]; top10 = sc.topk(10, dim=-1).indices
        for mode in claims:
            for n in range(N):
                cand = [(int(v), r_) for r_, v in enumerate(order3[n].tolist())] if mode == "top3_digits" else [(int((dig == t).nonzero()[0, 0]), r_) for r_, t in enumerate(top10[n].tolist()) if t in dig.tolist()]
                for v, r_ in cand:
                    cl = dict(item=n, layer=L, digit=v, right=bool(chain[n, v]), z=float(z[n, dig[v]]), rank=-(r_ + 1))
                    if rd in per:
                        i = int(per[rd][n, :, v].argmax()); c_ = cof if rd == "native" else cofr; s_ = sel if rd == "native" else selr
                        cl.update(word=i, coef_share=float(c_[n, i].abs() / xn[n]), coef_rank=-int((c_[n].abs() > c_[n, i].abs()).sum()), wtype=int(typ[s_[n, i]]), wblock=int(blk[s_[n, i]]))
                    claims[mode][rd].append(cl)
        if L == RECEIPT_L and rd in ("lens", "native"):
            for n in range(N):
                mine = [c for c in claims["top3_digits"][rd] if c["item"] == n and c["layer"] == L]
                if rd == "lens":
                    U = WU[dig[[c["digit"] for c in mine]]].float(); U = U / U.norm(dim=-1, keepdim=True); xs = X[n][None] - (X[n][None] @ U.T).T * U
                else: xs = torch.stack([X[n] - cof[n, c["word"]] * words[n, c["word"]] for c in mine])
                lp = torch.log_softmax(splice_last(model, arch, items[n]["ids"], L, torch.cat([X[n][None], xs])), -1); cc = int(dig[items[n]["c"]])
                for j, c in enumerate(mine): c["receipt"] = float(lp[0, cc] - lp[j + 1, cc])
    del S, per, sel, selr, words, wordsr; torch.cuda.empty_cache()
res = dict(model=name, n=N, k=K, layers=LAYERS, receipt_block=RECEIPT_L, modes={})
for mode, byrd in claims.items():
    out = {}
    for rd, cl in byrd.items():
        if not cl: out[rd] = dict(n=0, precision=None); continue
        right = [c["right"] for c in cl]; d = dict(n=len(cl), precision=sum(right) / len(cl), auc_z=auc(right, [c["z"] for c in cl]), auc_rank=auc(right, [c["rank"] for c in cl]), top_half_precision_z=top_half_precision(right, [c["z"] for c in cl]))
        rc = [c for c in cl if "receipt" in c]
        if rc: d.update(n_receipt=len(rc), precision_receipt_subset=sum(c["right"] for c in rc) / len(rc), auc_receipt=auc([c["right"] for c in rc], [c["receipt"] for c in rc]), top_half_precision_receipt=top_half_precision([c["right"] for c in rc], [c["receipt"] for c in rc]))
        if rd in ("native", "rotated"):
            d.update(auc_coef_share=auc(right, [c["coef_share"] for c in cl]), auc_coef_rank=auc(right, [c["coef_rank"] for c in cl]), top_half_precision_coef=top_half_precision(right, [c["coef_share"] for c in cl]))
            for tname, tcode in (("mlp", T_MLP), ("token", T_TOK), ("head", T_ATT)):
                sub = [c for c in cl if c["wtype"] == tcode]; d[f"share_{tname}"] = len(sub) / len(cl); d[f"precision_{tname}"] = (sum(c["right"] for c in sub) / len(sub)) if sub else None
        out[rd] = d
    res["modes"][mode] = out
    f = lambda v: "n/a" if v is None else f"{v:.2f}"
    log(f"{name} {mode}: " + " | ".join(f"{rd}: n {d['n']}, precision {f(d.get('precision'))}, AUC z {f(d.get('auc_z'))} rank {f(d.get('auc_rank'))} coef {f(d.get('auc_coef_share'))} coef-rank {f(d.get('auc_coef_rank'))} receipt {f(d.get('auc_receipt'))} (n {d.get('n_receipt', 0)}, precision there {f(d.get('precision_receipt_subset'))}); top-half precision z {f(d.get('top_half_precision_z'))} coef {f(d.get('top_half_precision_coef'))} receipt {f(d.get('top_half_precision_receipt'))}; by type mlp {f(d.get('share_mlp'))}/{f(d.get('precision_mlp'))} token {f(d.get('share_token'))}/{f(d.get('precision_token'))}" for rd, d in out.items()))
M3 = res["modes"]["top3_digits"]; g_ = lambda d, k: d.get(k) if d.get(k) is not None else 0.0
res["checks"] = dict(coef_auc_over_0_65=g_(M3["native"], "auc_coef_share") >= 0.65, receipt_auc_at_least_coef=g_(M3["native"], "auc_receipt") >= g_(M3["native"], "auc_coef_share"),
                     top_half_coef_raises_precision_0_1=g_(M3["native"], "top_half_precision_coef") >= g_(M3["native"], "precision") + 0.1, lens_z_at_least_native_coef=g_(M3["lens"], "auc_z") >= g_(M3["native"], "auc_coef_share"))
f = lambda v: "n/a" if v is None else f"{v:.2f}"
summ = (f"{name} n={N} chains | " + " || ".join(f"{mode}: precision lens {f(d['lens'].get('precision'))}, native {f(d['native'].get('precision'))}, rotated {f(d['rotated'].get('precision'))} ({d['rotated']['n']} claims); AUC right-vs-fabricated: lens z {f(d['lens'].get('auc_z'))} rank {f(d['lens'].get('auc_rank'))} receipt {f(d['lens'].get('auc_receipt'))}; native coefficient share {f(d['native'].get('auc_coef_share'))} rank {f(d['native'].get('auc_coef_rank'))} receipt {f(d['native'].get('auc_receipt'))} z {f(d['native'].get('auc_z'))}; rotated coefficient {f(d['rotated'].get('auc_coef_share'))}; "
        f"precision keeping the top half: lens by z {f(d['lens'].get('top_half_precision_z'))}, native by coefficient {f(d['native'].get('top_half_precision_coef'))} by receipt {f(d['native'].get('top_half_precision_receipt'))} by z {f(d['native'].get('top_half_precision_z'))}; native claims by type: MLP {f(d['native'].get('share_mlp'))} (precision {f(d['native'].get('precision_mlp'))}), token {f(d['native'].get('share_token'))} ({f(d['native'].get('precision_token'))})" for mode, d in res["modes"].items())
        + f" | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e478b_certificate_arith_{name}", res, summ)
