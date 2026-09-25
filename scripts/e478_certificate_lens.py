"""e478: a certificate for every claim. WorkspaceBench (LessWrong, 2026-09-23) judges workspace readers on what they
surface and on how often they state things the context contradicts, and finds that the expressive readers confabulate.
A native-word readout is a decomposition, so each concept it surfaces comes with numbers a text generator cannot offer:
the coefficient of the word that carries it, the word's provenance (type and block), and a causal receipt (what
removing that one word from the state does to the model's answer). This run asks whether those numbers separate right
claims from wrong ones better than the plain lens's own confidence does.
Setup: e420's two-hop items on Qwen2.5-7B (bf16; the bridge country is never written; capability gate as e420), read at
the subject's last token (blocks 12-20, where e420 found native words surfacing the bridge) and at the final position
(blocks 22-26, where every reader sees it). Readers: the plain logit lens, 16 native words (pooled per-word lens), 16
rotated words.
A claim is a country token in a reader's top 10; it is right if it is the item's country. Certificates:
- lens: the claimed token's logit z-score over the vocabulary, and its rank;
- lens receipt: remove the token's unembedding direction from the state at that cell (its lexical component) and
  measure the drop of the answer's log-probability at the final position;
- native: the coefficient share of the carrying word (|c| over the centred state's norm), its coefficient rank among
  the 16, its provenance (token embedding, MLP row or head basis; block), and the native receipt: remove that one word
  from the state and measure the same drop.
Receipts are computed at one block per position type (subject 16, final 24).
Scored: recall of the true country and precision of country claims per reader; the AUC of each certificate for right
against wrong claims (pooled over cells); precision after keeping only the top half of claims by each certificate; and
precision of native claims carried by MLP rows only (the "computed-only" reader).
Pre-registered (honest guesses):
- the native coefficient share separates right from wrong claims at AUC at least 0.65 (0.5);
- the native receipt's AUC is at least the coefficient share's (0.5);
- at the subject token the native coefficient's AUC beats the lens z-score's (0.5);
- keeping MLP-row claims only raises native precision by at least 0.05 (0.4)."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ws_common import *
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "e420_twohop_subject.py")).read()
exec(src[src.index("FACTS = ["): src.index("name = sys.argv[1]")], globals())
name = "qwen7"; K = 16; CELLS = {"subject": [12, 14, 16, 18, 20], "final": [22, 24, 26]}; RECEIPT = {"subject": 16, "final": 24}
model, tok, fam = load_bf16(name); arch = Arch(model, fam); layers = sorted(set(CELLS["subject"]) | set(CELLS["final"]))
first = lambda s: tok(s, add_special_tokens=False)["input_ids"][0]
def enc(text): return torch.tensor(tok(text, add_special_tokens=False)["input_ids"])
items = []
for lm, country, capital, lang, cur in FACTS:
    for rel, ans in [("capital", capital), ("language", lang), ("currency", cur)]:
        if ans is None: continue
        q, _ = TPL[rel]; shots = "".join(f"{q.format(a)} {b}.\n" for a, b in SHOT[rel]); head = shots + q.split("{}")[0] + lm
        items.append(dict(country=country, ids=enc(shots + q.format(lm)), spos=len(enc(head)) - 1, onehop=enc("".join(f"{ONEHOP[0].format(a)} {b}.\n" for a, b in ONEHOP[1]) + ONEHOP[0].format(lm[0].upper() + lm[1:])),
                          bridge_ids={first(" " + country), first(country)}, answer_ids={first(" " + ans), first(ans)}))
_, lg2 = last_states(model, arch, [it["ids"] for it in items], []); _, lg1 = last_states(model, arch, [it["onehop"] for it in items], [])
keep = [i for i, it in enumerate(items) if lg2[i].argmax().item() in it["answer_ids"] and lg1[i].argmax().item() in it["bridge_ids"]]
for i in keep: items[i]["ans_tok"] = lg2[i].argmax().item()
items = [items[i] for i in keep]; N = len(items); log(f"{name}: {N} of {len(lg2)} two-hop prompts pass the gate")
ev_all = torch.load("/workspace/wdd/eval_ids_all.pt", weights_only=False)["qwen05"]["eval_ids"][:4]
ref = ref_stats(model, arch, ev_all, layers); gain, WU = final_readout(model, arch)
A, blk, typ, ends = lean_dictionary(arch, max(layers))
g = torch.Generator().manual_seed(7); R = torch.linalg.qr(torch.randn(arch.D, arch.D, generator=g))[0].to(DEV)
def states_at(items, layers, where):
    out = {l: [] for l in layers}
    for it in items:
        cap = {}; p = it["spos"] if where == "subject" else -1
        hs = [arch.layers[l].register_forward_hook((lambda l_: lambda m, i, o: cap.__setitem__(l_, (o[0] if isinstance(o, tuple) else o)[0, p].float()))(l)) for l in layers]
        try:
            with torch.no_grad(): model(it["ids"][None].to(DEV))
        finally: [h.remove() for h in hs]
        for l in layers: out[l].append(cap[l])
    return {l: torch.stack(v) for l, v in out.items()}
def splice_at(ids, L, pos, x_new):
    B = x_new.shape[0]; ids_b = ids[None].expand(B, -1).to(DEV)
    def hk(m, i, o):
        xo = o[0] if isinstance(o, tuple) else o; y = xo.clone(); y[:, pos] = x_new.to(xo.dtype)
        return (y,) + tuple(o[1:]) if isinstance(o, tuple) else y
    h = arch.layers[L].register_forward_hook(hk)
    try:
        with torch.no_grad(): return model(ids_b).logits[:, -1].float()
    finally: h.remove()
countries = sorted({f[1] for f in FACTS}); ctok = {c: first(" " + c) for c in countries}; ctoks = torch.tensor(sorted(set(ctok.values())), device=DEV)
true_c = torch.tensor([ctok[it["country"]] for it in items], device=DEV)
def auc(right, score):
    r = torch.tensor(right, dtype=torch.bool); s = torch.tensor(score, dtype=torch.float)
    if r.all() or (~r).all(): return None
    a, b = s[r], s[~r]; return float(((a[:, None] > b[None, :]).float() + 0.5 * (a[:, None] == b[None, :]).float()).mean())
def top_half_precision(right, score):
    s = torch.tensor(score, dtype=torch.float); r = torch.tensor(right, dtype=torch.float); m = s >= s.median(); return float(r[m].mean()) if m.any() else None
res = dict(model=name, n=N, k=K, cells=CELLS, receipt_blocks=RECEIPT, by_position={})
for where, Ls in CELLS.items():
    X_all = states_at(items, Ls, where); claims = {"lens": [], "native": [], "rotated": []}; recall = {}
    for L in Ls:
        X = X_all[L]; mu = ref[L]["mu"]; Xc = X - mu; rms = X.pow(2).mean(-1).sqrt(); An = A[:ends[L]]; xn = Xc.norm(dim=-1)
        sel, _, _ = omp(Xc, An, K, batch=64, record_err=False); cof, _ = refit(Xc, An, sel); words = An[sel]
        selr, _, _ = omp(Xc @ R.T, An, K, batch=64, record_err=False); cofr, _ = refit(Xc @ R.T, An, selr); wordsr = An[selr] @ R
        S = dict(lens=lens_scores(X, gain, WU, rms), native=pooled_word_scores(words, cof, gain, WU, rms), rotated=pooled_word_scores(wordsr, cofr, gain, WU, rms))
        per = {"native": torch.stack([lens_scores(words[:, i] * cof[:, i:i + 1], gain, WU, rms)[:, ctoks] for i in range(K)], 1),          # [N, K, C]
               "rotated": torch.stack([lens_scores(wordsr[:, i] * cofr[:, i:i + 1], gain, WU, rms)[:, ctoks] for i in range(K)], 1)}
        pos = items[0]["spos"] if where == "subject" else -1
        for rd, sc in S.items():
            top10 = sc.topk(10, dim=-1).indices; z = (sc - sc.mean(-1, keepdim=True)) / sc.std(-1, keepdim=True)
            recall[f"{rd}@{L}"] = float((top10 == true_c[:, None]).any(-1).float().mean())
            rec_items = {}
            for n in range(N):
                for r_, t in enumerate(top10[n].tolist()):
                    if t not in ctok.values(): continue
                    cl = dict(item=n, layer=L, token=t, right=(t == int(true_c[n])), z=float(z[n, t]), rank=-(r_ + 1))
                    if rd in per:
                        ci = int((ctoks == t).nonzero()[0, 0]); i = int(per[rd][n, :, ci].argmax()); c_ = cof if rd == "native" else cofr; s_ = sel if rd == "native" else selr
                        cl.update(word=i, coef_share=float(c_[n, i].abs() / xn[n]), coef_rank=-int((c_[n].abs() > c_[n, i].abs()).sum()), wtype=int(typ[s_[n, i]]), wblock=int(blk[s_[n, i]]))
                    claims[rd].append(cl)
            # receipts at one block per position type
            if L == RECEIPT[where] and rd in ("lens", "native"):
                for n in range(N):
                    mine = [c for c in claims[rd] if c["item"] == n and c["layer"] == L]
                    if not mine: continue
                    if rd == "lens":
                        U = WU[[c["token"] for c in mine]].float(); U = U / U.norm(dim=-1, keepdim=True); xs = X[n][None] - (X[n][None] @ U.T).T * U
                    else: xs = torch.stack([X[n] - cof[n, c["word"]] * words[n, c["word"]] for c in mine])
                    lp = torch.log_softmax(splice_at(items[n]["ids"], L, items[n]["spos"] if where == "subject" else items[n]["ids"].numel() - 1, torch.cat([X[n][None], xs])), -1); a = items[n]["ans_tok"]
                    for j, c in enumerate(mine): c["receipt"] = float(lp[0, a] - lp[j + 1, a])
        del S, per, sel, selr, words, wordsr; torch.cuda.empty_cache()
    out = dict(recall_at10={k: v for k, v in recall.items()}, claims={})
    for rd, cl in claims.items():
        if not cl: out["claims"][rd] = dict(n=0); continue
        right = [c["right"] for c in cl]; d = dict(n=len(cl), precision=sum(right) / len(cl), auc_z=auc(right, [c["z"] for c in cl]), auc_rank=auc(right, [c["rank"] for c in cl]),
                                                   top_half_precision_z=top_half_precision(right, [c["z"] for c in cl]))
        rc = [c for c in cl if "receipt" in c]
        if rc: d.update(n_receipt=len(rc), auc_receipt=auc([c["right"] for c in rc], [c["receipt"] for c in rc]), top_half_precision_receipt=top_half_precision([c["right"] for c in rc], [c["receipt"] for c in rc]), precision_receipt_subset=sum(c["right"] for c in rc) / len(rc))
        if rd in ("native", "rotated"):
            d.update(auc_coef_share=auc(right, [c["coef_share"] for c in cl]), auc_coef_rank=auc(right, [c["coef_rank"] for c in cl]), top_half_precision_coef=top_half_precision(right, [c["coef_share"] for c in cl]))
            mlp = [c for c in cl if c["wtype"] == T_MLP]; tk = [c for c in cl if c["wtype"] == T_TOK]
            d.update(share_carried_by_mlp=len(mlp) / len(cl), share_carried_by_token=len(tk) / len(cl), precision_mlp_only=(sum(c["right"] for c in mlp) / len(mlp)) if mlp else None, precision_token_only=(sum(c["right"] for c in tk) / len(tk)) if tk else None,
                     right_claims_by_type=dict(mlp=sum(c["right"] for c in mlp), token=sum(c["right"] for c in tk), head=sum(c["right"] for c in cl if c["wtype"] == T_ATT)))
        out["claims"][rd] = d
    res["by_position"][where] = out
    f = lambda v: "n/a" if v is None else f"{v:.2f}"
    log(f"{name} {where}: recall@10 by cell " + ", ".join(f"{k} {v:.2f}" for k, v in recall.items()) + " | " + " | ".join(f"{rd}: n {d.get('n', 0)}, precision {f(d.get('precision'))}, AUC z {f(d.get('auc_z'))} rank {f(d.get('auc_rank'))} coef {f(d.get('auc_coef_share'))} coef-rank {f(d.get('auc_coef_rank'))} receipt {f(d.get('auc_receipt'))} (n {d.get('n_receipt', 0)}); "
                                                                                                                      f"precision top half by z {f(d.get('top_half_precision_z'))} coef {f(d.get('top_half_precision_coef'))} receipt {f(d.get('top_half_precision_receipt'))}; MLP-only {f(d.get('precision_mlp_only'))} (share {f(d.get('share_carried_by_mlp'))}), token-only {f(d.get('precision_token_only'))}" for rd, d in out["claims"].items()))
S_, F_ = res["by_position"]["subject"]["claims"], res["by_position"]["final"]["claims"]
g_ = lambda d, k: d.get(k) if d.get(k) is not None else 0.0
res["checks"] = dict(coef_auc_over_0_65=all(g_(d["native"], "auc_coef_share") >= 0.65 for d in (S_, F_)), receipt_auc_at_least_coef=all(g_(d["native"], "auc_receipt") >= g_(d["native"], "auc_coef_share") for d in (S_, F_)),
                     subject_native_coef_beats_lens_z=g_(S_["native"], "auc_coef_share") > g_(S_["lens"], "auc_z"), mlp_only_raises_precision=all(g_(d["native"], "precision_mlp_only") >= g_(d["native"], "precision") + 0.05 for d in (S_, F_)))
f = lambda v: "n/a" if v is None else f"{v:.2f}"
summ = (f"{name} n={N}: " + " || ".join(f"{where}: native claims {d['native']['n']} (precision {f(d['native'].get('precision'))}), lens claims {d['lens']['n']} ({f(d['lens'].get('precision'))}), rotated {d['rotated']['n']} ({f(d['rotated'].get('precision'))}); "
        f"AUC right-vs-wrong: lens z {f(d['lens'].get('auc_z'))}, lens receipt {f(d['lens'].get('auc_receipt'))}, native coefficient share {f(d['native'].get('auc_coef_share'))}, native coefficient rank {f(d['native'].get('auc_coef_rank'))}, native receipt {f(d['native'].get('auc_receipt'))}, rotated coefficient {f(d['rotated'].get('auc_coef_share'))}; "
        f"precision keeping the top half: lens by z {f(d['lens'].get('top_half_precision_z'))}, native by coefficient {f(d['native'].get('top_half_precision_coef'))}, by receipt {f(d['native'].get('top_half_precision_receipt'))}; native claims carried by MLP rows {f(d['native'].get('share_carried_by_mlp'))} with precision {f(d['native'].get('precision_mlp_only'))}, by token embeddings {f(d['native'].get('share_carried_by_token'))} with precision {f(d['native'].get('precision_token_only'))}; "
        f"true-country recall@10 best cell lens {max(v for k, v in res['by_position'][where]['recall_at10'].items() if k.startswith('lens')):.2f}, native {max(v for k, v in res['by_position'][where]['recall_at10'].items() if k.startswith('native')):.2f}" for where, d in ((w, res["by_position"][w]["claims"]) for w in ("subject", "final")))
        + f" | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e478_certificate_{name}", res, summ)
