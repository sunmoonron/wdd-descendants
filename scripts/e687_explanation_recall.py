"""e687 (session 155): the recall of a neuron's explanation, label-free. Gao's point: a pattern that fits every activating
example need not activate on every example that fits the pattern; auditors see precision, not recall. Here the
explanation is a surface rule learned from the neuron's own class (its top 0.5% of positions by activation on Pile
text): lookup tables of the current token, the previous token and the bigram, each entry's class rate, combined by the
maximum; scored on held-out text by precision at recall 0.9 and by the rule's size, the fewest entries that reach
recall 0.9 on the training text at precision 0.5 or better (none if no size does). A depth-6 decision tree on the
table scores, the position and token flags as the second explanation. Three depths, 300 neurons each.
Arguments: --model M [--smoke]. Pre-registered in e687_prereg.json."""
import argparse, re
from c11 import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_curve
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--smoke", action="store_true"); A_ = ap.parse_args(); SMOKE = A_.smoke
name = A_.model; TAG = f"e687_{name}" + ("_smoke" if SMOKE else ""); M = load(name); NB, D, DFF = M["NB"], M["D"], M["DFF"]; T = 256
NTR = 32 if SMOKE else 384; NTE = 16 if SMOKE else 192; BLOCKS = sorted(set([max(1, NB // 4), NB // 2, (3 * NB) // 4])); NNEU = 30 if SMOKE else 300; FRAC = 0.005; L_ = lambda s: log(TAG, s)
pile = pile_ids(M); trw = windows_from(pile[: len(pile) // 2], NTR, T, seed=687).to(DEV); tew = windows_from(pile[len(pile) // 2:], NTE, T, seed=688).to(DEV)
def acts(wins):
    out = {b: [] for b in BLOCKS}
    with Capture(M, BLOCKS, ["act"], stop_after=max(BLOCKS)) as C:
        for s0 in range(0, wins.shape[0], 8):
            run_base(M, wins[s0:s0 + 8])
            for b in BLOCKS: out[b].append(C.cap[("act", b)][:, 2:].float().reshape(-1, DFF).half())
    return {b: torch.cat(v) for b, v in out.items()}
ATR, ATE = acts(trw), acts(tew)
def feats(wins):
    cur = wins[:, 2:].reshape(-1); prv = wins[:, 1:-1].reshape(-1); prv2 = wins[:, :-2].reshape(-1); pos = torch.arange(2, T, device=DEV)[None].expand(wins.shape[0], -1).reshape(-1)
    return cur.cpu().numpy(), prv.cpu().numpy(), prv2.cpu().numpy(), pos.cpu().numpy()
ctr, ptr, p2tr, postr = feats(trw); cte, pte, p2te, poste = feats(tew); V = M["V"]
toks = [M["tok"].decode([t]) for t in range(V)]; is_punct = np.array([bool(re.fullmatch(r"\s*[^\w\s]+\s*", t)) for t in toks]); is_cap = np.array([bool(re.match(r"\s*[A-Z]", t)) for t in toks]); is_dig = np.array([bool(re.search(r"\d", t)) for t in toks]); is_sp = np.array([t.startswith(" ") for t in toks]); is_nl = np.array(["\n" in t for t in toks]); tlen = np.array([len(t.strip()) for t in toks])
big_tr = ptr.astype(np.int64) * V + ctr; big_te = pte.astype(np.int64) * V + cte
def table(keys, y, alpha=1.0):
    """class rate per key with Laplace smoothing toward the base rate"""
    base = y.mean(); k, inv = np.unique(keys, return_inverse=True); cnt = np.bincount(inv); hit = np.bincount(inv, weights=y); return dict(zip(k.tolist(), ((hit + alpha * base) / (cnt + alpha)).tolist())), base
def lookup(tab, keys, base): return np.array([tab.get(int(k), base) for k in keys])
def rule_size(scores_tr, y, entries_tr, precision_min=0.5, recall_target=0.9):
    """greedy: add entries (tokens or bigrams) by class rate until train recall >= target while precision >= min; the count, or None"""
    order = sorted(entries_tr.items(), key=lambda kv: -kv[1][0]); hit = 0; cov = 0; tot = int(y.sum())
    for n, (e, (rate, nh, nc)) in enumerate(order, 1):
        hit += nh; cov += nc
        if hit / max(tot, 1) >= recall_target: return n if hit / max(cov, 1) >= precision_min else None
        if hit / max(cov, 1) < precision_min and n > 5: return None
    return None
res = dict(model=M["hf"], n_blocks=NB, blocks={}, train_positions=int(len(ctr)), test_positions=int(len(cte)), class_fraction=FRAC)
for b in BLOCKS:
    A1, A2 = ATR[b], ATE[b]; mx = A1.float().max(0).values; cand = torch.nonzero(mx > (1.0 if M["fam"] != "gated" else float(mx.quantile(0.5))))[:, 0].tolist(); rng = np.random.default_rng(687 + b); neu = sorted(rng.choice(cand, min(NNEU, len(cand)), replace=False).tolist())
    K = max(20, int(A1.shape[0] * FRAC)); rows = []
    for j in neu:
        a1 = A1[:, j].float(); thr = float(a1.topk(K).values[-1]); y = (a1 >= thr).cpu().numpy().astype(float); y2 = (A2[:, j].float() >= thr).cpu().numpy().astype(float)
        if y2.sum() < 5: continue
        tc, base = table(ctr, y); tp, _ = table(ptr, y); tb, _ = table(big_tr, y)
        s_te = np.maximum.reduce([lookup(tc, cte, base), lookup(tp, pte, base), lookup(tb, big_te, base)]); s_tr = np.maximum.reduce([lookup(tc, ctr, base), lookup(tp, ptr, base), lookup(tb, big_tr, base)])
        pr, rc, _ = precision_recall_curve(y2, s_te); pat = lambda r_: float(pr[rc >= r_].max()) if (rc >= r_).any() else 0.0; p90, p70, p50 = pat(0.9), pat(0.7), pat(0.5); rat = lambda p_: float(rc[pr >= p_].max()) if (pr >= p_).any() else 0.0; r50, r80 = rat(0.5), rat(0.8); ap_ = float(-np.sum(np.diff(rc) * pr[:-1]))
        # entries: per token and per bigram, the class rate, hits and coverage on train
        ent = {}
        for keys, tab, kind in ((ctr, tc, "t"), (big_tr, tb, "b")):
            k_, inv = np.unique(keys, return_inverse=True); nh = np.bincount(inv, weights=y); nc = np.bincount(inv)
            for i_, kk in enumerate(k_.tolist()):
                if nh[i_] > 0: ent[(kind, kk)] = (nh[i_] / nc[i_], nh[i_], nc[i_])
        size = rule_size(s_tr, y, ent)
        Xtr = np.stack([lookup(tc, ctr, base), lookup(tp, ptr, base), lookup(tb, big_tr, base), lookup(tc, p2tr, base), np.log(postr), is_punct[ctr], is_cap[ctr], is_dig[ctr], is_sp[ctr], is_nl[ctr], tlen[ctr], is_punct[ptr]], 1)
        Xte = np.stack([lookup(tc, cte, base), lookup(tp, pte, base), lookup(tb, big_te, base), lookup(tc, p2te, base), np.log(poste), is_punct[cte], is_cap[cte], is_dig[cte], is_sp[cte], is_nl[cte], tlen[cte], is_punct[pte]], 1)
        tree = DecisionTreeClassifier(max_depth=6, class_weight="balanced", min_samples_leaf=5, random_state=0).fit(Xtr, y); st = tree.predict_proba(Xte)[:, 1]; pr2, rc2, _ = precision_recall_curve(y2, st); p90t = float(pr2[rc2 >= 0.9].max()) if (rc2 >= 0.9).any() else 0.0; r50t = float(rc2[pr2 >= 0.5].max()) if (pr2 >= 0.5).any() else 0.0
        one_tok = float(np.bincount(ctr[y > 0]).max() / y.sum()); unseen = float(np.mean([(int(k) not in tc) and (int(q) not in tb) for k, q in zip(cte[y2 > 0], big_te[y2 > 0])]))   # test positives whose token and bigram never fired in training
        rows.append(dict(neuron=j, n_test_class=int(y2.sum()), precision_at_recall90=p90, precision_at_recall70=p70, precision_at_recall50=p50, recall_at_precision50=r50, recall_at_precision80=r80, ap=ap_, rule_size=size, tree_precision_at_recall90=p90t, tree_recall_at_precision50=r50t, unseen_share=unseen, one_token_share=one_tok, top_token=toks[int(np.bincount(ctr[y > 0]).argmax())]))
    sizes = [r["rule_size"] for r in rows]; S = dict(n=len(rows), median_precision_at_recall90=med([r["precision_at_recall90"] for r in rows]), median_precision_at_recall70=med([r["precision_at_recall70"] for r in rows]), median_precision_at_recall50=med([r["precision_at_recall50"] for r in rows]),
                                              median_recall_at_precision50=med([r["recall_at_precision50"] for r in rows]), median_recall_at_precision80=med([r["recall_at_precision80"] for r in rows]), share_recall_at_precision50_above_09=mean([float(r["recall_at_precision50"] >= 0.9) for r in rows]), share_recall_at_precision50_below_05=mean([float(r["recall_at_precision50"] < 0.5) for r in rows]),
                                              share_precision90_above_06=mean([float(r["precision_at_recall90"] >= 0.6) for r in rows]), share_precision90_below_03=mean([float(r["precision_at_recall90"] < 0.3) for r in rows]), median_unseen_share=med([r["unseen_share"] for r in rows]),
                                              median_ap=med([r["ap"] for r in rows]), rule_exists_share=mean([float(s is not None) for s in sizes]), median_rule_size_when_exists=med([s for s in sizes if s is not None]), share_rule_size_le10=mean([float(s is not None and s <= 10) for s in sizes]),
                                              median_tree_precision_at_recall90=med([r["tree_precision_at_recall90"] for r in rows]), median_tree_recall_at_precision50=med([r["tree_recall_at_precision50"] for r in rows]), median_one_token_share=med([r["one_token_share"] for r in rows]))
    res["blocks"][str(b)] = dict(summary=S, neurons=rows); L_(f"block {b}: {S['n']} neurons; held-out precision at recall 0.9/0.7/0.5, median {S['median_precision_at_recall90']:.3f}/{S['median_precision_at_recall70']:.3f}/{S['median_precision_at_recall50']:.3f}; recall at precision 0.5/0.8, median {S['median_recall_at_precision50']:.3f}/{S['median_recall_at_precision80']:.3f} (recall at 0.5 >= 0.9 for {S['share_recall_at_precision50_above_09']:.2f}, < 0.5 for {S['share_recall_at_precision50_below_05']:.2f}); "
       f"test positives with an unseen token and bigram {S['median_unseen_share']:.2f}; AP {S['median_ap']:.3f}; a train rule exists for {S['rule_exists_share']:.2f}, median size {S['median_rule_size_when_exists']} (<= 10 for {S['share_rule_size_le10']:.2f}); tree recall at precision 0.5 {S['median_tree_recall_at_precision50']:.3f}; one-token share {S['median_one_token_share']:.2f}")
mid = res["blocks"][str(NB // 2)]["summary"]
V = dict(X1_gao_precision90_median_below_03=bool(mid["median_precision_at_recall90"] < 0.3), X1_nick_above_06=bool(mid["median_precision_at_recall90"] >= 0.6), X2_gao_rule_size_above_50_or_none=bool(mid["rule_exists_share"] < 0.5 or (mid["median_rule_size_when_exists"] or 1e9) > 50), X2_nick_rule_size_le10=bool(mid["share_rule_size_le10"] >= 0.5),
         X3_surface_covers_half_at_precision_half=bool(mid["median_recall_at_precision50"] >= 0.5))
res["verdicts"] = V
summ = f"[{name}] explanation recall, blocks {BLOCKS}: held-out recall a precision-0.5 surface rule reaches, median " + ", ".join(f"{b}: {res['blocks'][str(b)]['summary']['median_recall_at_precision50']:.2f} (precision at recall 0.9 {res['blocks'][str(b)]['summary']['median_precision_at_recall90']:.2f}; unseen-context positives {res['blocks'][str(b)]['summary']['median_unseen_share']:.2f}; train rule size {res['blocks'][str(b)]['summary']['median_rule_size_when_exists']})" for b in BLOCKS) + f"; verdicts {V}"
save(TAG, res, summ)
