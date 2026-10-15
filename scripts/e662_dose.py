"""e662 (session 145): the dose response of an attribute edit. How many rows does an attribute need, and does the WDD
vote order them better than a naive or random order? e659 found whole relay chains rescue edits a single row cannot
make, with the naive top row of each chain block nearly as good. Here the k first rows of three orders are negated
together, k = 1, 2, 3, 4, 6, 8, 12, 16, 24, 32: the WDD vote order (every row among the top ten of any level's vote,
ranked by its best share), a naive global order (every MLP row of blocks 0..L ranked by its normalized activation
difference between the poles on discovery items, no dictionary), and three random orders over the same blocks.
Scored on held-out items by starting pole (e661), with the wikitext KL from the unedited model. Reuses e661's setup.
Arguments: model [--smoke]."""
import sys
src = open("e661_markedness.py").read(); exec(src[src.index("import sys\n"): src.index("# ==== SETUP END ====")], globals())
KS = [1, 2, 4, 8] if SMOKE else [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
RES = dict(model=name, ks=KS, attrs={})
for a in ATTRS:
    best = {}
    for b in range(L + 1):
        for bb, r, sh in VOTE[a][b]:
            if sh > best.get((bb, r), -1): best[(bb, r)] = sh
    wdd_order = sorted(best, key=lambda x: -best[x])
    its = [it for it in ITEMS[a] if it["disc"]]; A_ = torch.stack([it["A"] for it in its]); s_ = torch.tensor([it["side"] for it in its])
    dn = (A_[s_ == 1].mean(0) - A_[s_ == 0].mean(0)).abs() / (A_.std(0) + 1e-6); flat = dn.reshape(-1).argsort(descending=True)[:max(KS)].tolist(); naive_order = [divmod(i_, DFF) for i_ in flat]
    rnd_orders = [[(int(rng.integers(L + 1)), int(rng.integers(DFF))) for _ in range(max(KS))] for _ in range(2 if SMOKE else 3)]
    out = {"wdd": {}, "naive": {}, "random": {}, "n_wdd_candidates": len(wdd_order)}
    for k in KS:
        if k <= len(wdd_order):
            with Negate(wdd_order[:k]): out["wdd"][str(k)] = by_pole(a) | {"kl": kl_off()}
        with Negate(naive_order[:k]): out["naive"][str(k)] = by_pole(a) | {"kl": kl_off()}
        rr = []
        for ro in rnd_orders:
            with Negate(ro[:k]): rr.append(by_pole(a) | {"kl": kl_off()})
        out["random"][str(k)] = {kk: float(np.mean([x[kk] for x in rr])) for kk in rr[0]}
        tot = lambda d: d["flip_from_0"] + d["flip_from_1"]
        log(f"{a} k={k}: flips (from pole0 + pole1) wdd {tot(out['wdd'][str(k)]) if str(k) in out['wdd'] else float('nan'):.2f}, naive {tot(out['naive'][str(k)]):.2f}, random {tot(out['random'][str(k)]):.2f}; KL wdd {out['wdd'].get(str(k), {}).get('kl', float('nan')):.4f} naive {out['naive'][str(k)]['kl']:.4f}")
    top8 = wdd_order[:8]; mag = {t: float(torch.stack([A_[s_ == t, bb, r].abs().mean() for bb, r in top8]).mean()) for t in (0, 1)}; out["marked"] = 1 if mag[1] >= mag[0] else 0; out["mag"] = mag
    out["wdd_order"] = wdd_order[:max(KS)]; out["naive_order"] = naive_order; RES["attrs"][a] = out
tot = lambda d: d["flip_from_0"] + d["flip_from_1"]
def kmin(a, kind, thr):
    for k in KS:
        d = RES["attrs"][a][kind].get(str(k))
        if d is not None and tot(d) >= thr: return k
    return None
RES["verdicts"] = dict(D1=sum(max(tot(RES["attrs"][a]["wdd"][k]) for k in RES["attrs"][a]["wdd"]) >= 0.3 for a in ATTRS) >= 3,
                       D2=sum(np.mean([tot(RES["attrs"][a]["wdd"][str(k)]) for k in KS if str(k) in RES["attrs"][a]["wdd"]]) > np.mean([tot(RES["attrs"][a]["naive"][str(k)]) for k in KS if str(k) in RES["attrs"][a]["wdd"]]) for a in ATTRS) >= 3,
                       D3=all(max(tot(RES["attrs"][a]["random"][str(k)]) for k in KS) <= 0.1 for a in ATTRS),
                       D4=all((lambda d, mk: d[f"flip_from_{mk}"] >= 3 * max(d[f"flip_from_{1 - mk}"], 0.01))(RES["attrs"][a]["wdd"][max(RES["attrs"][a]["wdd"], key=int)], RES["attrs"][a]["marked"]) for a in ATTRS))
summ = f"[{name}] dose response: best flips (wdd / naive / random over k) " + "; ".join(f"{a} {max(tot(v) for v in RES['attrs'][a]['wdd'].values()):.2f}/{max(tot(v) for v in RES['attrs'][a]['naive'].values()):.2f}/{max(tot(v) for v in RES['attrs'][a]['random'].values()):.2f}" for a in ATTRS) + f"; verdicts {RES['verdicts']} ({time.time() - t0:.0f}s)"
RES["summary"] = summ; RES["_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
json.dump(RES, open(f"results/e662_dose_{name}{'_smoke' if SMOKE else ''}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print(summ, flush=True); print("done", flush=True)
