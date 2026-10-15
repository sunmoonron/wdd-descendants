"""e664b (session 146; the literature check's missing baseline): does the WDD vote need OMP at all? The residual
difference between the two poles at block b is exactly a sum of per-row terms, each row's activation change times its
write, so rows can be ranked by their direct contribution with no dictionary and no pursuit. Two such orders, from
discovery items: the local direct contribution (activation change times the write's cosine with the mean pole
difference at the row's own block) and the read-out contribution (the same against the pole difference at the last
level, L). Also the depth profile of each order (mean block of its top 32 rows), to see whether the cumulative
dictionary simply favours early rows, an early row being eligible at every later level. The k first rows negated
(k = 1, 2, 4, 8, 16, 32) against the WDD vote order; held-out flips by starting pole, wikitext KL. Reuses e661's
setup. Arguments: model [--smoke]."""
import sys
src = open("e661_markedness.py").read(); exec(src[src.index("import sys\n"): src.index("# ==== SETUP END ====")], globals())
KS = [1, 4, 16] if SMOKE else [1, 2, 4, 8, 16, 32]; KMAX = max(KS)
def wdd_order(a):
    best = {}
    for b in range(L + 1):
        for bb, r, sh in VOTE[a][b]:
            if sh > best.get((bb, r), -1): best[(bb, r)] = sh
    return sorted(best, key=lambda x: -best[x])
WN = [layers[b].mlp.down_proj.weight.detach().float().cpu() for b in range(L + 1)]
RES = dict(model=name, ks=KS, attrs={})
for a in ATTRS:
    its = [it for it in ITEMS[a] if it["disc"]]; A_ = torch.stack([it["A"] for it in its]); s_ = torch.tensor([it["side"] for it in its]); da = A_[s_ == 1].mean(0) - A_[s_ == 0].mean(0)   # [L+1, DFF]
    dloc = torch.stack([DIFF[a][:, b].mean(0) for b in range(L + 1)]); dloc = dloc / dloc.norm(dim=1, keepdim=True).clamp_min(1e-12); dL = dloc[L]
    loc = torch.stack([da[b] * ((WN[b].T @ dloc[b]) / WN[b].norm(dim=0).clamp_min(1e-12)) for b in range(L + 1)])
    rd = torch.stack([da[b] * ((WN[b].T @ dL) / WN[b].norm(dim=0).clamp_min(1e-12)) for b in range(L + 1)])
    top = lambda S: [divmod(int(i_), DFF) for i_ in S.reshape(-1).argsort(descending=True)[:KMAX].tolist()]
    orders = dict(wdd=wdd_order(a)[:KMAX], direct_local=top(loc), direct_readout=top(rd))
    out = dict(mean_block={k: float(np.mean([b for b, _ in v])) for k, v in orders.items()}, overlap_with_wdd={k: len(set(v[:16]) & set(orders["wdd"][:16])) / 16 for k, v in orders.items()}, curves={})
    for kind, o in orders.items():
        out["curves"][kind] = {}
        for k in KS:
            with Negate(o[:k]): out["curves"][kind][str(k)] = by_pole(a) | {"kl": kl_off()}
    RES["attrs"][a] = out; tot = lambda d: d["flip_from_0"] + d["flip_from_1"]
    log(f"{a}: mean block of the top {KMAX} wdd {out['mean_block']['wdd']:.1f}, local {out['mean_block']['direct_local']:.1f}, readout {out['mean_block']['direct_readout']:.1f}; top-16 row overlap with wdd local {out['overlap_with_wdd']['direct_local']:.2f} readout {out['overlap_with_wdd']['direct_readout']:.2f}; flips at k={KS}: " + "; ".join(f"{kind} {[round(tot(out['curves'][kind][str(k)]), 2) for k in KS]}" for kind in orders))
tot = lambda d: d["flip_from_0"] + d["flip_from_1"]; mc = lambda a, kind: float(np.mean([tot(v) for v in RES["attrs"][a]["curves"][kind].values()]))
RES["verdicts"] = dict(X1=sum(mc(a, "wdd") > mc(a, "direct_local") + 0.05 for a in ATTRS) >= 3, X2=sum(mc(a, "direct_local") >= mc(a, "wdd") - 0.05 for a in ATTRS) >= 3, X3=sum(RES["attrs"][a]["mean_block"]["wdd"] < RES["attrs"][a]["mean_block"]["direct_local"] for a in ATTRS) >= 3)
summ = f"[{name}] direct-contribution baselines, mean flips over doses (wdd / direct local / direct readout): " + "; ".join(f"{a} {mc(a, 'wdd'):.2f}/{mc(a, 'direct_local'):.2f}/{mc(a, 'direct_readout'):.2f}" for a in ATTRS) + f"; verdicts {RES['verdicts']} ({time.time() - t0:.0f}s)"
RES["summary"] = summ; RES["_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
json.dump(RES, open(f"results/e664b_direct_{name}{'_smoke' if SMOKE else ''}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print(summ, flush=True); print("done", flush=True)
