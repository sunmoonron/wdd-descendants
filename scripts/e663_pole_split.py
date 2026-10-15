"""e663 (session 146; Gemini 4's pole-split sub-chains): is two-way weight surgery hiding in the WDD vote order? e662's
gender dose response peaked around k = 3 to 12 rows and then male items began to flip too (0.03 at k = 1 to 0.26 at
k = 32 in Qwen2.5-0.5B). Gemini's reading: the vote ranks rows by how they explain the pole difference, so rows that
fire on the marked pole and write +d and rows that fire on the unmarked pole and write -d both qualify; the unmarked
pole's rows are more diffuse and sit lower in the order, and negating them flips unmarked items toward the marked pole.
Here the top 32 rows of the vote order are split by the pole they are more active on (discovery items) into C1 (pole 1:
female, young, plural, the second polarity word) and C0; reported per row: rank, activation by pole, and its push (its
activation change between the poles times its write's cosine with the mean pole difference at its block). Each
sub-chain is negated cumulatively in vote order (k = 1, 2, 4, 8, 16, all), held-out flips by starting pole and wikitext
KL; and Gemini's four-quadrant composition on the kinship items: for each target (king, queen, prince, princess) the
gender sub-chain whose negation pushes toward the target's gender and the generation sub-chain that pushes toward its
generation are negated together, and the share of all items answering the target is read. Reuses e661's setup.
Arguments: model [--smoke]."""
import sys
src = open("e661_markedness.py").read(); exec(src[src.index("import sys\n"): src.index("# ==== SETUP END ====")], globals())
NTOP = 16 if SMOKE else 32
def wdd_order(a):
    best = {}
    for b in range(L + 1):
        for bb, r, sh in VOTE[a][b]:
            if sh > best.get((bb, r), -1): best[(bb, r)] = sh
    return sorted(best, key=lambda x: -best[x])
RES = dict(model=name, ntop=NTOP, attrs={}); SUB = {}
for a in ATTRS:
    order = wdd_order(a)[:NTOP]; its = [it for it in ITEMS[a] if it["disc"]]; A_ = torch.stack([it["A"] for it in its]); s_ = torch.tensor([it["side"] for it in its])
    prof = []
    for rank, (b, r) in enumerate(order):
        m0, m1 = float(A_[s_ == 0, b, r].abs().mean()), float(A_[s_ == 1, b, r].abs().mean()); d_ = DIFF[a][:, b].mean(0); dh = d_ / d_.norm().clamp_min(1e-12)
        w = layers[b].mlp.down_proj.weight[:, r].detach().float().cpu(); da = float(A_[s_ == 1, b, r].mean() - A_[s_ == 0, b, r].mean())
        prof.append(dict(rank=rank, row=[b, r], abs0=m0, abs1=m1, pole=1 if m1 > m0 else 0, push=da * float(w @ dh) / max(float(w.norm()), 1e-12)))
    C1 = [tuple(p["row"]) for p in prof if p["pole"] == 1]; C0 = [tuple(p["row"]) for p in prof if p["pole"] == 0]; SUB[a] = dict(C1=C1, C0=C0)
    out = dict(profile=prof, n_C1=len(C1), n_C0=len(C0), mean_rank_C1=float(np.mean([p["rank"] for p in prof if p["pole"] == 1])) if C1 else None, mean_rank_C0=float(np.mean([p["rank"] for p in prof if p["pole"] == 0])) if C0 else None,
               push_positive_C0=float(np.mean([p["push"] > 0 for p in prof if p["pole"] == 0])) if C0 else None, push_positive_C1=float(np.mean([p["push"] > 0 for p in prof if p["pole"] == 1])) if C1 else None, curves={"C1": {}, "C0": {}})
    for sub, rows in (("C1", C1), ("C0", C0)):
        for k in sorted(set([x for x in (1, 2, 4, 8, 16) if x <= len(rows)] + ([len(rows)] if rows else []))):
            with Negate(rows[:k]): out["curves"][sub][str(k)] = by_pole(a) | {"kl": kl_off()}
    RES["attrs"][a] = out
    cv = lambda sub: "; ".join(f"k={k} {v['flip_from_0']:.2f}/{v['flip_from_1']:.2f}" for k, v in out["curves"][sub].items())
    log(f"{a}: top {NTOP} split C1 {len(C1)} (mean rank {out['mean_rank_C1']}) / C0 {len(C0)} (mean rank {out['mean_rank_C0']}); push>0 C1 {out['push_positive_C1']} C0 {out['push_positive_C0']}; negate C1 flips from pole0/pole1: {cv('C1')} | negate C0: {cv('C0')}")
# ---- four-quadrant composition on the kinship items (index bit 1 = female, bit 2 = young)
def edit_for(t):
    g = SUB["gender"]["C0"] if t & 1 else SUB["gender"]["C1"]       # toward female: strip the male-active rows; toward male: strip the female-active rows
    y = SUB["generation"]["C0"] if t & 2 else SUB["generation"]["C1"]   # toward young: strip the old-active rows; toward old: strip the young-active rows
    return g + y
def land_on(rows, t):
    with Negate(rows):
        lps, _ = forward_items(KIN); by = np.zeros(4); nb = np.zeros(4)
        for lp, it in zip(lps, KIN): nb[it["i"]] += 1; by[it["i"]] += int(int(lp.argmax()) == t)
    return float(by.sum() / max(nb.sum(), 1)), (by / np.maximum(nb, 1)).tolist()
quad = {}
for t, nm in enumerate(("king", "queen", "prince", "princess")):
    share, by_start = land_on(edit_for(t), t); quad[nm] = dict(share=share, by_start=by_start)
RES["quadrants"] = quad; log("four-quadrant composition, share of all kinship items on the target (by starting item king/queen/prince/princess): " + "; ".join(f"{k} {v['share']:.2f} {[round(x, 2) for x in v['by_start']]}" for k, v in quad.items()))
A3 = RES["attrs"]
def full(a, sub):
    c = A3[a]["curves"][sub]; return c[max(c, key=int)] if c else None
RES["verdicts"] = dict(P1=A3["gender"]["n_C0"] > 0 and A3["gender"]["mean_rank_C0"] is not None and A3["gender"]["mean_rank_C1"] is not None and A3["gender"]["mean_rank_C0"] > A3["gender"]["mean_rank_C1"],
                       P2=all(full(a, "C1") is not None and full(a, "C1")["flip_from_0"] <= 0.05 for a in ("gender", "number")),
                       P3=full("gender", "C0") is not None and full("gender", "C0")["flip_from_0"] >= 0.3,
                       P4=full("number", "C0") is not None and full("number", "C0")["flip_from_0"] >= 0.3,
                       P5=all(v["share"] >= 0.5 for v in quad.values()),
                       P6=A3["gender"]["push_positive_C0"] is not None and A3["gender"]["push_positive_C0"] >= 0.7)
summ = (f"[{name}] pole-split sub-chains: gender C1/C0 {A3['gender']['n_C1']}/{A3['gender']['n_C0']} (mean rank {A3['gender']['mean_rank_C1']}/{A3['gender']['mean_rank_C0']}); negating C0 flips male->female {full('gender', 'C0')['flip_from_0'] if full('gender', 'C0') else None}, "
        f"singular->plural {full('number', 'C0')['flip_from_0'] if full('number', 'C0') else None}; quadrant shares " + ", ".join(f"{k} {v['share']:.2f}" for k, v in quad.items()) + f"; verdicts {RES['verdicts']} ({time.time() - t0:.0f}s)")
RES["summary"] = summ; RES["_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
json.dump(RES, open(f"results/e663_pole_split_{name}{'_smoke' if SMOKE else ''}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print(summ, flush=True); print("done", flush=True)
