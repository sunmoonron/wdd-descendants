"""e664 (session 146; Gemini 4's 'activation echo chamber'): is the WDD vote a better localizer of an attribute's writers
than the field's standard tools, or only than the naive one? e662 found the vote order flips attributes with a few
rows where a global ranking by normalized activation difference barely moves them; Gemini reads this as activation
differences catching downstream readers (echoes) and the cumulative dictionary crediting the first writer. The naive
ranking is not what the field uses. Here, per attribute, row orders built on discovery items only: the WDD vote order
(e662); the naive normalized activation difference (e662); attribution patching (Nanda 2023; Syed et al. 2023;
Kramar et al. 2024), the gradient of the log-odds of the other pole with respect to each row's activation at the word,
times the activation change to the other pole's item; an edit-specific attribution, the same gradient times minus twice
the activation (the first-order effect of negating the row's write at the word); causal block patching (each block's
MLP output at the word replaced by the other pole's, Meng et al. 2022 style), rows taken round-robin from the most
effective blocks by activation difference; and random. The k first rows of each are negated (k = 1, 2, 4, 8, 16, 32),
held-out flips by starting pole and wikitext KL. Reuses e661's setup. Arguments: model [--smoke]."""
import sys
src = open("e661_markedness.py").read(); exec(src[src.index("import sys\n"): src.index("# ==== SETUP END ====")], globals())
KS = [1, 4, 16] if SMOKE else [1, 2, 4, 8, 16, 32]; KMAX = max(KS)
def wdd_order(a):
    best = {}
    for b in range(L + 1):
        for bb, r, sh in VOTE[a][b]:
            if sh > best.get((bb, r), -1): best[(bb, r)] = sh
    return sorted(best, key=lambda x: -best[x])
def grads(its):
    """activations and the total derivative of the other pole's log-odds with respect to them, at each item's word position, blocks 0..L: two [n, L+1, DFF] CPU tensors.
    The embedding output is the graph's leaf, so every downstream path (later MLPs included) counts."""
    Aout = [None] * len(its); Gout = [None] * len(its); BSG = 4 if D > 1000 else 16
    with torch.enable_grad():
        for big in batches(its):
            for s0 in range(0, len(big), BSG):
                ch = big[s0:s0 + BSG]; store = {}
                he = model.get_input_embeddings().register_forward_hook(lambda m, a_, o: o.detach().requires_grad_(True))
                hs = [layers[b].mlp.down_proj.register_forward_pre_hook(lambda m, args, b=b: store.__setitem__(b, args[0])) for b in range(L + 1)]
                try:
                    lg = model(torch.tensor([its[n]["ids"] for n in ch], device=DEV), use_cache=False, logits_to_keep=1).logits[:, -1].float().log_softmax(-1)
                    obj = sum(lg[k, its[n]["cand"][1 - its[n]["side"]]] - lg[k, its[n]["cand"][its[n]["side"]]] for k, n in enumerate(ch))
                    gs = torch.autograd.grad(obj, [store[b] for b in range(L + 1)])
                finally: he.remove(); [h.remove() for h in hs]
                for k, n in enumerate(ch):
                    p = its[n]["p"]; Aout[n] = torch.stack([store[b][k, p].detach().float() for b in range(L + 1)]).cpu(); Gout[n] = torch.stack([gs[b][k, p].float() for b in range(L + 1)]).cpu()
                del store, gs, lg, obj
    return torch.stack(Aout), torch.stack(Gout)
def mlp_outs(its):
    out = [None] * len(its)
    for ch in batches(its):
        cap = {}; hs = [layers[b].mlp.register_forward_hook(lambda m, a_, o, b=b: cap.__setitem__(b, o.float())) for b in range(L + 1)]
        try: model(torch.tensor([its[n]["ids"] for n in ch], device=DEV), use_cache=False, logits_to_keep=1)
        finally: [h.remove() for h in hs]
        for k, n in enumerate(ch): out[n] = torch.stack([cap[b][k, its[n]["p"]] for b in range(L + 1)]).cpu()
    return torch.stack(out)
def block_effects(its, partner_out):
    """per block: mean change in the other pole's log-odds when the block's MLP output at the word is replaced by the partner item's"""
    eff = np.zeros(L + 1)
    for b in range(L + 1):
        tot = 0.0
        for ch in batches(its):
            P_ = partner_out[ch, b].to(DEV); pos = torch.tensor([its[n]["p"] for n in ch], device=DEV)
            def hk(m, a_, o, P_=P_, pos=pos): o = o.clone(); o[torch.arange(o.shape[0], device=DEV), pos] = P_; return o
            h = layers[b].mlp.register_forward_hook(hk)
            try: lg = model(torch.tensor([its[n]["ids"] for n in ch], device=DEV), use_cache=False, logits_to_keep=1).logits[:, -1].float().log_softmax(-1)
            finally: h.remove()
            for k, n in enumerate(ch):
                it = its[n]; tot += float((lg[k, it["cand"][1 - it["side"]]] - lg[k, it["cand"][it["side"]]]) - (it["lp0"][1 - it["side"]] - it["lp0"][it["side"]]))
        eff[b] = tot / len(its)
    return eff
RES = dict(model=name, ks=KS, attrs={})
for a in ATTRS:
    disc = [it for it in ITEMS[a] if it["disc"]]; by_key = {(it["lg"], it["pair"], it["t"], it["side"]): i_ for i_, it in enumerate(disc)}
    partner = [by_key.get((it["lg"], it["pair"], it["t"], 1 - it["side"])) for it in disc]; keep = [i_ for i_, pp in enumerate(partner) if pp is not None]
    disc = [disc[i_] for i_ in keep]; partner = [keep.index(partner[i_]) for i_ in keep]
    Aa, Gg = grads(disc); Ap = Aa[partner]; s_ = torch.tensor([it["side"] for it in disc])
    atp = ((Ap - Aa) * Gg).mean(0); neg = (-2 * Aa * Gg).mean(0)                       # [L+1, DFF]
    dn = (Aa[s_ == 1].mean(0) - Aa[s_ == 0].mean(0)).abs() / (Aa.std(0) + 1e-6)
    top = lambda S: [divmod(int(i_), DFF) for i_ in S.reshape(-1).argsort(descending=True)[:KMAX].tolist()]
    MO = mlp_outs(disc); eff = block_effects(disc, MO[partner]); border = np.argsort(-eff).tolist()
    dabs = (Aa[s_ == 1].mean(0) - Aa[s_ == 0].mean(0)).abs(); cb = []
    per_block = {b: dabs[b].argsort(descending=True).tolist() for b in border[:8]}
    for depth in range(KMAX):
        for b in border[:8]:
            if len(cb) < KMAX: cb.append((b, per_block[b][depth]))
    orders = dict(wdd=wdd_order(a)[:KMAX], naive=top(dn), atp_patch=top(atp), atp_negate=top(neg), causal_blocks=cb, random=[(int(rng.integers(L + 1)), int(rng.integers(DFF))) for _ in range(KMAX)])
    blk = lambda o: set(b for b, _ in o[:16])
    out = dict(block_effects=eff.tolist(), orders={k: v for k, v in orders.items()}, block_overlap_with_wdd={k: len(blk(v) & blk(orders["wdd"])) / max(len(blk(v) | blk(orders["wdd"])), 1) for k, v in orders.items()}, curves={})
    for kind, o in orders.items():
        out["curves"][kind] = {}
        for k in KS:
            if k <= len(o):
                with Negate(o[:k]): out["curves"][kind][str(k)] = by_pole(a) | {"kl": kl_off()}
    RES["attrs"][a] = out
    tot = lambda d: d["flip_from_0"] + d["flip_from_1"]
    log(f"{a}: block effects (top 4) {[(b, round(float(eff[b]), 2)) for b in border[:4]]}; flips summed over poles at k={KS}: " + "; ".join(f"{kind} {[round(tot(out['curves'][kind][str(k)]), 2) if str(k) in out['curves'][kind] else None for k in KS]}" for kind in orders))
tot = lambda d: d["flip_from_0"] + d["flip_from_1"]
mean_curve = lambda a, kind: float(np.mean([tot(v) for v in RES["attrs"][a]["curves"][kind].values()]))
RES["verdicts"] = dict(L1=sum(mean_curve(a, "wdd") > mean_curve(a, "atp_negate") for a in ATTRS) >= 3, L2=sum(mean_curve(a, "atp_negate") > mean_curve(a, "naive") for a in ATTRS) >= 3,
                       L3=sum(abs(mean_curve(a, "causal_blocks") - mean_curve(a, "wdd")) <= 0.1 for a in ATTRS) >= 3, L4=sum(mean_curve(a, "wdd") > mean_curve(a, "atp_patch") for a in ATTRS) >= 3)
summ = f"[{name}] localizers, mean flips over doses (wdd / atp-negate / atp-patch / causal blocks / naive / random): " + "; ".join(f"{a} " + "/".join(f"{mean_curve(a, k):.2f}" for k in ("wdd", "atp_negate", "atp_patch", "causal_blocks", "naive", "random")) for a in ATTRS) + f"; verdicts {RES['verdicts']} ({time.time() - t0:.0f}s)"
RES["summary"] = summ; RES["_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
json.dump(RES, open(f"results/e664_localizers_{name}{'_smoke' if SMOKE else ''}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print(summ, flush=True); print("done", flush=True)
