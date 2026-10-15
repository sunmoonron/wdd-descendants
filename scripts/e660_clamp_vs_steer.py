"""e660 (session 144; Gemini 4's 'value freeze'): a one-way sign clamp on the relay chain against a dense steering
vector. Negating a row's write is two-way (king to queen and queen to king). Gemini's alternative: force a pole by
clamping each chain row's activation to the target pole's sign, a <- s * |a|, applied at every position; since a row
writes in proportion to its activation, the clamp should inject little where the row is quiet (self-gating), where a
steering vector (the mean target-minus-other state at the peak block, added at every position) injects everywhere.
Qwen2.5-0.5B, the chains of e659 (e622's vote_by_block), gender, generation and number, each pole forced in turn. On
target (e622's held-out templates): the share of the other pole's items moved to the target and of the target pole's
items kept. Off target, on wikitext-2: the KL of the next-token distribution from the unedited model's, and for gender
the change in the she-minus-he logit. The steering vector is scaled over a grid and compared at the scale whose
on-target move best matches the clamp's. Also the self-gating premise: the chain rows' mean absolute activation on
wikitext against at the attribute words. Reuses e659's item harness. Arguments: model [--smoke]."""
import sys
src = open("e659_relay_chain.py").read(); exec(src[src.index("import sys, os, time"): src.index("# ---- edits and scoring")], globals())
ATT = ("gender", "generation", "number"); ALPHAS = [0.5, 1, 2, 4] if SMOKE else [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]
import datasets
WT = [t for t in datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if len(t.split()) > 60]
WIN = []
for t in WT:
    ids = tok(t)["input_ids"]
    if len(ids) >= 128: WIN.append(ids[:128])
    if len(WIN) >= (4 if SMOKE else 16): break
WIN = torch.tensor(WIN, device=DEV); SHE, HE = first_tok("she"), first_tok("he")
BASE = []
for s0 in range(0, WIN.shape[0], 4): BASE.append(model(WIN[s0:s0 + 4], use_cache=False).logits.float().log_softmax(-1).cpu().half())
def off_target():
    kl, dsh, n = 0.0, 0.0, 0
    for k, s0 in enumerate(range(0, WIN.shape[0], 4)):
        lp = model(WIN[s0:s0 + 4], use_cache=False).logits.float().log_softmax(-1); lb = BASE[k].to(DEV).float()
        kl += float((lb.exp() * (lb - lp)).sum(-1).sum()); dsh += float(((lp[..., SHE] - lp[..., HE]) - (lb[..., SHE] - lb[..., HE])).sum()); n += lp.shape[0] * lp.shape[1]; del lp, lb
    return kl / n, dsh / n
def on_target(a, t):
    its = [it for it in ITEMS[a] if not it["disc"]]; lps, _ = forward_items(its); mv, kp, nm, nk = 0, 0, 0, 0
    for it, lp in zip(its, lps):
        if it["side"] == t: kp += int(int(lp.argmax()) == t); nk += 1
        else: mv += int(int(lp.argmax()) == t); nm += 1
    return mv / max(nm, 1), kp / max(nk, 1)
class Clamp:
    def __init__(self, rows, signs): self.by = collections.defaultdict(list); [self.by[b].append((r, s)) for (b, r), s in zip(rows, signs)]; self.h = []
    def __enter__(self):
        for b, rs in self.by.items():
            idx = torch.tensor([r for r, _ in rs], device=DEV); sg = torch.tensor([s for _, s in rs], device=DEV, dtype=torch.float32)
            def hk(m, args, idx=idx, sg=sg): x = args[0].clone(); x[..., idx] = sg * x[..., idx].abs(); return (x,) + tuple(args[1:])
            self.h.append(layers[b].mlp.down_proj.register_forward_pre_hook(hk))
        return self
    def __exit__(self, *a_): [h.remove() for h in self.h]
class Steer:
    def __init__(self, b, v): self.b, self.v = b, v
    def __enter__(self):
        def hk(m, args, out): return (out[0] + self.v,) + tuple(out[1:]) if isinstance(out, tuple) else out + self.v
        self.h = layers[self.b].register_forward_hook(hk); return self
    def __exit__(self, *a_): self.h.remove()
def states_at(its, b):
    """residual state at the output of block b at each item's word position"""
    groups = collections.defaultdict(list); out = [None] * len(its)
    for n_, it in enumerate(its): groups[len(it["ids"])].append(n_)
    for Lq, idx in groups.items():
        for s0 in range(0, len(idx), BS):
            ch = idx[s0:s0 + BS]; cap = {}; h = layers[b].register_forward_hook(lambda m, a_, o: cap.__setitem__("x", (o[0] if isinstance(o, tuple) else o).float()))
            try: model(torch.tensor([its[n]["ids"] for n in ch], device=DEV), use_cache=False, logits_to_keep=1)
            finally: h.remove()
            for k, n in enumerate(ch): out[n] = cap["x"][k, its[n]["p"]]
    return torch.stack(out)
# the self-gating premise: chain rows' |activation| on wikitext against at the attribute words
def wt_acts(rows):
    cap = collections.defaultdict(list); hs = []
    for b in sorted(set(b for b, _ in rows)):
        idx = [r for bb, r in rows if bb == b]
        hs.append(layers[b].mlp.down_proj.register_forward_pre_hook(lambda m, a_, b=b, idx=idx: cap[b].append(a_[0][..., idx].abs().float().reshape(-1, len(idx)).mean(0))))
    try:
        for s0 in range(0, WIN.shape[0], 4): model(WIN[s0:s0 + 4], use_cache=False, logits_to_keep=1)
    finally: [h.remove() for h in hs]
    return {b: torch.stack(v).mean(0) for b, v in cap.items()}
RES = dict(model=name, alphas=ALPHAS, attrs={})
base_ot = off_target()
for a in ATT:
    rows = EDITS[a]["chain"]; bm = C[a]["block"]; disc = [it for it in ITEMS[a] if it["disc"]]
    A_ = torch.stack([it["A"] for it in disc]); sd = torch.tensor([it["side"] for it in disc])
    wa = wt_acts(rows); word_abs = [float(A_[:, b, r].abs().mean()) for b, r in rows]; wiki_abs = [float(wa[b][[rr for bb, rr in rows if bb == b].index(r)]) for b, r in rows]
    Xs = states_at(disc, bm); sdev = sd.to(DEV); RES["attrs"][a] = dict(rows=rows, peak_block=bm, word_abs=word_abs, wiki_abs=wiki_abs, gate_ratio=float(np.mean(wiki_abs) / max(np.mean(word_abs), 1e-9)), poles={})
    base_on = {t: on_target(a, t) for t in (0, 1)}
    for t in (0, 1):
        signs = [float(torch.sign(A_[sd == t, b, r].mean())) or 1.0 for b, r in rows]; flip_signs = sum(float(torch.sign(A_[sd == 1 - t, b, r].mean())) != s for (b, r), s in zip(rows, signs))
        with Clamp(rows, signs): cl_on = on_target(a, t); cl_off = off_target()
        v = Xs[sdev == t].mean(0) - Xs[sdev == 1 - t].mean(0); steer = {}
        for al in ALPHAS:
            with Steer(bm, al * v): steer[str(al)] = dict(on=on_target(a, t), off=off_target())
        match = min(ALPHAS, key=lambda al: (abs(steer[str(al)]["on"][0] - cl_on[0]), al))
        RES["attrs"][a]["poles"][str(t)] = dict(signs=signs, rows_whose_mean_sign_differs_by_pole=int(flip_signs), base_on=base_on[t], clamp=dict(on=cl_on, off=cl_off), steer=steer, matched_alpha=match, v_norm=float(v.norm()))
        s_ = steer[str(match)]; log(f"{a} force pole {t}: base moved {base_on[t][0]:.2f} kept {base_on[t][1]:.2f}; clamp moved {cl_on[0]:.2f} kept {cl_on[1]:.2f}, KL {cl_off[0]:.4f}, she-he {cl_off[1]:+.3f} ({flip_signs} of {len(rows)} rows change sign by pole); "
            f"steer at alpha {match} moved {s_['on'][0]:.2f} kept {s_['on'][1]:.2f}, KL {s_['off'][0]:.4f}, she-he {s_['off'][1]:+.3f}; gate ratio (wikitext over word |a|) {RES['attrs'][a]['gate_ratio']:.2f}")
g = RES["attrs"]["gender"]["poles"]; eff = [(a, t) for a in ATT for t in ("0", "1") if RES["attrs"][a]["poles"][t]["clamp"]["on"][0] >= 0.2]; P_ = lambda a, t: RES["attrs"][a]["poles"][t]; MS_ = lambda a, t: P_(a, t)["steer"][str(P_(a, t)["matched_alpha"])]
RES["effective_cases"] = eff
RES["verdicts"] = dict(K1=all(g[t]["clamp"]["on"][0] >= 0.3 and g[t]["clamp"]["on"][1] >= 0.9 for t in ("0", "1")),
                       K2=bool(eff) and all(P_(a, t)["clamp"]["off"][0] < MS_(a, t)["off"][0] for a, t in eff),
                       K3=any(a == "gender" for a, _ in eff) and all(abs(P_(a, t)["clamp"]["off"][1]) < abs(MS_(a, t)["off"][1]) for a, t in eff if a == "gender"),
                       K4=all(RES["attrs"][a]["gate_ratio"] < 0.1 for a in ATT),
                       K5=all(max(P_(a, t)["clamp"]["on"][0] for t in ("0", "1")) >= 0.2 and min(P_(a, t)["clamp"]["on"][0] for t in ("0", "1")) < 0.1 for a in ATT))
summ = f"[{name}] clamp vs steering: " + "; ".join(f"{a} pole {t}: clamp moved {RES['attrs'][a]['poles'][t]['clamp']['on'][0]:.2f} KL {RES['attrs'][a]['poles'][t]['clamp']['off'][0]:.4f}, steer@{RES['attrs'][a]['poles'][t]['matched_alpha']} moved {RES['attrs'][a]['poles'][t]['steer'][str(RES['attrs'][a]['poles'][t]['matched_alpha'])]['on'][0]:.2f} KL {RES['attrs'][a]['poles'][t]['steer'][str(RES['attrs'][a]['poles'][t]['matched_alpha'])]['off'][0]:.4f}" for a in ATT for t in ("0", "1")) + f"; verdicts {RES['verdicts']} ({time.time() - t0:.0f}s)"
RES["summary"] = summ; RES["_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
json.dump(RES, open(f"results/e660_clamp_vs_steer_{name}{'_smoke' if SMOKE else ''}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print(summ, flush=True); print("done", flush=True)
