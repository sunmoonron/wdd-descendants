"""e671 (session 151; Cammarata 2026, "Neural Networks Are Already Clean", read with e661): a SwiGLU neuron's activation
is a = SiLU(g) u, gate g and value u, so it has two poles and four sign regimes: gate open and positive (g > 0, u > 0),
gate open and negative (g > 0, u < 0), and two gate-closed leaks bounded by |SiLU(g)| <= 0.28 of the value, positive when
both are negative (the "double negative", which the essay finds fills the top of the positive list for about 4% of
Qwen3 neurons) and negative otherwise. e661 measured the attribute relay rows by |a| and found them one-pole detectors of
the marked form (female, plural), with generation symmetric. Part A, Qwen2.5-0.5B, the e622/e659 translation items via
wdd_demo's harness: the top 8 direct-contribution rows of gender, number and generation, each pole's signed activation
at the word and its regime shares; does the unmarked pole sit on the rows' negative side, does generation split by
sign, how much of the activation is gate-closed leak? Part B, Qwen2.5-0.5B and SmolLM2-135M, every MLP neuron over
wikitext windows: the regime of each neuron's top-20 positive and top-20 negative activations; the share of neurons
whose positive top-20 is mostly double-negative leak, and for those the peak positive over the peak negative magnitude.
Arguments: [--smoke]. Pre-registered in e671_prereg.json."""
import sys, time, json, collections
SMOKE = "--smoke" in sys.argv; sys.argv = [sys.argv[0], "--auto", "--no-color"]
import numpy as np, torch
import wdd_demo as wd
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_grad_enabled(False); t0 = time.time(); DEV = wd.DEV
def log(s): print(f"[e671] {s} ({time.time() - t0:.0f}s)", flush=True)
if SMOKE: wd.LANGS = ("fr",)
K = 4 if SMOKE else 8; RES = dict(smoke=SMOKE, partA={}, partB={})
# ---------------- Part A
lm = wd.LM("Qwen/Qwen2.5-0.5B")
def gen_items():
    """generation pairs (older, younger) of e622, built as LM.items builds gender and number"""
    its = []
    for lg in wd.LANGS:
        src = [(q[0], q[2]) for q in wd.QUADS[lg]] + [(q[1], q[3]) for q in wd.QUADS[lg]]; en = [(q[0], q[2]) for q in wd.QUADS["en"]] + [(q[1], q[3]) for q in wd.QUADS["en"]]
        for k, (pr, e) in enumerate(zip(src, en)):
            cand = [lm.ftok(w) for w in e]
            if cand[0] == cand[1]: continue
            for t in range(wd.NT):
                for i in (0, 1): ids, p = lm.prompt(lg, pr[i], t); its.append(dict(lg=lg, pair=k, t=t, side=i, word=pr[i], en=e, cand=cand, ids=ids, p=p, disc=t in wd.DISC))
    lm.run(its, capture=True); grp = collections.defaultdict(list)
    for it in its: grp[(it["lg"], it["pair"], it["t"])].append(it)
    return [it for g in grp.values() if len(g) == 2 and all(x["pred"] == x["side"] for x in g) for it in g]
POLES = dict(wd.POLES, generation=("older", "younger"))
IT = {a: lm.items(a) for a in ("gender", "number")}; IT["generation"] = gen_items(); SC = {a: lm.scan(IT[a]) for a in POLES}
def gate_up(its, rows):
    """gate and value pre-activations of the given (block, row) neurons at each item's word position"""
    by_b = collections.defaultdict(list)
    for b, r in rows: by_b[b].append(r)
    groups = collections.defaultdict(list); G = {br: [None] * len(its) for br in rows}; U = {br: [None] * len(its) for br in rows}
    for n, it in enumerate(its): groups[len(it["ids"])].append(n)
    for idx in groups.values():
        for s0 in range(0, len(idx), lm.BS):
            ch = idx[s0:s0 + lm.BS]; cap, hs = {}, []
            for b in by_b:
                hs.append(lm.layers[b].mlp.gate_proj.register_forward_hook(lambda m, a, o, b=b: cap.__setitem__(("g", b), o.float())))
                hs.append(lm.layers[b].mlp.up_proj.register_forward_hook(lambda m, a, o, b=b: cap.__setitem__(("u", b), o.float())))
            try: lm.model(torch.tensor([its[n]["ids"] for n in ch], device=DEV), use_cache=False, logits_to_keep=1)
            finally: [h.remove() for h in hs]
            for k, n in enumerate(ch):
                for b, r in rows: G[(b, r)][n] = float(cap[("g", b)][k, its[n]["p"], r]); U[(b, r)][n] = float(cap[("u", b)][k, its[n]["p"], r])
    return {br: (torch.tensor(G[br]), torch.tensor(U[br])) for br in rows}
silu = torch.nn.functional.silu
def regimes(g, u):
    return dict(open_pos=float(((g > 0) & (u > 0)).float().mean()), open_neg=float(((g > 0) & (u < 0)).float().mean()), closed_pos=float(((g < 0) & (u < 0)).float().mean()), closed_neg=float(((g < 0) & (u > 0)).float().mean()))
for a in POLES:
    disc = [it for it in IT[a] if it["disc"]]; s = torch.tensor([it["side"] for it in disc]); rows = SC[a]["order"][:K]; GU = gate_up(disc, rows); per = []
    for b, r in rows:
        g, u = GU[(b, r)]; act = silu(g) * u; m1, m0 = float(act[s == 1].mean()), float(act[s == 0].mean())
        absmass = act.abs(); leak_mass = float(absmass[(g < 0)].sum() / absmass.sum().clamp_min(1e-9))
        per.append(dict(row=[b, r], mean_marked=m1, mean_unmarked=m0, abs_marked=float(act[s == 1].abs().mean()), abs_unmarked=float(act[s == 0].abs().mean()), regimes_marked=regimes(g[s == 1], u[s == 1]), regimes_unmarked=regimes(g[s == 0], u[s == 0]), gate_closed_mass=leak_mass))
    big = lambda x: max(abs(x["mean_marked"]), abs(x["mean_unmarked"]))
    two_pole = [x for x in per if x["mean_marked"] * x["mean_unmarked"] < 0 and min(abs(x["mean_marked"]), abs(x["mean_unmarked"])) >= 0.25 * big(x)]
    RES["partA"][a] = dict(poles=POLES[a], n_items=len(disc), rows=per, n_two_pole=len(two_pole), n_opposite_sign=sum(x["mean_marked"] * x["mean_unmarked"] < 0 for x in per), gate_closed_mass_median=float(np.median([x["gate_closed_mass"] for x in per])),
                           abs_ratio=sum(x["abs_marked"] for x in per) / max(sum(x["abs_unmarked"] for x in per), 1e-9))
    log(f"{a} ({POLES[a][1]} vs {POLES[a][0]}, {len(disc)} items): top-{K} rows' mean signed activation marked / unmarked: " + "; ".join(f"b{x['row'][0]}r{x['row'][1]} {x['mean_marked']:+.2f}/{x['mean_unmarked']:+.2f}" for x in per)
        + f"; two-pole rows {len(two_pole)} of {K}, opposite signs {RES['partA'][a]['n_opposite_sign']}; |a| ratio {RES['partA'][a]['abs_ratio']:.1f}; gate-closed share of |a| (median) {RES['partA'][a]['gate_closed_mass_median']:.3f}")
del lm; torch.cuda.empty_cache()
# ---------------- Part B
import datasets
WT = [t for t in datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if len(t.split()) > 60]
for name, hf in (("smollm135", "HuggingFaceTB/SmolLM2-135M"),) + ((() if SMOKE else (("qwen05", "Qwen/Qwen2.5-0.5B"),))):
    tok = AutoTokenizer.from_pretrained(hf); model = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.float32).to(DEV).eval(); layers = model.model.layers; NL = len(layers); DFF = layers[0].mlp.gate_proj.out_features
    win = []
    for t in WT:
        ids = tok(t)["input_ids"]
        if len(ids) >= 256: win.append(ids[:256])
        if len(win) >= (4 if SMOKE else 128): break
    W = torch.tensor(win, device=DEV); TOP = 20; SNAP = 2 if SMOKE else 16; snap = None
    best = {(l, sgn): (torch.full((TOP, DFF), float("nan"), device=DEV), torch.zeros(TOP, DFF, dtype=torch.bool, device=DEV)) for l in range(NL) for sgn in (1, -1)}   # values (signed toward the pole) and gate-closed flags
    def make_hooks(l, cap):
        return [layers[l].mlp.gate_proj.register_forward_hook(lambda m, a, o: cap.__setitem__(("g", l), o.float())), layers[l].mlp.up_proj.register_forward_hook(lambda m, a, o: cap.__setitem__(("u", l), o.float()))]
    for s0 in range(0, W.shape[0], 4):
        cap = {}; hs = [h for l in range(NL) for h in make_hooks(l, cap)]
        try: model(W[s0:s0 + 4], use_cache=False, logits_to_keep=1)
        finally: [h.remove() for h in hs]
        for l in range(NL):
            g, u = cap[("g", l)][:, 1:].reshape(-1, DFF), cap[("u", l)][:, 1:].reshape(-1, DFF); act = silu(g) * u; closed = g < 0
            for sgn in (1, -1):
                v0, c0 = best[(l, sgn)]; v = torch.cat([torch.nan_to_num(v0, nan=-1e9), sgn * act]); c = torch.cat([c0, closed]); tv, ti = v.topk(TOP, dim=0); best[(l, sgn)] = (tv, torch.gather(c, 0, ti))
        del cap
        if s0 + 4 == SNAP: snap = sum(int((best[(l, 1)][1].float().mean(0) > 0.5).sum()) for l in range(NL)) / (NL * DFF)
    lay = []
    for l in range(NL):
        vp, cp = best[(l, 1)]; vn, cn = best[(l, -1)]
        dom = (cp.float().mean(0) > 0.5); ratio = vp[0] / vn[0].clamp_min(1e-9)
        lay.append(dict(layer=l, leak_dominated=float(dom.float().mean()), leak_in_top_pos=float(cp.float().mean()), leak_in_top_neg=float(cn.float().mean()), peak_ratio_leak_dominated=float(ratio[dom].median()) if dom.any() else None, n_dom=int(dom.sum())))
    ndom = sum(x["n_dom"] for x in lay); tot = NL * DFF
    rat = [x["peak_ratio_leak_dominated"] for x in lay if x["peak_ratio_leak_dominated"] is not None]
    RES["partB"][name] = dict(layers=lay, n_neurons=tot, share_leak_dominated=ndom / tot, share_leak_dominated_at_snapshot=snap, snapshot_positions=SNAP * (W.shape[1] - 1), leak_share_top_pos=float(np.mean([x["leak_in_top_pos"] for x in lay])), leak_share_top_neg=float(np.mean([x["leak_in_top_neg"] for x in lay])), median_layer_peak_ratio=float(np.median(rat)) if rat else None, positions=int(W.shape[0] * (W.shape[1] - 1)))
    log(f"{name}: at {SNAP * (W.shape[1] - 1)} positions {snap:.3f} of neurons are leak-dominated; at {W.shape[0] * (W.shape[1] - 1)} positions {ndom} of {tot} neurons ({ndom / tot:.3f}) have more than half of their top-20 positive activations in the gate-closed double negative; gate-closed share of all top-20 positive {RES['partB'][name]['leak_share_top_pos']:.3f}, of top-20 negative {RES['partB'][name]['leak_share_top_neg']:.3f}; for the leak-dominated neurons the peak positive is {RES['partB'][name]['median_layer_peak_ratio']} of the peak negative (median over layers); by layer " + " ".join(f"{x['leak_dominated']:.2f}" for x in lay))
    del model; torch.cuda.empty_cache()
RES["_time"] = time.strftime("%Y-%m-%d %H:%M:%S"); RES["seconds"] = time.time() - t0
json.dump(RES, open(f"results/e671_signed_poles{'_smoke' if SMOKE else ''}.json", "w"), indent=1); log("done"); print("done", flush=True)
