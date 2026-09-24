"""e392b: does the replacement model compound through the sinks? In e392 every block's output was replaced, during the
forward pass, by its k-sparse description in the model's own words (all positions but the first). The excess loss at
k 128 was +0.43 (GPT-2), +0.53 (SmolLM2) and +1.28 (Qwen-0.5B), but +3.26 in Pythia-410m and +3.77 in OLMo, the two
models e432 found to have sink positions beyond the first at middle depth. e437 showed that mis-describing a sink is
what made the step-4000 words false friends.
Same replacement model, 2 sequences, k 64 and 128, with the positions whose incoming state norm exceeds 10x the median
at that block either described like the rest (as e392) or passed through exactly. Centring means come from 8 other
sequences' typical positions.
Pre-registered: with the sinks exact, Pythia's and OLMo's excess at k 128 falls below 1.5 nats (to the level of the
models without sinks); GPT-2 is unchanged."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
name = sys.argv[1]; model, tok, fam = load_model(name); arch = Arch(model, fam); NB = arch.NB
EA = torch.load("/workspace/wdd/eval_ids_all.pt", weights_only=False)[name]; ev = EA["eval_ids"][:2].to(DEV); cen = EA["cen_ids"][:8].to(DEV); del EA
with torch.no_grad(): Lc = token_loss(model(ev).logits.float(), ev).mean().item()
LAY = list(range(NB - 1)); S = block_states(model, arch, cen, LAY, chunk=4); mus = {}
for l in LAY: x = S[l]; mus[l] = x[~sinkmask(x)].mean(0)
del S
A, lab = build_dictionary(arch, blocks=LAY); blk = lab["block"]; ends = {l: int((blk <= l).nonzero().max()) + 1 for l in LAY}
nsink = {}
def run(k, sinks_exact):
    hs = []; cnt = {}
    for l in LAY:
        def hk(m, i, o, l=l):
            xo = out_of(o); x = xo[:, 1:].float(); keep = ~sinkmask(x) if sinks_exact else torch.ones(x.shape[:2], dtype=torch.bool, device=DEV)
            cnt[l] = int(sinkmask(x).sum()); xp = x[keep] - mus[l][None]; Dct = A[:ends[l]]
            sel, cof, _ = omp(xp, Dct, k, batch=256, record_err=False); xh = x.clone(); xh[keep] = mus[l][None] + torch.einsum("nk,nkd->nd", cof, Dct[sel])
            y = xo.clone(); y[:, 1:] = xh.to(xo.dtype); return (y,) + tuple(o[1:]) if isinstance(o, tuple) else y
        hs.append(arch.layers[l].register_forward_hook(hk))
    try:
        with torch.no_grad(): out = token_loss(model(ev).logits.float(), ev).mean().item()
    finally: [h.remove() for h in hs]
    nsink[f"{k}_{sinks_exact}"] = cnt; return out
res = dict(model=name, clean=Lc, runs={})
for k in (64, 128):
    la, ls = run(k, False), run(k, True); res["runs"][str(k)] = dict(all_positions=la - Lc, sinks_exact=ls - Lc)
    log(f"{name} k{k}: excess loss all positions {la - Lc:+.3f}, sinks exact {ls - Lc:+.3f}")
res["sinks_per_layer_clean_run"] = {str(l): v for l, v in nsink["128_True"].items()}
res["checks"] = dict(sinks_exact_under_1_5=res["runs"]["128"]["sinks_exact"] < 1.5)
summ = (f"{name}: replacement model excess loss (clean {Lc:.3f}) k64 all {res['runs']['64']['all_positions']:+.2f} / sinks exact {res['runs']['64']['sinks_exact']:+.2f}; "
        f"k128 all {res['runs']['128']['all_positions']:+.2f} / sinks exact {res['runs']['128']['sinks_exact']:+.2f} | sink positions per block (k128 run): " + ",".join(str(v) for v in nsink["128_True"].values()) + f" | checks {json.dumps(res['checks'])}")
log(summ); record(f"e392b_replacement_sinks_{name}", res, summ)
