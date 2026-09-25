"""e464: does the network correct a native word written where it never occurs? This is the "illegal word" test,
suggested by an external review.
A native word here is an MLP neuron's write row in block b (a middle block, b = NB // 2 - 2).
- Legal contexts: positions where the neuron fires most (its top 20 activations in the evaluation text).
- Illegal contexts: positions where it writes almost nothing (absolute activation at or below its 5th percentile),
  matched to the legal ones on the state's norm. Norm matters because the per-block gain depends on it (e216).
At block b's output the same extra write, s times the word's unit row (s = the neuron's median natural write size in
its legal contexts), is added at the position. It is followed through the next blocks by the difference between the
injected and clean runs:
- survival: the difference's component along the word after 1, 2 and 4 blocks, over s;
- energy: the difference's norm over s (e221 found that energy is kept and the direction scattered);
- translation: at block b + 2, the share of the difference's energy that 8 native words explain, and whether the word
  itself is among them;
- effect: the KL divergence of the next-token prediction at the position.
Legal and illegal contexts are compared for 40 neurons (median over neurons of their per-context medians).
Models (argument): gpt2, qwen05.
Pre-registered (honest guesses):
- survival after 2 blocks differs by under 20% between legal and illegal contexts, because the contraction is generic
  (e214, e216) (0.6);
- the KL effect is larger in illegal contexts (0.5)."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; model, tok, fam = load_model(name); arch = Arch(model, fam); b = arch.NB // 2 - 2; D = arch.D
ev = eval_ids(name)[:16, :256].to(DEV); Bn, T = ev.shape
cap = {}
def pre(m, a):
    cap["h"] = a[0].detach().float(); return None                               # must return None
h1 = arch.mlp_lin(b).register_forward_pre_hook(pre); h2 = arch.layers[b].register_forward_hook(lambda m, i, o: cap.__setitem__("x", out_of(o).detach().float()))
acts, norms = [], []
for s in range(0, Bn, 4):
    with torch.no_grad(): model(ev[s:s + 4])
    acts.append(cap["h"]); norms.append(cap["x"].norm(dim=-1))
h1.remove(); h2.remove()
Aall = torch.cat(acts); Nall = torch.cat(norms)                                   # [B, T, DFF], [B, T]
valid = torch.zeros(Bn, T, dtype=torch.bool, device=DEV); valid[:, 8:] = True
valid &= Nall < 10 * Nall[:, 8:].median()                                        # no sink positions
W = arch.wdir(b); Wn = W / W.norm(dim=-1, keepdim=True)
fire = (Aall > 0).float()[valid].mean(0); cand = torch.nonzero((fire > 0.02) & (fire < 0.5))[:, 0]
g = torch.Generator(device=DEV).manual_seed(0); neurons = cand[torch.randperm(cand.numel(), device=DEV, generator=g)[:40]].tolist()
A, lab = build_dictionary(arch, blocks=list(range(b + 3))); Au = unitr(A); del A
def run(sq, p, add=None):
    """hidden states at blocks b..b+4 at position p and the next-token log-probs there; add: vector added at block b's output"""
    hs_, out = [], {}
    def ins(m, i, o):
        if add is None: return None
        y = out_of(o).clone(); y[0, p] = (y[0, p].float() + add).to(y.dtype); return (y,) + tuple(o[1:]) if isinstance(o, tuple) else y
    hs_.append(arch.layers[b].register_forward_hook(ins))
    for k in range(5):
        if b + k < arch.NB: hs_.append(arch.layers[b + k].register_forward_hook(lambda m, i, o, k=k: out.__setitem__(k, out_of(o)[0, p].detach().float())))
    try:
        with torch.no_grad(): lg = model(ev[sq:sq + 1, :p + 1]).logits[0, p].float()
    finally: [h.remove() for h in hs_]
    return out, torch.log_softmax(lg, -1)
res = dict(model=name, block=b, neurons=[], summary={})
agg = {"legal": {}, "illegal": {}}
for j in neurons:
    a = Aall[..., j].clone(); a[~valid] = float("nan"); flat = a.flatten(); ok = ~torch.isnan(flat)
    top = torch.nonzero(ok)[:, 0][flat[ok].topk(20).indices]
    lowthr = torch.quantile(flat[ok].abs(), 0.05); low = torch.nonzero(ok & (flat.abs() <= lowthr))[:, 0]   # the neuron writes (almost) nothing there
    nt = Nall.flatten()[top]; nl = Nall.flatten()[low]
    ill = torch.stack([low[(nl - v).abs().argmin()] for v in nt])                 # illegal positions matched on norm
    s_ = float((flat[top] * W[j].norm()).median()); add = s_ * Wn[j]
    per = {}
    for kind, poss in (("legal", top), ("illegal", ill)):
        vals = dict(surv1=[], surv2=[], surv4=[], energy2=[], energy4=[], expl8=[], self_in8=[], kl=[])
        for f in poss.tolist():
            sq, p = divmod(f, T); c0, lp0 = run(sq, p); c1, lp1 = run(sq, p, add)
            for k, key in ((1, "surv1"), (2, "surv2"), (4, "surv4")):
                if k in c1: vals[key].append(float((c1[k] - c0[k]) @ Wn[j]) / s_)
            for k, key in ((2, "energy2"), (4, "energy4")):
                if k in c1: vals[key].append(float((c1[k] - c0[k]).norm()) / s_)
            d2 = (c1[2] - c0[2])[None]; sel, _, _ = omp(d2, Au, 8, batch=8, record_err=False); _, err = refit(d2, Au, sel)
            vals["expl8"].append(1 - float(err[0] / d2.pow(2).sum())); vals["self_in8"].append(float(any(torch.allclose(Au[t_], Wn[j], atol=1e-4) for t_ in sel[0].tolist())))
            vals["kl"].append(float((lp0.exp() * (lp0 - lp1)).sum()))
        per[kind] = {k: (sorted(v)[len(v) // 2] if v else None) for k, v in vals.items()}
        for k, v in per[kind].items(): agg[kind].setdefault(k, []).append(v)
    res["neurons"].append(dict(neuron=j, write_size=s_, legal=per["legal"], illegal=per["illegal"]))
med = lambda v: sorted(x for x in v if x is not None)[len([x for x in v if x is not None]) // 2]
for kind in agg: res["summary"][kind] = {k: med(v) for k, v in agg[kind].items()}
Lg, Il = res["summary"]["legal"], res["summary"]["illegal"]
res["checks"] = dict(survival_within_20pct=abs(Il["surv2"] - Lg["surv2"]) <= 0.2 * abs(Lg["surv2"]), kl_larger_illegal=Il["kl"] > Lg["kl"])
f_ = lambda S: f"survival after 1/2/4 blocks {S['surv1']:.2f}/{S['surv2']:.2f}/{S['surv4']:.2f}, energy 2/4 {S['energy2']:.2f}/{S['energy4']:.2f}, 8 words explain {S['expl8']:.2f} (word itself among them {S['self_in8']:.2f}), KL {S['kl']:.4f}"
summ = f"{name}, words of block {b}, 40 neurons x 20 contexts: legal (where it fires): " + f_(Lg) + " || illegal (where it never fires, norm-matched): " + f_(Il) + f" | checks {_json.dumps(res['checks'])}"
log(summ); record(f"e464_illegal_{name}", res, summ)
