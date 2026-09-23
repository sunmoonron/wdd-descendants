"""e374: where does the flat-then-steep response to removing a head live, and is it a prominence crossing? For the
top-6 heads by zero-ablation effect on induction data, 3 random heads and the top-6 jointly, the head scale is swept
a = 1, 0.9, ..., 0 and recorded: the loss; the logits and the last-layer state (how far they have moved, relative to
full removal: 0.5 at a = 0.5 means linear); the loss predicted by moving the logits linearly between a = 1 and a = 0
(if it matches, the knee is made by the final softmax reading linearly moving logits); and, for the downstream head
whose attention changes most on removal, the share of queries whose top key has changed, against the share predicted
by moving its attention scores linearly between the two ends (if it matches, the knee is a prominence crossing at an
internal reader: each query's winner flips when its margin, moving linearly, crosses zero). Also the even part of the
loss around a = 1 (scaling up and down by t) and its local exponent: 2 for an ordinary quadratic minimum, higher for a
degenerate one."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p3_common import *
tag = sys.argv[1]; c = Cache(tag); model, tok, fam = load_eager(c.name); arch = Arch(model, fam); NB, NH = arch.NB, arch.NH; rng = random.Random(0); torch.manual_seed(0)
for p in model.parameters(): p.requires_grad_(False)
heads = [(l, h) for l in range(NB) for h in range(NH)]; nh = len(heads)
ind, off, half = induction_batch(tok, c, n=8, half=128, seed=0); pos = torch.arange(off + half, off + 2 * half - 1, device=DEV); tgt = ind[:, pos + 1]
def fwd(a, attn=False):
    hs = scale_hooks(arch, a)
    try:
        with torch.no_grad(): out = model(ind, output_attentions=attn)
    finally: [h.remove() for h in hs]
    lg = out.logits[:, pos].float(); return lg, (out.attentions if attn else None), out
def ce(lg): return -torch.log_softmax(lg, -1).gather(-1, tgt[..., None])[..., 0].mean().item()
hid = {}
def last_hook(m, args, kwargs):
    h = args[0] if len(args) > 0 else kwargs["hidden_states"]; hid["x"] = h.detach().float(); return None
fin = (model.transformer.ln_f if fam == "gpt2" else (model.model.norm if fam == "llama" else model.gpt_neox.final_layer_norm))
hk = fin.register_forward_pre_hook(last_hook, with_kwargs=True)
one = torch.ones(nh, device=DEV); lg1, att1, _ = fwd(one, True); L1 = ce(lg1); h1 = hid["x"][:, pos].clone()
E1 = torch.zeros(nh)
for i in range(nh):
    a = one.clone(); a[i] = 0; E1[i] = ce(fwd(a)[0]) - L1
order = E1.argsort(descending=True).tolist(); top = order[:6]; rnd = rng.sample(order[6:], 3)
alphas = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]; ts = [0.05, 0.1, 0.2, 0.4]
def study(idx, name):
    a0 = one.clone(); a0[idx] = 0; lg0, att0, _ = fwd(a0, True); h0 = hid["x"][:, pos].clone(); L0 = ce(lg0)
    lmin = min(heads[i][0] for i in idx); best, bestv = None, -1
    for l in range(lmin + 1, NB):
        tv = 0.5 * (att1[l][:, :, pos] - att0[l][:, :, pos]).abs().sum(-1).mean((0, 2))
        v, hh = tv.max(0)
        if v.item() > bestv: bestv, best = v.item(), (l, hh.item())
    curve = []
    for al in alphas:
        a = one.clone(); a[idx] = al; lg, att, _ = fwd(a, best is not None); hx = hid["x"][:, pos]
        row = dict(alpha=al, loss=ce(lg) - L1, acc=(lg.argmax(-1) == tgt).float().mean().item(), logit_move=(lg - lg1).norm(dim=-1).mean().item() / max((lg0 - lg1).norm(dim=-1).mean().item(), 1e-9), state_move=(hx - h1).norm(dim=-1).mean().item() / max((h0 - h1).norm(dim=-1).mean().item(), 1e-9))
        lam = 1 - al; row["loss_linear_logits"] = ce((1 - lam) * lg1 + lam * lg0) - L1
        if best is not None:
            l, h = best; A1, A0, Aa = att1[l][:, h, pos].clamp_min(1e-12).log(), att0[l][:, h, pos].clamp_min(1e-12).log(), att[l][:, h, pos]
            w1 = A1.argmax(-1); row["flip"] = (Aa.argmax(-1) != w1).float().mean().item(); row["flip_linear"] = (((1 - lam) * A1 + lam * A0).argmax(-1) != w1).float().mean().item(); row["attn_tv"] = 0.5 * (Aa - A1.exp()).abs().sum(-1).mean().item() / max(0.5 * (A0.exp() - A1.exp()).abs().sum(-1).mean().item(), 1e-9)
        curve.append(row)
    even = []
    for t in ts:
        up = one.clone(); up[idx] = 1 + t; dn = one.clone(); dn[idx] = 1 - t; even.append(ce(fwd(up)[0]) + ce(fwd(dn)[0]) - 2 * L1)
    ev = torch.tensor(even).abs().clamp_min(1e-9).log(); tt = torch.tensor(ts).log(); k_small = ((ev[1] - ev[0]) / (tt[1] - tt[0])).item(); k_large = ((ev[3] - ev[2]) / (tt[3] - tt[2])).item()
    quad_at_0 = abs(even[1]) / 2 / ts[1] ** 2
    def shape(key):
        v = {r["alpha"]: r.get(key) for r in curve}
        return None if v.get(0.0) in (None, 0) or v.get(0.5) is None else v[0.5] / v[0.0] if key not in ("loss", "loss_linear_logits") else (v[0.5] / v[0.0] if abs(v[0.0]) > 1e-9 else None)
    return dict(name=name, heads=[list(heads[i]) for i in idx], ablation_loss=L0 - L1, downstream_reader=list(best) if best else None, downstream_tv_at_removal=bestv, curve=curve,
                shape_loss=shape("loss"), shape_loss_linear_logits=shape("loss_linear_logits"), shape_logit_move=shape("logit_move"), shape_state_move=shape("state_move"), shape_flip=shape("flip"), shape_flip_linear=shape("flip_linear"), shape_attn_tv=shape("attn_tv"),
                even_part=even, k_small=k_small, k_large=k_large, quadratic_prediction_at_removal=quad_at_0, actual_over_quadratic=(L0 - L1) / max(quad_at_0, 1e-9))
res = dict(model=tag, base_loss=L1, studies=[study([i], f"top{k}") for k, i in enumerate(top)] + [study([i], f"random{k}") for k, i in enumerate(rnd)] + [study(top, "top6_joint")])
hk.remove()
def fmt(s): return f"{s['name']} {s['heads'] if len(s['heads']) == 1 else 'top-6'}: removal {s['ablation_loss']:+.3f}; shapes loss {s['shape_loss'] if s['shape_loss'] is None else round(s['shape_loss'], 2)} / loss from linear logits {s['shape_loss_linear_logits'] if s['shape_loss_linear_logits'] is None else round(s['shape_loss_linear_logits'], 2)} / logits {s['shape_logit_move']:.2f} / state {s['shape_state_move']:.2f} / reader {s['downstream_reader']} flips {s['shape_flip'] if s['shape_flip'] is None else round(s['shape_flip'], 2)} vs linear {s['shape_flip_linear'] if s['shape_flip_linear'] is None else round(s['shape_flip_linear'], 2)}; even-part exponent {s['k_small']:.1f} (small t) {s['k_large']:.1f} (large t); actual/quadratic {s['actual_over_quadratic']:.1f}"
for s in res["studies"]: log(f"{tag}: " + fmt(s))
tops = [s for s in res["studies"] if s["name"].startswith("top") and s["name"] != "top6_joint"]; med = lambda v: sorted(v)[len(v) // 2] if v else float("nan")
res["summary"] = {k: med([s[k] for s in tops if s[k] is not None]) for k in ("shape_loss", "shape_loss_linear_logits", "shape_logit_move", "shape_state_move", "shape_flip", "shape_flip_linear", "k_small", "k_large", "actual_over_quadratic")}
record(f"e374_scalemech_{tag}", res, "top heads (median): " + " ".join(f"{k} {v:.2f}" for k, v in res["summary"].items()))
