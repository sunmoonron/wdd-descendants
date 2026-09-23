"""e372: the shape of the loss along head scales, the direct view of why local quantities (e371) miss ablation
structure. For the top-8 heads by zero-ablation effect and 8 random heads, on induction data and natural text: the
loss at a_h in {1, 0.75, 0.5, 0.25, 0} for each head alone, and for the top-8 scaled together; the shape index
(L(0.5) - L(1)) / (L(0) - L(1)) (0.5 for a linear path, 0.25 for a quadratic path with zero slope, below 0.25 when the
loss rises only near zero, above 0.5 when it saturates); and the quadratic prediction from the gradient and the
second derivative at a = 1 (single heads: H_ii; the joint path: v'Hv along the top-8 direction)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pc_common import *
tag = sys.argv[1]; c = Cache(tag); model, tok, fam = load_eager(c.name); arch = Arch(model, fam); NB, NH = arch.NB, arch.NH; rng = random.Random(0); torch.manual_seed(0)
for p in model.parameters(): p.requires_grad_(False)
heads = [(l, h) for l in range(NB) for h in range(NH)]; nh = len(heads)
def run_scaled(ids, a):
    hs = []
    for l in range(NB):
        def pre(m, args, l=l):
            x = args[0]; B, T, W = x.shape; x = (x.reshape(B, T, NH, W // NH) * a[l * NH:(l + 1) * NH].to(x.dtype)[None, None, :, None]).reshape(B, T, W); return (x,) + tuple(args[1:])
        hs.append(arch.attn_lin(l).register_forward_pre_hook(pre))
    try: return model(ids).logits.float()
    finally: [h.remove() for h in hs]
ind, off, half = induction_batch(tok, c, n=8, half=128, seed=0); nat = c.s["eval_ids"][:4, :256].to(DEV)
datasets = {"induction": (ind, lambda lg, ids: induction_loss(lg, ids, off, half).mean()), "natural": (nat, lambda lg, ids: token_loss(lg, ids).mean())}
A = [1.0, 0.75, 0.5, 0.25, 0.0]; res = dict(model=tag, alphas=A)
for dn, (ids, lossf) in datasets.items():
    one = torch.ones(nh, device=DEV)
    with torch.no_grad():
        L0 = lossf(run_scaled(ids, one), ids).item(); E1 = torch.zeros(nh)
        for i in range(nh):
            a = one.clone(); a[i] = 0; E1[i] = lossf(run_scaled(ids, a), ids).item() - L0
    order = E1.argsort(descending=True).tolist(); top = order[:8]; rnd = rng.sample(order[8:], 8)
    def path(idx):
        out = []
        with torch.no_grad():
            for al in A:
                a = one.clone(); a[idx] = al; out.append(lossf(run_scaled(ids, a), ids).item() - L0)
        return out
    a = one.clone().requires_grad_(True)
    with torch.enable_grad():
        L = lossf(run_scaled(ids, a), ids); g = torch.autograd.grad(L, a, create_graph=True)[0]
        Hd = {i: torch.autograd.grad(g[i], a, retain_graph=True)[0][i].item() for i in top + rnd}
        v = torch.zeros(nh, device=DEV); v[top] = 1.0; gv = (g * v).sum(); vHv = (torch.autograd.grad(gv, a, retain_graph=True)[0] * v).sum().item(); gvv = gv.item()
    g = g.detach().cpu()
    def shape(p): return p[2] / p[4] if abs(p[4]) > 1e-9 else float("nan")
    def quad(gi, hii): return [(-(1 - al)) * gi + 0.5 * hii * (1 - al) ** 2 for al in A]
    singles = {"top": [], "random": []}
    for grp, idxs in (("top", top), ("random", rnd)):
        for i in idxs:
            p = path([i]); q = quad(g[i].item(), Hd[i]); singles[grp].append(dict(head=list(heads[i]), path=p, quadratic=q, shape=shape(p), quad_at_zero=q[4]))
    pj = path(top); qj = quad(gvv, vHv)
    med = lambda xs: float(torch.tensor([x for x in xs if x == x]).median()) if any(x == x for x in xs) else float("nan")
    res[dn] = dict(base_loss=L0, singles=singles, joint_top8=dict(path=pj, quadratic=qj, shape=shape(pj)), shape_top_median=med([s["shape"] for s in singles["top"]]), shape_random_median=med([s["shape"] for s in singles["random"]]),
                   quad_over_actual_top_median=med([s["quad_at_zero"] / s["path"][4] for s in singles["top"] if abs(s["path"][4]) > 1e-6]), joint_quad_over_actual=qj[4] / pj[4] if abs(pj[4]) > 1e-9 else float("nan"), joint_over_sum_singles=pj[4] / sum(s["path"][4] for s in singles["top"]))
def fm(d): return (f"single top-8 shape index median {d['shape_top_median']:.2f} (random {d['shape_random_median']:.2f}), quadratic/actual at a=0 {d['quad_over_actual_top_median']:.2f}; joint top-8 path " + " ".join(f"{x:+.3f}" for x in d['joint_top8']['path']) + f" (shape {d['joint_top8']['shape']:.2f}, quadratic/actual {d['joint_quad_over_actual']:.2f}, joint/sum of singles {d['joint_over_sum_singles']:.2f})")
log(f"{tag}: INDUCTION: " + fm(res["induction"]) + " | NATURAL: " + fm(res["natural"]))
record(f"e372_scalepath_{tag}", res, " | ".join(f"{dn}: shape top {res[dn]['shape_top_median']:.2f} rand {res[dn]['shape_random_median']:.2f} quad/actual {res[dn]['quad_over_actual_top_median']:.2f}; joint shape {res[dn]['joint_top8']['shape']:.2f} quad/actual {res[dn]['joint_quad_over_actual']:.2f} joint/sum {res[dn]['joint_over_sum_singles']:.2f}" for dn in ("induction", "natural")))
