"""e371: co-selection, curvature and ablation interaction in the space of head scales. Each attention head's write is
multiplied by a scale a_h (a = 1 is the model). The loss L(a) on a dataset gives, at a = 1: the gradient g (first-order
selection, e359-e368), the Hessian H (curvature coupling), and, over chunks of the data, the gradient correlation (the
co-selection matrix; at a well-specified optimum the per-example gradient covariance is the Fisher, which equals H).
Zero-ablating heads i and j is the move a_i = a_j = 0, so the second-order expansion predicts the single effect
E_i = -g_i + H_ii / 2 and the pairwise interaction E_ij - E_i - E_j = H_ij exactly. The test compares, on induction
data and on natural text, for the top-12 heads by single effect and 12 random heads: single effects against their
first- and second-order predictions, the measured interaction matrix against H_ij, and the co-selection correlation
against the normalised H and against the measured interactions. Where they agree, co-selection and curvature are
local versions of circuit interaction; where they disagree, the loss is not quadratic in the unit scales (saturation,
backups that engage only far from a = 1)."""
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
def spear(x, y):
    x, y = torch.as_tensor(x).float(), torch.as_tensor(y).float(); rx = x.argsort().argsort().float(); ry = y.argsort().argsort().float(); return torch.corrcoef(torch.stack([rx, ry]))[0, 1].item()
def r2(pred, true): pred, true = torch.as_tensor(pred).float(), torch.as_tensor(true).float(); return 1 - ((true - pred) ** 2).sum().item() / ((true - true.mean()) ** 2).sum().clamp_min(1e-12).item()
ind, off, half = induction_batch(tok, c, n=8, half=128, seed=0); nat = c.s["eval_ids"][:4, :256].to(DEV)
datasets = {"induction": (ind, lambda lg, ids: induction_loss(lg, ids, off, half).mean()), "natural": (nat, lambda lg, ids: token_loss(lg, ids).mean())}
res = dict(model=tag, n_heads=nh)
for dn, (ids, lossf) in datasets.items():
    one = torch.ones(nh, device=DEV)
    with torch.no_grad():
        L0 = lossf(run_scaled(ids, one), ids).item(); E1 = torch.zeros(nh)
        for i in range(nh):
            a = one.clone(); a[i] = 0; E1[i] = lossf(run_scaled(ids, a), ids).item() - L0
    order = E1.argsort(descending=True).tolist(); U = order[:12] + rng.sample(order[12:], 12); n = len(U)
    with torch.no_grad():
        S = torch.zeros(n, n)
        for x in range(n):
            for y in range(x + 1, n):
                a = one.clone(); a[U[x]] = 0; a[U[y]] = 0; S[x, y] = S[y, x] = lossf(run_scaled(ids, a), ids).item() - L0 - E1[U[x]].item() - E1[U[y]].item()
    a = one.clone().requires_grad_(True)
    with torch.enable_grad():
        L = lossf(run_scaled(ids, a), ids); g = torch.autograd.grad(L, a, create_graph=True)[0]; H = torch.zeros(n, n)
        for x in range(n): H[x] = torch.autograd.grad(g[U[x]], a, retain_graph=True)[0][U].detach().cpu()
    g = g.detach().cpu(); H = 0.5 * (H + H.T)
    # co-selection over chunks: per-chunk gradients of the chunk loss
    more = c.s["eval_ids"][:16, :256].to(DEV); chunks = [ids[i:i + 1] for i in range(ids.shape[0])] if dn == "induction" else [more[i:i + 1, s:s + 64] for i in range(more.shape[0]) for s in range(0, more.shape[1], 64)]
    if dn == "induction":
        extra = [induction_batch(tok, c, n=1, half=128, seed=1000 + k)[0] for k in range(40)]; chunks = chunks + extra
    G = []
    for ch in chunks:
        a2 = one.clone().requires_grad_(True)
        with torch.enable_grad():
            Lc = lossf(run_scaled(ch, a2), ch); G.append(torch.autograd.grad(Lc, a2)[0][U].detach().cpu())
    G = torch.stack(G); Gz = (G - G.mean(0)) / G.std(0).clamp_min(1e-12); CS = (Gz.T @ Gz) / (len(G) - 1)
    d = H.diagonal().clamp_min(1e-12).sqrt(); Hn = H / torch.outer(d, d); iu = torch.triu_indices(n, n, 1); top = (iu[0] < 12) & (iu[1] < 12)
    Ei = E1[U]; t1 = -g[U]; t2 = -g[U] + 0.5 * H.diagonal()
    Sv, Hv, CSv, Hnv = S[iu[0], iu[1]], H[iu[0], iu[1]], CS[iu[0], iu[1]], Hn[iu[0], iu[1]]
    res[dn] = dict(base_loss=L0, top_heads=[list(heads[i]) for i in U[:12]], single_effect_top_mean=Ei[:12].mean().item(), single_effect_random_mean=Ei[12:].mean().item(),
        single_first_order_spearman=spear(t1, Ei), single_first_order_r2=r2(t1, Ei), single_second_order_spearman=spear(t2, Ei), single_second_order_r2=r2(t2, Ei),
        interaction_vs_hessian_spearman_all=spear(Hv, Sv), interaction_vs_hessian_r2_all=r2(Hv, Sv), interaction_vs_hessian_spearman_top=spear(Hv[top], Sv[top]), interaction_vs_hessian_r2_top=r2(Hv[top], Sv[top]), interaction_abs_mean_top=Sv[top].abs().mean().item(), hessian_offdiag_abs_mean_top=Hv[top].abs().mean().item(),
        coselection_vs_hessian_spearman=spear(CSv, Hnv), coselection_vs_interaction_spearman=spear(CSv, Sv), coselection_vs_hessian_spearman_top=spear(CSv[top], Hnv[top]), coselection_vs_interaction_spearman_top=spear(CSv[top], Sv[top]), n_chunks=len(G), gradient_top_mean=g[U[:12]].mean().item(), hessian_diag_top_mean=H.diagonal()[:12].mean().item())
def fm(d): return (f"single effect top {d['single_effect_top_mean']:+.3f} vs random {d['single_effect_random_mean']:+.4f}; singles from first order Spearman {d['single_first_order_spearman']:+.2f} (R2 {d['single_first_order_r2']:.2f}), second order {d['single_second_order_spearman']:+.2f} (R2 {d['single_second_order_r2']:.2f}); "
                  f"pairwise interaction vs Hessian Spearman {d['interaction_vs_hessian_spearman_all']:+.2f} (R2 {d['interaction_vs_hessian_r2_all']:.2f}), top block {d['interaction_vs_hessian_spearman_top']:+.2f} (R2 {d['interaction_vs_hessian_r2_top']:.2f}; |interaction| {d['interaction_abs_mean_top']:.3f} vs |H_ij| {d['hessian_offdiag_abs_mean_top']:.3f}); "
                  f"co-selection vs normalised Hessian {d['coselection_vs_hessian_spearman']:+.2f} (top {d['coselection_vs_hessian_spearman_top']:+.2f}), co-selection vs interaction {d['coselection_vs_interaction_spearman']:+.2f} (top {d['coselection_vs_interaction_spearman_top']:+.2f}) over {d['n_chunks']} chunks")
log(f"{tag}: INDUCTION: " + fm(res["induction"]) + " | NATURAL: " + fm(res["natural"]))
record(f"e371_scalehess_{tag}", res, " | ".join(f"{dn}: singles 1st/2nd R2 {res[dn]['single_first_order_r2']:.2f}/{res[dn]['single_second_order_r2']:.2f}; interaction~H rho {res[dn]['interaction_vs_hessian_spearman_all']:+.2f} R2 {res[dn]['interaction_vs_hessian_r2_all']:.2f} (top rho {res[dn]['interaction_vs_hessian_spearman_top']:+.2f}); cosel~H rho {res[dn]['coselection_vs_hessian_spearman']:+.2f}, cosel~interaction rho {res[dn]['coselection_vs_interaction_spearman']:+.2f}" for dn in ("induction", "natural")))
