"""e377: are "backup" interactions a property of the circuit or of the metric? e374 found that the knee of a head's
removal curve comes mostly from the softmax reading linearly moving logits, and e376 found that head pairs in series
(one reads the other) look super-additive on the loss, like backups. Hypothesis: the convexity of cross-entropy makes
any two heads of one circuit super-additive, whatever their wiring. Test: the same pairs (top-12 by zero-ablation
effect and 12 random heads, induction data), interactions measured on three readouts: the loss (convex), the correct
token's logit (linear in the logits) and the correct token's centred logit; for each, the share of super- and
sub-additive pairs, how many pairs change sign against the loss, and the correlation of the signed interaction with
composition (series) and with write cosine (twins). If topology is visible only through a linear readout, composition
turns sub-additive and twins stay super-additive there."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p3_common import *
import torch.nn.functional as F
tag = sys.argv[1]; c = Cache(tag); model, tok, fam = load_eager(c.name); arch = Arch(model, fam); NB, NH, HD = arch.NB, arch.NH, arch.HD; rng = random.Random(0); torch.manual_seed(0)
for p in model.parameters(): p.requires_grad_(False)
heads = [(l, h) for l in range(NB) for h in range(NH)]; nh = len(heads); cap = Capture(model, arch)
ind, off, half = induction_batch(tok, c, n=8, half=128, seed=0); pos = torch.arange(off + half, off + 2 * half - 1, device=DEV); tgt = ind[:, pos + 1]
def metrics(a):
    hs = scale_hooks(arch, a)
    try:
        with torch.no_grad(): lg = model(ind).logits[:, pos].float()
    finally: [h.remove() for h in hs]
    cl = lg.gather(-1, tgt[..., None])[..., 0]
    return dict(loss=-torch.log_softmax(lg, -1).gather(-1, tgt[..., None])[..., 0].mean().item(), neg_logit=-cl.mean().item(), neg_centred_logit=-(cl - lg.mean(-1)).mean().item())
one = torch.ones(nh, device=DEV); B0 = metrics(one); E1 = {}
for i in range(nh):
    a = one.clone(); a[i] = 0; m = metrics(a); E1[i] = {k: m[k] - B0[k] for k in m}
order = sorted(range(nh), key=lambda i: -E1[i]["loss"]); U = order[:12] + rng.sample(order[12:], 12); n = len(U)
X, Z, M, out = cap(ind); P = ind.shape[0] * ind.shape[1]
O = {i: Z[heads[i][0]].reshape(P, -1)[:, heads[i][1] * HD:(heads[i][1] + 1) * HD] @ arch.wo(heads[i][0])[heads[i][1] * HD:(heads[i][1] + 1) * HD].to(DEV) for i in U}
RD = {j: {w: reader_dirs(arch, heads[j][0], heads[j][1], w, X[heads[j][0]].reshape(P, -1))[0] for w in ("Q", "K", "V")} for j in U if heads[j][0] > 0}
rows = []
for x in range(n):
    for y in range(x + 1, n):
        i, j = U[x], U[y]; a = one.clone(); a[i] = 0; a[j] = 0; m = metrics(a); S = {}
        for k in m:
            Eij = m[k] - B0[k]; S[k] = (Eij - E1[i][k] - E1[j][k]) / max(0.5 * (abs(E1[i][k]) + abs(E1[j][k])), 1e-9)
        same = heads[i][0] == heads[j][0]; lo, hi = (i, j) if heads[i][0] < heads[j][0] else (j, i)
        comp = max((O[lo] * RD[hi][w]).sum(-1).abs().mean().item() for w in ("Q", "K", "V")) if (not same and hi in RD) else 0.0
        mm = (O[i].norm(dim=-1) > 1e-6) & (O[j].norm(dim=-1) > 1e-6); wc = F.cosine_similarity(O[i][mm], O[j][mm], dim=-1).mean().item() if mm.any() else 0.0
        rows.append(dict(i=list(heads[i]), j=list(heads[j]), top_top=x < 12 and y < 12, S=S, comp=comp, write_cos=wc, same_layer=same))
def spear(a, b):
    a, b = torch.tensor(a).float(), torch.tensor(b).float(); return torch.corrcoef(torch.stack([a.argsort().argsort().float(), b.argsort().argsort().float()]))[0, 1].item()
res = dict(model=tag, base=B0, top_heads=[list(heads[i]) for i in U[:12]], pairs=rows, by_metric={})
for k in ("loss", "neg_logit", "neg_centred_logit"):
    for scope, rr in (("all", rows), ("top", [r for r in rows if r["top_top"]])):
        s = [r["S"][k] for r in rr]; cr = [r for r in rr if not r["same_layer"]]
        res["by_metric"][f"{k}_{scope}"] = dict(share_super=sum(v > 0.2 for v in s) / len(s), share_sub=sum(v < -0.2 for v in s) / len(s), mean_S=sum(s) / len(s),
            sign_flips_vs_loss=sum((r["S"][k] > 0) != (r["S"]["loss"] > 0) for r in rr) / len(rr), rho_S_comp_cross=spear([r["S"][k] for r in cr], [r["comp"] for r in cr]) if len(cr) > 2 else float("nan"), rho_S_writecos=spear(s, [r["write_cos"] for r in rr]))
def fm(d): return f"super {d['share_super']:.2f} sub {d['share_sub']:.2f} mean S {d['mean_S']:+.2f} sign flips vs loss {d['sign_flips_vs_loss']:.2f} rho(S, composition) {d['rho_S_comp_cross']:+.2f} rho(S, write cos) {d['rho_S_writecos']:+.2f}"
log(f"{tag}: " + " | ".join(f"{k.upper()}: all pairs {fm(res['by_metric'][k + '_all'])}; top-12 pairs {fm(res['by_metric'][k + '_top'])}" for k in ("loss", "neg_logit", "neg_centred_logit")))
record(f"e377_metricconv_{tag}", res, " | ".join(f"{k}: super {res['by_metric'][k + '_top']['share_super']:.2f} sub {res['by_metric'][k + '_top']['share_sub']:.2f} rho(S,comp) {res['by_metric'][k + '_all']['rho_S_comp_cross']:+.2f}" for k in ("loss", "neg_logit", "neg_centred_logit")))
