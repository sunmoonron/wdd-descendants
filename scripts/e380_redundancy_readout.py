"""e380: does the induction circuit's redundancy really grow through training, or does the loss manufacture it? Phase 2
(e365) found the top-8 induction heads' signed interaction rising from +0.08 at step 1000 to +0.94 at the end and joint
non-additivity growing, all measured on the loss. e377 showed the loss's convexity makes pairs look super-additive.
Here, at Pythia checkpoints 1000, 2000, 4000, 16000, 33000 and 143000, the same top-8 heads (by zero-ablation effect on
the induction loss) and 8 random heads are measured on three readouts (loss, the correct token's logit, its centred
logit): the mean signed pairwise interaction of the top-8, the share of super- and sub-additive top pairs, and joint
over sum for ablating all 8 together."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p3_common import *
rev = sys.argv[1]; c = Cache("pythia410"); model, tok, fam = load_eager("pythia410", revision=rev); arch = Arch(model, fam); NB, NH = arch.NB, arch.NH; rng = random.Random(0)
for p in model.parameters(): p.requires_grad_(False)
heads = [(l, h) for l in range(NB) for h in range(NH)]; nh = len(heads)
ind, off, half = induction_batch(tok, c, n=12, half=128, seed=0); pos = torch.arange(off + half, off + 2 * half - 1, device=DEV); tgt = ind[:, pos + 1]
def metrics(a):
    hs = scale_hooks(arch, a)
    try:
        with torch.no_grad(): lg = model(ind).logits[:, pos].float()
    finally: [h.remove() for h in hs]
    cl = lg.gather(-1, tgt[..., None])[..., 0]
    return dict(loss=-torch.log_softmax(lg, -1).gather(-1, tgt[..., None])[..., 0].mean().item(), logit=-cl.mean().item(), centred=-(cl - lg.mean(-1)).mean().item())
one = torch.ones(nh, device=DEV); B0 = metrics(one); E1 = {}
for i in range(nh):
    a = one.clone(); a[i] = 0; m = metrics(a); E1[i] = {k: m[k] - B0[k] for k in m}
order = sorted(range(nh), key=lambda i: -E1[i]["loss"]); top = order[:8]; rnd = rng.sample(order[8:], 8)
res = dict(rev=rev, base=B0, top=[list(heads[i]) for i in top])
for k in ("loss", "logit", "centred"):
    S = []
    for x in range(8):
        for y in range(x + 1, 8):
            i, j = top[x], top[y]; a = one.clone(); a[i] = 0; a[j] = 0; m = metrics(a); S.append((m[k] - B0[k] - E1[i][k] - E1[j][k]) / max(0.5 * (abs(E1[i][k]) + abs(E1[j][k])), 1e-9))
    a = one.clone(); a[top] = 0; mj = metrics(a); ar = one.clone(); ar[rnd] = 0; mr = metrics(ar)
    res[k] = dict(mean_S_top=sum(S) / len(S), share_super=sum(s > 0.2 for s in S) / len(S), share_sub=sum(s < -0.2 for s in S) / len(S), joint_over_sum=(mj[k] - B0[k]) / max(sum(E1[i][k] for i in top), 1e-9), joint_over_sum_random=(mr[k] - B0[k]) / max(abs(sum(E1[i][k] for i in rnd)), 1e-9), joint=(mj[k] - B0[k]), top_single_mean=sum(E1[i][k] for i in top) / 8)
log(f"{rev}: " + " | ".join(f"{k.upper()}: mean S {res[k]['mean_S_top']:+.2f} (super {res[k]['share_super']:.2f}, sub {res[k]['share_sub']:.2f}), joint/sum {res[k]['joint_over_sum']:.2f} (joint {res[k]['joint']:+.2f})" for k in ("loss", "logit", "centred")))
record(f"e380_redundread_{rev}", res, " | ".join(f"{k}: S {res[k]['mean_S_top']:+.2f} j/s {res[k]['joint_over_sum']:.2f}" for k in ("loss", "logit", "centred")))
