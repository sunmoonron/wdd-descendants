"""e185: anatomy of cancellation and reinforcement. For dominant block-b writes at level L (typical states), take the
main canceller block (from the e173 matrix: within-block for GPT-2, next block otherwise) and the block-0
reinforcer, and ask whether the contribution is concentrated (one neuron carries most of it: a dedicated
eraser/amplifier, and whether that neuron is the SAME across tokens) or diffuse (a crowd of small projections:
swamping). Reports the share of the negative contribution carried by the top-1 and top-5 neurons, the number of
neurons needed for 50%, the consistency (fraction of tokens whose top canceller neuron is the modal one for that
writer neuron), and the cosine of the top canceller row with the write direction."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); N = 6144; ids = sub(c.NT, N); Xraw = c.X(L, center=False)[ids]; typ = typical_mask(Xraw); A, lab = c.dictionary(L)
tb, tn, tc = c.top_writes(L, 1); tb, tn, ct = tb[ids, 0], tn[ids, 0], tc[ids, 0].to(DEV); row = c.atom_index(L, tb, tn).to(DEV); d = A[row]; tbd = tb.to(DEV)
within = tag.startswith("gpt2"); out = {}
def anatomy(b, bp, sign):
    m = typ & (tbd == b)
    if m.sum() < 30: return None
    acts = c.acts[bp][ids][m.cpu()].float().to(DEV); W = c.wdir_cpu(bp).to(DEV); dd = d[m]; cc = ct[m]
    contrib = acts * (W @ dd.T).T / cc[:, None]                                                     # [n, DFF] per-neuron contribution to survival
    if bp == b: contrib[torch.arange(len(contrib)), tn[m.cpu()].to(DEV)] = 0                    # remove the write itself
    part = contrib.clamp(max=0) if sign < 0 else contrib.clamp(min=0); tot = part.sum(1).abs().clamp_min(1e-9); srt = part.abs().sort(1, descending=True).values
    top1 = (srt[:, 0] / tot); top5 = (srt[:, :5].sum(1) / tot); cum = srt.cumsum(1) / tot[:, None]; n50 = ((cum < 0.5).sum(1) + 1).float()
    topn = part.abs().argmax(1); wn = tn[m.cpu()].to(DEV); cons = []
    for w in wn.unique():
        sel = wn == w
        if sel.sum() >= 5: cons.append((topn[sel] == torch.mode(topn[sel]).values).float().mean().item())
    cosr = ((W[topn] / W[topn].norm(dim=1, keepdim=True)) * dd).sum(1)
    return dict(n=int(m.sum()), top1=top1.median().item(), top5=top5.median().item(), n50=n50.median().item(), consistency=(sum(cons) / len(cons)) if cons else float("nan"), n_writers=len(cons), cos_top=cosr.median().item(), mean_total=(part.sum(1)).mean().item())
for b in range(1, min(L, 8) + 1):
    bp = b if within else b + 1
    if bp > L: continue
    r = anatomy(b, bp, -1); r0 = anatomy(b, 0, +1) if b > 0 else None
    if r is None: continue
    out[b] = dict(canceller_block=bp, cancel=r, reinforce0=r0)
    log(f"{tag} born b{b} canceller b{bp}: total {r['mean_total']:+.2f} top1 share {r['top1']:.2f} top5 {r['top5']:.2f} n50 {r['n50']:.0f} consistency {r['consistency']:.2f} ({r['n_writers']} writers) cos(top row, d) {r['cos_top']:+.2f}" + (f" | block-0 reinforcement {r0['mean_total']:+.2f} top1 {r0['top1']:.2f} n50 {r0['n50']:.0f} consistency {r0['consistency']:.2f} cos {r0['cos_top']:+.2f}" if r0 else ""))
import numpy as np
ks = list(out.keys()); med = lambda key, sub: float(np.median([out[b][sub][key] for b in ks if out[b][sub] is not None]))
record(f"e185_anatomy_{tag}", dict(model=tag, L=L, within=within, per_birth=out), f"canceller ({'within' if within else 'next'} block): top1 share {med('top1', 'cancel'):.2f} top5 {med('top5', 'cancel'):.2f} n50 {med('n50', 'cancel'):.0f} consistency {med('consistency', 'cancel'):.2f} cos(top canceller row, d) {med('cos_top', 'cancel'):+.2f} | block-0 reinforcement: top1 {med('top1', 'reinforce0'):.2f} n50 {med('n50', 'reinforce0'):.0f} consistency {med('consistency', 'reinforce0'):.2f} cos {med('cos_top', 'reinforce0'):+.2f}")
