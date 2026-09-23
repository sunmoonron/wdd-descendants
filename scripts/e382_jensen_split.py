"""e382: Jensen made quantitative. Cross-entropy is convex in the logits (it is a log-sum-exp minus a linear term), so
two ablations whose logit effects add exactly can still interact on the loss. For every pair of the top-12 heads (by
zero-ablation effect on induction data), with logit effects d_i, d_j and d_ij at the second-copy positions, the loss
interaction L_ij - L_i - L_j is split per position into: (1) the circuit term g . c, the loss gradient at the intact
model times the logit-space interaction c = d_ij - d_i - d_j (what the pair does jointly that it does not do
separately, read linearly); (2) the readout term d_i' H d_j, the softmax covariance of the two single effects (H =
diag(p) - p p', the curvature of the loss in the logits: positive when the two heads push the same tokens); (3) the
remainder (higher orders). Reported: each term's share of the loss interaction, and how often the readout term alone
sets the sign of a pair's interaction (it agrees with the loss while the circuit term disagrees)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p3_common import *
tag = sys.argv[1]; c = Cache(tag); model, tok, fam = load_eager(c.name); arch = Arch(model, fam); NB, NH = arch.NB, arch.NH; torch.manual_seed(0)
for p in model.parameters(): p.requires_grad_(False)
heads = [(l, h) for l in range(NB) for h in range(NH)]; nh = len(heads)
ind, off, half = induction_batch(tok, c, n=6, half=128, seed=0); pos = torch.arange(off + half, off + 2 * half - 1, device=DEV); tgt = ind[:, pos + 1].reshape(-1)
def logits(a):
    hs = scale_hooks(arch, a)
    try:
        with torch.no_grad(): return model(ind).logits[:, pos].float().reshape(-1, model.config.vocab_size if hasattr(model.config, "vocab_size") else -1)
    finally: [h.remove() for h in hs]
ce = lambda z: -torch.log_softmax(z, -1).gather(-1, tgt[:, None])[:, 0]
one = torch.ones(nh, device=DEV); z0 = logits(one); L0 = ce(z0); p0 = torch.softmax(z0, -1); g = p0.clone(); g[torch.arange(len(tgt)), tgt] -= 1
E1 = torch.zeros(nh)
for i in range(nh):
    a = one.clone(); a[i] = 0; E1[i] = (ce(logits(a)) - L0).mean()
top = E1.argsort(descending=True)[:12].tolist(); D = {}
for i in top:
    a = one.clone(); a[i] = 0; D[i] = (logits(a) - z0).half()
rows = []
for x in range(12):
    for y in range(x + 1, 12):
        i, j = top[x], top[y]; a = one.clone(); a[i] = 0; a[j] = 0; zij = logits(a); di, dj = D[i].float(), D[j].float(); dij = zij - z0
        Lint = (ce(zij) - ce(z0 + di) - ce(z0 + dj) + L0).mean().item(); cvec = dij - di - dj; circ = (g * cvec).sum(-1).mean().item()
        curv = ((p0 * di * dj).sum(-1) - (p0 * di).sum(-1) * (p0 * dj).sum(-1)).mean().item(); norm = max(0.5 * (abs(E1[i].item()) + abs(E1[j].item())), 1e-9)
        Radd = (ce(z0 + di + dj) - ce(z0 + di) - ce(z0 + dj) + L0).mean().item(); Bc = (ce(zij) - ce(z0 + di + dj)).mean().item(); Bc2 = (ce(z0 + cvec) - L0).mean().item()
        rows.append(dict(i=list(heads[i]), j=list(heads[j]), S_loss=Lint / norm, S_readout_exact=Radd / norm, S_circuit_exact=Bc / norm, S_circuit_first=Bc2 / norm, S_readout_second=(Lint - Bc2) / norm, S_circuit=circ / norm, S_readout=curv / norm, S_rest=(Lint - circ - curv) / norm, logit_interaction_norm=cvec.norm(dim=-1).mean().item(), single_norm=0.5 * (di.norm(dim=-1).mean().item() + dj.norm(dim=-1).mean().item())))
        del zij, dij, cvec
m = lambda k: sum(r[k] for r in rows) / len(rows); ma = lambda k: sum(abs(r[k]) for r in rows) / len(rows)
readout_sets_sign = sum((r["S_readout"] > 0) == (r["S_loss"] > 0) and (r["S_circuit"] > 0) != (r["S_loss"] > 0) for r in rows) / len(rows)
circuit_sets_sign = sum((r["S_circuit"] > 0) == (r["S_loss"] > 0) and (r["S_readout"] > 0) != (r["S_loss"] > 0) for r in rows) / len(rows)
res = dict(model=tag, top=[list(heads[i]) for i in top], pairs=rows, mean=dict(S_loss=m("S_loss"), S_readout_exact=m("S_readout_exact"), S_circuit_exact=m("S_circuit_exact"), S_circuit=m("S_circuit"), S_readout=m("S_readout"), S_rest=m("S_rest")),
           exact_readout_sets_sign=sum((r["S_readout_exact"] > 0) == (r["S_loss"] > 0) and (r["S_circuit_exact"] > 0) != (r["S_loss"] > 0) for r in rows) / len(rows), exact_circuit_sets_sign=sum((r["S_circuit_exact"] > 0) == (r["S_loss"] > 0) and (r["S_readout_exact"] > 0) != (r["S_loss"] > 0) for r in rows) / len(rows), share_super_circuit_exact=sum(r["S_circuit_exact"] > 0.2 for r in rows) / len(rows), share_sub_circuit_exact=sum(r["S_circuit_exact"] < -0.2 for r in rows) / len(rows), share_super_readout_exact=sum(r["S_readout_exact"] > 0.2 for r in rows) / len(rows), mean_circuit_first=m("S_circuit_first"), mean_readout_second=m("S_readout_second"), share_sub_circuit_first=sum(r["S_circuit_first"] < -0.2 for r in rows) / len(rows), share_super_circuit_first=sum(r["S_circuit_first"] > 0.2 for r in rows) / len(rows), mean_abs=dict(S_loss=ma("S_loss"), S_circuit=ma("S_circuit"), S_readout=ma("S_readout"), S_rest=ma("S_rest")),
           share_super_loss=sum(r["S_loss"] > 0.2 for r in rows) / len(rows), share_super_circuit=sum(r["S_circuit"] > 0.2 for r in rows) / len(rows), share_sub_circuit=sum(r["S_circuit"] < -0.2 for r in rows) / len(rows), readout_sets_sign=readout_sets_sign, circuit_sets_sign=circuit_sets_sign,
           relative_logit_interaction=sum(r["logit_interaction_norm"] / max(r["single_norm"], 1e-9) for r in rows) / len(rows))
log(f"{tag}: EXACT SPLIT: loss {res['mean']['S_loss']:+.2f} = readout (logits added exactly) {res['mean']['S_readout_exact']:+.2f} + circuit (logit-space interaction, read at the joint point) {res['mean']['S_circuit_exact']:+.2f}; super-additive pairs: loss {res['share_super_loss']:.2f}, readout part {res['share_super_readout_exact']:.2f}, circuit part {res['share_super_circuit_exact']:.2f} (circuit sub-additive {res['share_sub_circuit_exact']:.2f}); readout alone sets the sign in {res['exact_readout_sets_sign']:.2f} of pairs, circuit alone in {res['exact_circuit_sets_sign']:.2f} | REVERSE ORDER (circuit term applied at the intact model first): circuit {res['mean_circuit_first']:+.2f} (sub-additive {res['share_sub_circuit_first']:.2f}, super {res['share_super_circuit_first']:.2f}), readout remainder {res['mean_readout_second']:+.2f} || TAYLOR: mean relative interaction on the loss {res['mean']['S_loss']:+.2f} = circuit {res['mean']['S_circuit']:+.2f} + readout curvature {res['mean']['S_readout']:+.2f} + rest {res['mean']['S_rest']:+.2f} (mean |.|: {res['mean_abs']['S_loss']:.2f} = {res['mean_abs']['S_circuit']:.2f} / {res['mean_abs']['S_readout']:.2f} / {res['mean_abs']['S_rest']:.2f}); super-additive pairs on the loss {res['share_super_loss']:.2f}, on the circuit term {res['share_super_circuit']:.2f} (sub-additive {res['share_sub_circuit']:.2f}); readout alone sets the sign in {res['readout_sets_sign']:.2f} of pairs, circuit alone in {res['circuit_sets_sign']:.2f}; logit-space interaction norm relative to single effects {res['relative_logit_interaction']:.2f}")
record(f"e382_jensen_{tag}", res, f"exact: loss {res['mean']['S_loss']:+.2f} = readout {res['mean']['S_readout_exact']:+.2f} + circuit {res['mean']['S_circuit_exact']:+.2f}; super-additive loss {res['share_super_loss']:.2f} readout {res['share_super_readout_exact']:.2f} circuit {res['share_super_circuit_exact']:.2f}")
