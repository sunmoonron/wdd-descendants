"""e370: is the natural-text induction circuit saturated? On the 8 natural sequences of e369 and the 16 of e368: per
position loss and probability of the correct next token at induction-applicable positions against the rest (share
with p > 0.9, mean loss, mean p), and the local softmax gradient scale (1 - p) that multiplies every first-order
selection signal. Then, from the saved e368 (gradient) and e369 (ablation) token matrices: for each attention-defined
induction head, its mean gradient signal and its mean ablation effect at applicable positions, and their ratio
against the other heads, the direct measure of how much of the ablation effect the first-order signal misses."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pc_common import *
tag = sys.argv[1]; c = Cache(tag); model, tok, fam = load_eager(c.name); arch = Arch(model, fam)
def mask(nat):
    B, T = nat.shape; M = torch.zeros(B, T - 1, dtype=torch.bool)
    for b, row in enumerate(nat.tolist()):
        last = {}
        for t in range(T - 1):
            a = row[t]
            if a in last and row[last[a] + 1] == row[t + 1]: M[b, t] = True
            last[a] = t
    return M
nat = c.s["eval_ids"][:16]; res = dict(model=tag)
with torch.no_grad():
    L, P = [], []
    for ch in range(0, 16, 4):
        ids = nat[ch:ch + 4].to(DEV); lg = model(ids).logits.float(); lp = torch.log_softmax(lg[:, :-1], -1).gather(2, ids[:, 1:, None])[..., 0]; L.append(-lp.cpu()); P.append(lp.exp().cpu())
L, P = torch.cat(L), torch.cat(P); M = mask(nat); keep = torch.zeros_like(M); keep[:, 1:] = True
for nm, m in (("applicable", M & keep), ("other", ~M & keep)):
    res[nm] = dict(n=int(m.sum()), mean_loss=L[m].mean().item(), mean_p=P[m].mean().item(), share_p_above_0p9=(P[m] > 0.9).float().mean().item(), mean_one_minus_p=(1 - P[m]).mean().item())
for src, key in (("e368", "C"), ("e369", "C")):
    f = os.path.join(RESULTS, src, f"{tag}.pt")
    if not os.path.exists(f): res[src] = None; continue
    d = torch.load(f); Cm = d[key].float(); Mm = d["M"]; pf, pv = d["pref"], d["prev"]; nh = Cm.shape[-1]
    IND = [i for i in pf.argsort(descending=True).tolist() if pf[i] >= 0.2 and pf[i] >= pv[i]][:8]; others = [i for i in range(nh) if i not in IND]
    kk = torch.zeros_like(Mm); kk[:, 1:-1] = True; X = Cm[kk]; Mk = Mm[kk]; sgn = -1.0 if src == "e368" else 1.0
    app = sgn * X[Mk].mean(0); rest = sgn * X[~Mk].mean(0)
    res[src] = dict(induction_applicable=app[IND].mean().item() if IND else None, induction_rest=rest[IND].mean().item() if IND else None, others_applicable=app[others].mean().item(), others_rest=rest[others].mean().item())
if res.get("e368") and res.get("e369") and res["e368"]["induction_applicable"] is not None:
    res["gradient_to_ablation_ratio_induction"] = res["e368"]["induction_applicable"] / max(abs(res["e369"]["induction_applicable"]), 1e-12)
    res["gradient_to_ablation_ratio_others"] = res["e368"]["others_applicable"] / max(abs(res["e369"]["others_applicable"]), 1e-12)
a, o = res["applicable"], res["other"]
log(f"{tag}: applicable positions ({a['n']}): loss {a['mean_loss']:.2f}, p {a['mean_p']:.2f}, share p>0.9 {a['share_p_above_0p9']:.2f}, mean 1-p {a['mean_one_minus_p']:.2f} | other ({o['n']}): loss {o['mean_loss']:.2f}, p {o['mean_p']:.2f}, share p>0.9 {o['share_p_above_0p9']:.2f}, mean 1-p {o['mean_one_minus_p']:.2f}" + (f" | induction heads at applicable positions: selection (sign-flipped gradient) {res['e368']['induction_applicable']:+.4f} vs ablation effect {res['e369']['induction_applicable']:+.4f}; other heads {res['e368']['others_applicable']:+.4f} vs {res['e369']['others_applicable']:+.4f}" if res.get("e368") and res.get("e369") and res["e368"]["induction_applicable"] is not None else ""))
record(f"e370_saturation_{tag}", res, f"applicable p {a['mean_p']:.2f} (p>0.9 {a['share_p_above_0p9']:.2f}) vs other p {o['mean_p']:.2f}" + (f" | ind heads grad {res['e368']['induction_applicable']:+.4f} abl {res['e369']['induction_applicable']:+.4f}" if res.get("e368") and res.get("e369") and res["e368"]["induction_applicable"] is not None else ""))
