"""e9: predicted token against internal change (16 sequences, 8 loops). Per token and successive loops: does the top-1
prediction change, how far the next-token distribution moves (KL), how much the WDD code (16 atoms, before layer 12)
and the exit state change. Among tokens whose top-1 prediction stays the same: how much internal change remains, and
does it predict that the prediction changes in a later loop, or that the loss still improves."""
import sys; sys.path.insert(0, "/data/loopwdd/code")
from lw import *
m = LM("ouro"); C = torch.load(f"{ROOT}/cache/ouro_e1.pt"); K2 = torch.load(f"{ROOT}/cache/ouro_e2.pt")
NS, TLX, K = 16, 8, 16; ids = C["ids"][:NS]; T = ids.shape[1]
H = torch.empty(TLX, NS, T, m.d)
with torch.no_grad():
    for b in range(0, NS, 4):
        for t, h in enumerate(m.run(ids[b:b + 4].to(DEVM), loops=TLX)): H[t, b:b + 4] = h.cpu()
am = torch.empty(TLX, NS, T - 1, dtype=torch.long); lp = torch.empty(TLX, NS, T - 1); kl = torch.empty(TLX - 1, NS, T - 1)
with torch.no_grad():
    for s in range(NS):
        prev = None
        for t in range(TLX):
            ls = torch.log_softmax(H[t, s, :-1].to(DEVM) @ m.H.T, -1)
            am[t, s] = ls.argmax(-1).cpu(); lp[t, s] = ls.gather(1, ids[s, 1:, None].to(DEVM))[:, 0].cpu()
            if prev is not None: kl[t - 1, s] = (prev.exp() * (prev - ls)).sum(-1).cpu()
            prev = ls
def auc(score, y):
    y = y.bool(); n1, n0 = y.sum().item(), (~y).sum().item()
    if n1 == 0 or n0 == 0: return float("nan")
    r = score.argsort().argsort().float() + 1
    return ((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)).item()
cods = K2["codes"]; res = {}
for t in range(TLX - 1):
    a, b = cods[t]["sel"][:NS * (T - 1), :K].long(), cods[t + 1]["sel"][:NS * (T - 1), :K].long()
    inter = (a[:, :, None] == b[:, None, :]).any(-1).sum(-1).float(); cc = (1 - inter / (2 * K - inter)).view(NS, T - 1)
    dh = C["dh"][t + 1][:NS, :-1]                                           # exit-state change, loop t+1 vs t+2 (1-based)
    same = am[t] == am[t + 1]; gain = lp[t + 1] - lp[t]
    later = torch.zeros_like(same)
    for u in range(t + 2, min(TLX, 4)): later |= am[u] != am[t + 1]        # changes again before the trained exit
    sm = same[:, 1:].flatten(); ccf, dhf, klf, gf, lf = cc[:, :-1].flatten(), dh[:, 1:].flatten(), kl[t][:, 1:].flatten(), gain[:, 1:].flatten(), later[:, 1:].flatten()
    row = dict(top1_same=sm.float().mean().item(), kl_median=klf.median().item(),
               silent=dict(code_change=ccf[sm].median().item(), state_change=dhf[sm].median().item(), kl=klf[sm].median().item(),
                           loss_improves=(gf[sm] > 0.05).float().mean().item(), loss_worsens=(gf[sm] < -0.05).float().mean().item()),
               changed=dict(code_change=ccf[~sm].median().item(), state_change=dhf[~sm].median().item()))
    if t + 2 < 4:
        row["silent_predicts_later_change"] = dict(base_rate=lf[sm].float().mean().item(), auc_code_change=auc(ccf[sm], lf[sm]),
                                                   auc_state_change=auc(dhf[sm], lf[sm]), auc_kl=auc(klf[sm], lf[sm]))
    res[f"{t+1}->{t+2}"] = row
    log(f"loops {t+1}->{t+2}: top-1 same {row['top1_same']:.2f}; among same: code change {row['silent']['code_change']:.2f} (changed: {row['changed']['code_change']:.2f}), "
        f"state change {row['silent']['state_change']:.2f}, loss improves >0.05 for {row['silent']['loss_improves']:.2f}, worsens for {row['silent']['loss_worsens']:.2f}"
        + (f"; later change AUC code {row['silent_predicts_later_change']['auc_code_change']:.2f} state {row['silent_predicts_later_change']['auc_state_change']:.2f} KL {row['silent_predicts_later_change']['auc_kl']:.2f}" if "silent_predicts_later_change" in row else ""))
jdump(res, f"{ROOT}/results/e9_ouro.json")
