"""e54: how much of attention is DC? Per head (blocks 0..L): the energy share of the head's mean write vector over
typical tokens (the constant part that centering absorbs), the share of the variable part, and what fraction of the
CENTERED state's energy is attention after removing each head's mean. Also the cosine between each head's mean
write and the level mean mu."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); N = int(os.environ.get("WDD_N", 8192)); ids = sub(c.NT, N)
Xraw = c.X(L, center=False)[ids]; typ = typical_mask(Xraw); mu = c.s["mu"][L + 1].to(DEV); X = Xraw - mu
rows = []; att_var = torch.zeros_like(X); att_dc = torch.zeros_like(X)
for b in range(L + 1):
    HI = c.s["HI"][b][ids].float().to(DEV); WO = c.d["WO"][b].to(DEV)
    for h in range(c.NH):
        w = HI[:, h * c.HD:(h + 1) * c.HD] @ WO[h * c.HD:(h + 1) * c.HD]; m = w[typ].mean(0); e = (w[typ] ** 2).sum(1).mean()
        dc = (m ** 2).sum() / e; att_var += w - m; att_dc += m
        rows.append(dict(b=b, h=h, dc_share=dc.item(), mean_norm=m.norm().item(), rms_norm=e.sqrt().item(), cos_mean_mu=((m @ mu) / (m.norm() * mu.norm())).item()))
dc = torch.tensor([r["dc_share"] for r in rows]); per_block = {b: float(dc[[r["b"] == b for r in rows]].median()) for b in range(L + 1)}
mlp = sum((c.acts[b][ids].float().to(DEV) @ c.wdir_cpu(b).to(DEV)) for b in range(L + 1))
res = dict(model=tag, L=L, dc_share_median=dc.median().item(), dc_share_q90=dc.quantile(0.9).item(), frac_heads_dc_above_half=(dc > 0.5).float().mean().item(), per_block_median=per_block,
           energy_shares_vs_centered_state=dict(att_variable=((att_var[typ] ** 2).sum() / (X[typ] ** 2).sum()).item(), att_dc_total=((att_dc ** 2).sum() / (X[typ] ** 2).sum(1).mean()).item(),
                                                mlp_variable=(((mlp - mlp[typ].mean(0))[typ] ** 2).sum() / (X[typ] ** 2).sum()).item(), cos_attdc_mu=((att_dc[0] @ mu) / (att_dc[0].norm() * mu.norm())).item()),
           top_dc_heads=sorted(rows, key=lambda r: -r["dc_share"])[:5])
record(f"e54_attdc_{tag}", res, f"head DC share median {res['dc_share_median']:.2f} q90 {res['dc_share_q90']:.2f} heads>0.5: {res['frac_heads_dc_above_half']:.2f} | per block " + " ".join(f"b{b}:{v:.2f}" for b, v in per_block.items()) + f" | vs centered state: att-variable {res['energy_shares_vs_centered_state']['att_variable']:.2f} mlp-variable {res['energy_shares_vs_centered_state']['mlp_variable']:.2f} att-DC/state {res['energy_shares_vs_centered_state']['att_dc_total']:.2f} cos(attDC, mu) {res['energy_shares_vs_centered_state']['cos_attdc_mu']:.2f}")
