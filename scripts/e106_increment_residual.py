"""e106: what delta-WDD leaves in the increment. Per block, after OMP with k=8 over the block's atoms on the
(centered) increment: the residual's energy share, its split into the block's attention write vs MLP write
(least squares on the two exact components), and the residual FVU of the MLP-only increment."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); N = 4096; ids = sub(c.NT, N); rows = []
for b in range(c.NB):
    att = c.s["ATT"][b][ids].float().to(DEV); mlp = c.acts[b][ids].float().to(DEV) @ c.wdir_cpu(b).to(DEV)
    if c.d["mlp_bias"][b] is not None: mlp = mlp + c.d["mlp_bias"][b].to(DEV)
    D_ = att + mlp; Dc = D_ - D_.mean(0); typ = typical_mask(c.X(b, center=False)[ids]); Ab, _ = c.dictionary(b, blocks=[b], types=(T_MLP, T_ATT, T_BIAS))
    sel, cof, err = omp(Dc, Ab, 8); R = Dc - torch.einsum("nk,nkd->nd", cof, Ab[sel]); V = torch.stack([att - att.mean(0), mlp - mlp.mean(0)], 1)
    G = V @ V.transpose(1, 2) + 1e-3 * torch.eye(2, device=DEV); beta = torch.linalg.solve(G, V @ R[:, :, None])[:, :, 0]; e_r = (R ** 2).sum(1)
    share_att = ((beta[:, 0:1] * V[:, 0] * R).sum(1) / e_r.clamp_min(1e-6))[typ].mean().item(); share_mlp = ((beta[:, 1:2] * V[:, 1] * R).sum(1) / e_r.clamp_min(1e-6))[typ].mean().item()
    mc = mlp - mlp.mean(0); _, _, em = omp(mc, Ab, 8)
    rows.append(dict(b=b, residual_fvu8=fvu(err[:, 7], Dc, typ), residual_share_att=share_att, residual_share_mlp=share_mlp, att_share_of_increment=((att - att.mean(0))[typ] ** 2).sum().item() / (Dc[typ] ** 2).sum().item(), mlp_only_fvu8=fvu(em[:, 7], mc, typ)))
    log(f"{tag} b{b}: increment fvu8 {rows[-1]['residual_fvu8']:.2f} | residual is attention {share_att:.2f} mlp {share_mlp:.2f} (attention share of increment {rows[-1]['att_share_of_increment']:.2f}) | mlp-only increment fvu8 {rows[-1]['mlp_only_fvu8']:.2f}")
record(f"e106_incres_{tag}", dict(model=tag, rows=rows), " | ".join(f"b{r['b']}: fvu8 {r['residual_fvu8']:.2f} res-att {r['residual_share_att']:.2f} attshare {r['att_share_of_increment']:.2f} mlponly {r['mlp_only_fvu8']:.2f}" for r in rows))
