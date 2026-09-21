"""e117: block 0 as extended embedding. Per token: cos(block-0 MLP write, token embedding), the fraction of the
block-0 MLP write's energy along the token embedding direction, and how predictable the block-0 write is from the
token id alone (variance explained by the per-token mean over the centering slice's... here: over the eval set
split by halves). Relates to the growth of token-embedding survival and the block-0 atom over-selection."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); ids = sub(c.NT, 8192); tok = c.s["eval_ids"].reshape(-1)[ids]; E = c.d["emb"][0].to(DEV)[tok.to(DEV)]; e = E / E.norm(dim=1, keepdim=True)
w0 = c.acts[0][ids].float().to(DEV) @ c.wdir_cpu(0).to(DEV); att0 = c.s["ATT"][0][ids].float().to(DEV)
cosw = (w0 * e).sum(1) / w0.norm(dim=1).clamp_min(1e-6); along = ((w0 * e).sum(1) ** 2) / (w0 ** 2).sum(1).clamp_min(1e-6)
# predictability from the token id: split tokens into halves by sequence; per-token-id mean write from half A predicts half B
seq = (torch.arange(c.NT)[ids] // CTX).to(DEV); ha, hb = seq % 2 == 0, seq % 2 == 1; tokd = tok.to(DEV)
uk, inv = tokd.unique(return_inverse=True); sums = torch.zeros(len(uk), c.D, device=DEV).index_add_(0, inv[ha], w0[ha]); cnts = torch.zeros(len(uk), device=DEV).index_add_(0, inv[ha], torch.ones(int(ha.sum()), device=DEV))
seen = cnts[inv] > 0; pred = sums[inv] / cnts[inv].clamp_min(1)[:, None]; m = hb & seen
r2 = 1 - ((w0[m] - pred[m]) ** 2).sum() / ((w0[m] - w0[hb].mean(0)) ** 2).sum()
res = dict(model=tag, cos_mlp0_token_med=cosw.median().item(), frac_energy_along_token_med=along.median().item(), frac_positive_cos=(cosw > 0).float().mean().item(), mlp0_norm_over_emb_norm_med=(w0.norm(dim=1) / E.norm(dim=1)).median().item(),
           att0_norm_over_emb_norm_med=(att0.norm(dim=1) / E.norm(dim=1)).median().item(), r2_mlp0_from_token_id=r2.item(), frac_tokens_seen=m.float().sum().item() / hb.float().sum().item())
record(f"e117_extemb_{tag}", res, f"cos(MLP0 write, token emb) med {res['cos_mlp0_token_med']:+.2f} (positive {res['frac_positive_cos']:.2f}) energy along token {res['frac_energy_along_token_med']:.2f} | |MLP0|/|emb| {res['mlp0_norm_over_emb_norm_med']:.2f} |attn0|/|emb| {res['att0_norm_over_emb_norm_med']:.2f} | R2 of MLP0 write from token id alone {res['r2_mlp0_from_token_id']:.2f}")
