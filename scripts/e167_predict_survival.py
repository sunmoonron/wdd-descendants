"""e167: predict the provenance-survival curve (e131) from the law with zero parameters. For the token's top-k true
writes (k=1,3,8) at the mid layer: the fraction whose prominence in the centered STATE exceeds the covariance-matched
64th-order competitor level, and in the centered INCREMENT exceeds the block's own level (16th order), vs the
measured dual-reading recalls at state and increment level (e131). No fitting."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); N = 4096; ids = sub(c.NT, N); X = c.X(L)[ids]; typ = typical_mask(c.X(L, center=False)[ids]); A, lab = c.dictionary(L); g = torch.Generator(device=DEV).manual_seed(0)
def level(Xfit, Ad, kth):
    Sig = (Xfit.T @ Xfit) / Xfit.shape[0]; ev, V = torch.linalg.eigh(Sig); half = V @ torch.diag(ev.clamp_min(0).sqrt()) @ V.T; z = torch.randn(1024, c.D, generator=g, device=DEV) @ half; z = z / z.norm(dim=1, keepdim=True)
    return torch.cat([(z[s:s + 256] @ Ad.T).abs().topk(kth, dim=1).values[:, -1] for s in range(0, 1024, 256)]).mean().item()
tb, tn, tc = c.top_writes(L, 8); tb, tn = tb[ids], tn[ids]; rows8 = torch.stack([c.atom_index(L, tb[:, j], tn[:, j]) for j in range(8)], 1).to(DEV)
lvl_state = level(X[typ], A, 64); prom_state = torch.stack([(X * A[rows8[:, j]]).sum(1).abs() / X.norm(dim=1) for j in range(8)], 1)
pred_state = {k: (prom_state[typ][:, :k] > lvl_state).float().mean().item() for k in (1, 3, 8)}
prom_inc = torch.zeros(N, 8, device=DEV); lvl_inc = {}
for b in range(L + 1):
    D_ = c.s["H"][b + 1][ids].float().to(DEV) - c.s["H"][b][ids].float().to(DEV); Dc = D_ - D_.mean(0); Ab, labb = c.dictionary(b, blocks=[b], types=(T_MLP, T_ATT, T_BIAS)); mlp_rows = torch.nonzero(labb["type"] == T_MLP)[:, 0].to(DEV)
    lvl_inc[b] = level(Dc[typ], Ab, 16)
    for j in range(8):
        m = (tb[:, j] == b).to(DEV); r_b = mlp_rows[tn[:, j].to(DEV)]; prom_inc[m, j] = ((Dc * Ab[r_b]).sum(1).abs() / Dc.norm(dim=1))[m] - lvl_inc[b]   # store margin over the block's level
pred_inc = {k: (prom_inc[typ][:, :k] > 0).float().mean().item() for k in (1, 3, 8)}
meas = {}
try:
    e = json.load(open(os.path.join(RESULTS, f"e131_survival_{tag}.json"))); meas = {k: dict(state=e[f"top{k}"]["state_dual"], increment=e[f"top{k}"]["increment_dual"]) for k in (1, 3, 8)}
except Exception: pass
res = dict(model=tag, L=L, level_state=lvl_state, level_increment_by_block=lvl_inc, predicted=dict(state=pred_state, increment=pred_inc), measured=meas)
record(f"e167_predsurv_{tag}", res, " | ".join(f"top-{k}: state pred {pred_state[k]:.2f}" + (f" meas {meas[k]['state']:.2f}" if meas else "") + f"; increment pred {pred_inc[k]:.2f}" + (f" meas {meas[k]['increment']:.2f}" if meas else "") for k in (1, 3, 8)))
