"""e170: is GPT-2's within-block unreadability the massive-channel dynamics? Project the top-5 highest-variance
coordinates (the massive channels) out of both the states/increments and the atoms (re-normalized), and repeat:
(a) increment-level dominant-write recall per block (OMP@8, dual@8), (b) state-level recall at the mid layer
(OMP@64, dual@64), against the unprojected numbers. If GPT-2's within-block ceiling is the channel-447 build and
counter-write, its off-channel increment recall should rise toward the gated models'."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); N = 4096; ids = sub(c.NT, N)
Xraw = c.X(L, center=False)[ids]; typ = typical_mask(Xraw); var = Xraw[typ].var(0); top5 = var.topk(5).indices; keep = torch.ones(c.D, dtype=torch.bool, device=DEV); keep[top5] = False
def proj(M): return M[..., keep]
res = dict(model=tag, L=L, channels=top5.tolist(), top5_var_share=(var[top5].sum() / var.sum()).item(), increment={}, state={})
for b in range(L + 1):
    D_ = c.s["H"][b + 1][ids].float().to(DEV) - c.s["H"][b][ids].float().to(DEV); Dc = D_ - D_.mean(0); Ab, labb = c.dictionary(b, blocks=[b], types=(T_MLP, T_ATT, T_BIAS)); mlp_rows = torch.nonzero(labb["type"] == T_MLP)[:, 0].to(DEV)
    led = c.ledger(b, blocks=[b])[b][ids]; tn_ = led.abs().argmax(1).to(DEV); row = mlp_rows[tn_]; out = {}
    for nm, (V_, A_) in (("full", (Dc, Ab)), ("offchannel", (proj(Dc), proj(Ab) / proj(Ab).norm(dim=1, keepdim=True).clamp_min(1e-6)))):
        so, _, _ = omp(V_, A_, 8); S = A_.T @ A_; ev, Vv = torch.linalg.eigh(S); W = Vv @ torch.diag(1 / (ev + 1e-2 * ev[-1])) @ Vv.T; sd, _, _ = oneshot(V_, A_, 8, whiten=W)
        out[nm] = dict(omp=(so == row[:, None]).any(1)[typ].float().mean().item(), dual=(sd == row[:, None]).any(1)[typ].float().mean().item())
    res["increment"][b] = out; log(f"{tag} b{b}: increment recall omp {out['full']['omp']:.2f} -> off-channel {out['offchannel']['omp']:.2f} | dual {out['full']['dual']:.2f} -> {out['offchannel']['dual']:.2f}")
X = c.X(L)[ids]; A, lab = c.dictionary(L); tb, tn, tc = c.top_writes(L, 1); row = c.atom_index(L, tb[ids, 0], tn[ids, 0]).to(DEV)
for nm, (V_, A_) in (("full", (X, A)), ("offchannel", (proj(X), proj(A) / proj(A).norm(dim=1, keepdim=True).clamp_min(1e-6)))):
    so, _, eo = omp(V_, A_, 64); S = A_.T @ A_; ev, Vv = torch.linalg.eigh(S); W = Vv @ torch.diag(1 / (ev + 1e-2 * ev[-1])) @ Vv.T; sd, _, _ = oneshot(V_, A_, 64, whiten=W)
    res["state"][nm] = dict(omp=(so == row[:, None]).any(1)[typ].float().mean().item(), dual=(sd == row[:, None]).any(1)[typ].float().mean().item(), fvu64=fvu(eo[:, 63], V_, typ))
inc_full = float(np.mean([v["full"]["omp"] for v in res["increment"].values()])); inc_off = float(np.mean([v["offchannel"]["omp"] for v in res["increment"].values()]))
record(f"e170_offchannel_{tag}", res, f"top-5 channels carry {res['top5_var_share']:.2f} of typical variance | increment recall (omp@8, mean over blocks) {inc_full:.2f} -> off-channel {inc_off:.2f} | state recall omp {res['state']['full']['omp']:.2f} -> {res['state']['offchannel']['omp']:.2f}, dual {res['state']['full']['dual']:.2f} -> {res['state']['offchannel']['dual']:.2f}, fvu64 {res['state']['full']['fvu64']:.3f} -> {res['state']['offchannel']['fvu64']:.3f}")
