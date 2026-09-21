"""e131: the provenance survival curve. For the token's top-k true MLP writes (k = 1, 3, 8), the fraction
recovered at three observation levels with the dual-frame projection and with OMP: the WRITE level (each write
observed alone plus the block's other writes? no: the single write vector itself, dictionary of its block ->
trivially 1.0 unless twins), the INCREMENT level (the block's increment over the block's atoms, k=16), and the
STATE level (the full state over the full dictionary, k=64). Also the fraction of the top-8 writes' ENERGY
attributable at each level."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); N = 4096; ids = sub(c.NT, N); X = c.X(L)[ids]; typ = typical_mask(c.X(L, center=False)[ids]); A, lab = c.dictionary(L)
tb, tn, tc = c.top_writes(L, 8); tb, tn, tc = tb[ids], tn[ids], tc[ids].to(DEV); rows8 = torch.stack([c.atom_index(L, tb[:, j], tn[:, j]) for j in range(8)], 1).to(DEV)
S = A.T @ A; ev, V = torch.linalg.eigh(S); Winv = V @ torch.diag(1 / (ev + 1e-2 * ev[-1])) @ V.T
so, _, _ = omp(X, A, 64); sd, _, _ = oneshot(X, A, 64, whiten=Winv)
found_state = {nm: (s[:, :, None] == rows8[:, None, :]).any(1) for nm, s in (("omp", so), ("dual", sd))}                      # [N, 8]
# increment level: for each of the 8 writes, its block's increment over the block's atoms (k=16); reuse per block
found_inc = {nm: torch.zeros(N, 8, dtype=torch.bool, device=DEV) for nm in ("omp", "dual")}; found_write = torch.zeros(N, 8, dtype=torch.bool, device=DEV)
for b in range(L + 1):
    D_ = c.s["H"][b + 1][ids].float().to(DEV) - c.s["H"][b][ids].float().to(DEV); D_ = D_ - D_.mean(0); Ab, labb = c.dictionary(b, blocks=[b], types=(T_MLP, T_ATT, T_BIAS)); mlp_rows = torch.nonzero(labb["type"] == T_MLP)[:, 0].to(DEV)
    Sb = Ab.T @ Ab; evb, Vb = torch.linalg.eigh(Sb); Wb = Vb @ torch.diag(1 / (evb + 1e-2 * evb[-1])) @ Vb.T
    sob, _, _ = omp(D_, Ab, 16); sdb, _, _ = oneshot(D_, Ab, 16, whiten=Wb)
    for j in range(8):
        mb = (tb[:, j] == b).to(DEV); r_b = mlp_rows[tn[:, j].to(DEV)]
        found_inc["omp"][mb, j] = (sob == r_b[:, None]).any(1)[mb]; found_inc["dual"][mb, j] = (sdb == r_b[:, None]).any(1)[mb]
        # write level: the single write c*d over the block dictionary (top-1 correlation) -> identified unless a twin wins
        w = tc[:, j][:, None] * Ab[r_b]; best = (w @ Ab.T).abs().argmax(1); found_write[mb, j] = (best == r_b)[mb]
res = dict(model=tag, L=L, N=N)
for k in (1, 3, 8):
    res[f"top{k}"] = dict(write=found_write[typ][:, :k].float().mean().item(), increment_omp=found_inc["omp"][typ][:, :k].float().mean().item(), increment_dual=found_inc["dual"][typ][:, :k].float().mean().item(), state_omp=found_state["omp"][typ][:, :k].float().mean().item(), state_dual=found_state["dual"][typ][:, :k].float().mean().item())
e = tc ** 2; sh = lambda F: ((e * F.float())[typ].sum() / e[typ].sum()).item()
res["energy_share_of_top8_recovered"] = dict(write=sh(found_write), increment_dual=sh(found_inc["dual"]), state_dual=sh(found_state["dual"]), state_omp=sh(found_state["omp"]))
record(f"e131_survival_{tag}", res, " | ".join(f"top-{k}: write {res[f'top{k}']['write']:.2f} -> increment {res[f'top{k}']['increment_dual']:.2f} (omp {res[f'top{k}']['increment_omp']:.2f}) -> state {res[f'top{k}']['state_dual']:.2f} (omp {res[f'top{k}']['state_omp']:.2f})" for k in (1, 3, 8)) + f" || energy of top-8 recovered: write {res['energy_share_of_top8_recovered']['write']:.2f} increment {res['energy_share_of_top8_recovered']['increment_dual']:.2f} state {res['energy_share_of_top8_recovered']['state_dual']:.2f}")
