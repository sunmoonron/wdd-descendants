"""e162: differential WDD. For token ids occurring >= 2 times among typical tokens, pairs of occurrences: the
difference of the two centered states decomposed over the dictionary (k=32) vs the difference of the two ledgers:
recall of the largest differential MLP write (by |c1 - c2|), the share of the differential state's energy that is
token-lookup (block 0) vs later blocks, and the same recall for the plain single-state reading of that write.
Reads context by subtracting the shared token-lookup part."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); X = c.X(L); typ = typical_mask(c.X(L, center=False)); A, lab = c.dictionary(L); tok = c.s["eval_ids"].reshape(-1).to(DEV)
led = c.ledger(L); g = torch.Generator().manual_seed(0); idx_t = torch.nonzero(typ)[:, 0]; u, inv, cnt = tok[idx_t].unique(return_inverse=True, return_counts=True); multi = torch.nonzero(cnt >= 2)[:, 0]
pairs = []
for i in multi.tolist():
    occ = idx_t[inv == i]; perm = torch.randperm(len(occ), generator=g)[:2].to(DEV)
    if len(perm) == 2: pairs.append(occ[perm])
    if len(pairs) >= 3000: break
P = torch.stack(pairs); a, b = P[:, 0], P[:, 1]; Xd = X[a] - X[b]
C = torch.cat([led[bb] for bb in range(L + 1)], 1); Cd = (C[a.cpu()] - C[b.cpu()]).to(DEV); mlp_rows = torch.cat([c.atom_index(L, torch.full((c.DFF,), bb), torch.arange(c.DFF)) for bb in range(L + 1)]).to(DEV)
tdiff = Cd.abs().argmax(1); row_d = mlp_rows[tdiff]; S = A.T @ A; ev, V = torch.linalg.eigh(S); Winv = V @ torch.diag(1 / (ev + 1e-2 * ev[-1])) @ V.T
so, _, _ = omp(Xd, A, 32); sd, _, _ = oneshot(Xd, A, 32, whiten=Winv); s_single, _, _ = omp(X[a], A, 64); sd_single, _, _ = oneshot(X[a], A, 64, whiten=Winv)
promd = (Xd * A[row_d]).sum(1).abs() / Xd.norm(dim=1); prom_single = (X[a] * A[row_d]).sum(1).abs() / X[a].norm(dim=1)
birth = (tdiff // c.DFF); res = dict(model=tag, L=L, n_pairs=len(pairs), recall_diff=dict(omp32=(so == row_d[:, None]).any(1).float().mean().item(), dual32=(sd == row_d[:, None]).any(1).float().mean().item()),
           recall_same_write_in_single_state=dict(omp64=(s_single == row_d[:, None]).any(1).float().mean().item(), dual64=(sd_single == row_d[:, None]).any(1).float().mean().item()),
           prominence_of_diff_write=dict(in_difference=promd.median().item(), in_single_state=prom_single.median().item()), diff_write_birth_block_hist={int(bb): (birth == bb).float().mean().item() for bb in range(L + 1)},
           diff_energy_over_state_energy=((Xd ** 2).sum(1) / (X[a] ** 2).sum(1)).median().item(), block0_share_of_diff_ledger_energy=((Cd[:, :c.DFF] ** 2).sum(1) / (Cd ** 2).sum(1)).median().item())
record(f"e162_diff_{tag}", res, f"pairs {len(pairs)} | largest differential write: recall from the state DIFFERENCE omp {res['recall_diff']['omp32']:.2f} dual {res['recall_diff']['dual32']:.2f} vs from the single state {res['recall_same_write_in_single_state']['omp64']:.2f}/{res['recall_same_write_in_single_state']['dual64']:.2f} | its prominence in the difference {res['prominence_of_diff_write']['in_difference']:.2f} vs in the state {res['prominence_of_diff_write']['in_single_state']:.2f} | diff energy / state energy {res['diff_energy_over_state_energy']:.2f}; block-0 share of the differential ledger {res['block0_share_of_diff_ledger_energy']:.2f}; birth of the diff write " + " ".join(f"b{k}:{v:.2f}" for k, v in res['diff_write_birth_block_hist'].items()))
