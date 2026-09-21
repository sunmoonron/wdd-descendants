"""e168: peel the early increments (hidden states only): subtract H[b0+1] - H[0] (blocks 0..b0 exactly, including
their attention) from the level-L state, and read the remainder over the atoms of blocks b0+1..L. Target: the
largest true write among blocks b0+1..L. Recall with/without peeling (OMP@64 and dual), for b0 = 0 and 1; plus
what fraction of the peeled remainder's energy the later-block dictionary explains."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); N = 4096; ids = sub(c.NT, N); Xraw = c.X(L, center=False)[ids]; typ = typical_mask(Xraw); res = dict(model=tag, L=L, by_b0={})
led = c.ledger(L)
for b0 in (0, 1):
    blocks = list(range(b0 + 1, L + 1)); A, lab = c.dictionary(L, blocks=blocks); Cl = torch.cat([led[b][ids] for b in blocks], 1).to(DEV); t = Cl.abs().argmax(1); tbk = torch.tensor(blocks, device=DEV)[t // c.DFF]; tnk = t % c.DFF
    # rows in the restricted dictionary: embeddings first (blk<0) then blocks in order; build a row map for MLP atoms
    blkA, typA, idxA = lab["block"].to(DEV), lab["type"].to(DEV), lab["index"].to(DEV); lut = torch.full((c.NB, c.DFF), -1, dtype=torch.long, device=DEV); rows_m = torch.nonzero(typA == T_MLP)[:, 0]; lut[blkA[rows_m], idxA[rows_m]] = rows_m; row = lut[tbk, tnk]
    peeled = Xraw - c.s["H"][b0 + 1][ids].float().to(DEV); Xp = peeled - peeled.mean(0); Xs = Xraw - Xraw.mean(0)
    S = A.T @ A; ev, V = torch.linalg.eigh(S); Winv = V @ torch.diag(1 / (ev + 1e-2 * ev[-1])) @ V.T; out = {}
    for nm, V_ in (("state", Xs), ("peeled", Xp)):
        so, _, eo = omp(V_, A, 64); sd, _, _ = oneshot(V_, A, 64, whiten=Winv); out[nm] = dict(recall_omp=(so == row[:, None]).any(1)[typ].float().mean().item(), recall_dual=(sd == row[:, None]).any(1)[typ].float().mean().item(), fvu64=fvu(eo[:, 63], V_, typ), prom_med=((V_ * A[row]).sum(1).abs() / V_.norm(dim=1))[typ].median().item())
    res["by_b0"][b0] = dict(target_blocks=f"{b0 + 1}..{L}", peeled_energy_over_state=((peeled[typ] ** 2).sum() / (Xraw[typ] ** 2).sum()).item(), **{f"{k}_{m}": v for k, d in out.items() for m, v in d.items()})
    log(f"{tag} b0={b0}: largest write of blocks {b0 + 1}..{L}: recall state omp {out['state']['recall_omp']:.2f} dual {out['state']['recall_dual']:.2f} -> peeled omp {out['peeled']['recall_omp']:.2f} dual {out['peeled']['recall_dual']:.2f} | prominence {out['state']['prom_med']:.2f} -> {out['peeled']['prom_med']:.2f} | fvu {out['state']['fvu64']:.3f} -> {out['peeled']['fvu64']:.3f}")
record(f"e168_peel_{tag}", res, " | ".join(f"b0={b0}: omp {v['state_recall_omp']:.2f}->{v['peeled_recall_omp']:.2f} dual {v['state_recall_dual']:.2f}->{v['peeled_recall_dual']:.2f} prom {v['state_prom_med']:.2f}->{v['peeled_prom_med']:.2f}" for b0, v in res["by_b0"].items()))
