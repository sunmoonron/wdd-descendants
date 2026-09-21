"""e02: innovation (Kalman-style) decomposition. Decompose each block's increment Delta_b = H[b+1]-H[b] over ONLY
block b's atoms (MLP rows + head SVD bases + biases), which HF gives for free via hidden states, no hooks.
Question: how much of the observability ceiling is cross-block cancellation (removed here) vs within-block?
Compares recall of block b's top-1 MLP write from Delta_b (block dictionary, k=8..32) against the global
reading of the same write from the full state H[b+1] with the full dictionary at k=64."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; N = int(os.environ.get("WDD_N", 8192)); K = 32
c = Cache(tag); ids = sub(c.NT, N)
rows = []
for b in range(c.NB):
    D_ = c.s["H"][b + 1][ids].float().to(DEV) - c.s["H"][b][ids].float().to(DEV)
    Ab, labb = c.dictionary(b, blocks=[b], types=(T_MLP, T_ATT, T_BIAS))
    typ = typical_mask(c.X(b, center=False)[ids])
    sel, cof, err = omp(D_, Ab, K); selr, _, errr = omp(D_, rotate(Ab), K)
    led = c.ledger(b, blocks=[b])[b][ids]                       # [N, DFF] signed coefs of block b
    tc, tn = led.abs().max(1); tn = tn.to(DEV); tc = torch.gather(led, 1, tn[:, None].cpu())[:, 0].to(DEV)
    mlp_rows = torch.nonzero(labb["type"] == T_MLP)[:, 0].to(DEV)   # rows in Ab of neurons 0..DFF-1 (in order)
    true_row = mlp_rows[tn]
    rec = recall_curve(sel, true_row, ks=(1, 2, 4, 8, 16, 32))
    hit = sel == true_row[:, None]; idn = hit.any(1) & typ
    chat = (cof * hit).sum(1)[idn]; ct = tc[idn]
    rel = ((chat - ct).abs() / ct.abs()).median().item(); sign = ((chat * ct) > 0).float().mean().item()
    # second/third writes too
    top3 = led.abs().topk(3, 1).indices.to(DEV); rec3 = (sel[:, :, None] == mlp_rows[top3][:, None, :]).any(1).float().mean().item()
    # global reading of the same write: full state after block b, full dictionary, k=64 (cached per level)
    Xg = c.X(b)[ids]; Ag, labg = c.dictionary(b)
    selg, cofg, errg = omp(Xg, Ag, 64)
    true_g = c.atom_index(b, torch.full_like(tn.cpu(), b), tn.cpu()).to(DEV)
    recg = recall_curve(selg, true_g, ks=(32, 64))
    # energy split of the increment
    e_mlp = (led.to(DEV) ** 2).sum(1); e_att = (c.s["ATT"][b][ids].float().to(DEV) ** 2).sum(1); e_d = (D_ ** 2).sum(1)
    r = dict(b=b, atoms=Ab.shape[0], fvu8=fvu(err[:, 7], D_, typ), fvu16=fvu(err[:, 15], D_, typ), fvu32=fvu(err[:, 31], D_, typ),
             rot32=fvu(errr[:, 31], D_, typ), recall_delta=rec, recall_delta_top3any=rec3, sign=sign, med_rel_err=rel,
             recall_global=recg, self_cancel_mlp=(((led.to(DEV) @ (c.wdir_cpu(b).to(DEV) / c.d["WN"][b].to(DEV)[:, None])) ** 2).sum(1)[typ].sum() / e_mlp[typ].sum()).item(),
             att_share=(e_att[typ].sum() / e_d[typ].sum()).item(), support_mlp_frac=(labb["type"].to(DEV)[sel] == T_MLP).float().mean().item())
    rows.append(r); log(f"{tag} b{b}: delta recall@8 {rec[8]:.3f} @32 {rec[32]:.3f} (global@64 {recg[64]:.3f}) fvu32 {r['fvu32']:.3f} rot {r['rot32']:.3f} relerr {rel:.2f} attshare {r['att_share']:.2f}")
record(f"e02_delta_{tag}", dict(model=tag, N=N, rows=rows), " | ".join(f"b{r['b']}: d8 {r['recall_delta'][8]:.2f} d32 {r['recall_delta'][32]:.2f} g64 {r['recall_global'][64]:.2f}" for r in rows))
