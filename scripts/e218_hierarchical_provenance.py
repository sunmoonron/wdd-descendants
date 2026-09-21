"""e218: at which granularity is provenance reliably observable? For dominant MLP writes at level L: neuron-level
identification (the writer's atom in the support), block-level (a substantial atom, |coef| >= 20% of the token's
largest, from the birth block) vs a control block, the label granularity of the support atom closest in
direction to the write, and per-token block-energy recovery (Spearman across blocks between the support's energy
per block and the ledger's energy per block, vs shuffled block labels). For attention: head-level identification
of the block-L head with the largest write (an OV atom of that head in the support) vs a control head. OMP@64 and
dual@64. Kill rule: coarser labels help only if block-level recall exceeds its control by >= 0.2."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); N = 6144; ids = sub(c.NT, N); Xraw = c.X(L, center=False)[ids]; typ = typical_mask(Xraw); mu = c.s["mu"][L + 1].to(DEV); X = Xraw - mu; A, lab = c.dictionary(L); typA, blkA, idxA = lab["type"].to(DEV), lab["block"].to(DEV), lab["index"].to(DEV)
tb, tn, tc = c.top_writes(L, 1); tb, tn = tb[ids, 0], tn[ids, 0]; row = c.atom_index(L, tb, tn).to(DEV); tbd = tb.to(DEV); d = A[row]
led = c.ledger(L); E_led = torch.stack([(led[b][ids] ** 2).sum(1) for b in range(L + 1)], 1).to(DEV)                              # [N, L+1] ledger energy per block
sel_o, cof_o, _ = get_omp(c, L); sel_o, cof_o = sel_o[ids], cof_o[ids]; S = A.T @ A; ev, V = torch.linalg.eigh(S); Winv = V @ torch.diag(1 / (ev + 1e-2 * ev[-1])) @ V.T; sel_d, cof_d, _ = oneshot(X, A, 64, whiten=Winv)
# attention ground truth at block L: head with the largest write per token
HW = torch.stack([c.head_write(L, h, tokens=ids).norm(dim=1) for h in range(c.NH)], 1); top_head = HW.argmax(1).to(DEV); att_big = (HW.max(1).values > HW.sum(1) * 0.3).to(DEV)
def spearman_rows(a, b):
    ra = a.argsort(1).argsort(1).float(); rb = b.argsort(1).argsort(1).float(); ra = ra - ra.mean(1, keepdim=True); rb = rb - rb.mean(1, keepdim=True); return (ra * rb).sum(1) / (ra.norm(dim=1) * rb.norm(dim=1)).clamp_min(1e-9)
torch.manual_seed(0); out = {}
for nm, sel, cof in (("omp", sel_o, cof_o), ("dual", sel_d, cof_d)):
    big = cof.abs() >= 0.2 * cof.abs().max(1, keepdim=True).values; ismlp = typA[sel] == T_MLP; isatt = typA[sel] == T_ATT
    neuron = (sel == row[:, None]).any(1); block = (ismlp & big & (blkA[sel] == tbd[:, None])).any(1); ctrl_b = (tbd + torch.randint(1, L + 1, tbd.shape, device=DEV)) % (L + 1); block_ctrl = (ismlp & big & (blkA[sel] == ctrl_b[:, None])).any(1)
    cosv = torch.einsum("nkd,nd->nk", A[sel], d).abs(); best = cosv.argmax(1); bsel = sel.gather(1, best[:, None])[:, 0]; exact = bsel == row; sameblock = ~exact & (typA[bsel] == T_MLP) & (blkA[bsel] == tbd); other = ~exact & ~sameblock
    E_sup = torch.zeros(N, L + 1, device=DEV); m = ismlp; E_sup.scatter_add_(1, torch.where(m, blkA[sel], torch.zeros_like(sel)), (cof ** 2) * m.float())
    rho = spearman_rows(E_sup, E_led); perm = torch.randperm(L + 1, device=DEV); rho_ctrl = spearman_rows(E_sup[:, perm], E_led)
    head = (isatt & big & (blkA[sel] == L) & (idxA[sel] == top_head[:, None])).any(1); ctrl_h = (top_head + torch.randint(1, c.NH, top_head.shape, device=DEV)) % c.NH; head_ctrl = (isatt & big & (blkA[sel] == L) & (idxA[sel] == ctrl_h[:, None])).any(1)
    rec = dict(neuron_recall=neuron[typ].float().mean().item(), block_recall=block[typ].float().mean().item(), block_control=block_ctrl[typ].float().mean().item(), best_atom_exact=exact[typ].float().mean().item(), best_atom_same_block=sameblock[typ].float().mean().item(), best_atom_other=other[typ].float().mean().item(), best_atom_cos=cosv.max(1).values[typ].median().item(), block_energy_rho=rho[typ].median().item(), block_energy_rho_control=rho_ctrl[typ].median().item(), head_recall=head[typ & att_big].float().mean().item(), head_control=head_ctrl[typ & att_big].float().mean().item(), n_att=int((typ & att_big).sum()))
    out[nm] = rec
    log(f"{tag} {nm}: neuron-level recall {rec['neuron_recall']:.2f} | block-level {rec['block_recall']:.2f} vs control block {rec['block_control']:.2f} | closest support atom: exact {rec['best_atom_exact']:.2f}, same block {rec['best_atom_same_block']:.2f}, other {rec['best_atom_other']:.2f} (median cos {rec['best_atom_cos']:.2f}) | block-energy Spearman {rec['block_energy_rho']:.2f} vs shuffled {rec['block_energy_rho_control']:.2f} | head-level (block {L} top head, n {rec['n_att']}): {rec['head_recall']:.2f} vs control head {rec['head_control']:.2f}")
record(f"e218_hier_{tag}", dict(model=tag, L=L, results=out), " | ".join(f"{nm}: neuron {r['neuron_recall']:.2f}, block {r['block_recall']:.2f} (control {r['block_control']:.2f}), closest-atom exact/same-block/other {r['best_atom_exact']:.2f}/{r['best_atom_same_block']:.2f}/{r['best_atom_other']:.2f}, block-energy rho {r['block_energy_rho']:.2f} (shuffled {r['block_energy_rho_control']:.2f}), head {r['head_recall']:.2f} (control {r['head_control']:.2f})" for nm, r in out.items()))
