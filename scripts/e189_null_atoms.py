"""e189: known-null atoms as a clean false-positive control, and support repair. Ablate the MLP of block a
(a = 1 and a = L-2) in the forward pass; at level L the block-a atoms have exactly zero true coefficient. Reports
(i) the share of OMP@64 / dual@64 support falling on block-a atoms in the ablated run (false positives with a
known-zero ledger) vs the clean run's share and the dictionary share; (ii) the state change and the per-token
support Jaccard clean vs ablated; (iii) where the replacement atoms come from (block histogram)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV)
A, lab = c.dictionary(L); mu = c.s["mu"][L + 1].to(DEV); blkA = lab["block"].to(DEV); S = A.T @ A; ev, V = torch.linalg.eigh(S); Winv = V @ torch.diag(1 / (ev + 1e-2 * ev[-1])) @ V.T
def run(ablate_block=None):
    store = {}; hs = [arch.layers[L].register_forward_hook(lambda m, i, o: store.__setitem__("H", (o[0] if isinstance(o, tuple) else o).detach().float().reshape(-1, c.D)))]
    if ablate_block is not None: hs.append(arch.mlp_lin(ablate_block).register_forward_pre_hook(lambda m, inp: (torch.zeros_like(inp[0]),)))
    model(ids_seq); [h.remove() for h in hs]; return store["H"]
H0 = run(); typ = typical_mask(H0); X0 = H0 - mu; so0, _, _ = omp(X0, A, 64); sd0, _, _ = oneshot(X0, A, 64, whiten=Winv); rows = []
for a in sorted({1, L - 2}):
    H1 = run(a); X1 = H1 - mu; so1, _, _ = omp(X1, A, 64); sd1, _, _ = oneshot(X1, A, 64, whiten=Winv); dict_share = (blkA == a).float().mean().item(); rel = ((H1 - H0).norm(dim=1) / H0.norm(dim=1))[typ].median().item()
    def share(sel): return (blkA[sel] == a).float().mean(1)[typ].mean().item()
    def jacc(s0, s1):
        j = []
        for i in torch.nonzero(typ)[:, 0][:1500]: a0, a1 = set(s0[i].tolist()), set(s1[i].tolist()); j.append(len(a0 & a1) / len(a0 | a1))
        return float(sum(j) / len(j))
    # replacement atoms: atoms in the ablated support not in the clean support, by block
    newb = blkA[so1][~torch.isin(so1, so0[:, :1]) & ~(so1[..., None] == so0[:, None, :]).any(-1)] if False else blkA[so1[~(so1[..., None] == so0[:, None, :]).any(-1)]]
    hist = {int(b): round((newb == b).float().mean().item(), 3) for b in range(-1, L + 1) if (newb == b).any()}
    r = dict(ablated=a, dict_share=dict_share, state_change=rel, omp_share_clean=share(so0), omp_share_abl=share(so1), dual_share_clean=share(sd0), dual_share_abl=share(sd1), jaccard_omp=jacc(so0, so1), jaccard_dual=jacc(sd0, sd1), replacement_block_hist=hist); rows.append(r)
    log(f"{tag} ablate MLP b{a} (dict share {dict_share:.3f}, state change {rel:.2f}): block-a share of support clean->ablated: omp {r['omp_share_clean']:.3f}->{r['omp_share_abl']:.3f} dual {r['dual_share_clean']:.3f}->{r['dual_share_abl']:.3f} | support Jaccard omp {r['jaccard_omp']:.2f} dual {r['jaccard_dual']:.2f} | replacement atoms by block: " + " ".join(f"b{b}:{v:.2f}" for b, v in sorted(hist.items(), key=lambda kv: -kv[1])[:5]))
record(f"e189_nullatoms_{tag}", dict(model=tag, L=L, rows=rows), " | ".join(f"ablate b{r['ablated']}: FP share of known-null atoms omp {r['omp_share_abl']:.3f} (clean {r['omp_share_clean']:.3f}, dict {r['dict_share']:.3f}) dual {r['dual_share_abl']:.3f} (clean {r['dual_share_clean']:.3f}); state change {r['state_change']:.2f}; support Jaccard omp {r['jaccard_omp']:.2f} dual {r['jaccard_dual']:.2f}" for r in rows))
