"""e102: who are the filler atoms? The most-selected token atoms and the most-selected MLP atoms that are NOT
real writes at the token (spurious), their share of all selections, their embedding norm / training frequency
(token fillers) and their coherence with true writes they replace; then the recall/FVU after removing just the
top-200 filler atoms of each kind (a tiny, data-selected pruning)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); N = int(os.environ.get("WDD_N", 4096)); X = c.X(L); typ = typical_mask(c.X(L, center=False)); A, lab = c.dictionary(L)
sel, cof, err = get_omp(c, L, A=A, X=X); typA, blkA, idxA = lab["type"].to(DEV), lab["block"].to(DEV), lab["index"].to(DEV)
led = c.ledger(L); Cled = torch.cat([led[b] for b in range(L + 1)], 1).to(DEV); thr = 0.05 * Cled.abs().max(1, keepdim=True).values
ismlp = typA[sel] == T_MLP; key = torch.where(ismlp, blkA[sel] * c.DFF + idxA[sel], torch.zeros_like(sel)); real = ismlp & (Cled.gather(1, key).abs() >= thr)
tokat = typA[sel] == T_TOK; spur_mlp = ismlp & ~real
tok_u, tok_c = sel[typ][tokat[typ]].unique(return_counts=True); mlp_u, mlp_c = sel[typ][spur_mlp[typ]].unique(return_counts=True)
top_tok = tok_u[tok_c.topk(min(200, len(tok_u))).indices]; top_mlp = mlp_u[mlp_c.topk(min(200, len(mlp_u))).indices]
cnt = torch.bincount(c.s["cen_ids"].reshape(-1), minlength=c.n_tok_emb).to(DEV); enorm = c.d["emb"][0].norm(dim=1).to(DEV)
tok_share_top200 = tok_c.topk(min(200, len(tok_u))).values.sum().item() / tokat[typ].sum().item(); mlp_share_top200 = mlp_c.topk(min(200, len(mlp_u))).values.sum().item() / spur_mlp[typ].sum().item()
res = dict(model=tag, L=L, token_atoms_share_of_support=tokat[typ].float().mean().item(), top200_token_fillers_share_of_token_selections=tok_share_top200,
           token_fillers=dict(train_count_median=cnt[top_tok].float().median().item(), frac_unseen=(cnt[top_tok] == 0).float().mean().item(), emb_norm_median=enorm[top_tok].median().item(), emb_norm_median_all=enorm.median().item()),
           spurious_mlp_share_of_mlp_selections=(spur_mlp[typ].sum() / ismlp[typ].sum()).item(), top200_mlp_fillers_share_of_spurious=mlp_share_top200, n_distinct_spurious_mlp=len(mlp_u))
# prune the 400 fillers and re-run on a subsample
ids = sub(c.NT, N); keep = torch.ones(A.shape[0], dtype=torch.bool, device=DEV); keep[top_tok] = False; keep[top_mlp] = False
tb, tn, tc = c.top_writes(L, 1); row = c.atom_index(L, tb[ids, 0], tn[ids, 0]).to(DEV); newrow = (torch.cumsum(keep.long(), 0) - 1)[row]; ok = keep[row]
s2, c2, e2 = omp(X[ids], A[keep], 64); t2 = typ[ids]
res["after_pruning_400_fillers"] = dict(recall=((s2 == newrow[:, None]).any(1) & ok)[t2].float().mean().item(), fvu64=fvu(e2[:, 63], X[ids], t2), token_share_of_support=(typA[keep][s2] == T_TOK)[t2].float().mean().item(), true_atoms_removed=(~ok)[t2].float().mean().item())
s0, c0, e0 = omp(X[ids], A, 64); res["before"] = dict(recall=(s0 == row[:, None]).any(1)[t2].float().mean().item(), fvu64=fvu(e0[:, 63], X[ids], t2))
record(f"e102_fillers_{tag}", res, f"token atoms {res['token_atoms_share_of_support']:.2f} of support, top-200 token fillers = {tok_share_top200:.2f} of token selections (train count med {res['token_fillers']['train_count_median']:.0f}, unseen {res['token_fillers']['frac_unseen']:.2f}, emb norm {res['token_fillers']['emb_norm_median']:.2f} vs all {res['token_fillers']['emb_norm_median_all']:.2f}) | spurious MLP = {res['spurious_mlp_share_of_mlp_selections']:.2f} of MLP selections, top-200 = {mlp_share_top200:.2f} of spurious ({res['n_distinct_spurious_mlp']} distinct) | prune 400 fillers: recall {res['before']['recall']:.3f} -> {res['after_pruning_400_fillers']['recall']:.3f} fvu64 {res['before']['fvu64']:.3f} -> {res['after_pruning_400_fillers']['fvu64']:.3f} token share -> {res['after_pruning_400_fillers']['token_share_of_support']:.2f}")
