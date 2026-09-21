"""e13: attention. The paper scored only MLP writes. Here the exact per-head writes (head output slice times its
rows of W_O) give a ground truth for attention: is the dominant head's subspace selected by OMP, how much of its
write energy does the selected sub-basis capture, which SVD ranks get selected, and does attention energy in the
state get attributed to attention atoms at all?"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); N = int(os.environ.get("WDD_N", 8192)); ids = sub(c.NT, N)
X = c.X(L)[ids]; typ = typical_mask(c.X(L, center=False)[ids]); A, lab = c.dictionary(L)
sel, cof, err = get_omp(c, L, A=A, X=c.X(L)); sel, cof = sel[ids.to(DEV)], cof[ids.to(DEV)]
typA, blkA, idxA, rkA = (lab[k].to(DEV) for k in ("type", "block", "index", "rank"))
# per-head write norms [N, L+1, NH] and the dominant head per token
HW = torch.zeros(N, L + 1, c.NH, c.D, device=DEV)
for b in range(L + 1):
    HI = c.s["HI"][b][ids].float().to(DEV); WO = c.d["WO"][b].to(DEV)
    for h in range(c.NH): HW[:, b, h] = HI[:, h * c.HD:(h + 1) * c.HD] @ WO[h * c.HD:(h + 1) * c.HD]
hn = HW.norm(dim=3); flat = hn.view(N, -1); dom = flat.argmax(1); db, dh = dom // c.NH, dom % c.NH
# selected attention atoms per token: (block, head, rank)
isatt = typA[sel] == T_ATT; sb, sh, sr = blkA[sel], idxA[sel], rkA[sel]
head_hit = (isatt & (sb == db[:, None]) & (sh == dh[:, None])).any(1)
n_atoms_dom = (isatt & (sb == db[:, None]) & (sh == dh[:, None])).sum(1).float()
# energy of the dominant head's write captured by the selected atoms of that head (projection onto their span; orthonormal rows)
capt = torch.zeros(N, device=DEV); w = HW[torch.arange(N), db, dh]
for i in range(N):
    m = isatt[i] & (sb[i] == db[i]) & (sh[i] == dh[i])
    if m.any(): capt[i] = ((A[sel[i][m]] @ w[i]) ** 2).sum() / (w[i] ** 2).sum()
# energy of the dominant head write that lies in its own top-r SVD directions (r = 1, 4, 16): how low-rank are head writes?
res = dict(model=tag, L=L, N=N)
rows_att = torch.nonzero(typA == T_ATT)[:, 0]
def head_basis(b, h): return A[rows_att[(blkA[rows_att] == b) & (idxA[rows_att] == h)]]        # [HD, D] in rank order
lowrank = {r: [] for r in (1, 4, 16)}
for b in range(L + 1):
    for h in range(c.NH):
        m = (db == b) & (dh == h)
        if m.sum() < 5: continue
        B = head_basis(b, h); p = (HW[m, b, h] @ B.T) ** 2; tot = p.sum(1)
        for r in lowrank: lowrank[r].append((p[:, :r].sum(1) / tot).mean().item())
res["dominant_head"] = dict(selected_frac=head_hit[typ].float().mean().item(), atoms_when_selected_mean=n_atoms_dom[typ & head_hit].mean().item() if (typ & head_hit).any() else 0,
                            energy_captured_median=capt[typ & head_hit].median().item() if (typ & head_hit).any() else 0,
                            dom_head_share_of_att_energy=(hn.view(N, -1).max(1).values ** 2 / (hn ** 2).sum((1, 2)))[typ].median().item(),
                            rank_of_selected_atoms_median=sr[isatt & (sb == db[:, None]) & (sh == dh[:, None])].float().median().item() if head_hit.any() else None,
                            head_write_energy_in_top_r_svd={r: float(np.mean(v)) for r, v in lowrank.items()})
s1, _, _ = oneshot(X, A, 64); isatt1 = typA[s1] == T_ATT
res["dominant_head"]["selected_frac_oneshot"] = ((isatt1 & (blkA[s1] == db[:, None]) & (idxA[s1] == dh[:, None])).any(1))[typ].float().mean().item()
# attribution of energy: attention atoms' share of the reconstruction vs attention's true share of the state
rec_att = torch.einsum("nk,nkd->nd", cof * isatt, A[sel]); rec_all = torch.einsum("nk,nkd->nd", cof, A[sel])
true_att = sum(c.s["ATT"][b][ids].float().to(DEV) for b in range(L + 1)); mu = c.s["mu"][L + 1].to(DEV)
res["energy"] = dict(att_atoms_share_of_support=isatt[typ].float().mean().item(), att_atoms_share_of_recon_energy=((rec_att ** 2).sum(1)[typ].sum() / (rec_all ** 2).sum(1)[typ].sum()).item(),
                     true_att_share_of_state=((true_att ** 2).sum(1)[typ].sum() / (X ** 2).sum(1)[typ].sum()).item(),
                     cos_recatt_trueatt_median=((rec_att * true_att).sum(1) / (rec_att.norm(dim=1) * true_att.norm(dim=1)).clamp_min(1e-6))[typ].median().item())
record(f"e13_heads_{tag}", res, f"dominant head selected {res['dominant_head']['selected_frac']:.3f} (oneshot {res['dominant_head']['selected_frac_oneshot']:.3f}), atoms {res['dominant_head']['atoms_when_selected_mean']:.1f}, energy captured {res['dominant_head']['energy_captured_median']:.2f}, head-write energy in top-1/4/16 svd {res['dominant_head']['head_write_energy_in_top_r_svd'][1]:.2f}/{res['dominant_head']['head_write_energy_in_top_r_svd'][4]:.2f}/{res['dominant_head']['head_write_energy_in_top_r_svd'][16]:.2f} | att atoms {res['energy']['att_atoms_share_of_support']:.2f} of support, {res['energy']['att_atoms_share_of_recon_energy']:.2f} of recon energy vs true att share {res['energy']['true_att_share_of_state']:.2f}, cos(rec_att, true_att) {res['energy']['cos_recatt_trueatt_median']:.2f}")
