"""e03: the dictionary as an object (data-free), and the one scalar that should explain the rotation result.
(a) coherence profile (Donoho-Huo 2001, Tropp 2004): per-atom max |cos| and Babel function mu1(k); by atom type.
(b) frame operator S = A^T A (Duffin-Schaeffer 1952): spectrum, frame-bound ratio, and its alignment with the
    state covariance Sigma: cosF(S, Sigma) for weight / rotated / random dictionaries, plus principal-subspace overlap.
(c) matched-filter statistic: E_x max_i |<x, a_i>| / ||x|| (North 1943) for the three dictionaries.
(d) Tropp's exact recovery condition ERC(S) = max_{i not in S} ||A_S^+ a_i||_1 for each token's true top-3 support,
    and whether ERC (a data-free geometric quantity) predicts OMP identification."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c)
A, lab = c.dictionary(L); NA, D = A.shape; typ_atom = lab["type"].to(DEV)
X = c.X(L); Xraw = c.X(L, center=False); typ = typical_mask(Xraw); Xt = X[typ]
res = dict(model=tag, L=L, NA=NA, D=D)
# (a) coherence profile
maxc = torch.zeros(NA, device=DEV); partner = torch.zeros(NA, dtype=torch.long, device=DEV); top64 = torch.zeros(NA, 64, device=DEV)
CH = 2048
for s in range(0, NA, CH):
    G = A[s:s + CH] @ A.T; G[torch.arange(G.shape[0]), torch.arange(s, s + G.shape[0])] = 0
    v, i = G.abs().topk(64, dim=1); top64[s:s + CH] = v; maxc[s:s + CH] = v[:, 0]; partner[s:s + CH] = i[:, 0]
babel = top64.cumsum(1).max(0).values                       # mu1(k), k = 1..64
res["mutual_coherence"] = maxc.max().item(); res["babel"] = babel.tolist()
res["k_guaranteed"] = int(((babel[:-1] + top64.cumsum(1).max(0).values[1:]) < 1).sum())  # crude: mu1(k-1)+mu1(k) < 1 (Tropp 2004)
res["maxc_by_type"] = {int(t): dict(median=maxc[typ_atom == t].median().item(), q90=maxc[typ_atom == t].quantile(0.9).item(), max=maxc[typ_atom == t].max().item()) for t in typ_atom.unique().tolist()}
pt = typ_atom[partner]
res["partner_type_of_mlp_atoms"] = {int(t): ((pt == t) & (typ_atom == T_MLP)).sum().item() / (typ_atom == T_MLP).sum().item() for t in range(5)}
same_block = (lab["block"].to(DEV)[partner] == lab["block"].to(DEV)) & (typ_atom == T_MLP)
res["mlp_partner_same_block_frac"] = same_block.sum().item() / (typ_atom == T_MLP).sum().item()
# (b) frame operator and alignment with the state covariance
Sig = (Xt.T @ Xt) / Xt.shape[0]
def cosF(P, Q): return (P * Q).sum().item() / (P.norm() * Q.norm()).item()
def sub_overlap(P, Q, r):
    U = torch.linalg.eigh(P).eigenvectors[:, -r:]; V = torch.linalg.eigh(Q).eigenvectors[:, -r:]
    return ((U.T @ V) ** 2).sum().item() / r
S = A.T @ A; ev = torch.linalg.eigvalsh(S)
p = ev / ev.sum(); res["frame_eff_rank"] = math.exp(-(p * (p + 1e-12).log()).sum().item()); res["frame_bound_ratio"] = (ev[-1] / ev[0].clamp_min(1e-9)).item()
res["frame_top1_share"] = (ev[-1] / ev.sum()).item()
evs = torch.linalg.eigvalsh(Sig); ps = evs / evs.sum(); res["state_eff_rank"] = math.exp(-(ps * (ps + 1e-12).log()).sum().item())
res["align"] = dict(weight=cosF(S, Sig))
for sd in (7, 8, 9):
    Ar = rotate(A, sd); Sr = Ar.T @ Ar; res["align"][f"rot{sd}"] = cosF(Sr, Sig)
R = random_dict(NA, D); res["align"]["random"] = cosF(R.T @ R, Sig)
res["align_by_type"] = {int(t): cosF(A[typ_atom == t].T @ A[typ_atom == t], Sig) for t in typ_atom.unique().tolist()}
res["subspace_overlap"] = {r: dict(weight=sub_overlap(S, Sig, r), rot=sub_overlap(rotate(A).T @ rotate(A), Sig, r)) for r in (1, 4, 16, 64)}
# (c) matched-filter statistic on typical states
def maxcorr(Ad, n=4096):
    x = Xt[sub(Xt.shape[0], n)]; x = x / x.norm(dim=-1, keepdim=True)
    m = torch.zeros(x.shape[0], device=DEV); t8 = torch.zeros(x.shape[0], device=DEV)
    for s in range(0, x.shape[0], 1024):
        v = (x[s:s + 1024] @ Ad.T).abs().topk(8, dim=1).values; m[s:s + 1024] = v[:, 0]; t8[s:s + 1024] = v.mean(1)
    return dict(max_mean=m.mean().item(), max_median=m.median().item(), top8_mean=t8.mean().item())
res["matched_filter"] = dict(weight=maxcorr(A), rot=maxcorr(rotate(A)), random=maxcorr(R))
# (d) ERC of the true top-3 supports vs OMP identification
sel, cof, err = get_omp(c, L, A=A, X=X)
tb, tn, tc = c.top_writes(L, 3); rows3 = torch.stack([c.atom_index(L, tb[:, j], tn[:, j]) for j in range(3)], 1).to(DEV)
ids = sub(c.NT, 4096); erc = torch.zeros(len(ids), device=DEV); erc1 = torch.zeros(len(ids), device=DEV)
for s in range(0, len(ids), 256):
    ii = ids[s:s + 256].to(DEV); S3 = rows3[ii]; As = A[S3]                              # [n,3,D]
    G = As @ As.transpose(1, 2) + 1e-6 * torch.eye(3, device=DEV)
    P = torch.linalg.solve(G, As)                                                        # [n,3,D] = G^-1 A_S
    C = (P @ A.T).abs().sum(1)                                                            # [n, NA] l1 of pseudo-inverse coefficients
    C.scatter_(1, S3, 0.0); erc[s:s + 256] = C.max(1).values
    erc1[s:s + 256] = maxc[S3[:, 0]]
hit1 = (sel[ids] == rows3[ids][:, :1]).any(1); hit3 = (sel[ids][:, :, None] == rows3[ids][:, None, :]).any(1).all(1)
def auc(score, y):
    pos, neg = score[y], score[~y]
    if len(pos) == 0 or len(neg) == 0: return None
    return (pos[:, None] > neg[None, :]).float().mean().item()
t = typ[ids]
res["erc"] = dict(median=erc.median().item(), frac_below_1=(erc < 1).float().mean().item(), auc_top1_hit=auc(-erc[t], hit1[t]), auc_top3_hit=auc(-erc[t], hit3[t]),
                  auc_maxcoh_top1_hit=auc(-erc1[t], hit1[t]), hit1_rate=hit1[t].float().mean().item(), hit3_rate=hit3[t].float().mean().item())
record(f"e03_dict_{tag}", res, f"mu {res['mutual_coherence']:.3f} babel64 {babel[-1]:.1f} | align S~Sigma weight {res['align']['weight']:.3f} rot {res['align']['rot7']:.3f} rand {res['align']['random']:.3f} | maxcorr w {res['matched_filter']['weight']['max_mean']:.3f} r {res['matched_filter']['rot']['max_mean']:.3f} | ERC med {res['erc']['median']:.2f} <1: {res['erc']['frac_below_1']:.2f} AUC(top1) {res['erc']['auc_top1_hit']}")
