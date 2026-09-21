"""e238: provenance resolution and geometry preservation. (a) Inject a block-2 write vector w and a rotated copy
w' with cos(w, w') in (0.99, 0.95, 0.9, 0.75, 0.5, 0) at the same foreign tokens; at levels 3, 4, 6, L the cosine
between the two descendants vs the initial cosine (angle amplification or contraction), and the initial cosine at
which descendants fall below 0.5 (the resolution limit). (b) For the K neurons with >= 20 natural tokens: the
pairwise cosine matrix of their write vectors vs that of their descendant centroids at L: Spearman over pairs and
5-nearest-neighbour overlap. Split: angle-preserving transport in a rotated frame vs angle-distorting mixing."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from desc_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; b = 2; levels = sorted({b + 1, b + 2, b + 4, L}); run = make_runner(model, arch, c, ids_seq, levels, NT); lab = c.d["lab"]; A = c.d["A"]; R = A[(lab["type"] == T_MLP) & (lab["block"] == b)].float().to(DEV)
led, tn, tc = dominant(model, arch, c, ids_seq, b); S0 = run(); S1 = run(b, tn); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5)
# (a) separation
torch.manual_seed(0); k = torch.randint(0, c.DFF, (NT,), device=DEV); s = tc.abs().median(); w = R[k]; u = torch.randn_like(w); u = unit(u - (u * w).sum(1, keepdim=True) * w); base = run(inject=s * w, inject_block=b + 1); sep = {}
for cs in (0.99, 0.95, 0.9, 0.75, 0.5, 0.0):
    w2 = cs * w + math.sqrt(1 - cs * cs) * u; S2 = run(inject=s * w2, inject_block=b + 1); sep[cs] = {}
    for lv in levels:
        Fa = base[lv] - S0[lv]; Fb = S2[lv] - S0[lv]; cosd = ((Fa * Fb).sum(1) / (Fa.norm(dim=1) * Fb.norm(dim=1)).clamp_min(1e-9)); sep[cs][lv] = cosd[typ].median().item()
    log(f"{tag} initial cosine {cs:.2f}: descendant cosine by level " + " ".join(f"{lv}:{sep[cs][lv]:.2f}" for lv in levels))
res_limit = {}
for lv in levels:
    lim = None
    for cs in (0.99, 0.95, 0.9, 0.75, 0.5, 0.0):
        if sep[cs][lv] < 0.5: lim = cs; break
    res_limit[lv] = lim
# (b) geometry preservation
idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep); F = S0[L] - S1[L]; cents = centroids(F[idx], lab_i, K); Rw = unit(R[keep]); Gw = Rw @ Rw.T; Gd = cents @ cents.T; iu = torch.triu_indices(K, K, 1, device=DEV); a_ = Gw[iu[0], iu[1]]; b_ = Gd[iu[0], iu[1]]
ra = a_.argsort().argsort().float(); rb = b_.argsort().argsort().float(); rho = torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
nn_w = (Gw - 2 * torch.eye(K, device=DEV)).topk(5, dim=1).indices; nn_d = (Gd - 2 * torch.eye(K, device=DEV)).topk(5, dim=1).indices; overlap = float(sum(len(set(nn_w[i].tolist()) & set(nn_d[i].tolist())) for i in range(K)) / (5 * K))
log(f"{tag} geometry ({K} neurons): Spearman(pairwise cos of write vectors, of descendant centroids) {rho:+.2f}; 5-NN overlap {overlap:.2f} (chance {5 / (K - 1):.2f}) | resolution limit (first initial cosine with descendant cosine < 0.5) by level " + " ".join(f"{lv}:{res_limit[lv]}" for lv in levels))
record(f"e238_separation_{tag}", dict(model=tag, b=b, L=L, separation={str(k_): {str(kk): vv for kk, vv in v.items()} for k_, v in sep.items()}, resolution_limit={str(k_): v for k_, v in res_limit.items()}, geometry=dict(K=K, spearman=rho, knn5_overlap=overlap, chance=5 / (K - 1))), "descendant cosine at L for initial cos 0.99/0.95/0.9/0.75/0.5/0: " + " ".join(f"{sep[cs][L]:.2f}" for cs in (0.99, 0.95, 0.9, 0.75, 0.5, 0.0)) + f" | at +2: " + " ".join(f"{sep[cs][b + 2]:.2f}" for cs in (0.99, 0.95, 0.9, 0.75, 0.5, 0.0)) + f" | geometry Spearman {rho:+.2f}, 5-NN overlap {overlap:.2f} (chance {5 / (K - 1):.2f})")
