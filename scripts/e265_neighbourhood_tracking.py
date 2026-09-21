"""e265: (1) inverse test: are logit-footprint neighbours descendant neighbours? For each neuron its 3 nearest L-
neighbours: the fraction that are among its 5 nearest D-neighbours (chance 5/(K-1)) and the mean D-similarity of L-
neighbours vs random; the many-to-one cell (high L-similarity, low D-similarity) share. (2) Neighbour identity
tracking: for each neuron, the rank of its W-nearest neighbour in D-space at every level, and the rank of its L-
nearest neighbour in D-space at every level; the depth where the W-neighbour's median D-rank first exceeds K/4 and
where the L-neighbour's median D-rank first falls below K/4. (3) Causal-influence capacity: dimensions of the
descendant (top principal components, at L) needed to predict a token's removal KL by kNN regression (Spearman),
alongside identity and function."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from func_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; NB = c.NB; b = 2; D = c.D; lab = c.d["lab"]; A = c.d["A"]; R = A[(lab["type"] == T_MLP) & (lab["block"] == b)].float().to(DEV); levels = list(range(b, NB))
led, tn, tc = dominant(model, arch, c, ids_seq, b); run = make_runner_logits(model, arch, c, ids_seq, levels, NT, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep)
S0 = run(positions=idx); S1 = run(tn, positions=idx); dl = S0["lg"] - S1["lg"]; kl = (torch.log_softmax(S0["lg"], -1).exp() * (torch.log_softmax(S0["lg"], -1) - torch.log_softmax(S1["lg"], -1))).sum(1); dl = dl - dl.mean(1, keepdim=True); Cl = unit(centroids(dl, lab_i, K)); Gl = Cl @ Cl.T; Rw = unit(R[keep]); Gw = Rw @ Rw.T
def nn_of(G, k):
    G2 = G.clone(); G2.fill_diagonal_(-2); return G2.topk(k, dim=1).indices
def rank_in(G, target):
    G2 = G.clone(); G2.fill_diagonal_(-2); order = G2.argsort(dim=1, descending=True); return (order == target[:, None]).float().argmax(1).float()
FL = S0[L] - S1[L]; CdL = unit(centroids(FL[idx], lab_i, K)); GdL = CdL @ CdL.T; nnL = nn_of(Gl, 3); nnD5 = nn_of(GdL, 5); frac = (nnL[:, :, None] == nnD5[:, None, :]).any(2).float().mean().item(); torch.manual_seed(0); rnd = torch.randint(0, K, (K, 3), device=DEV)
inv = dict(frac_L_neighbours_in_D5=frac, chance=5.0 / (K - 1), mean_D_sim_of_L_neighbours=GdL[torch.arange(K, device=DEV)[:, None], nnL].mean().item(), mean_D_sim_random=GdL[torch.arange(K, device=DEV)[:, None], rnd].mean().item()); iu = torch.triu_indices(K, K, 1, device=DEV); gl, gd = Gl[iu[0], iu[1]], GdL[iu[0], iu[1]]; inv["many_to_one_share"] = ((gl > gl.quantile(0.75)) & (gd < gd.quantile(0.25))).float().sum().item() / max((gl > gl.quantile(0.75)).float().sum().item(), 1)
wnn = nn_of(Gw, 1)[:, 0]; lnn = nnL[:, 0]; track = {}
for lv in levels:
    Fl = S0[lv] - S1[lv]; Cd = unit(centroids(Fl[idx], lab_i, K)); Gd = Cd @ Cd.T; track[lv] = dict(w_neighbour_rank_in_D=rank_in(Gd, wnn).median().item(), l_neighbour_rank_in_D=rank_in(Gd, lnn).median().item())
first_w = next((lv for lv in levels if track[lv]["w_neighbour_rank_in_D"] > K / 4), None); first_l = next((lv for lv in levels if track[lv]["l_neighbour_rank_in_D"] < K / 4), None)
torch.manual_seed(0); split = torch.rand(len(idx), device=DEV) < 0.5; tr, te = torch.nonzero(split)[:, 0], torch.nonzero(~split)[:, 0]; Fc = FL[idx][tr] - FL[idx][tr].mean(0, keepdim=True); Uf = torch.linalg.svd(Fc, full_matrices=False)[2]; Fi = FL[idx]
def spearman(a_, b_):
    ra = a_.argsort().argsort().float(); rb = b_.argsort().argsort().float(); return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
cap = {}
for d in [dd for dd in (1, 2, 4, 8, 16, 32, 64, 128, 256, D) if dd <= D]:
    P = Uf[:d].T; Ptr, Pte = Fi[tr] @ P, Fi[te] @ P; nn5 = torch.cdist(Pte, Ptr).topk(5, dim=1, largest=False).indices; cap[d] = dict(causal_kl_spearman=spearman(kl[tr][nn5].mean(1), kl[te]), identity=accuracy(Pte, centroids(Ptr, lab_i[tr], K), lab_i[te]))
log(f"{tag} (K {K}): L-neighbours among the 5 D-neighbours {inv['frac_L_neighbours_in_D5']:.2f} (chance {inv['chance']:.2f}); D-similarity of L-neighbours {inv['mean_D_sim_of_L_neighbours']:.2f} vs random {inv['mean_D_sim_random']:.2f}; many-to-one share of high-L pairs {inv['many_to_one_share']:.2f} | W-neighbour's median rank in D-space by level " + " ".join(f"{lv}:{track[lv]['w_neighbour_rank_in_D']:.0f}" for lv in levels[::max(1, len(levels) // 8)]) + f" (first > K/4 at level {first_w}); L-neighbour's rank " + " ".join(f"{lv}:{track[lv]['l_neighbour_rank_in_D']:.0f}" for lv in levels[::max(1, len(levels) // 8)]) + f" (first < K/4 at level {first_l}) | causal-influence capacity (Spearman with removal KL) by d " + " ".join(f"{d}:{v['causal_kl_spearman']:.2f}" for d, v in cap.items()))
record(f"e265_track_{tag}", dict(model=tag, b=b, L=L, K=K, inverse=inv, tracking={str(k): v for k, v in track.items()}, first_w_over=first_w, first_l_under=first_l, capacity={str(k): v for k, v in cap.items()}), f"K {K}: L-neighbours in D5 {inv['frac_L_neighbours_in_D5']:.2f} (chance {inv['chance']:.2f}), many-to-one {inv['many_to_one_share']:.2f} | W-neighbour lost in D at level {first_w}, L-neighbour found in D at level {first_l} | KL capacity by d " + " ".join(f"{d}:{v['causal_kl_spearman']:.2f}" for d, v in cap.items() if d in (4, 32, 256, D)))
