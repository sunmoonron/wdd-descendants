"""e475: three geometries. Does WDD induce its own distance between states, one that predicts behaviour beyond plain
activation distance? From an external review ("WDD-induced geometry", "activation / WDD / behaviour triangle", and the
split of identities from coefficients). e151 and e169 compared support overlap with state cosine as kNN next-token
predictors, and e460 found ledgers add nothing for near-identical states; neither had a rotated control.
Setup: 8 x 256 evaluation tokens, typical positions (about 2000), depths b = NB/4, NB/2, 3NB/4. For every pair of
positions:
- behaviour d_B: 1 - Bhattacharyya coefficient of the two next-token distributions (the squared Hellinger distance);
- future d_F: 1 - cosine of the centred states at b + 4;
- activation d_X: 1 - cosine of the centred states at b;
- native WDD (16 words over the dictionary up to b): d_C, 1 - cosine of the sparse coefficient vectors (identities and
  coefficients); d_S, 1 - Jaccard of the word sets (identities only); d_M, the distance between the sorted normalised
  coefficient magnitudes (coefficients only); d_R, 1 - cosine of the reconstructions;
- the same for rotated words (same Gram matrix, no provenance), and d_Z, 1 - cosine in the top 16 principal components
  (fitted on other sequences).
Measured: Spearman correlation of each distance with d_B and with d_F; the partial Spearman correlation with d_B given
d_X (what the distance adds beyond activation distance); the mean d_B of each position's 10 nearest neighbours under each
distance; and the quadrants: among activation-near pairs (d_X in the lowest 5%), the mean d_B of the pairs with the
larger against the smaller half of d_C (does WDD separate states that behave differently?), and among WDD-near pairs, the
same split by d_X.
v3 adds, after the magnitude-only channel showed information at the deepest depth: the gap between the two next-token
entropies as a distance, the rank correlation of the native top word's coefficient share with the entropy, and partial
correlations with behaviour given both activation distance and the entropy gap.
Models (argument): gpt2, qwen05, smollm2.
Pre-registered (honest guesses):
- activation distance predicts behaviour at least as well as native d_C at every depth (0.7);
- native d_C adds under 0.05 partial correlation beyond d_X (0.6);
- native d_C's partial correlation exceeds rotated's (0.5)."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; model, tok, fam = load_model(name); arch = Arch(model, fam); NB = arch.NB; K = 16
E = eval_ids(name); ids = E[:8, :256].to(DEV); fit = E[8:16, :256].to(DEV); B_, T_ = ids.shape
DEPTHS = sorted({NB // 4, NB // 2, (3 * NB) // 4}); FUT = {b: min(b + 4, NB - 1) for b in DEPTHS}
blocks = sorted(set(DEPTHS) | set(FUT.values()))
S_ = block_states(model, arch, ids, blocks, chunk=4); SF = block_states(model, arch, fit, DEPTHS, chunk=4)
keep = ~sinkmask(S_[DEPTHS[len(DEPTHS) // 2]].reshape(-1, arch.D))
for b in blocks: keep &= ~sinkmask(S_[b].reshape(-1, arch.D))
N = int(keep.sum()); rows = []
for s0 in range(0, B_, 2):
    with torch.no_grad(): lg = model(ids[s0:s0 + 2]).logits[:, 1:].float()
    k2 = keep.view(B_, T_ - 1)[s0:s0 + 2].reshape(-1); rows.append(torch.softmax(lg, -1).reshape(-1, lg.shape[-1])[k2].sqrt()); del lg
Sq = torch.cat(rows); dB = (1 - Sq @ Sq.T).clamp_min(0); Hn = -(Sq.pow(2) * 2 * Sq.clamp_min(1e-30).log()).sum(-1); del Sq, rows   # Hn: next-token entropy
iu = torch.triu_indices(N, N, 1, device=DEV); pick = lambda M: M[iu[0], iu[1]]
def rank(v):                                                          # average ranks (ties matter for the Jaccard distance)
    u, inv, cnt = torch.unique(v, return_inverse=True, return_counts=True); e_ = cnt.cumsum(0).double(); return ((2 * e_ - cnt.double() - 1) / 2)[inv].float()
def corr(a, b): a = a - a.mean(); b = b - b.mean(); return float((a * b).sum() / (a.norm() * b.norm()))
def resid(a, x): a = a - a.mean(); x = x - x.mean(); return a - (a * x).sum() / (x * x).sum() * x
def cosd(M): U = unitr(M); return 1 - U @ U.T
def resid2(a, x1, x2):                                                # residual of a after least squares on [1, x1, x2]
    Xd = torch.stack([torch.ones_like(x1), x1 - x1.mean(), x2 - x2.mean()], 1).double(); beta = torch.linalg.lstsq(Xd, a.double()[:, None]).solution; return (a.double() - (Xd @ beta)[:, 0]).float()
pB = pick(dB); rB = rank(pB); dH = (Hn[:, None] - Hn[None, :]).abs(); rH = rank(pick(dH))
res = dict(model=name, depths=DEPTHS, n_positions=N, n_pairs=int(pB.numel()), by_depth={})
for b in DEPTHS:
    X = S_[b].reshape(-1, arch.D)[keep]; mu = X.mean(0); Xc = X - mu
    Fu = S_[FUT[b]].reshape(-1, arch.D)[keep]; Fc = Fu - Fu.mean(0)
    Ff = SF[b].reshape(-1, arch.D); Ff = Ff[~sinkmask(Ff)]; P16 = pcs(Ff, K)[0]
    A, lab = build_dictionary(arch, blocks=list(range(b + 1))); Au = unitr(A); Ar = unitr(rotate(A, seed=7)); del A
    dist = {"X": cosd(Xc), "Z_pca16": cosd((Xc - (Ff.mean(0) - mu)) @ P16), "H_entropy_gap": dH}
    for kind, Dct in (("native", Au), ("rotated", Ar)):
        sel, _, _ = omp(Xc, Dct, K, batch=1024, record_err=False); cof, _ = refit(Xc, Dct, sel)
        uq, inv = torch.unique(sel, return_inverse=True); C = torch.zeros(N, uq.numel(), device=DEV); C.scatter_(1, inv, cof)
        Bm = torch.zeros_like(C); Bm.scatter_(1, inv, 1.0); inter = Bm @ Bm.T
        mag = torch.sort(cof.abs() / cof.norm(dim=-1, keepdim=True), 1, descending=True).values
        if kind == "native": top1_vs_entropy = corr(rank(mag[:, 0]), rank(Hn))
        dist[f"C_{kind}"] = cosd(C); dist[f"S_{kind}"] = 1 - inter / (2 * K - inter); dist[f"M_{kind}"] = torch.cdist(mag, mag); dist[f"R_{kind}"] = cosd(torch.einsum("nk,nkd->nd", cof, Dct[sel]))
        del C, Bm, inter
    del Au, Ar; torch.cuda.empty_cache()
    pF = pick(cosd(Fc)); rF = rank(pF); rX = rank(pick(dist["X"])); eB = resid(rB, rX)
    out = {}
    for key, M in dist.items():
        pv = pick(M); rv = rank(pv)
        Mn = M.clone(); Mn.fill_diagonal_(float("inf")); nn_ = Mn.topk(10, dim=1, largest=False).indices
        out[key] = dict(spearman_behaviour=corr(rv, rB), spearman_future=corr(rv, rF), partial_behaviour_given_X=(corr(resid(rv, rX), eB) if key != "X" else None),
                        knn10_behaviour=float(dB.gather(1, nn_).mean()))
        del Mn
    out["random_pair_behaviour"] = float(pB.mean()); out["native_top1_share_vs_entropy_spearman"] = top1_vs_entropy
    eBH = resid2(rB, rX, rH)
    for key in ("M_native", "M_rotated", "C_native", "C_rotated", "S_native", "S_rotated"): out[key]["partial_behaviour_given_X_and_entropy"] = corr(resid2(rank(pick(dist[key])), rX, rH), eBH)
    pX = pick(dist["X"]); qx = pX <= pX.quantile(0.05)
    for kind in ("native", "rotated"):
        # ">=" puts ties at the top (pairs sharing no word have d_C = 1 exactly) into the far half
        pC = pick(dist[f"C_{kind}"]); sub = pC[qx]; hi = qx.clone(); hi[qx] = sub >= sub.median(); lo = qx & ~hi
        qc = pC <= pC.quantile(0.05); subx = pX[qc]; hx = qc.clone(); hx[qc] = subx >= subx.median(); lx = qc & ~hx
        out[f"quadrants_{kind}"] = dict(activation_near_wdd_far=float(pB[hi].mean()), activation_near_wdd_near=float(pB[lo].mean()), activation_near_far_share=float(hi.sum() / qx.sum()),
                                        activation_near_no_shared_word=float((qx & (pick(dist[f"S_{kind}"]) >= 1)).sum() / qx.sum()),
                                        wdd_near_activation_far=float(pB[hx].mean()), wdd_near_activation_near=float(pB[lx].mean()))
    res["by_depth"][b] = out
    log(f"{name} depth {b}: " + " | ".join(f"{k_} rho_B {v['spearman_behaviour']:+.3f}" + (f" partial {v['partial_behaviour_given_X']:+.3f}" if v.get('partial_behaviour_given_X') is not None else "") + f" rho_F {v['spearman_future']:+.3f} knn {v['knn10_behaviour']:.3f}"
                                         for k_, v in out.items() if isinstance(v, dict) and "spearman_behaviour" in v) + f" | random {out['random_pair_behaviour']:.3f} | quadrants native {out['quadrants_native']} rotated {out['quadrants_rotated']}")
    del dist; torch.cuda.empty_cache()
BD = res["by_depth"]
res["checks"] = dict(activation_at_least_native=all(v["X"]["spearman_behaviour"] >= v["C_native"]["spearman_behaviour"] for v in BD.values()),
                     native_partial_under_0_05=all(v["C_native"]["partial_behaviour_given_X"] < 0.05 for v in BD.values()),
                     native_partial_over_rotated=all(v["C_native"]["partial_behaviour_given_X"] > v["C_rotated"]["partial_behaviour_given_X"] for v in BD.values()))
KEYS = ["X", "C_native", "S_native", "M_native", "R_native", "C_rotated", "S_rotated", "R_rotated", "Z_pca16"]
summ = (f"{name}, {N} positions: " + " || ".join(f"depth {b}: rho with behaviour (partial given X) / knn10 d_B: " + ", ".join(f"{k_} {v[k_]['spearman_behaviour']:+.2f}" + (f" ({v[k_]['partial_behaviour_given_X']:+.3f})" if k_ != "X" else "") + f"/{v[k_]['knn10_behaviour']:.3f}" for k_ in KEYS)
        + f"; random pairs d_B {v['random_pair_behaviour']:.3f}; activation-near pairs, d_B when WDD far vs near: native {v['quadrants_native']['activation_near_wdd_far']:.3f} vs {v['quadrants_native']['activation_near_wdd_near']:.3f}, rotated {v['quadrants_rotated']['activation_near_wdd_far']:.3f} vs {v['quadrants_rotated']['activation_near_wdd_near']:.3f} (share with no shared word native {v['quadrants_native']['activation_near_no_shared_word']:.2f}, rotated {v['quadrants_rotated']['activation_near_no_shared_word']:.2f}); "
        + f"WDD-near pairs, d_B when activation far vs near: native {v['quadrants_native']['wdd_near_activation_far']:.3f} vs {v['quadrants_native']['wdd_near_activation_near']:.3f}; "
        + f"entropy gap rho {v['H_entropy_gap']['spearman_behaviour']:+.2f} (partial given X {v['H_entropy_gap']['partial_behaviour_given_X']:+.3f}); native top-word share vs entropy rho {v['native_top1_share_vs_entropy_spearman']:+.2f}; partial given X and entropy: M native {v['M_native']['partial_behaviour_given_X_and_entropy']:+.3f}, C native {v['C_native']['partial_behaviour_given_X_and_entropy']:+.3f}, C rotated {v['C_rotated']['partial_behaviour_given_X_and_entropy']:+.3f}, S native {v['S_native']['partial_behaviour_given_X_and_entropy']:+.3f}, S rotated {v['S_rotated']['partial_behaviour_given_X_and_entropy']:+.3f}" for b, v in BD.items()) + f" | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e475_threegeometries_{name}", res, summ)
