"""e248: an explicit linear transport operator, fitted by the analysis (not by the model). Pairs (v, image) from random
unit directions injected at the block-3 input and read at levels 4 and L (two passes, one direction per token) fit a
ridge map T. Tests: (A) held-out random directions: cos(T v, mean image) vs the global-mean baseline; (B) zero-shot
provenance of natural footprints by nearest PREDICTED descendant T w_k of the K candidate neurons (no neuron ever
injected); (C) inversion: a ridge inverse U maps natural footprints back to write space, source identified by the
nearest native atom, cos(U F, w) reported; (D) composition: T_{2->4} fitted from block-3 injections read at 4,
T_{4->6} from block-5 injections read at 6, composed vs the direct T_{2->6} on held-out neuron atoms and on natural
footprint identification at level 6; (E) Gram preservation: Spearman and relative Frobenius error between V V^T and
(T V)(T V)^T for 64 held-out directions; (F) a+b: inject unit(a+b) for 32 neuron pairs and classify against images
of a, b, a+b, a-b. Kill: T no better than the global-mean baseline, or predicted-descendant provenance near chance."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from desc_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; b = 2; lv6 = min(6, L); levels = sorted({4, lv6, L}); run = make_runner(model, arch, c, ids_seq, levels, NT); lab = c.d["lab"]; A = c.d["A"]; R = A[(lab["type"] == T_MLP) & (lab["block"] == b)].float().to(DEV); D = c.D
led, tn, tc = dominant(model, arch, c, ids_seq, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); S1 = run(b, tn); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep); s_inj = tc.abs().median(); pool = torch.nonzero(typ)[:, 0]
def collect(inject_block, n_dirs, passes, seed):
    torch.manual_seed(seed); V = unit(torch.randn(n_dirs, D, device=DEV)); X = []; Y = {lv: [] for lv in levels}; assigns = []
    for p in range(passes):
        a = torch.randint(0, n_dirs, (len(pool),), device=DEV); inj = torch.zeros(NT, D, device=DEV); inj[pool] = s_inj * V[a]; S2 = run(inject=inj, inject_block=inject_block); X.append(V[a]); assigns.append(a)
        for lv in levels: Y[lv].append((S2[lv] - S0[lv])[pool] / s_inj)
    return V, torch.cat(X), {lv: torch.cat(Y[lv]) for lv in levels}, torch.cat(assigns)
def ridge(X, Y, lam=1e-2):
    G = X.T @ X; return torch.linalg.solve(G + lam * G.diagonal().mean() * torch.eye(X.shape[1], device=DEV), X.T @ Y)
Vtr, Xtr, Ytr, Atr = collect(b + 1, 1024, 2, 0); Vte, Xte, Yte, Ate = collect(b + 1, 256, 2, 1); out = {}
for lv in levels:
    T = ridge(Xtr, Ytr[lv]); U = ridge(Ytr[lv], Xtr)                                                  # forward and inverse ridge maps
    mean_te = torch.zeros(256, D, device=DEV).index_add_(0, Ate, Yte[lv]) / torch.bincount(Ate, minlength=256).clamp_min(1)[:, None]; gmean = unit(Ytr[lv].mean(0, keepdim=True))
    r = dict(A_pred_cos=((unit(Vte @ T) * unit(mean_te)).sum(1)).median().item(), A_baseline_cos=((unit(mean_te) @ gmean.T)[:, 0]).median().item(), A_pertoken_cos=((unit(Xte @ T) * unit(Yte[lv])).sum(1)).median().item())
    F = S0[lv] - S1[lv]; pred = unit(R[keep] @ T); r["B_predicted_descendant_provenance"] = accuracy(F[idx], pred, lab_i); r["B_chance"] = 1.0 / K
    Fi = unit(F[idx] @ U); Rk = unit(R[keep]); r["C_inverse_provenance"] = ((Fi @ Rk.T).argmax(1) == lab_i).float().mean().item(); r["C_inverse_cos"] = ((Fi * Rk[lab_i]).sum(1)).median().item(); r["C_native_atom_on_footprint"] = ((unit(F[idx]) @ Rk.T).argmax(1) == lab_i).float().mean().item()
    G1 = Vte[:64] @ Vte[:64].T; P = unit(Vte[:64] @ T); G2 = P @ P.T; iu = torch.triu_indices(64, 64, 1, device=DEV); a_, b_ = G1[iu[0], iu[1]], G2[iu[0], iu[1]]; ra = a_.argsort().argsort().float(); rb = b_.argsort().argsort().float(); r["E_gram_spearman"] = torch.corrcoef(torch.stack([ra, rb]))[0, 1].item(); r["E_gram_rel_frobenius"] = ((G1 - G2).norm() / G1.norm()).item()
    out[lv] = r
    log(f"{tag} level {lv} (K {K}): A held-out random directions: cos(T v, mean image) {r['A_pred_cos']:.2f} (per-token {r['A_pertoken_cos']:.2f}; global-mean baseline {r['A_baseline_cos']:.2f}) | B natural footprints by predicted descendants {r['B_predicted_descendant_provenance']:.2f} (chance {r['B_chance']:.2f}) | C inverse map: source by nearest native atom {r['C_inverse_provenance']:.2f}, cos(U F, w) {r['C_inverse_cos']:.2f} (footprint vs native atom directly {r['C_native_atom_on_footprint']:.2f}) | E Gram preservation Spearman {r['E_gram_spearman']:.2f}, rel. Frobenius {r['E_gram_rel_frobenius']:.2f}")
# (D) composition 2->4, 4->6 vs direct 2->6
if lv6 > 4:
    T24 = ridge(Xtr, Ytr[4]); Vm, Xm, Ym, Am = collect(5, 1024, 2, 2); T46 = ridge(Xm, Ym[lv6]); T26 = ridge(Xtr, Ytr[lv6]); Wk = R[keep]; comp = unit(Wk @ T24 @ T46); direct = unit(Wk @ T26); F6 = S0[lv6] - S1[lv6]
    out["D"] = dict(cos_composed_vs_direct=((comp * direct).sum(1)).median().item(), provenance_composed=accuracy(F6[idx], comp, lab_i), provenance_direct=accuracy(F6[idx], direct, lab_i))
    log(f"{tag} D composition 2->4->6 vs direct 2->6: cos {out['D']['cos_composed_vs_direct']:.2f}; natural-footprint provenance at level {lv6} by composed {out['D']['provenance_composed']:.2f} vs direct {out['D']['provenance_direct']:.2f}")
# (F) a+b
P = min(32, K); torch.manual_seed(5); pa = torch.randperm(K, device=DEV)[:P]; pb = (pa + torch.randint(1, K, (P,), device=DEV)) % K; a_v, b_v = R[keep[pa]], R[keep[pb]]; cand = torch.stack([a_v, b_v, unit(a_v + b_v), unit(a_v - b_v)], 1)   # [P, 4, D]
imgs = torch.zeros(P, 4, D, device=DEV); cnt = torch.zeros(P, 4, device=DEV); torch.manual_seed(6)
for p in range(3):
    ap = torch.randint(0, P, (len(pool),), device=DEV); aj = torch.randint(0, 4, (len(pool),), device=DEV); inj = torch.zeros(NT, D, device=DEV); inj[pool] = s_inj * cand[ap, aj]; S2 = run(inject=inj, inject_block=b + 1); img = (S2[L] - S0[L])[pool]; imgs.index_put_((ap, aj), img, accumulate=True); cnt.index_put_((ap, aj), torch.ones(len(pool), device=DEV), accumulate=True)
imgs = unit(imgs / cnt.clamp_min(1)[:, :, None]); ap = torch.randint(0, P, (len(pool),), device=DEV); inj = torch.zeros(NT, D, device=DEV); inj[pool] = s_inj * cand[ap, 2]; S3 = run(inject=inj, inject_block=b + 1); Fq = unit((S3[L] - S0[L])[pool]); sims = torch.einsum("nd,nkd->nk", Fq, imgs[ap]); pick = sims.argmax(1)
out["F"] = {nm: (pick == j).float().mean().item() for j, nm in enumerate(("a", "b", "a_plus_b", "a_minus_b"))}
log(f"{tag} F inject unit(a+b): classified as a {out['F']['a']:.2f}, b {out['F']['b']:.2f}, a+b {out['F']['a_plus_b']:.2f}, a-b {out['F']['a_minus_b']:.2f}")
record(f"e248_operator_{tag}", dict(model=tag, b=b, L=L, K=K, results={str(k): v for k, v in out.items()}), " | ".join(f"lv{lv}: pred cos {v['A_pred_cos']:.2f} (baseline {v['A_baseline_cos']:.2f}), predicted-descendant provenance {v['B_predicted_descendant_provenance']:.2f} (chance {v['B_chance']:.2f}), inverse provenance {v['C_inverse_provenance']:.2f} (cos {v['C_inverse_cos']:.2f}), Gram Spearman {v['E_gram_spearman']:.2f}" for lv, v in out.items() if isinstance(lv, int)) + (f" | composition cos {out['D']['cos_composed_vs_direct']:.2f}, provenance composed {out['D']['provenance_composed']:.2f} vs direct {out['D']['provenance_direct']:.2f}" if "D" in out else "") + f" | a+b classified as a+b {out['F']['a_plus_b']:.2f} (a {out['F']['a']:.2f}, b {out['F']['b']:.2f})")
