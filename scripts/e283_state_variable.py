"""e283: is the descendant a state variable? The per-token natural descendant at a later level lf is predicted from:
(a) the descendant at L pushed through the operator T(L->lf) fitted on random injections; (b) the signed write pushed
through T(3->lf); (c) a ridge from the top-64 state PCs at L (fit on half the tokens); (d) a random vector of the
descendant's norm pushed through T(L->lf); (e) descendant law plus a state ridge on its residual; (f) descendant
law plus write law with least-squares weights. Per-token cosine to the actual future descendant on held-out tokens,
centroid-level cosine, and the future functional test: the logit effect of the centroid predicted by (a) and (b)
injected at block lf+1 against the effect of the actual future centroid."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from func_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; NB = c.NB; b = 2; D = c.D; lab = c.d["lab"]; A = c.d["A"]; R = A[(lab["type"] == T_MLP) & (lab["block"] == b)].float().to(DEV); futures = [lf for lf in (L + 2, L + 4) if lf < NB - 1]
led, tn, tc = dominant(model, arch, c, ids_seq, b); run = make_runner_logits(model, arch, c, ids_seq, [L] + futures, NT, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep); S1 = run(tn); smed = torch.stack([tc[idx][lab_i == k].median() for k in range(K)]); sgn = torch.sign(smed)
pool = torch.nonzero(typ)[:, 0]; s_inj = tc.abs().median(); torch.manual_seed(0); split = torch.rand(len(idx), device=DEV) < 0.5
def fit_T(blk, lv, seed):
    torch.manual_seed(seed); Vr = unit(torch.randn(1024, D, device=DEV)); X = []; Y = []
    for p in range(2):
        a = torch.randint(0, 1024, (len(pool),), device=DEV); inj = torch.zeros(NT, D, device=DEV); inj[pool] = s_inj * Vr[a]; S2 = run(inject=inj, inject_block=blk); X.append(Vr[a]); Y.append((S2[lv] - S0[lv])[pool] / s_inj)
    X, Y = torch.cat(X), torch.cat(Y); G = X.T @ X; return torch.linalg.solve(G + 1e-2 * G.diagonal().mean() * torch.eye(D, device=DEV), X.T @ Y)
def pcs(S, d=64):
    Sc = S - S.mean(0, keepdim=True); U = torch.linalg.svd(Sc, full_matrices=False)[2][:d]; return Sc @ U.T
def ridge_fit(Xf, Y):
    Xtr, Ytr = Xf[split], Y[split]; mx, my = Xtr.mean(0, keepdim=True), Ytr.mean(0, keepdim=True); Xc, Yc = Xtr - mx, Ytr - my; G = Xc.T @ Xc; Wm = torch.linalg.solve(G + 1e-1 * G.diagonal().mean() * torch.eye(G.shape[0], device=DEV), Xc.T @ Yc); return (Xf - mx) @ Wm + my
pcos = lambda P, Q: ((P * Q).sum(1) / (P.norm(dim=1) * Q.norm(dim=1)).clamp_min(1e-9))
FL = (S0[L] - S1[L])[idx]; XL = pcs(S0[L][idx]); foreign = torch.nonzero(typ & ~torch.isin(tn, keep))[:, 0][:1536]; NF = len(foreign); base = run(positions=foreign); lg0 = base["lg"]; out = {}
for lf in futures:
    TLf = fit_T(L + 1, lf, 10 + lf); T3f = fit_T(b + 1, lf, 20 + lf); Ff = (S0[lf] - S1[lf])[idx]
    pa = FL @ TLf; pb = (sgn[:, None] * (R[keep] @ T3f))[lab_i]; pc_ = ridge_fit(XL, Ff); torch.manual_seed(3); pd = unit(torch.randn(len(idx), D, device=DEV)) * FL.norm(dim=1, keepdim=True) @ TLf; pe = pa + ridge_fit(XL, Ff - pa)
    ua, ub = unit(pa), unit(pb); a00 = (ua * ua).sum(1); a01 = (ua * ub).sum(1); a11 = (ub * ub).sum(1); b0 = (Ff * ua).sum(1); b1 = (Ff * ub).sum(1); det = (a00 * a11 - a01 ** 2).clamp_min(1e-6); c0 = ((a11 * b0 - a01 * b1) / det); c1 = ((a00 * b1 - a01 * b0) / det); w0, w1 = c0[split].median(), c1[split].median(); pf = w0 * ua + w1 * ub
    preds = dict(descendant_law=pa, write_law=pb, state_ridge=pc_, random_law=pd, descendant_plus_state=pe, descendant_plus_write=pf); rec = {nm: pcos(P, Ff)[~split].median().item() for nm, P in preds.items()}
    Ca = centroids(Ff[~split], lab_i[~split], K); rec["centroid_descendant_law"] = ((unit(centroids(pa, lab_i, K)) * Ca).sum(1)).median().item(); rec["centroid_write_law"] = ((unit(centroids(pb, lab_i, K)) * Ca).sum(1)).median().item(); rec["centroid_random_law"] = ((unit(centroids(pd, lab_i, K)) * Ca).sum(1)).median().item()
    ynorm = Ff.norm(dim=1).median(); asg = torch.randint(0, K, (NF,), device=DEV)
    def effect(Cv):
        inj = torch.zeros(NT, D, device=DEV); inj[foreign] = ynorm * unit(Cv)[asg]; r = run(positions=foreign, inject=inj, inject_block=lf + 1); dl = r["lg"] - lg0; return dl - dl.mean(1, keepdim=True)
    Eact = effect(centroids(Ff, lab_i, K)); rec["effect_descendant_law_vs_actual"] = pcos(effect(centroids(pa, lab_i, K)), Eact).median().item(); rec["effect_write_law_vs_actual"] = pcos(effect(centroids(pb, lab_i, K)), Eact).median().item(); rec["effect_random_law_vs_actual"] = pcos(effect(centroids(pd, lab_i, K)), Eact).median().item()
    out[lf] = rec; log(f"{tag} L {L} -> {lf}: per-token cos to the actual future descendant: descendant law {rec['descendant_law']:.2f}, write law {rec['write_law']:.2f}, state ridge {rec['state_ridge']:.2f}, random law {rec['random_law']:.2f}, descendant+state {rec['descendant_plus_state']:.2f}, descendant+write {rec['descendant_plus_write']:.2f} | centroid cos: descendant law {rec['centroid_descendant_law']:.2f}, write law {rec['centroid_write_law']:.2f}, random {rec['centroid_random_law']:.2f} | future effect (cos to the actual future centroid's effect): descendant law {rec['effect_descendant_law_vs_actual']:.2f}, write law {rec['effect_write_law_vs_actual']:.2f}, random {rec['effect_random_law_vs_actual']:.2f}")
record(f"e283_statevar_{tag}", dict(model=tag, b=b, L=L, K=K, per_future={str(k): v for k, v in out.items()}), " | ".join(f"{L}->{lf}: token cos D-law {v['descendant_law']:.2f} W-law {v['write_law']:.2f} state {v['state_ridge']:.2f} random {v['random_law']:.2f} D+state {v['descendant_plus_state']:.2f} D+W {v['descendant_plus_write']:.2f}; centroid D {v['centroid_descendant_law']:.2f} W {v['centroid_write_law']:.2f}; effect D {v['effect_descendant_law_vs_actual']:.2f} W {v['effect_write_law_vs_actual']:.2f} random {v['effect_random_law_vs_actual']:.2f}" for lf, v in out.items()))
