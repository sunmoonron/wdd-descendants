"""e534: the anatomy of a native word's basin. e533 found that half the native words survive a rotation of 26-29
degrees toward a random orthogonal direction, twice a random dictionary's words, and read the difference as the margin
above the competitors. Three things it did not test: whether the basin is directionally structured (the same angle
toward the row's own future direction, into the cloud's principal subspace, out of it, toward the nearest competitor
atom, away from it, against a random direction); whether the basin is the row's or its competitors' (all non-word
atoms rotated instead of the word; each word's nearest competitors rotated toward it); and whether the margin
explains the native words' wider basin (per-word survival against the word's margin over the floor, its breadth, its
margin over the best competitor at its best position, its nearest-neighbour cosine and its usage, and native against
random words within bins of margin).
Setup as e533: Pythia-410m, block 12, checkpoints 4000, 8000, 12000; the native dictionary at the checkpoint and a
fixed random dictionary; words the 256 most-used row atoms; OMP rerun on the same states (static survival) and on the
states four thousand steps later with the rotated atoms frozen (prospective); angles 20 and 30 degrees.
Pre-registered (honest guesses), static, 20 degrees:
- rotation toward the row's own future direction loses at least 0.15 fewer words than a random direction (0.5);
- rotation into the top-32 subspace loses fewer words than random by 0.10 or more, and out of it more (0.5);
- within bins of the margin over the floor, native and random words survive alike, within 0.10 (0.4);
- the nearest-neighbour cosine reads survival at AUC 0.6 or below (0.6);
- rotating every non-word atom by 20 degrees leaves 0.9 or more of the words (0.6).
Arguments: name."""
import sys, os, json as _json, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; B = 12; NWORD = 256; K = 16; H = 4000; ANG = [20, 30]; CKS = [4000, 8000, 12000]; NPC = 32; NNC = 5
idsB = eval_ids(name)[:8, :256].to(DEV); gen = torch.Generator(device=DEV).manual_seed(0)
def gabs(m, T=14.0, n=28001):
    t = torch.linspace(0, T, n, dtype=torch.float64); F = torch.special.erf(t / math.sqrt(2)); return float(torch.trapezoid(1 - F.pow(m), t))
def auc(pos, neg):
    if pos.numel() < 5 or neg.numel() < 5: return None
    x = torch.cat([pos, neg]); r = x.argsort().argsort().double() + 1; npos, nneg = pos.numel(), neg.numel(); return float((r[:npos].sum() - npos * (npos + 1) / 2) / (npos * nneg))
def states_at(n):
    model, tok, fam = load_model(name, revision=f"step{n}"); arch = Arch(model, fam)
    for p in model.parameters(): p.requires_grad_(False)
    X = block_states(model, arch, idsB, [B], chunk=4)[B].reshape(-1, arch.D); A, lab = build_dictionary(arch, blocks=list(range(B + 1))); del model; torch.cuda.empty_cache()
    keep = ~sinkmask(X); Xk = X[keep]; return Xk - Xk.mean(0), unitr(A), lab, arch.D, arch.DFF
def rows_index(lab, DFF):
    typ = lab["type"].to(DEV); blk = lab["block"].to(DEV); idx = lab["index"].to(DEV); gidx = torch.full((B + 1, DFF), -1, device=DEV, dtype=torch.long)
    for bb in range(B + 1): mm = (typ == T_MLP) & (blk == bb); gidx[bb, idx[mm]] = torch.nonzero(mm)[:, 0]
    return gidx.reshape(-1)
def usage_of(Xc, A, rows):
    sel, _, _ = omp(Xc, A, K, batch=1024, record_err=False); return torch.bincount(sel.reshape(-1), minlength=A.shape[0])[rows].float()
def topset(u):
    w = torch.zeros(u.numel(), dtype=torch.bool, device=u.device); w[torch.nonzero(u > 0)[:, 0][u[u > 0].argsort(descending=True)[:NWORD]]] = True; return w
def floor_of(U, A):
    m = A.shape[0]; Ar = unitr(rotate(A, seed=11)); CA = A.T @ A / m; CR = Ar.T @ Ar / m; gm = gabs(m); N = U.shape[0]
    s2 = ((U @ CA) * U).sum(1); s2r = ((U @ CR) * U).sum(1); Mr = torch.cat([(U[s:s + 128] @ Ar.T).abs().max(1).values for s in range(0, N, 128)]); r_cal = float(Mr.mean() / (s2r.clamp_min(1e-12).sqrt() * gm).mean()); return s2.clamp_min(1e-12).sqrt() * gm * r_cal
orth = lambda v, w: unitr(v - (v * w).sum(1, keepdim=True) * w)
res = dict(model=name, block=B, angles=ANG, horizon=H, by_checkpoint={})
for n in CKS:
    Xc0, A0, lab0, D, DFF = states_at(n); Xc1, A1, _, _, _ = states_at(n + H); rows = rows_index(lab0, DFF); U0 = unitr(Xc0); L0 = floor_of(U0, A0); N0 = U0.shape[0]
    ev, V = torch.linalg.eigh(Xc0.T @ Xc0 / N0); V = V.flip(1)[:, :NPC]
    Ar = unitr(torch.randn(A0.shape[0], D, device=DEV, generator=gen)); out = {}
    for dname, A, Afut in (("native", A0, A1), ("random", Ar, None)):
        u0 = usage_of(Xc0, A, rows); W0 = topset(u0); widx = rows[torch.nonzero(W0)[:, 0]]; nw = widx.numel(); w = A[widx]
        base1 = usage_of(Xc1, A, rows); W1 = topset(base1)
        # per-word features: margin over the floor, breadth, margin over the best competitor at the best position, nearest-neighbour cosine, usage
        P = (U0 @ w.T); ratio = P.abs() / L0[:, None]; S, a = ratio.max(0); cnt = (ratio > 1).sum(0).float()
        Pall = (U0[a] @ A.T).abs(); Pall[torch.arange(nw, device=DEV), widx] = 0; best_other = Pall.max(1).values; comp_margin = P.abs()[a, torch.arange(nw, device=DEV)] / best_other.clamp_min(1e-9)
        G = (w @ A.T).abs(); G[torch.arange(nw, device=DEV), widx] = 0; nn_cos, nn_idx = G.max(1); nn5 = G.topk(NNC, dim=1).indices
        feats = dict(S=S, breadth=cnt, competitor_margin=comp_margin, nn_cos=nn_cos, usage=u0[W0], survives_4000=W1[W0].float())
        # directions, all orthogonal to the word
        rnd = orth(torch.randn(nw, D, device=DEV, generator=gen), w)
        dirs = {"random": rnd, "into_top32": orth((rnd @ V) @ V.T, w), "out_of_top32": orth(rnd - (rnd @ V) @ V.T, w), "toward_nearest": orth(A[nn_idx], w), "away_from_nearest": orth(-A[nn_idx], w)}
        if Afut is not None: dirs["toward_future_row"] = orth(Afut[widx], w)
        surv = {}
        for th in ANG:
            aa = math.radians(th)
            for dn, r in dirs.items():
                Ap = A.clone(); Ap[widx] = math.cos(aa) * w + math.sin(aa) * r; Ws = topset(usage_of(Xc0, Ap, rows)); Wp = topset(usage_of(Xc1, Ap, rows))
                surv[f"{dn}_{th}"] = dict(static=Ws[W0].float(), prospective=Wp[W0].float())
        # competitors perturbed instead of the word (static)
        aa = math.radians(20); Ap = A.clone(); nonw = torch.nonzero(~W0)[:, 0]; nidx = rows[nonw]; rr = orth(torch.randn(nidx.numel(), D, device=DEV, generator=gen), A[nidx]); Ap[nidx] = math.cos(aa) * A[nidx] + math.sin(aa) * rr
        surv["competitors_random_20"] = dict(static=topset(usage_of(Xc0, Ap, rows))[W0].float())
        Ap = A.clone(); flat = nn5.reshape(-1); tgt = w.repeat_interleave(NNC, 0); rr = orth(tgt, A[flat]); Ap[flat] = math.cos(aa) * A[flat] + math.sin(aa) * rr
        surv["nearest5_toward_word_20"] = dict(static=topset(usage_of(Xc0, Ap, rows))[W0].float())
        rec = dict(n_words=nw, frozen_baseline_prospective=float(W1[W0].float().mean()), survival={k: {kk: float(v.mean()) for kk, v in d.items()} for k, d in surv.items()}, features_median={k: float(v.median()) for k, v in feats.items()},
                   feature_auc_for_survival_random_20={k: auc(feats[k][surv["random_20"]["static"] > 0.5], feats[k][surv["random_20"]["static"] <= 0.5]) for k in ("S", "breadth", "competitor_margin", "nn_cos", "usage")},
                   per_word={"S": S.cpu().tolist(), "breadth": cnt.cpu().tolist(), "competitor_margin": comp_margin.cpu().tolist(), "nn_cos": nn_cos.cpu().tolist(), "random_20_static": surv["random_20"]["static"].cpu().tolist(), "random_30_static": surv["random_30"]["static"].cpu().tolist()})
        out[dname] = rec
        log(f"{name} step{n} {dname}: static survival at 20 / 30 degrees by direction: " + ", ".join(f"{dn} {rec['survival'][f'{dn}_20']['static']:.2f}/{rec['survival'][f'{dn}_30']['static']:.2f}" for dn in dirs) + f"; prospective at +{H} (baseline {rec['frozen_baseline_prospective']:.2f}): " + ", ".join(f"{dn} {rec['survival'][f'{dn}_20']['prospective']:.2f}/{rec['survival'][f'{dn}_30']['prospective']:.2f}" for dn in dirs) + f"; competitors rotated 20 degrees, words kept {rec['survival']['competitors_random_20']['static']:.2f}; nearest five competitors rotated toward the word, kept {rec['survival']['nearest5_toward_word_20']['static']:.2f}; features (median) S {rec['features_median']['S']:.2f}, breadth {rec['features_median']['breadth']:.0f}, competitor margin {rec['features_median']['competitor_margin']:.2f}, nearest-neighbour cosine {rec['features_median']['nn_cos']:.2f}; AUC of features for survival at 20 degrees: " + ", ".join(f"{k} {v:.2f}" if v is not None else f"{k} n/a" for k, v in rec["feature_auc_for_survival_random_20"].items()))
    # margin-matched comparison: native against random words within bins of S (pooled quantiles)
    Sn = torch.tensor(out["native"]["per_word"]["S"]); Sr = torch.tensor(out["random"]["per_word"]["S"]); sn = torch.tensor(out["native"]["per_word"]["random_20_static"]); sr = torch.tensor(out["random"]["per_word"]["random_20_static"])
    edges = torch.cat([Sn, Sr]).quantile(torch.linspace(0, 1, 5)); bins = []
    for i in range(4):
        mn = (Sn >= edges[i]) & (Sn <= edges[i + 1]); mr = (Sr >= edges[i]) & (Sr <= edges[i + 1])
        bins.append(dict(S_range=[float(edges[i]), float(edges[i + 1])], n_native=int(mn.sum()), n_random=int(mr.sum()), native_survival=float(sn[mn].mean()) if int(mn.sum()) else None, random_survival=float(sr[mr].mean()) if int(mr.sum()) else None))
    out["margin_matched"] = bins
    log(f"{name} step{n}: survival at 20 degrees within bins of S (pooled quartiles), native / random: " + " | ".join(f"S {b_['S_range'][0]:.2f}-{b_['S_range'][1]:.2f}: {b_['native_survival'] if b_['native_survival'] is None else round(b_['native_survival'], 2)} ({b_['n_native']}) / {b_['random_survival'] if b_['random_survival'] is None else round(b_['random_survival'], 2)} ({b_['n_random']})" for b_ in bins))
    res["by_checkpoint"][n] = out; del Xc0, Xc1, A0, A1, Ar; torch.cuda.empty_cache()
nat = lambda n, k, kk="static": res["by_checkpoint"][n]["native"]["survival"][k][kk]
gap = []
for n in CKS:
    for b_ in res["by_checkpoint"][n]["margin_matched"]:
        if b_["native_survival"] is not None and b_["random_survival"] is not None and b_["n_native"] >= 20 and b_["n_random"] >= 20: gap.append(abs(b_["native_survival"] - b_["random_survival"]))
res["checks"] = dict(future_row_safer_0_15=all(nat(n, "toward_future_row_20") - nat(n, "random_20") >= 0.15 for n in CKS), into_top32_safer_out_worse=all(nat(n, "into_top32_20") - nat(n, "random_20") >= 0.10 and nat(n, "out_of_top32_20") < nat(n, "random_20") for n in CKS),
                     margin_matched_alike=all(g <= 0.10 for g in gap) if gap else False, nn_cos_auc_under_0_6=all((res["by_checkpoint"][n]["native"]["feature_auc_for_survival_random_20"]["nn_cos"] or 0.5) <= 0.6 for n in CKS), competitors_rotated_keep_0_9=all(nat(n, "competitors_random_20") >= 0.9 for n in CKS))
summ = f"{name} block {B}: " + " | ".join(f"step {n}: native static survival at 20 degrees random/future row/into top-32/out of top-32/toward nearest/away from nearest " + "/".join(f"{nat(n, k + '_20'):.2f}" for k in ("random", "toward_future_row", "into_top32", "out_of_top32", "toward_nearest", "away_from_nearest")) + f"; competitors rotated {nat(n, 'competitors_random_20'):.2f}, nearest five toward the word {nat(n, 'nearest5_toward_word_20'):.2f}; feature AUCs S/breadth/competitor margin/nn cos " + "/".join(f"{(res['by_checkpoint'][n]['native']['feature_auc_for_survival_random_20'][k] or 0.5):.2f}" for k in ("S", "breadth", "competitor_margin", "nn_cos")) + "; margin-matched native vs random " + " ".join(f"{b_['native_survival'] if b_['native_survival'] is None else round(b_['native_survival'], 2)}:{b_['random_survival'] if b_['random_survival'] is None else round(b_['random_survival'], 2)}" for b_ in res["by_checkpoint"][n]["margin_matched"]) for n in CKS) + f" | checks {_json.dumps(res['checks'])}"
log(summ); record(f"e534_basin_anatomy_{name}", res, summ)
