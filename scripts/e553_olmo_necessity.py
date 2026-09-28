"""e553: the necessity test on a second model. e548 on OLMo-1B from the e550 cache (steps 1000-16000, block 8, the MLP
rows of blocks 0-8): at every origin checkpoint the non-words that enter at t+1, t+2, t+4, t+6 or t+8 and 4,000 random
non-words that do not, with ten features at t (S and the count over three quarters of the floor; the neuron's mean
positive activation, the kurtosis and the top-five-percent share of its activation, the coherence of the states at its
sixteen most active positions, the row's norm and its growth; the row's cosine with the mean centred state at those
sixteen positions; the coalition's coherence at its four most active positions); single-feature AUCs by horizon and
logistic regressions on within-checkpoint percentile ranks out of sample (even origins to odd and the reverse).
Pre-registered (honest guesses): WDD adds 0.05 or more to the activation side and the reverse adds under 0.03 (0.6);
S and breadth lead at every horizon (0.6).
Arguments: none."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from olmo_cache_common import *
name = "olmo1b"; C = load_olmo_cache(); T, N, m, D, DFF, B = C["T"], C["N"], C["m"], C["D"], C["DFF"], C["B"]; ACT, XS, RU, NORM, WORDS, SS, RATIO = (C[k] for k in ("ACT", "XS", "RU", "NORM", "WORDS", "SS", "RATIO")); steps = C["steps"]
HS = [1, 2, 4, 6, 8]; words = [WORDS[i] for i in range(T)]; NSAMP = 4000; gs = torch.Generator().manual_seed(11); FEATS = ["S", "cnt75", "act_mean_pos", "act_kurtosis", "act_top5_share", "context_coherence", "row_norm", "norm_growth", "read_align", "coalition_coherence"]; rows_all = []
def auc(pos, neg):
    if pos.numel() < 5 or neg.numel() < 5: return None
    x = torch.cat([pos, neg]); r = x.argsort().argsort().double() + 1; npos, nneg = pos.numel(), neg.numel(); return float((r[:npos].sum() - npos * (npos + 1) / 2) / (npos * nneg))
for t in range(T - 1):
    nonw = ~words[t]; labels = {}; cand = torch.zeros(m, dtype=torch.bool)
    for H in HS:
        if t + H < T: labels[H] = nonw & words[t + H]; cand |= labels[H]
    rest = torch.nonzero(nonw & ~cand)[:, 0]; samp = rest[torch.randperm(rest.numel(), generator=gs)[:NSAMP]]; cand[samp] = True; Cc = torch.nonzero(cand)[:, 0]; nC = Cc.numel()
    A = ACT[t][:, Cc].float().to(DEV); Uc = XS[t].to(DEV) - XS[t].to(DEV).mean(0); Un = unitr(Uc); Ru = RU[t].float().to(DEV); Ru_c = Ru[Cc.to(DEV)]; nrm = NORM[t].to(DEV); S_ = SS[t][Cc].to(DEV); c75 = (RATIO[t][:, Cc].float().to(DEV) > 0.75).sum(0).float()
    pos = A.clamp_min(0); mean_pos = pos.mean(0); mu = A.mean(0); var = A.var(0); kurt = ((A - mu[None]) ** 4).mean(0) / var.clamp_min(1e-12) ** 2 - 3; k5 = max(N // 20, 1); top5 = pos.topk(k5, dim=0).values.sum(0) / pos.sum(0).clamp_min(1e-9); topk16 = A.topk(16, dim=0).indices
    ctx = torch.zeros(nC, device=DEV); ralign = torch.zeros(nC, device=DEV); coal = torch.zeros(nC, device=DEV); iu = torch.triu_indices(16, 16, 1, device=DEV); iu4 = torch.triu_indices(4, 4, 1, device=DEV); ACTt = ACT[t]
    for j in range(nC):
        P = topk16[:, j]; Sm = Un[P] @ Un[P].T; ctx[j] = Sm[iu[0], iu[1]].mean(); ms = Uc[P].mean(0); ralign[j] = (ms @ Ru_c[j]) / ms.norm().clamp_min(1e-9)
        r_ = int(Cc[j]); g = nrm * (Ru @ Ru[r_]); a4 = ACTt[P[:4].cpu()].float().to(DEV); cv = a4 * g[None]; cn = cv / cv.norm(dim=1, keepdim=True).clamp_min(1e-9); Sc = cn @ cn.T; coal[j] = Sc[iu4[0], iu4[1]].mean()
    growth = (NORM[t][Cc] / NORM[t - 1][Cc].clamp_min(1e-9)) if t > 0 else torch.ones(nC)
    F = dict(S=S_.cpu(), cnt75=c75.cpu(), act_mean_pos=mean_pos.cpu(), act_kurtosis=kurt.cpu(), act_top5_share=top5.cpu(), context_coherence=ctx.cpu(), row_norm=NORM[t][Cc], norm_growth=growth, read_align=ralign.cpu(), coalition_coherence=coal.cpu())
    rows_all.append(dict(t=t, C=Cc, F=F, labels={H: v[Cc] for H, v in labels.items()})); del A, Uc, Un, Ru, Ru_c; torch.cuda.empty_cache(); log(f"{name} origin step{steps[t]}: {nC} candidates, entrants at H=4 {int(labels[4].sum()) if 4 in labels else 0}")
f2 = lambda x: "n/a" if x is None else f"{x:.2f}"; res = dict(model=name, block=B, features=FEATS, auc_by_horizon={}, logistic={})
for H in HS:
    out = {}
    for f in FEATS:
        pos_, neg_ = [], []
        for R_ in rows_all:
            if H not in R_["labels"]: continue
            y = R_["labels"][H]; x = R_["F"][f]; pos_.append(x[y]); neg_.append(x[~y])
        out[f] = auc(torch.cat(pos_), torch.cat(neg_)) if pos_ else None
    res["auc_by_horizon"][str(H)] = out; log(f"{name} AUC for entry at H={H}: " + ", ".join(f"{f} {f2(v)}" for f, v in out.items()))
def ranks(x): return (x.argsort().argsort().double() / max(x.numel() - 1, 1)).float()
SETS = {"wdd": ["S", "cnt75"], "activation": ["act_mean_pos", "act_kurtosis", "act_top5_share", "context_coherence", "row_norm", "norm_growth"], "activation+geometry+coalition": ["act_mean_pos", "act_kurtosis", "act_top5_share", "context_coherence", "row_norm", "norm_growth", "read_align", "coalition_coherence"], "all": FEATS}
def fit_eval(train, test, fs, H=4):
    def mat(Rs):
        X, Y = [], []
        for R_ in Rs:
            if H not in R_["labels"]: continue
            X.append(torch.stack([ranks(R_["F"][f]) for f in fs], 1)); Y.append(R_["labels"][H].float())
        return torch.cat(X).to(DEV), torch.cat(Y).to(DEV)
    Xtr, Ytr = mat(train); Xte, Yte = mat(test); w = torch.zeros(len(fs), device=DEV, requires_grad=True); b_ = torch.zeros(1, device=DEV, requires_grad=True); opt = torch.optim.LBFGS([w, b_], lr=0.5, max_iter=200); pw = float((Ytr == 0).sum() / Ytr.sum().clamp_min(1))
    def closure():
        opt.zero_grad(); z = Xtr @ w + b_; loss = torch.nn.functional.binary_cross_entropy_with_logits(z, Ytr, pos_weight=torch.tensor(pw, device=DEV)) + 1e-3 * (w ** 2).sum(); loss.backward(); return loss
    with torch.enable_grad(): opt.step(closure)
    with torch.no_grad(): sc = Xte @ w + b_; return auc(sc[Yte == 1], sc[Yte == 0])
even = [R_ for R_ in rows_all if R_["t"] % 2 == 0]; odd = [R_ for R_ in rows_all if R_["t"] % 2 == 1]
for sname, fs in SETS.items():
    a1 = fit_eval(even, odd, fs); a2 = fit_eval(odd, even, fs); res["logistic"][sname] = dict(auc_even_to_odd=a1, auc_odd_to_even=a2, auc_mean=(a1 + a2) / 2 if (a1 is not None and a2 is not None) else None); log(f"{name} logistic on {sname}: AUC {f2(a1)} (even to odd), {f2(a2)} (odd to even)")
L = res["logistic"]; res["increments"] = dict(wdd_adds=L["all"]["auc_mean"] - L["activation+geometry+coalition"]["auc_mean"], activation_side_adds=L["all"]["auc_mean"] - L["wdd"]["auc_mean"]); inc = res["increments"]; A4, A8 = res["auc_by_horizon"]["4"], res["auc_by_horizon"]["8"]; g_ = lambda x: -9 if x is None else x
res["checks"] = dict(wdd_adds=inc["wdd_adds"] >= 0.05 and inc["activation_side_adds"] < 0.03, wdd_leads_all_horizons=all(max(g_(res["auc_by_horizon"][h]["S"]), g_(res["auc_by_horizon"][h]["cnt75"])) >= max(g_(res["auc_by_horizon"][h][f]) for f in FEATS[2:]) for h in ("1", "2", "4", "6", "8")))
summ = f"{name} block {B}: AUC at H=4: " + ", ".join(f"{f} {f2(v)}" for f, v in A4.items()) + "; at H=8: S " + f"{f2(A8['S'])}, breadth {f2(A8['cnt75'])}, best of the rest {f2(max(g_(A8[f]) for f in FEATS[2:]))}" + " | logistic out of sample at H=4: " + ", ".join(f"{k} {f2(v['auc_mean'])}" for k, v in L.items()) + f" | WDD adds {inc['wdd_adds']:+.3f}, activation side adds {inc['activation_side_adds']:+.3f} | checks {_json.dumps(res['checks'])}"
log(summ); record(f"e553_olmo_necessity_{name}", res, summ)
