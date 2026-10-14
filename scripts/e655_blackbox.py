"""e655 (session 142; Gemini 4's idea 3, against DESIGN_PROXY section 2's "API-only models: nothing"): the final ledger
from the top-M output logprobs alone. The logits z = W_U y are a linear measurement of the post-final-norm state y, so
with open weights and more than D+1 logprobs y is determined exactly (prior art: Finlayson et al. 2024, Carlini et al.
2024); the new question is the underdetermined regime M <= D, where only a sparse prior can help. The reference ledger
is the 16-word OMP description of unit(y - ybar) over the native rows as the unembedding reads them (unit(g(row)), g the
final norm's centring and gain), ybar the mean over wikitext train positions; also reported is its overlap with the
usual hidden-state ledger (OMP of unit(h - mu) over the raw rows) and its depth profile (e128/e198). From the top-M
logprobs (M = 20, 64, 256, 1024, 2048, all) two recoveries: R1, the minimum-norm deviation from the prior [ybar; c0]
(c the unknown log-partition) then OMP; R2 (Gemini's), OMP directly in R^M over the restricted, centred signatures
W_U[top-M] g(row) (M <= 1024). Controls: the same recovery fed another position's observation (the chance level with the
recurring words), R1 from M random tokens instead of the top M, and the reference against another position's reference.
Scored by the state cosine, the support Jaccard, whether the reference's largest word is recovered, and the recall of
its four largest words. Arguments: model [--smoke]."""
from bx_common import *
name = sys.argv[1]; SMOKE = "--smoke" in sys.argv; t0 = time.time(); rng = np.random.default_rng(655)
M = load(name); WU = M["WU"]; V = WU.shape[0]; D = M["D"]; T = 128; NPOS = 32 if SMOKE else 256
A, blk = dictionary(M); Ay = unitr(gain(M, A)); NB = M["NB"]
TR = wikitext(M, "train", 8 if SMOKE else 64, T); TE = wikitext(M, "test", 6 if SMOKE else 24, T)
def states(W):
    Hs = []
    for s0 in range(0, W.shape[0], 4): h, _ = final_states(M, W[s0:s0 + 4]); h = h[:, 4:].reshape(-1, D); Hs.append(h[~sinkmask(h)])
    return torch.cat(Hs)
Htr = states(TR); Hte = states(TE); Hte = Hte[torch.tensor(rng.choice(Hte.shape[0], min(NPOS, Hte.shape[0]), replace=False), device=DEV)]; N = Hte.shape[0]
Ytr = M["norm"](Htr); ybar = Ytr.mean(0); c0 = float(torch.cat([torch.logsumexp(Ytr[s:s + 256] @ WU.T, -1) for s in range(0, Ytr.shape[0], 256)]).mean()); mu = Htr.mean(0)  # chunked: all positions x a 152k vocabulary is 4.5 GB
Y = M["norm"](Hte); LP = (Y @ WU.T).log_softmax(-1)
selT, cofT = omp(unitr(Y - ybar), Ay); selH, _ = omp(unitr(Hte - mu), A)
order = cofT.abs().argsort(1, descending=True); top1 = selT.gather(1, order[:, :1]); top4 = selT.gather(1, order[:, :4])
perm = torch.tensor(np.r_[1:N, 0], device=DEV)  # a fixed derangement: the other position whose observation feeds the control
jac = lambda s1, s2: np.mean([len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(s1.tolist(), s2.tolist())])
res = dict(model=name, n=N, D=D, V=V, reference=dict(jaccard_with_hidden_state_ledger=float(jac(selT, selH)), jaccard_with_other_position=float(jac(selT, selT[perm])),
           share_last4_blocks=float((blk.to(DEV)[selT] >= NB - 4).float().mean()), block_histogram=torch.bincount(blk.to(DEV)[selT].reshape(-1), minlength=NB).tolist()), by_M={})
print(f"[{name}] {N} positions; reference ledger vs hidden-state ledger Jaccard {res['reference']['jaccard_with_hidden_state_ledger']:.3f}, vs another position {res['reference']['jaccard_with_other_position']:.3f}; share from the last 4 blocks {res['reference']['share_last4_blocks']:.2f} ({time.time() - t0:.0f}s)", flush=True)
WU64 = WU.double(); BF = torch.cat([WU64, -torch.ones(V, 1, device=DEV, dtype=torch.float64)], 1); GF = BF.T @ BF; lamF = 1e-9 * float(GF.diagonal().mean())
def r1(idx, lpM):
    """minimum-norm deviation from the prior [ybar; c0] explaining the observed logprobs (c the unknown log-partition), in float64"""
    yb = ybar.double(); lpM = lpM.double()
    if idx.numel() == V:  # every token observed: the normal matrix is the same for every position
        r = torch.empty(V, device=DEV, dtype=torch.float64); r[idx] = lpM - (WU64[idx] @ yb - c0); d = torch.linalg.solve(GF + lamF * torch.eye(D + 1, device=DEV, dtype=torch.float64), BF.T @ r)
    else:
        B = BF[idx]; r = lpM - (WU64[idx] @ yb - c0)
        if B.shape[0] <= D + 1: G = B @ B.T; d = B.T @ torch.linalg.solve(G + 1e-9 * float(G.diagonal().mean()) * torch.eye(B.shape[0], device=DEV, dtype=torch.float64), r)
        else: G = B.T @ B; d = torch.linalg.solve(G + 1e-9 * float(G.diagonal().mean()) * torch.eye(D + 1, device=DEV, dtype=torch.float64), B.T @ r)
    return (yb + d[:D]).float()
def r2(idx, lpM):
    """Gemini's: OMP in R^M over the restricted centred signatures, the offset removed by centring over the M tokens"""
    Wt = WU[idx] - WU[idx].mean(0, keepdim=True); bt = (lpM - lpM.mean()) - Wt @ ybar; Pi = Ay @ Wt.T; nrm = Pi.norm(dim=1).clamp_min(1e-12)
    sel, cof = omp(bt[None], Pi / nrm[:, None], batch=1); return sel[0], ybar + ((cof[0] / nrm[sel[0]])[:, None] * Ay[sel[0]]).sum(0)
def score(sel, yh):
    s, t1, t4 = sel.tolist(), top1.tolist(), top4.tolist()
    return dict(cos=float(torch.nn.functional.cosine_similarity(yh - ybar, Y - ybar).mean()), jaccard=float(jac(sel, selT)), top1=float(np.mean([a[0] in set(b) for a, b in zip(t1, s)])), top4=float(np.mean([len(set(a) & set(b)) / 4 for a, b in zip(t4, s)])))
Ms = [20, 64, 256, 1024, 2048, V]
for m in Ms:
    out = {}
    for meth in ("R1", "R2"):
        if meth == "R2" and m > 1024: continue
        for ctl in ("top", "other_position", "random_tokens"):
            if ctl == "random_tokens" and (meth == "R2" or m == V): continue
            YH, SEL = [], []
            for i in range(N):
                j = int(perm[i]) if ctl == "other_position" else i
                idx = torch.tensor(rng.choice(V, m, replace=False), device=DEV) if ctl == "random_tokens" else LP[j].topk(m).indices
                if meth == "R1": yh = r1(idx, LP[j][idx]); s_, _ = omp(unitr(yh - ybar)[None], Ay); s_ = s_[0]
                else: s_, yh = r2(idx, LP[j][idx])
                YH.append(yh); SEL.append(s_)
            out[f"{meth}_{ctl}"] = score(torch.stack(SEL), torch.stack(YH))
    res["by_M"][str(m)] = out
    print(f"[{name}] M={m}: " + "; ".join(f"{k} cos {v['cos']:.3f} Jaccard {v['jaccard']:.3f} top1 {v['top1']:.2f} top4 {v['top4']:.2f}" for k, v in out.items()) + f" ({time.time() - t0:.0f}s)", flush=True)
bm = res["by_M"]
res["verdicts"] = dict(B1=bm["20"]["R2_top"]["top1"] >= 0.5, B2=bm["20"]["R2_top"]["jaccard"] - bm["20"]["R2_other_position"]["jaccard"] >= 0.10,
                       B3=bm["64"]["R2_top"]["jaccard"] > bm["64"]["R1_top"]["jaccard"] and bm["256"]["R2_top"]["jaccard"] > bm["256"]["R1_top"]["jaccard"],
                       B4=bm["2048"]["R1_top"]["cos"] >= 0.99 and bm["2048"]["R1_top"]["jaccard"] >= 0.95, B5=res["reference"]["jaccard_with_hidden_state_ledger"] >= 0.5, B6=res["reference"]["share_last4_blocks"] >= 0.6)
summ = (f"[{name}] the final ledger from top-M logprobs, {N} positions: support Jaccard with the reference ledger, R1 (min-norm then OMP) / R2 (OMP in logit space) / other-position control: "
        + "; ".join(f"M={m} " + "/".join(f"{bm[str(m)][k]['jaccard']:.2f}" if k in bm[str(m)] else "-" for k in ("R1_top", "R2_top", "R1_other_position")) for m in Ms)
        + f"; reference vs hidden-state ledger {res['reference']['jaccard_with_hidden_state_ledger']:.2f}, last-4-block share {res['reference']['share_last4_blocks']:.2f}; verdicts {res['verdicts']} ({time.time() - t0:.0f}s)")
save(f"e655_blackbox_{name}" + ("_smoke" if SMOKE else ""), res, summ); print("done", flush=True)
