"""Shared loaders for session 101 (e556-e564): language-model states after a block with the MLP write rows of blocks
0..B as the dictionary (rows only, as in sessions 89-100), the OMP statistics of ex_common, the floor for new states
from a calibration on the Pile states, and the three-dictionary comparison (native rows, the rows rotated, Gaussian
atoms with the rows' second moment)."""
import sys, os, time, math, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ex_common import *
import wdd_common
from wdd_common import load_model, Arch, omp, refit, rotate, log, record, DEV
from ma_common import block_states, sinkmask
from lr_common import eval_ids
wdd_common.MODELS.update({"pythia160": ("EleutherAI/pythia-160m", "neox"), "pythia160d": ("EleutherAI/pythia-160m-deduped", "neox")})
MID = {"pythia410": 12, "pythia160": 6, "pythia160d": 6, "gpt2": 6, "olmo1b": 8}
K = 16

def rows_of(arch, B):
    R = torch.cat([arch.wdir(b) for b in range(B + 1)]); return unitr(R), R.norm(dim=1)

def pile_ids(name, nseq=8, T=256, start=0):
    key = "pythia410" if name.startswith("pythia") else name
    return eval_ids(key)[start:start + nseq, :T].to(DEV)

def lm_states(name, B=None, revision=None, ids=None, model=None, nseq=8, T=256):
    """states after block B at positions 1: of each sequence, sinks dropped, centred and unit; the unit rows of blocks
    0..B and their norms; the token ids at the kept positions."""
    B = MID[name] if B is None else B
    if model is None: model, tok, fam = load_model(name, revision=revision)
    else: fam = wdd_common.MODELS[name][1]
    arch = Arch(model, fam)
    if ids is None: ids = pile_ids(name, nseq, T)
    X = block_states(model, arch, ids, [B])[B].reshape(-1, arch.D); keep = ~sinkmask(X)
    A, norms = rows_of(arch, B); mu = X[keep].mean(0); U = unitr(X[keep] - mu)
    return dict(X=X, keep=keep, U=U, A=A, norms=norms, ids=ids, model=model, arch=arch, mu=mu, B=B, tok=ids[:, 1:].reshape(-1)[keep])

def floor_calibration(U, A, seed=11):
    """r_cal, gm and the rows' second moment CA from a set of unit states (the same recipe as ex_common.stats), so that
    the floor of new states is L = sqrt(u' CA u) * gm * r_cal."""
    m = A.shape[0]; Ar = unitr(rotate(A, seed=seed)); CA = A.T @ A / m; CR = Ar.T @ Ar / m; gm = gabs(m); N = U.shape[0]
    s2r = ((U @ CR) * U).sum(1); Mr = torch.cat([(U[s:s + 256] @ Ar.T).abs().max(1).values for s in range(0, N, 256)])
    r_cal = float(Mr.mean() / (s2r.clamp_min(1e-12).sqrt() * gm).mean()); return dict(r_cal=r_cal, gm=gm, CA=CA)

def floor_of(U, cal): return (((U @ cal["CA"]) * U).sum(1)).clamp_min(1e-12).sqrt() * cal["gm"] * cal["r_cal"]

def dict_compare(U, A, Kk=16, with_stats=True):
    """FVU at K with the native rows, the rotated rows and Gaussian atoms with the rows' second moment, the top-256 usage
    share of each, and (native only) the extreme-value statistics of the 256 words."""
    m, D = A.shape; out = {}; tot = float(U.pow(2).sum())
    ev, V = torch.linalg.eigh((A.T @ A / m).double()); Rm = ((V * ev.clamp_min(0).sqrt()) @ V.T).float()
    g = torch.Generator(device="cpu").manual_seed(13); G = unitr(torch.randn(m, D, generator=g).to(DEV) @ Rm)
    for nm, Dd in (("own", A), ("rot", unitr(rotate(A, seed=7))), ("covA", G)):
        sel, cof, err = omp(U, Dd, Kk, batch=1024, record_err=True); out[f"{nm}_fvu"] = float(err[:, -1].sum() / tot); out[f"{nm}_fvu_k4"] = float(err[:, 3].sum() / tot)
        usage = torch.bincount(sel.reshape(-1), minlength=m).float(); us = usage.sort(descending=True).values
        out[f"{nm}_top256_usage_share"] = float(us[:256].sum() / us.sum()); out[f"{nm}_atoms_used"] = int((usage > 0).sum())
    out["advantage_fvu"] = out["rot_fvu"] - out["own_fvu"]; out["advantage_over_covA"] = out["covA_fvu"] - out["own_fvu"]
    if with_stats:
        st = stats(U, A, Kk); w = wordset(st["usage"]); Sw = st["S"][w]
        out["words"] = dict(n=int(w.sum()), median_S=float(Sw.median()), q10_S=float(Sw.quantile(0.1)), share_S_over_1=float((Sw >= 1).float().mean()), median_cnt=float(st["cnt"][w].median()),
                            median_breadth=float((st["cnt75"][w] / st["cnt"][w].clamp_min(1)).median()), rows_over_floor_anywhere=int((st["S"] >= 1).sum()), median_S_all=float(st["S"].median()),
                            usage_threshold=float(st["usage"][w].min()), median_cut=float(st["cut"].median()))
        str_ = stats(U, unitr(rotate(A, seed=7)), Kk); wr = wordset(str_["usage"]); Sr = str_["S"][wr]
        out["rotated_words"] = dict(median_S=float(Sr.median()), share_S_over_1=float((Sr >= 1).float().mean()), rows_over_floor_anywhere=int((str_["S"] >= 1).sum()), median_S_all=float(str_["S"].median()))
        out["_st"] = st
    return out

def summarize_compare(tag, c):
    w, r = c.get("words"), c.get("rotated_words")
    s = f"{tag}: FVU at K=16 own/rot/covA {c['own_fvu']:.3f}/{c['rot_fvu']:.3f}/{c['covA_fvu']:.3f} (advantage {c['advantage_fvu']:.3f}), top-256 usage share own/rot/covA {c['own_top256_usage_share']:.2f}/{c['rot_top256_usage_share']:.2f}/{c['covA_top256_usage_share']:.2f}"
    if w: s += f"; 256 words: median S {w['median_S']:.2f} (q10 {w['q10_S']:.2f}, share over the floor {w['share_S_over_1']:.2f}, median count {w['median_cnt']:.0f}, breadth {w['median_breadth']:.2f}), rows over the floor anywhere {w['rows_over_floor_anywhere']} vs rotated {r['rows_over_floor_anywhere']} (rotated words' median S {r['median_S']:.2f})"
    return s

def auc(score, y):
    """area under the ROC curve by the rank statistic (ties averaged)"""
    score = torch.as_tensor(score, dtype=torch.float64); y = torch.as_tensor(y).bool(); n1 = int(y.sum()); n0 = int((~y).sum())
    if n1 == 0 or n0 == 0: return None
    r = torch.empty_like(score); o = score.argsort(); s = score[o]; ranks = torch.arange(1, len(s) + 1, dtype=torch.float64)
    # average ranks over ties
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]: j += 1
        ranks[i:j + 1] = (i + j + 2) / 2; i = j + 1
    r[o] = ranks; return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def logreg(Xtr, ytr, Xte, l2=1e-2, iters=300):
    """L2 logistic regression on standardized features (LBFGS); returns scores on Xte"""
    mu = Xtr.mean(0); sd = Xtr.std(0).clamp_min(1e-6); Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    with torch.enable_grad():
        w = torch.zeros(Xtr.shape[1], device=Xtr.device, requires_grad=True); b = torch.zeros(1, device=Xtr.device, requires_grad=True); yt = ytr.float()
        opt = torch.optim.LBFGS([w, b], lr=0.5, max_iter=iters, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad(); loss = torch.nn.functional.binary_cross_entropy_with_logits(Xtr @ w + b, yt) + l2 * w.pow(2).sum(); loss.backward(); return loss
        opt.step(closure)
    return (Xte @ w + b).detach()
