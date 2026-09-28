"""e532: what property of the native rows makes them persistent extreme axes? e531 found the prospective criterion four
fifths generic and the native directions' distinction to be stability (half the word set kept over four thousand
steps against a sixth for random directions), carried by the directions at one checkpoint. Two controls it lacked:
random atoms with the native dictionary's own second-order structure (drawn from a Gaussian with the dictionary's
covariance, so that the anisotropy of the native rows is kept and only their individual orientations are lost), and
random atoms with the state cloud's covariance (atoms concentrated where the states are). If either matches the native
retention, the native advantage is second-order geometry; if neither does, it is in the individual axes. Alongside,
the half-life of every dictionary's words (retention at one, two, four and eight thousand steps) and the temporal
kernel of native dictionaries frozen at 1000, 4000, 8000 and 16000, each evaluated on the states of every checkpoint,
with the overlap of the words it finds and the words the contemporaneous native dictionary finds.
Setup as e531: sixteen Pythia checkpoints a thousand steps apart, 8 x 256 measurement tokens, OMP with 16 atoms,
words the 256 most-used row atoms, S with each dictionary's own floor; blocks 12 and 6.
Pre-registered (honest guesses), block 12, four thousand steps:
- covariance-matched random atoms keep 0.25-0.35 of their words, between isotropic random (0.17) and native (0.50) (0.5);
- state-covariance random atoms keep no more than the dictionary-covariance ones (0.5);
- neither matched control reaches within 0.10 of the native retention (0.6);
- the frozen native dictionaries' word sets overlap the contemporaneous native words at 0.5 or more up to four
  thousand steps away in either direction and decay beyond (0.5).
Arguments: name."""
import sys, os, json as _json, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; CDIR = f"/workspace/wdd/cache/e524_{name}"; BL = [12, 6]; LB = 12; NWORD = 256; K = 16; FREEZE = [1000, 4000, 8000, 16000]; HS = [1000, 2000, 4000, 8000]
steps = list(range(1000, 16001, 1000)); ck = {n: torch.load(f"{CDIR}/step{n}.pt") for n in steps}
idsB = eval_ids(name)[:8, :256].to(DEV)
def gabs(m, T=14.0, n=28001):
    t = torch.linspace(0, T, n, dtype=torch.float64); F = torch.special.erf(t / math.sqrt(2)); return float(torch.trapezoid(1 - F.pow(m), t))
def auc(pos, neg):
    if pos.numel() < 5 or neg.numel() < 5: return None
    x = torch.cat([pos, neg]); r = x.argsort().argsort().double() + 1; npos, nneg = pos.numel(), neg.numel(); return float((r[:npos].sum() - npos * (npos + 1) / 2) / (npos * nneg))
def spearman(a, b):
    ra = a.argsort().argsort().double(); rb = b.argsort().argsort().double(); ra = ra - ra.mean(); rb = rb - rb.mean(); return float((ra * rb).sum() / (ra.norm() * rb.norm()).clamp_min(1e-12))
Xs = {}; AD = {}
for n in steps:
    model, tok, fam = load_model(name, revision=f"step{n}"); arch = Arch(model, fam); D, DFF = arch.D, arch.DFF
    for p in model.parameters(): p.requires_grad_(False)
    X0 = block_states(model, arch, idsB, BL, chunk=4); Xs[n] = {b: X0[b].reshape(-1, D) for b in BL}
    if n in FREEZE: A, lab = build_dictionary(arch, blocks=list(range(LB + 1))); AD[n] = (unitr(A), lab); del A
    del model, X0; torch.cuda.empty_cache()
log(f"{name}: states at {len(steps)} checkpoints; dictionaries frozen at {FREEZE}")
gen = torch.Generator(device=DEV).manual_seed(0); A8, lab8 = AD[8000]; m = A8.shape[0]
def gaussian_atoms(C):
    ev, V = torch.linalg.eigh(C.double()); ev = ev.clamp_min(0); z = torch.randn(m, D, device=DEV, generator=gen, dtype=torch.float64); return unitr((z * ev.sqrt()[None]) @ V.T).float()
X8 = Xs[8000][12]; kp8 = ~sinkmask(X8); Xc8 = X8[kp8] - X8[kp8].mean(0)
DICTS = {f"frozen_{n}": AD[n] for n in FREEZE}
DICTS.update({"rotated": (unitr(rotate(A8, seed=7)), lab8), "random_isotropic": (unitr(torch.randn(m, D, device=DEV, generator=gen)), lab8), "random_dictionary_covariance": (gaussian_atoms(A8.T @ A8 / m), lab8), "random_state_covariance": (gaussian_atoms(Xc8.T @ Xc8 / Xc8.shape[0]), lab8)})
del AD
def restrict(Au, lab, b):
    typ = lab["type"].to(DEV); blk = lab["block"].to(DEV); idx = lab["index"].to(DEV); mk = (blk <= b); Ab = Au[mk]; typb, blkb, idxb = typ[mk], blk[mk], idx[mk]
    gidx = torch.full((b + 1, DFF), -1, device=DEV, dtype=torch.long)
    for bb in range(b + 1): mm = (typb == T_MLP) & (blkb == bb); gidx[bb, idxb[mm]] = torch.nonzero(mm)[:, 0]
    return Ab, gidx.reshape(-1)
PRE = {}
for k, (Au, lab) in DICTS.items():
    PRE[k] = {}
    for b in BL:
        Ab, rows = restrict(Au, lab, b); mb = Ab.shape[0]; Ar = unitr(rotate(Ab, seed=11)); PRE[k][b] = dict(A=Ab, rows=rows, CA=Ab.T @ Ab / mb, CR=Ar.T @ Ar / mb, Ar=Ar, gm=gabs(mb), Am=Ab[rows])
USG = {k: {b: {} for b in BL} for k in DICTS}; SS = {k: {b: {} for b in BL} for k in DICTS}
for n in steps:
    for b in BL:
        X = Xs[n][b]; keep = ~sinkmask(X); Xk = X[keep]; N = Xk.shape[0]; mu = Xk.mean(0); Xc = Xk - mu; U = unitr(Xc)
        for k in DICTS:
            d = PRE[k][b]; sel, _, _ = omp(Xc, d["A"], K, batch=1024, record_err=False); usage = torch.bincount(sel.reshape(-1), minlength=d["A"].shape[0])[d["rows"]].float()
            s2 = ((U @ d["CA"]) * U).sum(1); s2r = ((U @ d["CR"]) * U).sum(1); Mr = torch.cat([(U[s:s + 128] @ d["Ar"].T).abs().max(1).values for s in range(0, N, 128)])
            r_cal = float(Mr.mean() / (s2r.clamp_min(1e-12).sqrt() * d["gm"]).mean()); L = s2.clamp_min(1e-12).sqrt() * d["gm"] * r_cal
            P = U @ d["Am"].T; S = (P.abs() / L[:, None]).max(0).values; USG[k][b][n] = usage.cpu(); SS[k][b][n] = S.cpu(); del P
    log(f"{name} step{n}: usage and S for {len(DICTS)} dictionaries")
USG["native"] = {b: {n: ck[n]["blocks"][b]["usage"].float() for n in steps} for b in BL}; SS["native"] = {b: {n: ck[n]["blocks"][b]["S"].float() for n in steps} for b in BL}
def wordset(u):
    w = torch.zeros(u.numel(), dtype=torch.bool); w[torch.nonzero(u > 0)[:, 0][u[u > 0].argsort(descending=True)[:NWORD]]] = True; return w
res = dict(model=name, steps=steps, dictionaries=list(USG), by_block={})
for b in BL:
    W = {k: {n: wordset(USG[k][b][n]) for n in steps} for k in USG}; out = {}
    for k in USG:
        per = {}
        for H in HS:
            aucs, nents, pers, ovl = [], [], [], []
            for n in steps:
                if n + H not in W[k]: continue
                S0 = SS[k][b][n]; nonw = ~W[k][n]; ent = nonw & W[k][n + H]; a = auc(S0[ent], S0[nonw & ~ent])
                if a is not None: aucs.append(a)
                nents.append(int(ent.sum())); pers.append(spearman(S0, SS[k][b][n + H])); ovl.append(float((W[k][n] & W[k][n + H]).sum() / NWORD))
            per[H] = dict(prospective_auc_mean=float(sum(aucs) / len(aucs)) if aucs else None, entries_mean=float(sum(nents) / len(nents)), persistence_of_S_mean=float(sum(pers) / len(pers)), retention_mean=float(sum(ovl) / len(ovl)))
        out[k] = per
    kernel = {f"frozen_{n0}": {n: float((W[f"frozen_{n0}"][n] & W["native"][n]).sum() / NWORD) for n in steps} for n0 in FREEZE}
    res["by_block"][b] = dict(per_dictionary=out, kernel_overlap_with_contemporaneous_native_words=kernel)
    fm = lambda x: "n/a" if x is None else f"{x:.2f}"
    log(f"{name} block {b}: retention of the word set at 1000/2000/4000/8000 steps (prospective AUC at 4000; entries per 4000; persistence of S at 4000): " + " | ".join(f"{k} " + "/".join(f"{out[k][H]['retention_mean']:.2f}" for H in HS) + f" ({fm(out[k][4000]['prospective_auc_mean'])}; {out[k][4000]['entries_mean']:.0f}; {out[k][4000]['persistence_of_S_mean']:.2f})" for k in USG))
    log(f"{name} block {b}: overlap of a frozen native dictionary's words with the contemporaneous native words, by checkpoint: " + " | ".join(f"frozen at {n0}: " + " ".join(f"{kernel[f'frozen_{n0}'][n]:.2f}" for n in steps) for n0 in FREEZE))
P12 = res["by_block"][12]["per_dictionary"]; r4 = lambda k: P12[k][4000]["retention_mean"]; kern = res["by_block"][12]["kernel_overlap_with_contemporaneous_native_words"]
res["checks"] = dict(dict_cov_between=0.25 <= r4("random_dictionary_covariance") <= 0.35, state_cov_no_more=r4("random_state_covariance") <= r4("random_dictionary_covariance") + 0.02, neither_within_0_10=max(r4("random_dictionary_covariance"), r4("random_state_covariance")) < r4("native") - 0.10, kernel_half_within_4000=all(kern[f"frozen_{n0}"][n] >= 0.5 for n0 in FREEZE for n in steps if abs(n - n0) <= 4000))
fm = lambda x: "n/a" if x is None else f"{x:.2f}"
summ = f"{name}: " + " | ".join(f"block {b}: retention at 4000 steps native {o['per_dictionary']['native'][4000]['retention_mean']:.2f}, frozen 4000 {o['per_dictionary']['frozen_4000'][4000]['retention_mean']:.2f}, rotated {o['per_dictionary']['rotated'][4000]['retention_mean']:.2f}, isotropic random {o['per_dictionary']['random_isotropic'][4000]['retention_mean']:.2f}, dictionary-covariance random {o['per_dictionary']['random_dictionary_covariance'][4000]['retention_mean']:.2f}, state-covariance random {o['per_dictionary']['random_state_covariance'][4000]['retention_mean']:.2f}; prospective AUC at 4000 " + "/".join(fm(o['per_dictionary'][k][4000]['prospective_auc_mean']) for k in ('native', 'frozen_4000', 'rotated', 'random_isotropic', 'random_dictionary_covariance', 'random_state_covariance')) + "; half-life: native retention at 1000/2000/4000/8000 " + "/".join(f"{o['per_dictionary']['native'][H]['retention_mean']:.2f}" for H in HS) + ", isotropic " + "/".join(f"{o['per_dictionary']['random_isotropic'][H]['retention_mean']:.2f}" for H in HS) for b, o in res["by_block"].items()) + f" | checks {_json.dumps(res['checks'])}"
log(summ); record(f"e532_covariance_matched_{name}", res, summ)
