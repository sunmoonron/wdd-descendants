"""e666 (session 148; Gemini 4's baseline for the provenance core). e632 part C: given a class (the positions where the
eventual word's row is over the floor at step 16000), two checkpoints before recruitment the recruit is in the top ten
of the non-word rows by mean class ratio for 0.77 to 0.84 of the eventual words on Pythia-160m seeds 0, 1 and 2
(median rank 1), against 0.22 to 0.31 by usage and 0.02 to 0.04 by an autoencoder's decoder; the activation side was
never ranked given the class. Here, at the same checkpoint and among the same non-word rows, the recruit's rank by: the
mean class ratio (e632, reproduced); the cosine of its write row with the class direction at that checkpoint (e629's
route); its activation selectivity at the class positions (the z-score of e632 part A); its mean activation there; its
direct write along the class direction, the mean over the class of a_j(t) w_j . d_C (e664b's baseline, Gemini's
formula). Controls: the same ranks at 16000, where the class is defined (each score's ceiling); a class defined by the
activation side instead (the positions where the eventual row fires most at 16000, as many as its projection class),
which hands the home advantage to selectivity; the recruit's own direct write as a share of the class states'
projection onto its row (e546). The vectors went with the box, so e629's longitudinal cache is rebuilt from the
checkpoints: states after block 6 on the same 24 Pile sequences of 512 tokens, the MLP write rows of blocks 0..6, the
words read from the e629 tables. Arguments: tag (pythia160 | pythia160s1 | pythia160s2) [--smoke]. Pre-registered in
e666_prereg.json."""
import sys, os, time, json, math
import torch
from transformers import AutoModelForCausalLM
torch.set_grad_enabled(False); DEV = "cuda"; t0 = time.time()
tag = sys.argv[1]; SMOKE = "--smoke" in sys.argv; sfx = "_smoke" if SMOKE else ""
HF = {"pythia160": "EleutherAI/pythia-160m", "pythia160s1": "EleutherAI/pythia-160m-seed1", "pythia160s2": "EleutherAI/pythia-160m-seed2"}[tag]
REF = {"pythia160": dict(N=12253, n_tracked=194, top10=0.835), "pythia160s1": dict(N=12250, n_tracked=193, top10=0.782), "pythia160s2": dict(N=12255, n_tracked=185, top10=0.768)}[tag]   # e632's recorded part C
B = 6; NSEQ = 8 if SMOKE else 24; T = 512; LATE = list(range(1000, 16001, 1000)); last = LATE[-1]
CD = f"cache/e666_{tag}{sfx}"; os.makedirs(CD, exist_ok=True); os.makedirs("results", exist_ok=True)
METHODS = ("ratio", "cos", "sel", "act", "dc", "dc_abs")
def log(s): print(f"[{time.strftime('%H:%M:%S')}] {s} ({time.time() - t0:.0f}s)", flush=True)
# ---------------- the helpers of s101_common / ex_common / wdd_common, verbatim in effect
def unitr(X): return X / X.norm(dim=-1, keepdim=True).clamp_min(1e-12)
def rotate(A, seed=7):
    g = torch.Generator(device="cpu").manual_seed(seed); Q, _ = torch.linalg.qr(torch.randn(A.shape[1], A.shape[1], generator=g)); return A @ Q.to(A.device)
def gabs(m, Tt=14.0, n=28001):
    t = torch.linspace(0, Tt, n, dtype=torch.float64); F = torch.special.erf(t / math.sqrt(2)); return float(torch.trapezoid(1 - F.pow(m), t))
def floor_calibration(U, A, seed=11):
    m = A.shape[0]; Ar = unitr(rotate(A, seed=seed)); CA = A.T @ A / m; CR = Ar.T @ Ar / m; gm = gabs(m); N = U.shape[0]
    s2r = ((U @ CR) * U).sum(1); Mr = torch.cat([(U[s:s + 256] @ Ar.T).abs().max(1).values for s in range(0, N, 256)])
    r_cal = float(Mr.mean() / (s2r.clamp_min(1e-12).sqrt() * gm).mean()); return dict(r_cal=r_cal, gm=gm, CA=CA)
def floor_of(U, cal): return (((U @ cal["CA"]) * U).sum(1)).clamp_min(1e-12).sqrt() * cal["gm"] * cal["r_cal"]
def sinkmask(x, f=10.0):
    n = x.norm(dim=-1); return n > f * n.median()
def med(v): return float(torch.tensor(v, dtype=torch.float64).median()) if len(v) else None
def mean(v): return float(torch.tensor(v, dtype=torch.float64).mean()) if len(v) else None
def first_hold(flags):
    for i in range(len(flags)):
        if flags[i] and all(flags[i:]): return i
    return None
# ---------------- 1. the longitudinal cache (e629's: block-6 states at positions 1:, sinks marked, unit write rows and norms)
ids = torch.load("eval_ids_all.pt", weights_only=False)["pythia410"]["eval_ids"][:NSEQ, :T].to(DEV)
class Stop(Exception): pass
def load(n): return AutoModelForCausalLM.from_pretrained(HF, revision=f"step{n}", dtype=torch.float32).to(DEV).eval()
def forward(model, want_act):
    layers = model.gpt_neox.layers; Xs, As = [], []
    for s0 in range(0, ids.shape[0], 4):
        cap = {}
        def hk(mm, i, o):
            cap["x"] = (o[0] if isinstance(o, tuple) else o)[:, 1:].detach().float(); raise Stop
        hs = [layers[B].register_forward_hook(hk)]
        if want_act: hs += [layers[b].mlp.dense_4h_to_h.register_forward_pre_hook((lambda b_: lambda mm, a: cap.__setitem__(b_, a[0].detach()[:, 1:].half()))(b)) for b in range(B + 1)]
        try: model(ids[s0:s0 + 4])
        except Stop: pass
        finally: [h.remove() for h in hs]
        Xs.append(cap["x"].reshape(-1, cap["x"].shape[-1]))
        if want_act: As.append(torch.cat([cap[b].reshape(-1, cap[b].shape[-1]) for b in range(B + 1)], 1))
    return torch.cat(Xs), (torch.cat(As) if want_act else None)
for n in LATE:
    f = f"{CD}/step{n}.pt"
    if os.path.exists(f): continue
    model = load(n); X, _ = forward(model, False)
    W = torch.cat([model.gpt_neox.layers[b].mlp.dense_4h_to_h.weight.detach().float().T for b in range(B + 1)])
    torch.save(dict(X=X.half().cpu(), keep=(~sinkmask(X)).cpu(), rows=unitr(W).half().cpu(), norms=W.norm(dim=1).cpu()), f + ".tmp"); os.replace(f + ".tmp", f)
    del model, X, W; torch.cuda.empty_cache(); log(f"cached step {n}")
ck = {n: torch.load(f"{CD}/step{n}.pt") for n in LATE}; keepc = torch.stack([ck[n]["keep"] for n in LATE]).all(0); kd = keepc.to(DEV); N = int(keepc.sum())
U, MU = {}, {}
for n in LATE:
    Xn = ck[n]["X"][keepc].float().to(DEV); MU[n] = Xn.mean(0); U[n] = unitr(Xn - MU[n]); del Xn
m = ck[last]["rows"].shape[0]; DFF = m // (B + 1)
tab = torch.load(f"e629_tables/e629_scores_{tag}_B6.pt", weights_only=False); SCw = {int(k): v for k, v in tab["scores"].items()}; assert all(n in SCw for n in LATE)
WORDS = {n: SCw[n]["words"].bool().cpu() for n in LATE}; USAGE = {n: SCw[n]["usage"].float().cpu() for n in LATE}; assert WORDS[last].numel() == m
CAL = {}
def cal_of(n):
    if n not in CAL: CAL[n] = floor_calibration(U[n], ck[n]["rows"].float().to(DEV))
    return CAL[n]
# ---------------- 2. e632's eventual words and their classes at 16000
Al = ck[last]["rows"].float().to(DEV); Ll = floor_of(U[last], cal_of(last))
R16 = torch.cat([(U[last][s:s + 1024] @ Al.T).abs() / Ll[s:s + 1024, None] for s in range(0, N, 1024)]).half().cpu(); del Al
wl = torch.nonzero(WORDS[last])[:, 0]; csize = (R16[:, wl].float() > 1).sum(0); tracked = []
for w in wl[(csize >= 10) & ~WORDS[LATE[0]][wl]].tolist():
    j = first_hold([bool(WORDS[n][w]) for n in LATE])
    if j is not None and j >= 2: tracked.append((w, j))
CLS = {w: torch.nonzero(R16[:, w].float() > 1)[:, 0] for w, _ in tracked}; del R16
log(f"{tag}: {N} positions (e632 {REF['N']}), {m} rows, {len(tracked)} eventual words recruited at index 2 or later (e632 {REF['n_tracked']})")
# ---------------- 3. at 16000 (first, for the activation-defined classes) and two checkpoints before each recruitment
def scores(n, pos, A, nr, act, mu, sd):
    p = pos.to(DEV); Up = U[n][p]; L = floor_of(Up, cal_of(n)); ratio = torch.cat([((Up[s:s + 512] @ A.T).abs() / L[s:s + 512, None]).sum(0, keepdim=True) for s in range(0, p.numel(), 512)]).sum(0) / p.numel()
    d = unitr(Up.mean(0, keepdim=True))[0]; cos = A @ d; am = act[p].mean(0); dc = am * nr * cos
    return dict(ratio=ratio, cos=cos, sel=(am - mu) / sd, act=am, dc=dc, dc_abs=dc.abs())
NEED = [last] + sorted({LATE[j - 2] for _, j in tracked}); PW = {w: dict(step=LATE[j], two_before=LATE[j - 2]) for w, j in tracked}; CLS_A = {}
for n in NEED:
    model = load(n); _, a0 = forward(model, True); del model; torch.cuda.empty_cache()
    act = a0[kd].float(); del a0; mu = act.mean(0); sd = act.std(0).clamp_min(1e-6); A = ck[n]["rows"].float().to(DEV); nr = ck[n]["norms"].to(DEV); Xc = ck[n]["X"][keepc].float().to(DEV) - MU[n]
    if n == last: CLS_A = {w: act[:, w].topk(int(CLS[w].numel())).indices.cpu() for w, _ in tracked}
    for w, j in tracked:
        n2 = LATE[j - 2]
        if n not in (n2, last): continue
        lab = "two_before" if n == n2 else "final"; nonw = ~WORDS[n2]
        for cname, pos in (("proj", CLS[w]), ("act", CLS_A[w])):
            S = {k: v.cpu() for k, v in scores(n, pos, A, nr, act, mu, sd).items()}
            PW[w][f"{lab}_{cname}"] = dict(ranks={k: int(((v > v[w]) & nonw).sum()) + 1 for k, v in S.items()}, vals={k: float(v[w]) for k, v in S.items()})
        p = CLS[w].to(DEV); own = float(((act[p, w] - mu[w]) * nr[w]).mean()); tot = float((Xc[p] @ A[w]).mean())
        PW[w][f"{lab}_own"] = dict(own=own, total=tot, share=own / tot if abs(tot) > 1e-9 else None, cos_dir_row=float(unitr(U[n][p].mean(0, keepdim=True))[0] @ A[w]))
        if lab == "two_before": PW[w]["rank_usage"] = int(((USAGE[n2] > USAGE[n2][w]) & nonw).sum()) + 1
    del act, A, nr, Xc; torch.cuda.empty_cache(); log(f"step {n} done")
for w, _ in tracked:
    a_, p_ = set(CLS_A[w].tolist()), set(CLS[w].tolist()); PW[w]["jaccard_act_proj"] = len(a_ & p_) / len(a_ | p_)
# ---------------- 4. summaries
def summ(lab, cname):
    out = {}
    for k in METHODS:
        r = [PW[w][f"{lab}_{cname}"]["ranks"][k] for w, _ in tracked]
        out[k] = dict(median=med(r), top1=mean([float(x == 1) for x in r]), top10=mean([float(x <= 10) for x in r]), top100=mean([float(x <= 100) for x in r]))
    for k in METHODS[1:]:
        a_ = [PW[w][f"{lab}_{cname}"]["ranks"]["ratio"] for w, _ in tracked]; b_ = [PW[w][f"{lab}_{cname}"]["ranks"][k] for w, _ in tracked]
        out[k]["ratio_better"] = mean([float(x < y) for x, y in zip(a_, b_)]); out[k]["ratio_worse"] = mean([float(x > y) for x, y in zip(a_, b_)])
    return out
R = {f"{lab}_{c}": summ(lab, c) for lab in ("two_before", "final") for c in ("proj", "act")}
OWN = {lab: dict(median_share=med([PW[w][f"{lab}_own"]["share"] for w, _ in tracked if PW[w][f"{lab}_own"]["share"] is not None]), share_below_half=mean([float(PW[w][f"{lab}_own"]["share"] is not None and PW[w][f"{lab}_own"]["share"] < 0.5) for w, _ in tracked]),
              positive_total=mean([float(PW[w][f"{lab}_own"]["total"] > 0) for w, _ in tracked]), median_cos_dir_row=med([PW[w][f"{lab}_own"]["cos_dir_row"] for w, _ in tracked])) for lab in ("two_before", "final")}
usage = [PW[w]["rank_usage"] for w, _ in tracked]; USG = dict(median=med(usage), top10=mean([float(x <= 10) for x in usage]))
res = dict(model=tag, smoke=SMOKE, N=N, m=m, n_tracked=len(tracked), reference=REF, ranks=R, own=OWN, usage=USG, jaccard_act_proj_median=med([PW[w]["jaccard_act_proj"] for w, _ in tracked]),
           class_size_median=med([int(CLS[w].numel()) for w, _ in tracked]), per_word={str(w): v for w, v in PW.items()})
r2, f2, ra = R["two_before_proj"], R["final_proj"], R["two_before_act"]
summary = (f"[{tag}] {len(tracked)} eventual words (e632 {REF['n_tracked']}), {N} positions; two checkpoints before recruitment, among the non-word rows, given the projection class: top 10 (median rank) by class ratio {r2['ratio']['top10']:.2f} ({r2['ratio']['median']:.0f}; e632 {REF['top10']:.2f}), by write cosine {r2['cos']['top10']:.2f} ({r2['cos']['median']:.0f}), by selectivity {r2['sel']['top10']:.2f} ({r2['sel']['median']:.0f}), by mean activation {r2['act']['top10']:.2f} ({r2['act']['median']:.0f}), by direct write {r2['dc']['top10']:.2f} ({r2['dc']['median']:.0f}), |direct write| {r2['dc_abs']['top10']:.2f}, usage {USG['top10']:.2f} ({USG['median']:.0f}); "
           f"ceiling at 16000: ratio {f2['ratio']['top10']:.2f}, cosine {f2['cos']['top10']:.2f}, selectivity {f2['sel']['top10']:.2f}, direct write {f2['dc']['top10']:.2f}; activation-defined class (Jaccard with the projection class {res['jaccard_act_proj_median']:.2f}) two before: ratio {ra['ratio']['top10']:.2f}, cosine {ra['cos']['top10']:.2f}, selectivity {ra['sel']['top10']:.2f}, direct write {ra['dc']['top10']:.2f}; "
           f"the recruit's own direct write as a share of the class projection onto its row {OWN['two_before']['median_share']:.2f} two before, {OWN['final']['median_share']:.2f} at 16000")
res["summary"] = summary; res["_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
json.dump(res, open(f"results/e666_provenance_{tag}{sfx}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); log(summary); print("done", flush=True)
