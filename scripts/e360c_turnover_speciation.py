"""e360c: turnover and speciation of MLP neurons across training (CPU), on the per-checkpoint files of e360 (Pythia) or
e362_<tag> (other models). Argument: the results subdirectory (e360 or e362_olmo1b). Per block and checkpoint:
(1) turnover, the post's and Brendan Long's question: of the final top-5% neurons by loss attribution, the share
already in the top 5% at that step, the step at which each first entered and stayed, and the share of dead neurons
(mean |activation| under 1% of the block median); (2) speciation, cdt's question: spherical k-means (k=16) of the
logit signatures at each step and at the final step on a fixed 1024-neuron subsample; split entropy (how a step's
cluster spreads over final clusters) against merge entropy (how a final cluster draws from the step's clusters), both
in bits, with a label-permutation null; and kNN neighbourhood persistence (k=10) of the signature graph against
chance (10/1023)."""
import sys, os, json, glob, re, torch, time, math
torch.set_num_threads(16); RES = os.environ.get("WDD_RESULTS", "/workspace/wdd/results"); sd = sys.argv[1]
files = glob.glob(os.path.join(RES, sd, "*.pt")); step = lambda f: int(re.search(r"step(\d+)", os.path.basename(f)).group(1))
files = sorted(files, key=step); data = {step(f): torch.load(f) for f in files}; steps = sorted(data); T = max(steps); fin = data[T]; blocks = fin["blocks"]
unit = lambda X: X / X.norm(dim=-1, keepdim=True).clamp_min(1e-9)
def skmeans(X, k=16, seed=0, iters=60, reps=5):
    X = unit(X.float()); g = torch.Generator().manual_seed(seed); best = None
    for r in range(reps):
        C = X[torch.randperm(len(X), generator=g)[:k]].clone()
        for it in range(iters):
            lab = (X @ C.T).argmax(1); C = torch.stack([unit(X[lab == j].sum(0)) if (lab == j).any() else C[j] for j in range(k)])
        score = (X * C[lab]).sum().item()
        if best is None or score > best[0]: best = (score, lab.clone())
    return best[1]
def cond_entropy(a, b, k=16):
    """mean over clusters of a (weighted by size) of the entropy (bits) of b within the cluster"""
    H = 0.0; n = len(a)
    for j in range(k):
        m = a == j; nj = int(m.sum())
        if nj == 0: continue
        p = torch.bincount(b[m], minlength=k).float() / nj; p = p[p > 0]; H += nj / n * float(-(p * p.log2()).sum())
    return H
def knn_persist(Xa, Xb, k=10):
    A = unit(Xa.float()); B = unit(Xb.float()); Ka = A @ A.T; Kb = B @ B.T; Ka.fill_diagonal_(-2); Kb.fill_diagonal_(-2); na = Ka.topk(k, 1).indices; nb = Kb.topk(k, 1).indices
    return float(sum(len(set(na[i].tolist()) & set(nb[i].tolist())) for i in range(len(A))) / (k * len(A)))
g = torch.Generator().manual_seed(0); sub = {l: torch.randperm(fin["sig"][l].shape[0], generator=g)[:1024] for l in blocks}
labT = {l: skmeans(fin["sig"][l][sub[l]]) for l in blocks}; out = dict(subdir=sd, steps=steps, final=T, blocks=blocks, per_step={}, first_entry={})
for l in blocks:
    at = fin["attr"][l]; K = int(0.05 * len(at)); topT = at.argsort(descending=True)[:K]; inTop = {s: set(data[s]["attr"][l].argsort(descending=True)[:K].tolist()) for s in steps}
    fe = []
    for i in topT.tolist():
        e = None
        for s in steps:
            if i in inTop[s]:
                if e is None: e = s
            else: e = None
        fe.append(e if e is not None else T)
    fe = torch.tensor(fe, dtype=torch.float); out["first_entry"][str(l)] = dict(median=float(fe.median()), q25=float(fe.quantile(0.25)), q75=float(fe.quantile(0.75)), share_stable_since_first_quarter=float((fe <= steps[len(steps) // 4]).float().mean()))
for s in steps:
    d = data[s]; row = {}
    for l in blocks:
        at = d["attr"][l]; K = int(0.05 * len(at)); topS = set(at.argsort(descending=True)[:K].tolist()); topT = set(fin["attr"][l].argsort(descending=True)[:K].tolist()); am = d["actmag"][l]
        lab = skmeans(d["sig"][l][sub[l]]); split = cond_entropy(lab, labT[l]); merge = cond_entropy(labT[l], lab); perm = labT[l][torch.randperm(len(lab), generator=g)]
        row[str(l)] = dict(final_top_share=len(topS & topT) / K, dead_share=float((am < 0.01 * am.median()).float().mean()), split_entropy=split, merge_entropy=merge, split_null=cond_entropy(lab, perm), knn_persistence=knn_persist(d["sig"][l][sub[l]], fin["sig"][l][sub[l]]), knn_chance=10 / 1023)
    out["per_step"][str(s)] = row
json.dump(dict(out, _exp=f"e360c_{sd}", _time=time.strftime("%Y-%m-%d %H:%M:%S")), open(os.path.join(RES, f"e360c_turnover_speciation_{sd}.json"), "w"), indent=1)
pick = sorted(set([steps[0], steps[1] if len(steps) > 1 else steps[0], steps[len(steps) // 4], steps[len(steps) // 2], steps[3 * len(steps) // 4], steps[-2] if len(steps) > 1 else steps[-1]]))
line = f"e360c turnover/speciation {sd} ({len(steps)} ckpts): " + " | ".join(f"step{s}: " + " ".join(f"b{l} top5 {out['per_step'][str(s)][str(l)]['final_top_share']:.2f} dead {out['per_step'][str(s)][str(l)]['dead_share']:.2f} split {out['per_step'][str(s)][str(l)]['split_entropy']:.2f}/merge {out['per_step'][str(s)][str(l)]['merge_entropy']:.2f} (null {out['per_step'][str(s)][str(l)]['split_null']:.2f}) knn {out['per_step'][str(s)][str(l)]['knn_persistence']:.2f};" for l in blocks) for s in pick) + " | first entry of final top-5% (median step): " + " ".join(f"b{l} {v['median']:.0f}" for l, v in out["first_entry"].items())
open(os.path.join(RES, "FINDINGS.log"), "a").write(time.strftime("%Y-%m-%d %H:%M:%S") + " " + line + "\n"); print(line)
