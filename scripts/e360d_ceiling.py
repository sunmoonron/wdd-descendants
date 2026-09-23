"""e360d (CPU): reliability ceilings for the e360 logit signatures and the reliability-corrected drift curves. For each
checkpoint with a second, independently seeded signature estimate (e360_ceil/<rev>_1.pt: same weights, different random
assignment of neurons to tokens): per block, the median per-neuron cosine between the two estimates (the ceiling for
any signature comparison involving that checkpoint), the Spearman between their pairwise-cosine geometries, and their
kNN neighbourhood persistence. Then the e360b signature and functional-geometry curves against the final checkpoint,
divided by the geometric mean of the two checkpoints' ceilings (the ceiling of the nearest measured checkpoint is used
for the others)."""
import os, glob, json, re, time, torch
torch.set_num_threads(16); RES = os.environ.get("WDD_RESULTS", "/workspace/wdd/results")
unit = lambda X: X / X.norm(dim=-1, keepdim=True).clamp_min(1e-9)
def spearman(a, b): ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float(); return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
g = torch.Generator().manual_seed(0); iu = torch.triu_indices(1024, 1024, 1); pairs = torch.randperm(iu.shape[1], generator=g)[:200000]
def kern(X): Xu = unit(X.float()); K = Xu @ Xu.T; return K[iu[0][pairs], iu[1][pairs]]
def knn(Xa, Xb, k=10):
    A = unit(Xa.float()); B = unit(Xb.float()); Ka = A @ A.T; Kb = B @ B.T; Ka.fill_diagonal_(-2); Kb.fill_diagonal_(-2); na = Ka.topk(k, 1).indices; nb = Kb.topk(k, 1).indices
    return float(sum(len(set(na[i].tolist()) & set(nb[i].tolist())) for i in range(len(A))) / (k * len(A)))
ceil = {}
for f in sorted(glob.glob(f"{RES}/e360_ceil/*_1.pt")):
    rev = os.path.basename(f)[:-5]; s = int(re.search(r"step(\d+)", rev).group(1)); a = torch.load(f"{RES}/e360/{rev}.pt"); b = torch.load(f); row = {}
    for l in a["blocks"]:
        sa, sb = a["sig"][l].float(), b["sig"][l].float(); sub = torch.randperm(sa.shape[0], generator=torch.Generator().manual_seed(1))[:1024]
        row[str(l)] = dict(sig_cos_median=(unit(sa) * unit(sb)).sum(1).median().item(), fgeo_spearman=spearman(kern(sa[sub]), kern(sb[sub])), knn=knn(sa[sub], sb[sub]))
    ceil[s] = row
e = json.load(open(f"{RES}/e360b_drift_analysis.json")); steps = e["steps"]; T = max(steps); meas = sorted(ceil)
def near(s): return min(meas, key=lambda m: abs(m - s)) if meas else None
corr = {}
for s in steps:
    if not meas: break
    r = e["vs_final"][str(s)]; out = {}
    for l in ("2", "8", "16"):
        ct, cT = ceil[near(s)][l], ceil[T][l] if T in ceil else ceil[near(T)][l]
        out[l] = dict(sig=r[l]["signature_cos_median"], sig_corrected=r[l]["signature_cos_median"] / max((ct["sig_cos_median"] * cT["sig_cos_median"]) ** 0.5, 1e-6), fgeo=r[l]["functional_geometry_spearman"], fgeo_corrected=r[l]["functional_geometry_spearman"] / max((ct["fgeo_spearman"] * cT["fgeo_spearman"]) ** 0.5, 1e-6), write_aligned=r[l]["write_cos_aligned_median"], attribution=r[l]["attribution_spearman"])
    corr[str(s)] = out
json.dump(dict(ceilings={str(k): v for k, v in ceil.items()}, corrected=corr, _exp="e360d_ceiling", _time=time.strftime("%Y-%m-%d %H:%M:%S")), open(f"{RES}/e360d_ceiling.json", "w"), indent=1)
pick = [s for s in steps if s in (0, 512, 1000, 2000, 4000, 8000, 16000, 33000, 63000, 93000, 123000, 138000)]
line = "e360d signature ceilings (two seeds, same weights): " + " | ".join(f"step{s}: " + " ".join(f"b{l} sig {v['sig_cos_median']:.2f} fgeo {v['fgeo_spearman']:.2f} knn {v['knn']:.3f}" for l, v in ceil[s].items()) for s in sorted(ceil)) + " || corrected vs final: " + " | ".join(f"step{s}: " + " ".join(f"b{l} sig {corr[str(s)][l]['sig_corrected']:.2f} fgeo {corr[str(s)][l]['fgeo_corrected']:.2f} write {corr[str(s)][l]['write_aligned']:.2f}" for l in ("2", "8", "16")) for s in pick if str(s) in corr)
open(f"{RES}/FINDINGS.log", "a").write(time.strftime("%Y-%m-%d %H:%M:%S") + " " + line + "\n"); print(line)
