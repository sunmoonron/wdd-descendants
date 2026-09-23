"""e364b: persistence of co-selection across training (CPU). For the heads of Pythia-410m, which keep their index at
every checkpoint: the residual co-selection correlation matrix (global mode removed, natural batches and mixed
batches) at each checkpoint; its stability between checkpoints (Spearman of the pairwise entries, against a
head-permuted null) and the stability of its communities (adjusted Rand index of spectral clusters); whether the
induction-selected heads of the final checkpoint co-select at earlier checkpoints; and whether co-selection
persists better than the heads' individual selection strength (the mean selection signal per head)."""
import sys, os, glob, json, time, torch, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(16); RES = os.environ.get("WDD_RESULTS", "/workspace/wdd/results")
from pc_common import spectral, ari
files = sorted(glob.glob(os.path.join(RES, "e364", "pythia410_step*.pt")), key=lambda f: int(os.path.basename(f).split("step")[1][:-3])); steps = [int(os.path.basename(f).split("step")[1][:-3]) for f in files]
D = {s: torch.load(f) for s, f in zip(steps, files)}
def rcorr(X):
    Z = (X - X.mean(0)) / X.std(0).clamp_min(1e-12); U, S, Vh = torch.linalg.svd(Z, full_matrices=False); Zr = Z - (U[:, :1] * S[:1]) @ Vh[:1]; C = (Zr.T @ Zr) / (len(Zr) - 1); d = C.diagonal().clamp_min(1e-12).sqrt(); return C / torch.outer(d, d)
def spearman(a, b):
    ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float(); return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
out = {}
for nm, sel in (("natural", lambda d: d["TY"] == 0), ("mixed", lambda d: torch.ones_like(d["TY"], dtype=torch.bool))):
    C = {s: rcorr(D[s]["SH"][sel(D[s])]) for s in steps}; n = C[steps[0]].shape[0]; iu = torch.triu_indices(n, n, 1); cl = {s: spectral(C[s].clamp_min(0), kmax=8)[0] for s in steps}; fin = steps[-1]
    strength = {s: D[s]["SH"][sel(D[s])].mean(0) for s in steps}; g = torch.Generator().manual_seed(0); perm = torch.randperm(n, generator=g)
    rows = {}
    for s in steps:
        v, vf = C[s][iu[0], iu[1]], C[fin][iu[0], iu[1]]; Cp = C[fin][perm][:, perm]; vp = Cp[iu[0], iu[1]]
        rows[str(s)] = dict(corr_stability_vs_final=spearman(v, vf), permuted_null=spearman(v, vp), community_ari_vs_final=ari(cl[s], cl[fin]), strength_spearman_vs_final=spearman(strength[s], strength[fin]))
    nbr = {f"{a}->{b}": dict(corr_stability=spearman(C[a][iu[0], iu[1]], C[b][iu[0], iu[1]]), community_ari=ari(cl[a], cl[b])) for a, b in zip(steps[:-1], steps[1:])}
    ind = torch.tensor([h[0] * D[fin]["NH"] + h[1] for h in json.load(open(os.path.join(RES, "e364_cosel_pythia410_step%d.json" % fin)))["induction_selected_heads"]])
    gc = lambda Cm, gi: ((Cm[gi][:, gi].sum() - Cm[gi][:, gi].diagonal().sum()) / (len(gi) ** 2 - len(gi))).item()
    out[nm] = dict(vs_final=rows, neighbours=nbr, final_induction_heads_corr_by_step={str(s): gc(C[s], ind) for s in steps})
json.dump(dict(out, steps=steps, _exp="e364b_persistence", _time=time.strftime("%Y-%m-%d %H:%M:%S")), open(os.path.join(RES, "e364b_persistence.json"), "w"), indent=1)
line = " || ".join(f"{nm}: " + " ".join(f"{s}: stab {v['corr_stability_vs_final']:.2f} (null {v['permuted_null']:.2f}) ARI {v['community_ari_vs_final']:.2f} strength {v['strength_spearman_vs_final']:.2f};" for s, v in o["vs_final"].items()) + " ind-heads corr by step " + " ".join(f"{s}:{c:+.2f}" for s, c in o["final_induction_heads_corr_by_step"].items()) for nm, o in out.items())
print(line)
with open(os.path.join(RES, "FINDINGS.log"), "a") as f: f.write(time.strftime("%Y-%m-%d %H:%M:%S") + " e364b_persistence: " + line + "\n")
