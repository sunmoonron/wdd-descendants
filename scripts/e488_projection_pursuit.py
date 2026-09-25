"""e488: native atoms as projection-pursuit directions (Friedman and Tukey 1974; Huber 1985; kurtosis as the FastICA
criterion, Hyvarinen 1999). Projection pursuit looks for directions along which the data are least Gaussian; kurtosis
is its classical index. An MLP row is written only when its neuron fires, so the states' projections on it should be
heavy-tailed. This run asks whether the native atoms are the non-Gaussian directions of the state cloud, which no
fitting would have to find, and whether OMP's usage follows that index.
Setup: middle depth, 16 x 512 evaluation tokens, typical positions, centred states. Directions: 4096 MLP rows from
the blocks up to the middle (random subset), the same rows rotated, 4096 random unit directions, the 16 top principal
directions, and 4096 token embeddings. For each direction the excess kurtosis of the projections; for the native rows
also their usage in 16-word descriptions of the same states.
Reported: median and 90th-percentile kurtosis per family; the share of native rows above the 99th percentile of random
directions; Spearman of usage with kurtosis over native rows; the kurtosis of used against never-used rows.
Models (argument): the five.
Pre-registered (honest guesses):
- native MLP rows have a higher median kurtosis than rotated rows and random directions in all five (0.8);
- usage correlates with kurtosis (Spearman above 0.3) (0.5);
- the principal directions are less kurtotic than the median native row (0.6)."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; model, tok, fam = load_model(name); arch = Arch(model, fam); L = arch.NB // 2; K = 16; D = arch.D
E = eval_ids(name)[:16, :512].to(DEV)
X = block_states(model, arch, E, [L], chunk=4)[L].reshape(-1, D); keep = ~sinkmask(X); Xc = X[keep] - X[keep].mean(0); N = Xc.shape[0]
A, lab = build_dictionary(arch, blocks=list(range(L + 1))); Au = unitr(A); Ar = unitr(rotate(A, seed=7)); typ = lab["type"].to(DEV); del A
g = torch.Generator(device=DEV).manual_seed(0)
mlp = torch.nonzero(typ == T_MLP)[:, 0]; mlp = mlp[torch.randperm(mlp.numel(), generator=g, device=DEV)[:4096]]
tk = torch.nonzero(typ == T_TOK)[:, 0]; tk = tk[torch.randperm(tk.numel(), generator=g, device=DEV)[:4096]]
P16, _ = pcs(Xc, 16)
def kurt(Dirs):
    out = []
    for s in range(0, Dirs.shape[0], 1024):
        z = Xc @ Dirs[s:s + 1024].T; z = z - z.mean(0); v = z.pow(2).mean(0); out.append(z.pow(4).mean(0) / v.pow(2).clamp_min(1e-12) - 3)
    return torch.cat(out)
fam_dirs = {"native_mlp": Au[mlp], "rotated_mlp": Ar[mlp], "random": unitr(torch.randn(4096, D, generator=g, device=DEV)), "pca16": P16.T, "token_embeddings": Au[tk]}
kk = {k: kurt(v) for k, v in fam_dirs.items()}
sel, _, _ = omp(Xc, Au, K, batch=1024, record_err=False); usage = torch.bincount(sel.reshape(-1), minlength=Au.shape[0]).float()[mlp]
def spear(x, y):
    rx, ry = x.argsort().argsort().double(), y.argsort().argsort().double(); rx, ry = rx - rx.mean(), ry - ry.mean(); return float((rx * ry).sum() / (rx.norm() * ry.norm()).clamp_min(1e-9))
q99 = float(kk["random"].quantile(0.99))
res = dict(model=name, level=L, n=N, kurtosis={k: dict(median=float(v.median()), p90=float(v.quantile(0.9)), mean=float(v.mean())) for k, v in kk.items()},
           native_above_random_99=float((kk["native_mlp"] > q99).float().mean()), rotated_above_random_99=float((kk["rotated_mlp"] > q99).float().mean()),
           usage_vs_kurtosis_spearman=spear(usage, kk["native_mlp"]), used_share=float((usage > 0).float().mean()), kurtosis_used=float(kk["native_mlp"][usage > 0].median()) if (usage > 0).any() else None, kurtosis_unused=float(kk["native_mlp"][usage == 0].median()) if (usage == 0).any() else None)
Kz = res["kurtosis"]
res["checks"] = dict(native_over_rotated_and_random=Kz["native_mlp"]["median"] > max(Kz["rotated_mlp"]["median"], Kz["random"]["median"]), usage_tracks_kurtosis=res["usage_vs_kurtosis_spearman"] > 0.3, pca_less_kurtotic_than_native=Kz["pca16"]["median"] < Kz["native_mlp"]["median"])
summ = (f"{name} L{L}, {N} positions: excess kurtosis of the states' projections, median (90th pct): native MLP rows {Kz['native_mlp']['median']:.2f} ({Kz['native_mlp']['p90']:.2f}), rotated rows {Kz['rotated_mlp']['median']:.2f} ({Kz['rotated_mlp']['p90']:.2f}), random directions {Kz['random']['median']:.2f} ({Kz['random']['p90']:.2f}), top-16 PCs {Kz['pca16']['median']:.2f}, token embeddings {Kz['token_embeddings']['median']:.2f} ({Kz['token_embeddings']['p90']:.2f}); "
        f"native rows above the random 99th percentile {res['native_above_random_99']:.2f} (rotated {res['rotated_above_random_99']:.2f}); usage-kurtosis Spearman {res['usage_vs_kurtosis_spearman']:+.2f}, used rows {res['used_share']:.2f} with median kurtosis {res['kurtosis_used']} against unused {res['kurtosis_unused']} | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e488_pursuit_{name}", res, summ)
