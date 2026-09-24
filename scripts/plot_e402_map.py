"""Local plot of e402: the intelligibility map (vocabulary of stage i describing the states of stage j, loss recovered at
k 16) and the same map minus each target's rotation null, with cells below the lowest rotation marked."""
import json, sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.dirname(here)
d = json.load(open(os.path.join(root, "results", "e402_intelmap_pythia410.json")))
revs = d["revs"]; n = len(revs); lab = [r.replace("step", "") for r in revs]
for k in ["8", "16"]:
    M = np.array([[d["cells"][f"{i}->{j}"][k]["rec"] for j in revs] for i in revs])
    nl = np.array([[d["null"][f"{j}:{s}"][k]["rec"] for s in (7, 11, 13)] for j in revs])
    D = M - nl.mean(1)[None, :]; low = M < nl.min(1)[None, :]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.6))
    for a, X, t, cm, v in [(ax[0], M, f"loss recovered, k {k}", "viridis", (min(0, M.min()), 1)), (ax[1], D, f"minus the target's rotation null, k {k}", "RdBu", (-np.abs(D).max(), np.abs(D).max()))]:
        im = a.imshow(X, cmap=cm, vmin=v[0], vmax=v[1]); a.set_xticks(range(n)); a.set_xticklabels(lab, rotation=60); a.set_yticks(range(n)); a.set_yticklabels(lab)
        a.set_xlabel("states of step"); a.set_ylabel("vocabulary of step"); a.set_title(t); plt.colorbar(im, ax=a, fraction=0.046)
        for i in range(n):
            for j in range(n):
                a.text(j, i, f"{X[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="k" if cm == "RdBu" else "w")
                if low[i, j] and cm == "RdBu": a.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="k", lw=2))
    fig.suptitle("Pythia-410m, middle depth: can one training stage's own words describe another stage's states? (boxed: below every rotation)")
    fig.tight_layout(); out = os.path.join(root, "results", f"e402_map_k{k}.png"); fig.savefig(out, dpi=130); print(out)
