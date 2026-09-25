"""e468: does a native description keep what later tokens need from a position? This is the "predictive" reading of a
description, suggested by an external review. e388 and e465 spliced every position at once. Here one position is
replaced, and the damage is measured at that position and at the next 8. Later positions read the replaced one only
through attention in the blocks above the splice.
Setup: the middle depth L, 96 positions t (8 evaluation sequences, t between 32 and 200, typical positions). The state
at t is replaced by:
- its 16-word native description;
- its 16 rotated words;
- its top 16 principal components (fitted on other sequences);
- the mean state (everything removed).
Measured: the KL divergence of the next-token prediction from the clean run, at t itself and averaged over t+1..t+8.
Each description's damage is also reported relative to the mean state's (1 = as bad as removing everything).
Models (argument): gpt2, qwen05.
Pre-registered (honest guesses):
- the native description damages later positions less than rotated words or PCA do (0.6);
- relative to removing everything, the damage to later positions is smaller than at t itself (0.6)."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; model, tok, fam = load_model(name); arch = Arch(model, fam); L = arch.NB // 2; K = 16; D = arch.D; H = 8
E = eval_ids(name); ids = E[:8, :256].to(DEV); fit = E[8:16, :256].to(DEV)
F_ = block_states(model, arch, fit, [L], chunk=4)[L].reshape(-1, D); mu = F_[~sinkmask(F_)].mean(0)
ev_, U_ = torch.linalg.eigh(torch.cov((F_[~sinkmask(F_)] - mu).T.double(), correction=0)); P = U_.flip(-1)[:, :K].float()
X = block_states(model, arch, ids, [L], chunk=4)[L]                            # [B, T-1, D]
A, lab = build_dictionary(arch, blocks=list(range(L + 1))); Au = unitr(A); Ar = unitr(rotate(A, seed=7)); del A
g = torch.Generator().manual_seed(0); picks = []
for s in range(ids.shape[0]):
    for t in torch.randperm(168, generator=g)[:12].tolist():
        t = t + 32; x = X[s, t - 1]
        if x.norm() < 10 * X[s].norm(dim=-1).median(): picks.append((s, t))
xs = torch.stack([X[s, t - 1] for s, t in picks]) - mu
def desc(Dct):
    sel, _, _ = omp(xs, Dct, K, batch=256, record_err=False); cof, _ = refit(xs, Dct, sel); return torch.einsum("nk,nkd->nd", cof, Dct[sel])
reps = {"native": mu + desc(Au), "rotated": mu + desc(Ar), "pca": mu + xs @ P @ P.T, "mean": mu.expand(len(picks), -1)}
def lps(s, t, new=None):
    def hk(m, i, o):
        if new is None: return None
        y = out_of(o).clone(); y[0, t] = new.to(y.dtype); return (y,) + tuple(o[1:]) if isinstance(o, tuple) else y
    h = arch.layers[L].register_forward_hook(hk)
    try:
        with torch.no_grad(): lg = model(ids[s:s + 1, :t + H + 1]).logits[0, t:t + H + 1].float()
    finally: h.remove()
    return torch.log_softmax(lg, -1)                                               # [H+1, V]: predictions at t..t+H
kl = lambda p, q: (p.exp() * (p - q)).sum(-1)
out = {k: dict(own=[], future=[]) for k in reps}
for n_, (s, t) in enumerate(picks):
    c = lps(s, t)
    for k, R in reps.items():
        d = kl(c, lps(s, t, R[n_])); out[k]["own"].append(float(d[0])); out[k]["future"].append(float(d[1:].mean()))
mean = lambda v: sum(v) / len(v)
res = dict(model=name, level=L, k=K, n=len(picks), horizon=H, kl={k: dict(own=mean(v["own"]), future=mean(v["future"])) for k, v in out.items()})
M = res["kl"]["mean"]
for k in reps: res["kl"][k]["own_rel_mean"] = res["kl"][k]["own"] / M["own"]; res["kl"][k]["future_rel_mean"] = res["kl"][k]["future"] / M["future"]
Kk = res["kl"]
res["checks"] = dict(native_least_future_damage=Kk["native"]["future"] < min(Kk["rotated"]["future"], Kk["pca"]["future"]), future_smaller_share_than_own=Kk["native"]["future_rel_mean"] < Kk["native"]["own_rel_mean"])
summ = (f"{name} L{L}, one position replaced ({len(picks)} positions), KL at the position / mean over the next {H}: "
        + " | ".join(f"{k} {v['own']:.3f} / {v['future']:.4f} (relative to mean {v['own_rel_mean']:.2f} / {v['future_rel_mean']:.2f})" for k, v in Kk.items()) + f" | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e468_futuretokens_{name}", res, summ)
