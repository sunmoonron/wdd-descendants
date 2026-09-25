"""e472: is the 16-word native description a fixed point of WDD? From an external review's "recursive WDD" list: the
fixed point, the self-consistency triangle and the attractor test.
The operator D maps a centred state to its 16-word OMP description, least-squares refitted. D(x) lies in the span of the
words chosen for x, so D(D(x)) = D(x) exactly when OMP, run on D(x), chooses the same words. The iteration
x_{t+1} = D(x_t) is then a sequence of projections: it can only lose energy, and it stops once the words repeat. Whether
it stops is a question about the geometry of the chosen word sets. The rotated dictionary has the same Gram matrix, so it
is the control for whether provenance matters.
Setup: middle depth L, 8 x 256 evaluation tokens, typical positions. Five iterations of:
- native and rotated words, no perturbation;
- the attractor test: Gaussian noise of 5% of the state's norm added before each step, two noise seeds, native and
  rotated;
- a word-sized nudge: a random dictionary atom (up to block L) scaled to 5% of the state's norm added before each step,
  native.
Per step: the share of positions whose words equal the previous step's (fixed points), the Jaccard overlap with the
previous step's words and with the first step's, the energy relative to the original, the cosine to the original, and
loss recovered when the iterate is spliced in. For the noisy runs: the Jaccard between the two seeds' final words, and
with the noise-free run's final words.
Models (argument): gpt2, qwen05.
Pre-registered (honest guesses):
- the fixed-point share after one step is similar for native and rotated words (within 0.1), since it is set by the
  shared geometry (0.6);
- loss recovered changes by under 0.05 after the first step (0.6);
- noisy trajectories from different seeds end with mostly the same words (Jaccard at least 0.5) (0.5)."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; model, tok, fam = load_model(name); arch = Arch(model, fam); L = arch.NB // 2; K = 16; STEPS = 5
ids = eval_ids(name)[:8, :256].to(DEV)
sp = Splicer(model, arch, ids, L, chunk=2)
X = block_states(model, arch, ids, [L], chunk=4)[L]; flat = X.reshape(-1, arch.D); keep = ~sinkmask(flat); mu = flat[keep].mean(0); X0 = flat[keep] - mu
lm = sp.lossmask(keep.view(X.shape[0], -1)); base = sp.c["loss"][lm].mean().item(); ms = flat.clone(); ms[keep] = mu; mean_loss = sp.run(ms.view_as(X))["loss"][lm].mean().item()
A, lab = build_dictionary(arch, blocks=list(range(L + 1))); Au = unitr(A); Ar = unitr(rotate(A, seed=7)); del A
def D(Xc, Dct):
    sel, _, _ = omp(Xc, Dct, K, batch=1024, record_err=False); cof, _ = refit(Xc, Dct, sel); return torch.einsum("nk,nkd->nd", cof, Dct[sel]), torch.sort(sel, 1).values
def rec(Xc):
    new = flat.clone(); new[keep] = mu + Xc; return (mean_loss - sp.run(new.view_as(X))["loss"][lm].mean().item()) / max(mean_loss - base, 1e-9)
def jac(S1, S2):
    inter = torch.tensor([len(set(a.tolist()) & set(b.tolist())) for a, b in zip(S1, S2)], dtype=torch.float); return float((inter / (2 * K - inter)).mean())
def perturb(x, kind, g):
    if kind == "none": return x
    if kind == "gauss": return x + 0.05 * x.norm(dim=-1, keepdim=True) * unitr(torch.randn(x.shape, generator=g, device=DEV))
    j = torch.randint(0, Au.shape[0], (x.shape[0],), generator=g, device=DEV); sgn = torch.randint(0, 2, (x.shape[0], 1), generator=g, device=DEV).float() * 2 - 1
    return x + 0.05 * x.norm(dim=-1, keepdim=True) * sgn * Au[j]
RUNS = [("native", Au, "none", 0), ("rotated", Ar, "none", 0), ("native_noise_s1", Au, "gauss", 1), ("native_noise_s2", Au, "gauss", 2),
        ("rotated_noise_s1", Ar, "gauss", 1), ("rotated_noise_s2", Ar, "gauss", 2), ("native_word_nudge", Au, "word", 3)]
res = dict(model=name, level=L, k=K, steps=STEPS, runs={}); final = {}
for run_name, Dct, kind, seed in RUNS:
    g = torch.Generator(device=DEV).manual_seed(seed); x, prev, first, traj = X0.clone(), None, None, []
    for t in range(STEPS):
        xn, S = D(perturb(x, kind, g), Dct)
        row = dict(step=t + 1, energy=float(xn.norm(dim=-1).pow(2).sum() / X0.norm(dim=-1).pow(2).sum()), cos_to_original=float((unitr(xn) * unitr(X0)).sum(-1).median()), loss_recovered=rec(xn))
        if prev is not None: row.update(fixed_share=float((S == prev).all(1).float().mean()), jaccard_prev=jac(S, prev), jaccard_first=jac(S, first))
        else: first = S
        traj.append(row); prev = S; x = xn
    res["runs"][run_name] = traj; final[run_name] = prev
    log(f"{name} {run_name}: " + " | ".join(f"step {r['step']}: energy {r['energy']:.3f}, cos {r['cos_to_original']:.3f}, rec {r['loss_recovered']:.3f}" + (f", fixed {r['fixed_share']:.2f}, jac prev {r['jaccard_prev']:.2f}, jac first {r['jaccard_first']:.2f}" if 'fixed_share' in r else "") for r in traj))
res["attractor"] = {d_: dict(seeds_final_jaccard=jac(final[f"{d_}_noise_s1"], final[f"{d_}_noise_s2"]), seed1_vs_clean_final=jac(final[f"{d_}_noise_s1"], final[d_])) for d_ in ("native", "rotated")}
R = res["runs"]
res["checks"] = dict(fixed_share_similar=abs(R["native"][1]["fixed_share"] - R["rotated"][1]["fixed_share"]) <= 0.1, loss_stable_after_step1=abs(R["native"][-1]["loss_recovered"] - R["native"][0]["loss_recovered"]) < 0.05,
                     noisy_seeds_converge=res["attractor"]["native"]["seeds_final_jaccard"] >= 0.5)
f_ = lambda tr: (f"rec {tr[0]['loss_recovered']:.3f} -> {tr[-1]['loss_recovered']:.3f}, energy {tr[0]['energy']:.3f} -> {tr[-1]['energy']:.3f}, fixed share steps 2..5 " + "/".join(f"{r['fixed_share']:.2f}" for r in tr[1:])
                 + ", jaccard with previous " + "/".join(f"{r['jaccard_prev']:.2f}" for r in tr[1:]) + f", with first {tr[-1]['jaccard_first']:.2f}")
summ = (f"{name} L{L}, iterated 16-word descriptions: " + " || ".join(f"{rn}: " + f_(R[rn]) for rn in R)
        + " || attractor: " + "; ".join(f"{d_} seeds' final Jaccard {v['seeds_final_jaccard']:.2f} (vs clean final {v['seed1_vs_clean_final']:.2f})" for d_, v in res["attractor"].items()) + f" | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e472_fixedpoint_{name}", res, summ)
