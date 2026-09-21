"""e301: second-order WDD. For the largest-coefficient candidate and two typical ones, the write is scaled at its own
tokens by delta in {-1, -0.5, -0.25, +0.25, +0.5, +1, +2} times its natural coefficient (delta = -1 is full removal),
and the state response at L and the centred logit response are fitted per dimension as a*delta + b*delta^2. Reported:
the quadratic-to-linear energy ratio at the natural amplitude, the variance explained by the quadratic fit beyond the
linear one, and the evenness cos(R(+1), R(-1)); the same sweep at foreign tokens for comparison."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from func_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; b = 2; D = c.D; lab = c.d["lab"]; A = c.d["A"]; R = A[(lab["type"] == T_MLP) & (lab["block"] == b)].float().to(DEV)
led, tn, tc = dominant(model, arch, c, ids_seq, b); run = make_runner_logits(model, arch, c, ids_seq, [L], NT, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep); smed = torch.stack([tc[idx][lab_i == k].median() for k in range(K)]); med = smed.abs(); sgn = torch.sign(smed); order = med.argsort(); picks = {"largest": int(med.argmax()), "typical_a": int(order[len(order) // 2]), "typical_b": int(order[len(order) // 2 - 1])}
foreign = torch.nonzero(typ & ~torch.isin(tn, keep))[:, 0][:1024]; deltas = [-1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 2.0]; out = {}
def fit(Rs):
    X = torch.tensor([[dd, dd * dd] for dd in deltas], device=DEV); Y = torch.stack(Rs); coef = torch.linalg.lstsq(X, Y).solution; a_, b_ = coef[0], coef[1]; lin = torch.linalg.lstsq(X[:, :1], Y).solution[0]; res_q = ((Y - X @ coef) ** 2).sum(); res_l = ((Y - X[:, :1] * lin[None]) ** 2).sum(); tot = (Y ** 2).sum(); return dict(quadratic_over_linear=(b_.norm() / a_.norm().clamp_min(1e-9)).item(), r2_linear=1 - (res_l / tot).item(), r2_quadratic=1 - (res_q / tot).item())
for nm, k in picks.items():
    own = idx[lab_i == k]; w = unit(R[keep[k]][None])[0] * sgn[k]; rec = dict(neuron=int(keep[k]), coef=med[k].item())
    for where, rows in (("own", own), ("foreign", foreign)):
        Rs, Ls = [], []; base = run(positions=rows); lg0 = base["lg"]
        for dd in deltas:
            inj = torch.zeros(NT, D, device=DEV); inj[rows] = dd * med[k] * w[None]; r = run(positions=rows, inject=inj, inject_block=b + 1); Rs.append((r[L] - S0[L])[rows].mean(0)); dlr = r["lg"] - lg0; Ls.append((dlr - dlr.mean(1, keepdim=True)).mean(0))
        fs, fl = fit(Rs), fit(Ls); ev = lambda Rv: (unit(Rv[deltas.index(1.0)][None]) @ unit(Rv[deltas.index(-1.0)][None]).T).item(); rec[where] = dict(state=fs, logits=fl, evenness_state=ev(Rs), evenness_logits=ev(Ls))
    out[nm] = rec; log(f"{tag} {nm} neuron {int(keep[k])} (|coef| {med[k]:.1f}): " + " | ".join(f"{where}: state quadratic/linear {v['state']['quadratic_over_linear']:.2f}, R2 linear {v['state']['r2_linear']:.2f} -> quadratic {v['state']['r2_quadratic']:.2f}, evenness {v['evenness_state']:+.2f}; logits quadratic/linear {v['logits']['quadratic_over_linear']:.2f}, R2 {v['logits']['r2_linear']:.2f} -> {v['logits']['r2_quadratic']:.2f}, evenness {v['evenness_logits']:+.2f}" for where, v in ((w_, rec[w_]) for w_ in ("own", "foreign"))))
record(f"e301_secondorder_{tag}", dict(model=tag, b=b, L=L, K=K, per_neuron=out), " | ".join(f"{nm} ({v['coef']:.1f}): own q/l {v['own']['state']['quadratic_over_linear']:.2f} even {v['own']['evenness_state']:+.2f} (logits {v['own']['evenness_logits']:+.2f}); foreign q/l {v['foreign']['state']['quadratic_over_linear']:.2f} even {v['foreign']['evenness_state']:+.2f}" for nm, v in out.items()))
