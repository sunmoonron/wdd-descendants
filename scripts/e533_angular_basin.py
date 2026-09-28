"""e533: the angular basin of a native word. e531 found that a native row frozen at one checkpoint keeps its wordhood
as well as the moving row, though the row rotates by some thirty degrees over four thousand steps; and that random
directions lose their words five times faster. How precisely oriented must a row be to be a word? Here each word atom
at a checkpoint (the 256 most-used rows of the native dictionary, and of a fixed random dictionary for comparison) is
rotated by an angle theta toward a random orthogonal direction, the rest of the dictionary unchanged, and OMP is
rerun on the same checkpoint's states (static retention: the fraction of the rotated atoms still among the 256
most used) and on the states four thousand steps later with the rotated atoms frozen (prospective retention, against
the unrotated frozen baseline); the rotated atoms' largest projection over the floor is recorded as well. Angles
0, 5, 10, 15, 20, 30, 45, 60 and 90 degrees; block 12; checkpoints 4000, 8000, 12000.
Pre-registered (honest guesses):
- the angle at which half the native words are lost statically is 20-30 degrees (0.4);
- the random dictionary's words have a narrower basin, their half-loss angle smaller by 5 degrees or more (0.5);
- the prospective retention at 10 degrees is within 0.05 of the unrotated frozen baseline (0.6).
Arguments: name."""
import sys, os, json as _json, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; B = 12; NWORD = 256; K = 16; H = 4000; ANG = [0, 5, 10, 15, 20, 30, 45, 60, 90]; CKS = [4000, 8000, 12000]
idsB = eval_ids(name)[:8, :256].to(DEV); gen = torch.Generator(device=DEV).manual_seed(0)
def gabs(m, T=14.0, n=28001):
    t = torch.linspace(0, T, n, dtype=torch.float64); F = torch.special.erf(t / math.sqrt(2)); return float(torch.trapezoid(1 - F.pow(m), t))
def states_at(n):
    model, tok, fam = load_model(name, revision=f"step{n}"); arch = Arch(model, fam)
    for p in model.parameters(): p.requires_grad_(False)
    X = block_states(model, arch, idsB, [B], chunk=4)[B].reshape(-1, arch.D); A, lab = build_dictionary(arch, blocks=list(range(B + 1))); del model; torch.cuda.empty_cache()
    keep = ~sinkmask(X); Xk = X[keep]; return Xk - Xk.mean(0), unitr(A), lab, arch.D, arch.DFF
def rows_index(lab, DFF):
    typ = lab["type"].to(DEV); blk = lab["block"].to(DEV); idx = lab["index"].to(DEV); gidx = torch.full((B + 1, DFF), -1, device=DEV, dtype=torch.long)
    for bb in range(B + 1): mm = (typ == T_MLP) & (blk == bb); gidx[bb, idx[mm]] = torch.nonzero(mm)[:, 0]
    return gidx.reshape(-1)
def usage_of(Xc, A, rows):
    sel, _, _ = omp(Xc, A, K, batch=1024, record_err=False); return torch.bincount(sel.reshape(-1), minlength=A.shape[0])[rows].float()
def topset(u):
    w = torch.zeros(u.numel(), dtype=torch.bool, device=u.device); w[torch.nonzero(u > 0)[:, 0][u[u > 0].argsort(descending=True)[:NWORD]]] = True; return w
def floor_of(U, A):
    m = A.shape[0]; Ar = unitr(rotate(A, seed=11)); CA = A.T @ A / m; CR = Ar.T @ Ar / m; gm = gabs(m); N = U.shape[0]
    s2 = ((U @ CA) * U).sum(1); s2r = ((U @ CR) * U).sum(1); Mr = torch.cat([(U[s:s + 128] @ Ar.T).abs().max(1).values for s in range(0, N, 128)]); r_cal = float(Mr.mean() / (s2r.clamp_min(1e-12).sqrt() * gm).mean()); return s2.clamp_min(1e-12).sqrt() * gm * r_cal
res = dict(model=name, block=B, angles=ANG, horizon=H, by_checkpoint={})
for n in CKS:
    Xc0, A0, lab0, D, DFF = states_at(n); Xc1, _, _, _, _ = states_at(n + H); rows = rows_index(lab0, DFF); U0 = unitr(Xc0); L0 = floor_of(U0, A0)
    Ar = unitr(torch.randn(A0.shape[0], D, device=DEV, generator=gen)); out = {}
    for dname, A in (("native", A0), ("random", Ar)):
        u0 = usage_of(Xc0, A, rows); W0 = topset(u0); widx = rows[torch.nonzero(W0)[:, 0]]; nw = widx.numel()
        base1 = usage_of(Xc1, A, rows); W1 = topset(base1); frozen_ret = float((W0 & W1).sum() / nw)
        r = torch.randn(nw, D, device=DEV, generator=gen); w = A[widx]; r = r - (r * w).sum(1, keepdim=True) * w; r = unitr(r); rec = {}
        for th in ANG:
            a = math.radians(th); Ap = A.clone(); Ap[widx] = math.cos(a) * w + math.sin(a) * r
            us = usage_of(Xc0, Ap, rows); Ws = topset(us); static_ret = float(Ws[W0].float().mean())
            up = usage_of(Xc1, Ap, rows); Wp = topset(up); pros_ret = float(Wp[W0].float().mean())
            Sr = ((U0 @ Ap[widx].T).abs() / L0[:, None]).max(0).values; S0 = ((U0 @ w.T).abs() / L0[:, None]).max(0).values
            rec[th] = dict(static_retention=static_ret, prospective_retention=pros_ret, median_S_ratio=float((Sr / S0.clamp_min(1e-9)).median()), fraction_over_floor=float((Sr > 1).float().mean()))
        # the half-loss angle, interpolated on the static curve
        def half_angle(key):
            ys = [rec[th][key] for th in ANG]; target = ys[0] / 2
            for i in range(1, len(ANG)):
                if ys[i] <= target:
                    y0, y1 = ys[i - 1], ys[i]; return ANG[i - 1] + (ANG[i] - ANG[i - 1]) * ((y0 - target) / (y0 - y1) if y0 != y1 else 0.5)
            return None
        out[dname] = dict(n_words=nw, frozen_baseline_prospective=frozen_ret, by_angle=rec, half_loss_angle_static=half_angle("static_retention"), half_loss_angle_prospective=half_angle("prospective_retention"))
        log(f"{name} step{n} {dname}: static retention by angle " + ", ".join(f"{th}: {rec[th]['static_retention']:.2f}" for th in ANG) + f"; prospective at +{H} " + ", ".join(f"{th}: {rec[th]['prospective_retention']:.2f}" for th in ANG) + f" (frozen baseline {frozen_ret:.2f}); S ratio " + ", ".join(f"{th}: {rec[th]['median_S_ratio']:.2f}" for th in ANG) + f"; half-loss angle static {out[dname]['half_loss_angle_static']}, prospective {out[dname]['half_loss_angle_prospective']}")
    res["by_checkpoint"][n] = out; del Xc0, Xc1, A0, Ar; torch.cuda.empty_cache()
hn = [res["by_checkpoint"][n]["native"]["half_loss_angle_static"] for n in CKS]; hr = [res["by_checkpoint"][n]["random"]["half_loss_angle_static"] for n in CKS]; g_ = lambda x: -1 if x is None else x
res["checks"] = dict(native_half_angle_20_30=all(20 <= g_(x) <= 30 for x in hn), random_narrower_by_5=all(g_(a) - g_(b) >= 5 for a, b in zip(hn, hr)), prospective_10deg_within_0_05=all(abs(res["by_checkpoint"][n]["native"]["by_angle"][10]["prospective_retention"] - res["by_checkpoint"][n]["native"]["frozen_baseline_prospective"]) <= 0.05 for n in CKS))
fm = lambda x: "n/a" if x is None else f"{x:.0f}"
summ = f"{name} block {B}: " + " | ".join(f"step {n}: native static retention at 5/10/20/30/45 degrees " + "/".join(f"{o['native']['by_angle'][th]['static_retention']:.2f}" for th in (5, 10, 20, 30, 45)) + f" (half-loss {fm(o['native']['half_loss_angle_static'])} deg), random " + "/".join(f"{o['random']['by_angle'][th]['static_retention']:.2f}" for th in (5, 10, 20, 30, 45)) + f" ({fm(o['random']['half_loss_angle_static'])} deg); prospective at +4000 native " + "/".join(f"{o['native']['by_angle'][th]['prospective_retention']:.2f}" for th in (0, 10, 20, 30, 45)) + f" (baseline {o['native']['frozen_baseline_prospective']:.2f})" for n, o in res["by_checkpoint"].items()) + f" | checks {_json.dumps(res['checks'])}"
log(summ); record(f"e533_angular_basin_{name}", res, summ)
