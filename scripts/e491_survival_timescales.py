"""e491: how many timescales does provenance loss have? (Prony 1795.) A write is read at birth and fades over later
blocks (e21, e131, e188: a causal half-life of about two blocks, then a floor). Prony's method fits a sum of
exponentials to a sampled curve by linear prediction; with two modes it says whether the fading has one rate or two
(a fast loss of the direct trace and a slow loss of the re-written part), and where the floor is.
Setup: the block at a quarter depth as the birth block; the top MLP write of each typical position at that block (its
activation times its row norm); then, at each of the next eight blocks, whether that write's atom is in the 16-word
native description of the state there (the dictionary up to that block). The recall curve r(tau), tau = 0..8, is
fitted by one exponential plus a floor and by Prony's two-mode linear prediction.
Reported: the curve, the single-exponential half-life and R^2, Prony's two rates (as per-block retention factors) and
the R^2 of the two-mode fit, and the floor.
Models (argument): the five.
Pre-registered (honest guesses):
- a second mode is needed: the two-mode R^2 exceeds the single-exponential R^2 by at least 0.05 in at least three
  models (0.5);
- the fast rate keeps under half of the trace per block and the slow rate over 0.8 (0.5);
- recall at birth (tau = 0) is at least 0.8 in all five (0.6)."""
import sys, os, json as _json, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; model, tok, fam = load_model(name); arch = Arch(model, fam); K = 16; D = arch.D; DFF = arch.DFF
b0 = arch.NB // 4; H = min(8, arch.NB - 2 - b0); blocks = list(range(b0, b0 + H + 1))
ids = eval_ids(name)[:8, :256].to(DEV)
acts = []
def pre(m, a): acts.append(a[0].detach().float()[:, 1:].reshape(-1, DFF)); return None       # must return None; block_states runs in chunks, so accumulate (v2 fix)
h = arch.mlp_lin(b0).register_forward_pre_hook(pre)
try: S_ = block_states(model, arch, ids, blocks, chunk=4)
finally: h.remove()
rown = arch.wdir(b0).float().norm(dim=-1); c = torch.cat(acts) * rown[None]; top = c.abs().argmax(1); prom_birth = c.abs().max(1).values
A, lab = build_dictionary(arch, blocks=list(range(b0 + H + 1))); Au = unitr(A); del A; blk = lab["block"].to(DEV); typ = lab["type"].to(DEV); idx = lab["index"].to(DEV)
atom_of = torch.full((DFF,), -1, device=DEV, dtype=torch.long); m = (typ == T_MLP) & (blk == b0); atom_of[idx[m]] = torch.nonzero(m)[:, 0]; target = atom_of[top]
keep = torch.ones(target.numel(), dtype=torch.bool, device=DEV)
for b in blocks: keep &= ~sinkmask(S_[b].reshape(-1, D))
curve = []
for b in blocks:
    Xb = S_[b].reshape(-1, D)[keep]; Xc = Xb - Xb.mean(0); An = Au[:int((blk <= b).sum())]
    sel, _, _ = omp(Xc, An, K, batch=1024, record_err=False); curve.append(float((sel == target[keep][:, None]).any(1).float().mean()))
r = torch.tensor(curve); tau = torch.arange(len(curve)).float()
# single exponential with a floor: r = c + A exp(-tau / T); grid over T and c
best = None
for T in torch.linspace(0.3, 12, 118):
    for cfl in torch.linspace(0, float(r.min()), 21):
        e = torch.exp(-tau / T); Aamp = float(((r - cfl) * e).sum() / (e * e).sum()); fit = cfl + Aamp * e; sse = float(((r - fit) ** 2).sum())
        if best is None or sse < best[0]: best = (sse, float(T), float(cfl), Aamp)
sst = float(((r - r.mean()) ** 2).sum()); r2_single = 1 - best[0] / sst; half_life = best[1] * math.log(2)
# Prony, two modes: r(t+2) = p1 r(t+1) + p2 r(t), least squares; roots z1, z2 are the per-block retention factors
Y = r[2:]; Xp = torch.stack([r[1:-1], r[:-2]], 1); p = torch.linalg.lstsq(Xp, Y[:, None]).solution[:, 0]
disc = p[0] ** 2 + 4 * p[1]; roots = [float((p[0] + s * disc.abs().sqrt()) / 2) for s in (1, -1)] if disc >= 0 else None
if roots is not None:
    V = torch.stack([torch.tensor(roots[0]) ** tau, torch.tensor(roots[1]) ** tau], 1); amps = torch.linalg.lstsq(V, r[:, None]).solution[:, 0]; fit2 = V @ amps; r2_prony = float(1 - ((r - fit2) ** 2).sum() / sst)
else: amps, r2_prony = None, None
res = dict(model=name, birth_block=b0, horizon=H, n=int(keep.sum()), recall=curve, prominence_at_birth_median=float(prom_birth[keep].median() / S_[b0].reshape(-1, D)[keep].norm(dim=-1).median()),
           single=dict(T=best[1], half_life=half_life, floor=best[2], amplitude=best[3], r2=r2_single), prony=dict(roots=roots, amplitudes=(amps.tolist() if amps is not None else None), r2=r2_prony))
fast, slow = (sorted(roots) if roots else (None, None))
res["checks"] = dict(second_mode_needed=(r2_prony is not None and r2_prony - r2_single >= 0.05), fast_under_half_slow_over_0_8=(roots is not None and fast < 0.5 and slow > 0.8), recall_at_birth_0_8=curve[0] >= 0.8)
summ = (f"{name}, birth block {b0}, {int(keep.sum())} positions: recall of the top write over blocks +0..+{H}: " + "/".join(f"{v:.2f}" for v in curve) + f" | one exponential plus floor: half-life {half_life:.2f} blocks, floor {best[2]:.2f}, R^2 {r2_single:.3f} | Prony two modes: retention per block "
        + (f"{fast:.2f} and {slow:.2f}, amplitudes {amps[0]:.2f}, {amps[1]:.2f}, R^2 {r2_prony:.3f}" if roots else "complex (no real two-mode fit)") + f" | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e491_survival_{name}", res, summ)
