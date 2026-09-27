"""e529: the curvature along the drift and along the oscillation. e527 found the thousand-step update no flatter than a
random direction, but e526 and e528 found that update to be mostly a mean-reverting oscillation (consecutive increments
anti-correlated) with a small persistent drift (increments two to six thousand steps apart correlated at 0.1). A
thousand-step displacement therefore measures the oscillation's geometry, not the drift's. The oscillation cancels over
intervals while the drift adds, so the eight-thousand-step displacement is mostly drift, and the difference of two
consecutive increments is mostly oscillation. This run measures the directional curvature of the mean loss in the
row's coordinates (central differences on 8 x 512 fixed tokens, displacement a twentieth of the row's norm) along the
thousand-step update, the two-thousand-step displacement (the sum of two increments), the difference of the two
increments (the oscillation axis), the eight-thousand-step displacement (the drift), and a random direction; sixty
entrants within three thousand steps and sixty S-matched non-entrants, block 12 rows; origins 4000 and 8000.
Pre-registered (honest guesses):
- the oscillation axis is steeper than a random direction (median ratio above 1.5) (0.5);
- the eight-thousand-step displacement is flatter than the thousand-step one (ratio of their curvatures at or below
  0.8) (0.5);
- the eight-thousand-step displacement is flatter than a random direction (ratio below 0.9) (0.4);
- entrants and matched rows do not differ in any of these (AUC within 0.4-0.6) (0.6).
Arguments: name step (the origin checkpoint, thousands)."""
import sys, os, json as _json, math, time; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; t0 = int(sys.argv[2]) * 1000; CDIR = f"/workspace/wdd/cache/e524_{name}"; LB = 12; NWORD = 256; NSEL = 60; EPS_REL = 0.05; B = 12
ck = {k: torch.load(f"{CDIR}/step{t0 + 1000 * k}.pt") for k in (0, 1, 2, 3, 4, 8)}
model, tok, fam = load_model(name, revision=f"step{t0}"); arch = Arch(model, fam); D, DFF = arch.D, arch.DFF; R = (LB + 1) * DFF
for p in model.parameters(): p.requires_grad_(False)
lins = [arch.mlp_lin(bb) for bb in range(LB + 1)]; ids = eval_ids(name)[:8].to(DEV)
def auc(pos, neg):
    if pos.numel() < 5 or neg.numel() < 5: return None
    x = torch.cat([pos, neg]); r = x.argsort().argsort().double() + 1; npos, nneg = pos.numel(), neg.numel(); return float((r[:npos].sum() - npos * (npos + 1) / 2) / (npos * nneg))
def backward_all(weights):
    """gradients of the mean loss on the text with respect to the given weights (fp32, no autocast)"""
    for w in weights: w.grad = None; w.requires_grad_(True)
    ntok = ids.shape[0] * (ids.shape[1] - 1)
    for s in range(0, ids.shape[0], 4):
        c = ids[s:s + 4]; wgt = c.shape[0] * (c.shape[1] - 1) / ntok
        with torch.enable_grad(): l = model(c, labels=c, use_cache=False).loss * wgt
        l.backward()
    out = [w.grad.detach().clone() for w in weights]
    for w in weights: w.grad = None; w.requires_grad_(False)
    return out
col = lambda Wg, j: (Wg[j] if fam == "gpt2" else Wg[:, j])
def set_col(bb, j, v):
    with torch.no_grad():
        if fam == "gpt2": lins[bb].weight[j] = v
        else: lins[bb].weight[:, j] = v
# selection: entrants within three thousand steps and S-matched non-entrants, block 12 rows
Bk = {k: ck[k]["blocks"][B] for k in ck}; words = {}
for k in ck:
    u = Bk[k]["usage"]; wd = torch.zeros(R, dtype=torch.bool); wd[torch.nonzero(u > 0)[:, 0][u[u > 0].argsort(descending=True)[:NWORD]]] = True; words[k] = wd
S0 = Bk[0]["S"].float(); nonw0 = ~words[0]; ent = torch.zeros(R, dtype=torch.bool); stay = nonw0.clone()
for k in (1, 2, 3):
    e = stay & ~words[k - 1] & words[k] & words[k + 1]; ent |= e; stay &= ~words[k]
never = nonw0 & ~words[1] & ~words[2] & ~words[3] & ~words[4]; g = torch.Generator().manual_seed(0)
ent_idx = torch.nonzero(ent)[:, 0]; ent_idx = ent_idx[torch.randperm(ent_idx.numel(), generator=g)[:NSEL]]; pool = torch.nonzero(never)[:, 0]; taken = torch.zeros(R, dtype=torch.bool); matched = []
for i in ent_idx.tolist():
    c = pool[~taken[pool]]; j = c[(S0[c] - S0[i]).abs().argmin()]; taken[j] = True; matched.append(int(j))
sel = torch.cat([ent_idx, torch.tensor(matched, dtype=torch.long)]); is_ent = torch.cat([torch.ones(ent_idx.numel()), torch.zeros(len(matched))]).bool()
Wk = {k: (ck[k]["rows"].float() * ck[k]["norms"][:, None]) for k in (0, 1, 2, 8)}; dW = Wk[1] - Wk[0]; dW2 = Wk[2] - Wk[0]; dDiff = 2 * Wk[1] - Wk[0] - Wk[2]; dW8 = Wk[8] - Wk[0]
log(f"{name} step{t0}: {ent_idx.numel()} entrants and {len(matched)} matched rows selected")
gen = torch.Generator(device=DEV).manual_seed(1)
DIRS = ["update_1000", "sum_2000", "difference", "update_8000", "random_1"]; kap = {d: [] for d in DIRS}; t_start = time.time()
for n_, r in enumerate(sel.tolist()):
    bb, j = r // DFF, r % DFF; w = lins[bb].weight.detach().clone(); wcol = col(w, j).clone(); eps = EPS_REL * float(wcol.norm())
    dirs = {"update_1000": dW[r].to(DEV), "sum_2000": dW2[r].to(DEV), "difference": dDiff[r].to(DEV), "update_8000": dW8[r].to(DEV), "random_1": torch.randn(D, device=DEV, generator=gen)}
    for d, v in dirs.items():
        v = v / v.norm().clamp_min(1e-12); gs = []
        for sg in (1.0, -1.0):
            set_col(bb, j, wcol + sg * eps * v); gs.append(col(backward_all([lins[bb].weight])[0], j))
        set_col(bb, j, wcol); kap[d].append(float(((gs[0] - gs[1]) @ v) / (2 * eps)))
    if n_ % 20 == 0: log(f"{name} step{t0}: row {n_ + 1}/{sel.numel()} done ({time.time() - t_start:.0f} s)")
K = {d: torch.tensor(v) for d, v in kap.items()}; rnd = K["random_1"].abs().clamp_min(1e-12)
res = dict(model=name, origin=t0, n_entrants=int(is_ent.sum()), n_matched=int((~is_ent).sum()), eps_rel=EPS_REL, groups={})
for gname, gm in (("entrants", is_ent), ("matched", ~is_ent), ("all", torch.ones_like(is_ent))):
    res["groups"][gname] = dict(median_curvature={d: float(K[d][gm].median()) for d in DIRS}, median_ratio_to_random={d: float((K[d][gm] / rnd[gm]).median()) for d in DIRS[:4]}, ratio_8000_over_1000=float((K["update_8000"][gm] / K["update_1000"][gm].abs().clamp_min(1e-12)).median()), ratio_difference_over_sum=float((K["difference"][gm] / K["sum_2000"][gm].abs().clamp_min(1e-12)).median()),
                                fraction_flatter_than_random={d: float((K[d][gm] < K["random_1"][gm]).float().mean()) for d in DIRS[:4]}, fraction_negative={d: float((K[d][gm] < 0).float().mean()) for d in DIRS})
res["auc_matched_over_entrants"] = {d: auc((K[d] / rnd)[~is_ent], (K[d] / rnd)[is_ent]) for d in DIRS[:4]}
e_, m_, a_ = res["groups"]["entrants"], res["groups"]["matched"], res["groups"]["all"]; fm = lambda x: "n/a" if x is None else f"{x:.2f}"
res["checks"] = dict(oscillation_steeper_than_random=a_["median_ratio_to_random"]["difference"] > 1.5, drift_flatter_than_update=a_["ratio_8000_over_1000"] <= 0.8, drift_flatter_than_random=a_["median_ratio_to_random"]["update_8000"] < 0.9, no_group_difference=all(0.4 <= (v or 0.5) <= 0.6 for v in res["auc_matched_over_entrants"].values()))
summ = (f"{name} step{t0} ({res['n_entrants']} entrants, {res['n_matched']} matched): curvature ratio to a random direction, all rows: thousand-step update {a_['median_ratio_to_random']['update_1000']:.2f}, two-thousand-step displacement {a_['median_ratio_to_random']['sum_2000']:.2f}, oscillation axis (difference of consecutive increments) {a_['median_ratio_to_random']['difference']:.2f}, eight-thousand-step displacement {a_['median_ratio_to_random']['update_8000']:.2f} (flatter than random for {a_['fraction_flatter_than_random']['update_8000']:.2f} of rows); eight-thousand over thousand {a_['ratio_8000_over_1000']:.2f}, difference over sum {a_['ratio_difference_over_sum']:.2f}; entrants "
        + "/".join(f"{e_['median_ratio_to_random'][d]:.2f}" for d in DIRS[:4]) + ", matched " + "/".join(f"{m_['median_ratio_to_random'][d]:.2f}" for d in DIRS[:4]) + "; AUC matched over entrants " + "/".join(fm(res['auc_matched_over_entrants'][d]) for d in DIRS[:4]) + f"; negative curvature along the eight-thousand-step displacement for {a_['fraction_negative']['update_8000']:.2f} of rows | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e529_drift_curvature_{name}_step{t0}", res, summ)
