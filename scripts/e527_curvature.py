"""e527: does a row's thousand-step motion lie in flat directions of the loss? e525 showed the motion is not the row's
mean gradient, and e524b that it is directed for the rows near the floor. One way a small persistent force produces a
large directed motion while no batch's gradient points along it: the motion is along directions of low curvature, where
nothing pushes back and displacements accumulate, while the gradient at any time points along steep directions that the
optimiser corrects within steps. This run measures the curvature of the loss in a row's own coordinates along five
directions: the row's actual thousand-step update, the negative gradient of the loss on the text, the row's own
direction (growth), and two random directions. The curvature along v is the directional second derivative
v' H v, estimated by central differences of the gradient of the mean loss on 8 x 512 fixed evaluation tokens with the
row's parameters displaced by plus and minus a twentieth of the row's norm along v (the same text, so the difference
is exact up to arithmetic). Entrants within three thousand steps against S-matched non-entrants, sixty of each,
block 12 rows; Pythia-410m, origins 4000, 8000, 12000.
Pre-registered (honest guesses), block 12:
- the thousand-step update is flatter than a random direction: median ratio of its curvature to the random
  directions' below 0.7 for both groups (0.5);
- the gradient direction is steeper than random: median ratio above 2 (0.6);
- the entrants' updates are flatter than the matched rows' (AUC of the ratio, matched over entrants, 0.6 or above) (0.4);
- the row's own direction is steeper than random (median ratio above 1.5) (0.5).
Arguments: name step (the origin checkpoint, thousands)."""
import sys, os, json as _json, math, time; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; t0 = int(sys.argv[2]) * 1000; CDIR = f"/workspace/wdd/cache/e524_{name}"; LB = 12; NWORD = 256; NSEL = 60; EPS_REL = 0.05; B = 12
ck = {k: torch.load(f"{CDIR}/step{t0 + 1000 * k}.pt") for k in range(0, 5)}
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
W0 = (ck[0]["rows"].float() * ck[0]["norms"][:, None]); W1 = (ck[1]["rows"].float() * ck[1]["norms"][:, None]); dW = (W1 - W0)
log(f"{name} step{t0}: {ent_idx.numel()} entrants and {len(matched)} matched rows selected; baseline gradient")
G0 = backward_all([l.weight for l in lins]); gen = torch.Generator(device=DEV).manual_seed(1)
DIRS = ["update_1000", "negative_gradient", "own_direction", "random_1", "random_2"]; kap = {d: [] for d in DIRS}; t_start = time.time()
for n_, r in enumerate(sel.tolist()):
    bb, j = r // DFF, r % DFF; w = lins[bb].weight.detach().clone(); wcol = col(w, j).clone(); eps = EPS_REL * float(wcol.norm())
    dirs = {"update_1000": dW[r].to(DEV), "negative_gradient": -col(G0[bb], j), "own_direction": wcol, "random_1": torch.randn(D, device=DEV, generator=gen), "random_2": torch.randn(D, device=DEV, generator=gen)}
    for d, v in dirs.items():
        v = v / v.norm().clamp_min(1e-12); gs = []
        for sg in (1.0, -1.0):
            set_col(bb, j, wcol + sg * eps * v); gs.append(col(backward_all([lins[bb].weight])[0], j))
        set_col(bb, j, wcol); kap[d].append(float(((gs[0] - gs[1]) @ v) / (2 * eps)))
    if n_ % 20 == 0: log(f"{name} step{t0}: row {n_ + 1}/{sel.numel()} done ({time.time() - t_start:.0f} s)")
K = {d: torch.tensor(v) for d, v in kap.items()}; rnd = (K["random_1"] + K["random_2"]) / 2
res = dict(model=name, origin=t0, n_entrants=int(is_ent.sum()), n_matched=int((~is_ent).sum()), eps_rel=EPS_REL, groups={})
for gname, gm in (("entrants", is_ent), ("matched", ~is_ent), ("all", torch.ones_like(is_ent))):
    res["groups"][gname] = dict(median_curvature={d: float(K[d][gm].median()) for d in DIRS}, median_ratio_to_random={d: float((K[d][gm] / rnd[gm].abs().clamp_min(1e-12)).median()) for d in ("update_1000", "negative_gradient", "own_direction")},
                                fraction_flatter_than_random={d: float((K[d][gm] < rnd[gm]).float().mean()) for d in ("update_1000", "negative_gradient", "own_direction")}, random_agreement=float(((K["random_1"][gm] - K["random_2"][gm]).abs() / rnd[gm].abs().clamp_min(1e-12)).median()), fraction_negative_curvature={d: float((K[d][gm] < 0).float().mean()) for d in DIRS})
res["auc_matched_over_entrants"] = {d: auc((K[d] / rnd.abs().clamp_min(1e-12))[~is_ent], (K[d] / rnd.abs().clamp_min(1e-12))[is_ent]) for d in ("update_1000", "negative_gradient", "own_direction")}
e_, m_, a_ = res["groups"]["entrants"], res["groups"]["matched"], res["groups"]["all"]; fm = lambda x: "n/a" if x is None else f"{x:.2f}"
res["checks"] = dict(update_flatter_than_random=e_["median_ratio_to_random"]["update_1000"] < 0.7 and m_["median_ratio_to_random"]["update_1000"] < 0.7, gradient_steeper_than_random=a_["median_ratio_to_random"]["negative_gradient"] > 2, entrants_flatter_than_matched=(res["auc_matched_over_entrants"]["update_1000"] or 0) >= 0.6, own_direction_steeper=a_["median_ratio_to_random"]["own_direction"] > 1.5)
summ = (f"{name} step{t0} ({res['n_entrants']} entrants, {res['n_matched']} matched; curvature of the mean loss in the row's coordinates, central differences at {EPS_REL} of the row's norm): median curvature along the thousand-step update / negative gradient / own direction / random, entrants " + "/".join(f"{e_['median_curvature'][d]:.2e}" for d in DIRS[:4]) + ", matched " + "/".join(f"{m_['median_curvature'][d]:.2e}" for d in DIRS[:4])
        + f"; ratio to the random directions' curvature, all rows: update {a_['median_ratio_to_random']['update_1000']:.2f} (flatter than random for {a_['fraction_flatter_than_random']['update_1000']:.2f}), gradient {a_['median_ratio_to_random']['negative_gradient']:.2f}, own direction {a_['median_ratio_to_random']['own_direction']:.2f}; entrants {e_['median_ratio_to_random']['update_1000']:.2f}/{e_['median_ratio_to_random']['negative_gradient']:.2f}/{e_['median_ratio_to_random']['own_direction']:.2f}, matched {m_['median_ratio_to_random']['update_1000']:.2f}/{m_['median_ratio_to_random']['negative_gradient']:.2f}/{m_['median_ratio_to_random']['own_direction']:.2f}; AUC matched over entrants (update) {fm(res['auc_matched_over_entrants']['update_1000'])}; the two random directions differ by {a_['random_agreement']:.2f} of their mean; negative curvature along the update for {a_['fraction_negative_curvature']['update_1000']:.2f} of rows, along random {a_['fraction_negative_curvature']['random_1']:.2f} | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e527_curvature_{name}_step{t0}", res, summ)
