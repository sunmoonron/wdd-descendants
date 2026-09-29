"""e589 (session 109): the mechanism of the turn. The record has: the class exists, the row nearest to it is recruited
(e587), and the row turns toward directions where the cloud already has variance (e571). Missing is the learning signal.
For the tracked words of e584 recruited at checkpoint index 3 or later, at the checkpoint two before recruitment, the
gradient of the language-model loss (24 sequences of 512 tokens) with respect to the row's write column, decomposed by
the positions the writes were made at: the full gradient and the part contributed at the class's positions (the
class = the row's over-the-floor positions at 16000). The step direction (minus the gradient) against the class's own
direction at that checkpoint (the unit mean of its unit states, row-free) and against the row's direction at 16000;
the share of the gradient's size contributed at the class's positions against the class's share of positions.
Controls: random rows paired with the same classes; matched non-recruits (non-words at the same checkpoint with the
same cosine to the class's direction that are never words by 16000); and a class-shuffled null (each recruit's
gradient against another recruit's class direction). Pre-registered (probabilities are honest guesses):
 M1 (0.6) the eventual row's step points toward the class's direction two checkpoints before recruitment (median
    cosine 0.2 or more), above the matched non-recruits and the shuffled null;
 M2 (0.6) the class's positions contribute a share of the row's gradient several times their share of positions
    (the turn is driven by the class's own loss);
 M3 (0.5) the step points toward the row's 16000 direction more than toward its current direction (the turn, not
    growth)."""
from s101_common import *
t0 = time.time(); CD = "/workspace/wdd/cache/e582_pythia410"; steps = list(range(1000, 16001, 1000)); B = 12; ids = eval_ids("pythia410")[:24, :512].to(DEV); T_ = ids.shape[1]
ck = {n: torch.load(f"{CD}/step{n}.pt", map_location="cpu") for n in steps}; keepc = torch.stack([ck[n]["keep"] for n in steps]).all(0); kidx = torch.nonzero(keepc)[:, 0]; N = int(keepc.sum())
U = {n: unitr((ck[n]["X"][keepc].float() - ck[n]["X"][keepc].float().mean(0)).to(DEV)) for n in steps}; A = {n: ck[n]["rows"].float().to(DEV) for n in steps}; DFF = ck[steps[0]]["DFF"]; m = A[steps[0]].shape[0]
ev = json.load(open("/workspace/wdd/results/e584_longitudinal.json"))["events"]; st16 = stats(U[16000], A[16000], K); R16 = st16["ratio"].float(); W16 = wordset(st16["usage"])
tracked = [(int(w), e["row"]) for w, e in ev.items() if e["row"] is not None and e["row"] >= 3]; by_ck = {}
for w, tr in tracked: by_ck.setdefault(steps[tr - 2], []).append(w)
never = torch.zeros(m, dtype=torch.bool); never[torch.nonzero(~torch.stack([wordset(stats(U[n], A[n], K)["usage"]) for n in (4000, 8000, 12000, 16000)]).any(0))[:, 0]] = True; log(f"{len(tracked)} tracked words over {len(by_ck)} checkpoints; {int(never.sum())} rows never words at 4000-16000")
g = torch.Generator().manual_seed(0); res = dict(groups={g_: [] for g_ in ("recruits", "matched", "random", "shuffled")})
def grad_pieces(n, rows_needed):
    """one forward and backward at checkpoint n: for every MLP neuron of blocks 0-12 the activation at each position and the loss gradient at the down-projection's output"""
    model, _, fam = load_model("pythia410", revision=f"step{n}"); arch = Arch(model, fam)
    for p_ in model.parameters(): p_.requires_grad_(False)
    ACT, DEL = {}, {}; hs = []
    for b in range(B + 1):
        lin = arch.mlp_lin(b)
        hs.append(lin.register_forward_pre_hook((lambda b_: lambda mm, a: ACT.__setitem__(b_, a[0].detach()[:, 1:].reshape(-1, a[0].shape[-1])))(b)))
        hs.append(lin.register_forward_hook((lambda b_: lambda mm, i, o: (o.requires_grad_(True), o.register_hook(lambda gr: DEL.__setitem__(b_, gr.detach()[:, 1:].reshape(-1, gr.shape[-1]))))[0])(b)))
    with torch.enable_grad():
        lg = model(ids).logits.float(); loss = torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1), reduction="sum"); loss.backward()
    [h.remove() for h in hs]; del model, lg, loss; torch.cuda.empty_cache()
    out = {}
    for r in rows_needed:
        b, j = r // DFF, r % DFF; a = ACT[b][:, j].float()[keepc.to(DEV)]; d = DEL[b].float()[keepc.to(DEV)]   # [N], [N, D]
        out[r] = dict(a=a, d=d)
    del ACT, DEL; torch.cuda.empty_cache(); return out
for n, ws in sorted(by_ck.items()):
    dC = {w: unitr(U[n][torch.nonzero(R16[:, w] > 1)[:, 0].to(DEV)].mean(0, keepdim=True))[0] for w in ws}; cosall = {w: (A[n] @ dC[w]).cpu() for w in ws}
    matched = {}; rnd = {}
    for w in ws:
        c = cosall[w]; target = float(c[w]); cand = torch.nonzero(never & (c - target).abs().lt(0.02))[:, 0]; cand = cand[cand != w]
        matched[w] = int(cand[torch.randint(0, cand.numel(), (1,), generator=g)]) if cand.numel() else None; rnd[w] = int(torch.randint(0, m, (1,), generator=g))
    need = sorted(set(ws) | set(v for v in matched.values() if v is not None) | set(rnd.values())); GP = grad_pieces(n, need)
    def measure(r, w, tag):
        a, d = GP[r]["a"], GP[r]["d"]; cls = torch.nonzero(R16[:, w] > 1)[:, 0].to(DEV); gfull = (a[:, None] * d).sum(0); gcls = (a[cls, None] * d[cls]).sum(0); step_full, step_cls = -gfull, -gcls
        return dict(cos_step_class=float(unitr(step_full[None])[0] @ dC[w]), cos_stepcls_class=float(unitr(step_cls[None])[0] @ dC[w]) if gcls.norm() > 0 else 0.0, cos_step_row16=float(unitr(step_full[None])[0] @ A[16000][r]), cos_step_rownow=float(unitr(step_full[None])[0] @ A[n][r]), cos_row_class=float(A[n][r] @ dC[w]),
                    class_share_of_grad=float(gcls.norm() / gfull.norm().clamp_min(1e-12)), class_share_of_positions=float(cls.numel() / N), class_share_of_activation=float(a[cls].abs().sum() / a.abs().sum().clamp_min(1e-12)), grad_norm=float(gfull.norm()), checkpoint=n)
    for w in ws:
        res["groups"]["recruits"].append(measure(w, w, "recruit"))
        if matched[w] is not None: res["groups"]["matched"].append(measure(matched[w], w, "matched"))
        res["groups"]["random"].append(measure(rnd[w], w, "random"))
    if len(ws) > 1:
        for i, w in enumerate(ws): w2 = ws[(i + 1) % len(ws)]; r_ = measure(w, w2, "shuffled"); res["groups"]["shuffled"].append(r_)
    del GP; torch.cuda.empty_cache(); log(f"checkpoint {n}: {len(ws)} recruits done ({time.time() - t0:.0f}s)")
agg = {}
for g_, L in res["groups"].items():
    if not L: continue
    agg[g_] = {k: med([o[k] for o in L]) for k in ("cos_step_class", "cos_stepcls_class", "cos_step_row16", "cos_step_rownow", "cos_row_class", "class_share_of_grad", "class_share_of_positions", "class_share_of_activation")}; agg[g_]["n"] = len(L); agg[g_]["share_step_toward_class_over_02"] = mean([float(o["cos_step_class"] > 0.2) for o in L])
    a_ = agg[g_]; log(f"{g_} ({a_['n']}): step vs class direction {a_['cos_step_class']:.3f} (over 0.2 for {a_['share_step_toward_class_over_02']:.2f}; the class positions' own part {a_['cos_stepcls_class']:.3f}), step vs the row's 16000 direction {a_['cos_step_row16']:.3f}, vs its current direction {a_['cos_step_rownow']:.3f}; row vs class now {a_['cos_row_class']:.3f}; class share of the gradient {a_['class_share_of_grad']:.3f} (of positions {a_['class_share_of_positions']:.3f}, of activation {a_['class_share_of_activation']:.3f})")
res["aggregate"] = agg; Rc, Mc, Rd, Sh = agg["recruits"], agg.get("matched", {}), agg["random"], agg.get("shuffled", {})
summ = (f"the gradient two checkpoints before recruitment ({Rc['n']} recruits): the step points toward the class's direction at cosine {Rc['cos_step_class']:.3f} (over 0.2 for {Rc['share_step_toward_class_over_02']:.2f}) against matched non-recruits {Mc.get('cos_step_class', float('nan')):.3f}, random rows {Rd['cos_step_class']:.3f}, class-shuffled {Sh.get('cos_step_class', float('nan')):.3f}; toward the row's 16000 direction {Rc['cos_step_row16']:.3f} (matched {Mc.get('cos_step_row16', float('nan')):.3f}) and its current direction {Rc['cos_step_rownow']:.3f}; the class's positions contribute {Rc['class_share_of_grad']:.3f} of the gradient's size against {Rc['class_share_of_positions']:.3f} of positions and {Rc['class_share_of_activation']:.3f} of the row's activation (matched {Mc.get('class_share_of_grad', float('nan')):.3f}); the class part of the step alone points at the class at {Rc['cos_stepcls_class']:.3f} | {time.time() - t0:.0f}s")
log(summ); record("e589_gradient_turn", res, summ)
