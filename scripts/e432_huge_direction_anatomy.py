"""e432: what are the huge directions? The survey of e000-e431 found M (the states' top-8 principal directions at middle
depth: 0.87 of the variance in Pythia-410m, 0.48 in OLMo, 0.24-0.29 in the others, with 0.015 of the Fisher trace in
Pythia) behind a dozen results that were never connected:
- phase 1's sink tokens and massive channels (e020, e118, e126, e170). Phase 1 always dropped positions above 10x the
  median norm; phase 3 never did.
- the set point (e290, e301, e331): removing SmolLM2's massive write moves the state as adding more does.
- normalisation (e344), the knee (e372, e374), false friends (e404) and the compounding replacement model (e392).
Five accounts, one pass per model at middle depth, 6 sequences x 512 tokens, subspaces fitted on 6 other sequences:
 H4 sinks: M is a handful of sink positions (norm above 10x the median).
    Audit: sinks per sequence and their share of the variance; M's share without them; the overlap of M (all
    positions) with M_ns (non-sink positions).
    Loss when M is mean-ablated at every position, M at non-sink positions only, M_ns at non-sink positions, and when
    the sinks alone are replaced by the typical mean state.
 H0 quadratic: M only has large variance and ordinary curvature.
    The centred M_ns part at non-sink positions is scaled by a (sinks exact). The quadratic fitted at a = 1 +- 0.1 and
    1 +- 0.25 predicts the cost at a = 0.
    Controls: principal directions 9-16 and 33-40, a random 8-dimensional subspace, and random displacements
    in it with M_ns's energy.
 H1 set point: the response to a small scaling is even per position, and the network restores the displacement
    downstream (retained share of the displacement in the subspace at later blocks, against the controls).
 H2 ballast: removing M costs through the norm it gives the state, not its direction. The M_ns mean-ablation is split
    into a direction-only change (ablate, then restore each state's norm as the next block's norm sees it) and a
    norm-only change (the unablated state rescaled to the ablated norm).
 H5 temperature: moving M changes the output's scale, not its ranking. Every intervention reports the entropy change,
    the ratio of centred-logit norms and their correlation with the clean logits.
Pre-registered:
 (H4) in Pythia and OLMo, sinks hold over half the variance and M_ns shares under half of M;
 (H0) the quadratic from a = 0.75/1.25 under-predicts M_ns's a = 0 cost by over 2x, and does not for the controls;
 (H1) along M_ns the per-position odd part is under a third of the even part at a = 1 +- 0.1, and above it for the
      controls;
 (H2) the norm-only change carries under a third of the cost;
 (H5) at a = 0.5 along M_ns the logit correlation stays above 0.9 while the entropy moves monotonically in a."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sd_common import *
from lr_common import eval_ids
name = sys.argv[1]; model, tok, fam = load_model(name); arch = Arch(model, fam); L = arch.NB // 2; D = arch.D
for p in model.parameters(): p.requires_grad_(False)
E = eval_ids(name); ev = E[:6].to(DEV); fit = E[6:12].to(DEV); B, T = ev.shape
TRACK = sorted(set(min(b, arch.NB - 1) for b in (L + 1, L + 2, L + arch.NB // 4, arch.NB - 1)))
nxt = arch.layers[min(L + 1, arch.NB - 1)]; RMS = "rms" in type(nxt.ln_1 if fam == "gpt2" else nxt.input_layernorm).__name__.lower()

def states(ids):
    out = {}; h = arch.layers[L].register_forward_hook(lambda m, i, o: out.__setitem__("x", (o[0] if isinstance(o, tuple) else o).detach().float()))
    try:
        with torch.no_grad(): model(ids)
    finally: h.remove()
    return out["x"][:, 1:]                                                   # positions 1..T-1: [B, T-1, D]
CLEAN, CH = {}, 2
def run(Xnew=None, track=False):
    """ev with block L's output at positions 1: replaced by Xnew [B, T-1, D] (None: clean). Per-position loss and entropy
    [B, T-1] (position j predicts token j+1), centred-logit norm ratio and cosine to the clean run; (track) the states at
    the TRACK blocks."""
    loss, ent, lnr, lcos, tr = [], [], [], [], {b: [] for b in TRACK}
    for s0 in range(0, B, CH):
        ids = ev[s0:s0 + CH]; hs = []; cap = {}
        if Xnew is not None:
            Xs = Xnew[s0:s0 + CH]
            def hk(m, i, o, Xs=Xs):
                xo = o[0] if isinstance(o, tuple) else o; y = xo.clone(); y[:, 1:] = Xs.to(xo.dtype)
                return (y,) + tuple(o[1:]) if isinstance(o, tuple) else y
            hs.append(arch.layers[L].register_forward_hook(hk))
        if track:
            for b in TRACK: hs.append(arch.layers[b].register_forward_hook(lambda m, i, o, b=b: cap.__setitem__(b, (o[0] if isinstance(o, tuple) else o)[:, 1:].detach().float())))
        try:
            with torch.no_grad(): lg = model(ids).logits.float()
        finally: [h.remove() for h in hs]
        lp = torch.log_softmax(lg[:, :-1], -1); loss.append(-lp.gather(2, ids[:, 1:, None])[..., 0]); ent.append(-(lp.exp() * lp).sum(-1))
        z = lg[:, :-1] - lg[:, :-1].mean(-1, keepdim=True)
        if Xnew is None: CLEAN[s0] = z.to(torch.bfloat16)
        zc = CLEAN[s0].float(); lnr.append(z.norm(dim=-1) / zc.norm(dim=-1)); lcos.append((z * zc).sum(-1) / (z.norm(dim=-1) * zc.norm(dim=-1)).clamp_min(1e-9))
        if track:
            for b in TRACK: tr[b].append(cap[b])
        del lg, lp, z, zc
    out = dict(loss=torch.cat(loss), ent=torch.cat(ent), lnr=torch.cat(lnr), lcos=torch.cat(lcos))
    if track: out["track"] = {b: torch.cat(v) for b, v in tr.items()}
    return out

# ---- audit (H4)
xf, xe = states(fit), states(ev)
def sinkmask(x): n = x.norm(dim=-1); return n > 10 * n.median(), n
skf, nf = sinkmask(xf); ske, ne = sinkmask(xe); ns = ~ske
Xf = xf.reshape(-1, D); mf = skf.reshape(-1)
C = torch.cov((Xf - Xf.mean(0)).T.double(), correction=0); U_all = torch.linalg.eigh(C)[1][:, -8:].float()
Cn = torch.cov((Xf[~mf] - Xf[~mf].mean(0)).T.double(), correction=0); evn, Un = torch.linalg.eigh(Cn); Un = Un.flip(-1).float(); evn = evn.flip(-1)
U_ns, U_9, U_33 = Un[:, :8].contiguous(), Un[:, 8:16].contiguous(), Un[:, 32:40].contiguous()
U_rand = torch.linalg.qr(torch.randn(D, 8, generator=torch.Generator().manual_seed(0)))[0].to(DEV)
mu_all, mu_ns = xe.reshape(-1, D).mean(0), xe[ns].mean(0)
Xa, Xn = xe.reshape(-1, D) - mu_all, xe[ns] - mu_ns
sh = lambda X, U: ((X @ U).pow(2).sum() / X.pow(2).sum()).item()
sk_pos = torch.nonzero(ske)[:24].tolist()
audit = dict(level=L, rms=RMS, n_sinks_ev=int(ske.sum()), n_sinks_fit=int(skf.sum()), sinks_per_seq=ske.sum(1).tolist(), max_norm_over_median=(ne.max() / ne.median()).item(),
             sink_tokens=[(s, p + 1, tok.decode([int(ev[s, p + 1])]), round((ne[s, p] / ne.median()).item(), 1)) for s, p in sk_pos],
             sink_share_of_variance=(Xa[ske.reshape(-1)].pow(2).sum() / Xa.pow(2).sum()).item() if ske.any() else 0.0,
             M_all_share_all=sh(Xa, U_all), M_all_share_ns=sh(Xn, U_all), M_ns_share_ns=sh(Xn, U_ns), pc9_16_share_ns=sh(Xn, U_9), pc33_40_share_ns=sh(Xn, U_33), rand8_share_ns=sh(Xn, U_rand),
             overlap_M_all_M_ns=((U_all.T @ U_ns).pow(2).sum() / 8).item(), mean_direction_in_M_ns=((U_ns.T @ (mu_ns / mu_ns.norm())).pow(2).sum()).item())
ch = mu_ns.abs().topk(3).indices; audit["massive_channels"] = ch.tolist(); audit["M_ns_mass_on_massive_channels"] = (U_ns[ch].pow(2).sum() / 8).item()
en = (Xn @ U_ns).pow(2).sum(-1); nn_ = xe[ns].norm(dim=-1).pow(2); audit["corr_M_ns_energy_state_norm"] = torch.corrcoef(torch.stack([en, nn_]))[0, 1].item()
W = torch.cat([unitr(arch.wdir(b)) for b in range(L + 1)]); ws = (W @ U_ns).pow(2).sum(-1); top = ws.topk(5)
audit["top_writers_M_ns"] = [(int(i) // arch.DFF, int(i) % arch.DFF, round(v.item(), 3)) for v, i in zip(top.values, top.indices)]; del W
log(f"{name} L{L} audit: " + json.dumps({k: v for k, v in audit.items() if k != "sink_tokens"})); log("sinks: " + str(audit["sink_tokens"][:12]))

# ---- runs
c0 = run(None, track=True); LC = c0["loss"]
lm = {k: torch.zeros(B, T - 1, dtype=torch.bool, device=DEV) for k in ("all", "ns", "sk")}
lm["all"][:, 1:] = True; lm["ns"][:, 1:] = ns[:, :-1]; lm["sk"][:, 1:] = ske[:, :-1]
def stats(r, tag=""):
    d = r["loss"] - LC; o = dict(dL_all=d[lm["all"]].mean().item(), dL_ns=d[lm["ns"]].mean().item(), dL_sk=d[lm["sk"]].mean().item() if lm["sk"].any() else None,
                                 dH_ns=(r["ent"] - c0["ent"])[lm["ns"]].mean().item(), logit_norm_ratio=r["lnr"][lm["ns"]].mean().item(), logit_cos=r["lcos"][lm["ns"]].mean().item())
    if tag: log(f"{name} {tag}: " + " ".join(f"{k} {v:+.4f}" for k, v in o.items() if v is not None))
    return o
PR = lambda U: U @ U.T
res = dict(model=name, audit=audit, track_blocks=TRACK, ablate={}, scale={}, norm_split={}, restore={})
X = xe.clone(); X = X - (X - mu_all) @ PR(U_all); res["ablate"]["M_all_everywhere"] = stats(run(X), "M_all mean-ablated everywhere")
X = xe.clone(); X[ns] = (xe - (xe - mu_all) @ PR(U_all))[ns]; res["ablate"]["M_all_nonsink"] = stats(run(X), "M_all at non-sink positions")
X = xe.clone(); X[ns] = (xe - (xe - mu_ns) @ PR(U_ns))[ns]; res["ablate"]["M_ns_nonsink"] = stats(run(X), "M_ns at non-sink positions")
if ske.any(): X = xe.clone(); X[ske] = mu_ns; res["ablate"]["sinks_to_mean"] = stats(run(X), "sinks replaced by the typical mean")
X = xe.clone(); X[ns] = mu_ns; res["ablate"]["whole_state_nonsink"] = stats(run(X), "whole state mean-ablated at non-sink positions")

# ---- scale sweeps (H0, H1, H5) at non-sink positions, sinks exact
AS = [0.0, 0.25, 0.5, 0.75, 0.9, 1.1, 1.25, 1.5]
sig2 = (Xn @ U_ns).pow(2).sum(-1).mean() / 8
Z = torch.randn(B, T - 1, 8, generator=torch.Generator().manual_seed(1)).to(DEV) * sig2.sqrt(); NOISE = Z @ U_rand.T
fams = dict(M_ns=U_ns, pc9_16=U_9, pc33_40=U_33, rand8=U_rand, rand8_matched=None)
per = {}
for fn, U in fams.items():
    disp = NOISE if U is None else (xe - mu_ns) @ PR(U); Uf = U_rand if U is None else U; res["scale"][fn] = {}
    for a in AS:
        X = xe.clone(); X[ns] = (xe + (a - 1) * disp)[ns]; tr = a in (0.5, 1.5); r = run(X, track=tr)
        res["scale"][fn][str(a)] = stats(r); per[(fn, a)] = r["loss"]
        if tr:
            d0 = ((a - 1) * disp)[ns]; e0 = (d0 @ Uf).pow(2).sum(); t0 = d0.pow(2).sum(); rr = {}
            for b in TRACK:
                db = (r["track"][b] - c0["track"][b]); rr[str(b)] = dict(in_subspace=((db[ns] @ Uf).pow(2).sum() / e0).item(), total=(db.pow(2).sum() / t0).item())
            res["restore"][f"{fn}@{a}"] = rr
        del r
    s = res["scale"][fn]; q = {}
    for eps in (0.1, 0.25):
        up, dn = s[str(round(1 + eps, 2))]["dL_ns"], s[str(round(1 - eps, 2))]["dL_ns"]; g_, h_ = (up - dn) / (2 * eps), (up + dn) / (2 * eps * eps)
        q[f"pred_a0_eps{eps}"] = -g_ + h_; q[f"pred_a05_eps{eps}"] = -0.5 * g_ + 0.25 * h_
    q["meas_a0"], q["meas_a05"] = s["0.0"]["dL_ns"], s["0.5"]["dL_ns"]
    q["knee_ratio_eps0.25"] = q["meas_a0"] / max(abs(q["pred_a0_eps0.25"]), 1e-6); q["knee_ratio_eps0.1"] = q["meas_a0"] / max(abs(q["pred_a0_eps0.1"]), 1e-6)
    m_ = lm["ns"]; odd = (per[(fn, 1.1)] - per[(fn, 0.9)]) / 2; even = (per[(fn, 1.1)] + per[(fn, 0.9)]) / 2 - LC
    q["odd_abs_per_pos"], q["even_per_pos"], q["odd_mean"] = odd[m_].abs().mean().item(), even[m_].mean().item(), odd[m_].mean().item()
    q["odd_over_even_per_pos"] = q["odd_abs_per_pos"] / max(abs(q["even_per_pos"]), 1e-9)
    q["entropy_by_a"] = [s[str(a)]["dH_ns"] for a in AS]; q["cost_per_variance_a0"] = q["meas_a0"] / max(audit["M_ns_share_ns"] if fn in ("M_ns", "rand8_matched") else audit[{"pc9_16": "pc9_16_share_ns", "pc33_40": "pc33_40_share_ns", "rand8": "rand8_share_ns"}[fn]], 1e-9)
    res["scale"][fn]["_fit"] = q
    log(f"{name} {fn}: dL_ns by a " + " ".join(f"{a}:{s[str(a)]['dL_ns']:+.3f}" for a in AS) + f" | a=0 measured {q['meas_a0']:+.3f} vs quadratic {q['pred_a0_eps0.25']:+.3f} (eps .25) {q['pred_a0_eps0.1']:+.3f} (eps .1)"
        f" | per-position odd/even {q['odd_over_even_per_pos']:.2f} | logit cos at a=.5 {s['0.5']['logit_cos']:.3f}, dH {s['0.5']['dH_ns']:+.3f}")
del per

# ---- norm or direction (H2)
def nrm(X): return (X.norm(dim=-1, keepdim=True) if RMS else (X - X.mean(-1, keepdim=True)).norm(dim=-1, keepdim=True)).clamp_min(1e-6)
for tag, U, mu, mask in [("M_ns_nonsink", U_ns, mu_ns, ns), ("M_all_everywhere", U_all, mu_all, torch.ones_like(ns))]:
    X0 = xe.clone(); X0[mask] = (xe - (xe - mu) @ PR(U))[mask]
    full = stats(run(X0)); dirn = stats(run(X0 * (nrm(xe) / nrm(X0)))); norm_ = stats(run(xe * (nrm(X0) / nrm(xe))))
    res["norm_split"][tag] = dict(full=full, direction_only=dirn, norm_only=norm_, norm_ratio_mean=(nrm(X0) / nrm(xe))[mask].mean().item())
    log(f"{name} {tag} norm split: full {full['dL_all']:+.3f}, direction-only {dirn['dL_all']:+.3f}, norm-only {norm_['dL_all']:+.3f} (ablated/original norm {res['norm_split'][tag]['norm_ratio_mean']:.3f})")

# ---- summary and pre-registered checks
F = {fn: res["scale"][fn]["_fit"] for fn in fams}; ab = res["ablate"]; nsx = res["norm_split"]["M_ns_nonsink"]
res["checks"] = dict(
    H4_sinks_hold_half=audit["sink_share_of_variance"] > 0.5, H4_Mns_shares_under_half=audit["overlap_M_all_M_ns"] < 0.5,
    H0_knee_Mns=F["M_ns"]["knee_ratio_eps0.25"] > 2, H0_knee_controls=[fn for fn in fams if fn != "M_ns" and F[fn]["knee_ratio_eps0.25"] > 2],
    H1_even_Mns=F["M_ns"]["odd_over_even_per_pos"] < 1 / 3, H1_even_controls=[fn for fn in fams if fn != "M_ns" and F[fn]["odd_over_even_per_pos"] < 1 / 3],
    H2_norm_under_third=abs(nsx["norm_only"]["dL_all"]) < abs(nsx["full"]["dL_all"]) / 3,
    H5_temperature=res["scale"]["M_ns"]["0.5"]["logit_cos"] > 0.9 and (all(x <= y for x, y in zip(F["M_ns"]["entropy_by_a"], F["M_ns"]["entropy_by_a"][1:])) or all(x >= y for x, y in zip(F["M_ns"]["entropy_by_a"], F["M_ns"]["entropy_by_a"][1:]))))
rb = lambda fn, a: res["restore"][f"{fn}@{a}"][str(TRACK[0])]["in_subspace"]
summ = (f"{name} L{L}: sinks {audit['n_sinks_ev']} (share of variance {audit['sink_share_of_variance']:.2f}; max norm {audit['max_norm_over_median']:.0f}x median) | M share all {audit['M_all_share_all']:.2f}, "
        f"M at non-sink {audit['M_all_share_ns']:.2f}, M_ns {audit['M_ns_share_ns']:.2f}, overlap M/M_ns {audit['overlap_M_all_M_ns']:.2f} | ablations dL: M everywhere {ab['M_all_everywhere']['dL_all']:+.3f}, M non-sink {ab['M_all_nonsink']['dL_all']:+.3f}, "
        f"M_ns non-sink {ab['M_ns_nonsink']['dL_all']:+.3f}" + (f", sinks->mean {ab['sinks_to_mean']['dL_all']:+.3f}" if "sinks_to_mean" in ab else "") + f", whole non-sink state {ab['whole_state_nonsink']['dL_all']:+.3f} | "
        f"knee (a=0 measured/quadratic): " + " ".join(f"{fn} {F[fn]['knee_ratio_eps0.25']:.1f}" for fn in fams) + " | odd/even per position: " + " ".join(f"{fn} {F[fn]['odd_over_even_per_pos']:.2f}" for fn in fams)
        + f" | cost per unit variance at a=0: " + " ".join(f"{fn} {F[fn]['cost_per_variance_a0']:.2f}" for fn in fams if fn != "rand8_matched") + f", matched noise dL {F['rand8_matched']['meas_a0']:+.3f}"
        + f" | M_ns norm split full/dir/norm {nsx['full']['dL_all']:+.3f}/{nsx['direction_only']['dL_all']:+.3f}/{nsx['norm_only']['dL_all']:+.3f} | M_ns a=.5 logit cos {res['scale']['M_ns']['0.5']['logit_cos']:.3f} norm ratio {res['scale']['M_ns']['0.5']['logit_norm_ratio']:.3f}"
        + f" | retained at L{TRACK[0]} (a=.5): " + " ".join(f"{fn} {rb(fn, 0.5):.2f}" for fn in fams) + f" | checks {json.dumps(res['checks'])}")
log(summ); record(f"e432_manatomy_{name}", res, summ)
