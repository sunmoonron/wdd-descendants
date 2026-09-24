"""e433: is M the identity channel? Here M is the top-8 principal directions at typical positions, middle depth.
What e432 found about M:
- Removed, it costs 2-4x its local quadratic prediction; a random move of the same energy stays quadratic.
- The next block carries a displacement along M forward whole, while erasing 25-50% of a random one.
- Its strongest writers are block-0 MLP rows in GPT-2 (M-share 0.82-0.85) and blocks 0-1 in OLMo.
Hypothesis: M keeps the token's identity (and, in GPT-2, its position). Downstream reads identity categorically, so
small moves along M do not change which token it is but full removal does (the knee), and identity has to survive to
the output (persistence).
Per model: typical positions only (sinks excluded); token and position means from the 64 extra sequences (cen);
evaluated on 6 sequences.
 (a) Out of sample, the share of the variance in M explained by the token's mean, by the position's mean, by both
     (additive), and by the state after block 0 (ridge regression: how much of M the embedding and block 0 already
     wrote). The same for principal directions 33-40, a random 8-dimensional subspace and the whole state.
 (b) M split into its token part P(m(v) - mu) and the within-token rest P(x - m(v)), plus the position part. For each:
     the cost of removing it, its quadratic prediction from a = 0.75/1.25 (knee ratio), and the share of a
     half-removal the next block keeps.
Pre-registered:
- token identity explains over half of M's variance and over twice the share in directions 33-40, in all five;
- in GPT-2, position explains more than token;
- removing M's token part costs over two thirds of removing M and shows the knee (ratio > 2); the within-token rest
  does not (ratio < 1.5)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
name = sys.argv[1]; model, tok, fam = load_model(name); arch = Arch(model, fam); L = arch.NB // 2; D = arch.D
for p in model.parameters(): p.requires_grad_(False)
EA = torch.load("/workspace/wdd/eval_ids_all.pt", weights_only=False)[name]; ev = EA["eval_ids"][:6].to(DEV); cen = EA["cen_ids"].to(DEV); del EA
S = block_states(model, arch, cen, [0, L], chunk=4); xc, h0c = S[L], S[0]
S = block_states(model, arch, ev, [0, L], chunk=4); xe, h0e = S[L], S[0]; del S
nc, ne_ = ~sinkmask(xc), ~sinkmask(xe)
Xc = xc[nc]; mu = Xc.mean(0); U40, evs = pcs(Xc, 40); U_M, U_33 = U40[:, :8].contiguous(), U40[:, 32:40].contiguous()
U_R = torch.linalg.qr(torch.randn(D, 8, generator=torch.Generator().manual_seed(0)))[0].to(DEV)
vc = cen[:, 1:][nc]; uniq, inv, cnt = torch.unique(vc, return_inverse=True, return_counts=True)
tmean = torch.zeros(len(uniq), D, device=DEV).index_add_(0, inv, Xc) / cnt[:, None]
posc = torch.arange(cen.shape[1] - 1, device=DEV)[None].expand(cen.shape[0], -1)[nc]
pmean = torch.zeros(cen.shape[1] - 1, D, device=DEV).index_add_(0, posc, Xc) / torch.bincount(posc, minlength=cen.shape[1] - 1).clamp_min(1)[:, None]
Xe = xe[ne_]; ve = ev[:, 1:][ne_]; pe = torch.arange(ev.shape[1] - 1, device=DEV)[None].expand(ev.shape[0], -1)[ne_]
ix = torch.searchsorted(uniq, ve).clamp_max(len(uniq) - 1); known = (uniq[ix] == ve) & (cnt[ix] >= 3)
Mt = torch.where(known[:, None], tmean[ix], mu[None]); Mp = pmean[pe]
Hc, He = h0c[nc], h0e[ne_]; m0 = Hc.mean(0)
res = dict(model=name, level=L, n_typical_ev=int(ne_.sum()), known_token_share=known.float().mean().item(), var_share=dict(M=(evs[:8].sum() / evs.sum()).item()), explained={}, parts={})
subs = dict(M=U_M, pc33_40=U_33, rand8=U_R, whole=None)
for sn, U in subs.items():
    pr = (lambda Z: Z @ U) if U is not None else (lambda Z: Z)
    y = pr(Xe - mu); tot = y.pow(2).sum()
    r2 = lambda pred: 1 - ((y - pred).pow(2).sum() / tot).item()
    Yc = pr(Xc - mu); Hcc = Hc - m0; lam = 1e-3 * Hcc.pow(2).sum() / Hcc.shape[0]
    Wb = torch.linalg.solve((Hcc.T @ Hcc).double() + lam * torch.eye(D, device=DEV, dtype=torch.float64), (Hcc.T @ Yc).double()).float()
    res["explained"][sn] = dict(token=r2(pr(Mt - mu)), position=r2(pr(Mp - mu)), token_plus_position=r2(pr(Mt + Mp - 2 * mu)), block0_state=r2((He - m0) @ Wb),
                                token_known_only=1 - ((y - pr(Mt - mu))[known].pow(2).sum() / y[known].pow(2).sum()).item())
    del Yc, Wb
    log(f"{name} {sn}: explained by token {res['explained'][sn]['token']:.2f} (known tokens {res['explained'][sn]['token_known_only']:.2f}), position {res['explained'][sn]['position']:.2f}, "
        f"both {res['explained'][sn]['token_plus_position']:.2f}, block-0 state {res['explained'][sn]['block0_state']:.2f}")
# (b) functional split of M at typical positions (sinks exact)
sp = Splicer(model, arch, ev, L, track=[min(L + 1, arch.NB - 1)]); lm = sp.lossmask(ne_); PM = U_M @ U_M.T; b1 = min(L + 1, arch.NB - 1)
parts = dict(full=(Xe - mu) @ PM, token=(Mt - mu) @ PM, within_token=(Xe - Mt) @ PM, position=(Mp - mu) @ PM)
for pn, part in parts.items():
    out = {}
    for a in (0.0, 0.5, 0.75, 1.25):
        X = xe.clone(); X[ne_] = Xe + (a - 1) * part; r = sp.run(X, track=(a == 0.5)); out[str(a)] = sp.stats(r, lm)
        if a == 0.5:
            d0 = (a - 1) * part; db = r["track"][b1][ne_] - sp.c["track"][b1][ne_]
            out["kept_next_block"] = ((db @ U_M).pow(2).sum() / (d0 @ U_M).pow(2).sum().clamp_min(1e-9)).item()
        del r
    up, dn = out["1.25"]["dL"], out["0.75"]["dL"]; g_, h_ = (up - dn) / 0.5, (up + dn) / (2 * 0.0625); pred = -g_ + h_
    out["quad_pred_a0"] = pred; out["knee"] = out["0.0"]["dL"] / max(abs(pred), 1e-6); out["energy_share_of_M"] = (part.pow(2).sum() / parts["full"].pow(2).sum()).item()
    res["parts"][pn] = out
    log(f"{name} M part {pn}: energy {out['energy_share_of_M']:.2f} of M | removal dL {out['0.0']['dL']:+.3f} (quadratic {pred:+.3f}, knee {out['knee']:.1f}) | half-removal kept by the next block {out['kept_next_block']:.2f}")
E_, P_ = res["explained"], res["parts"]
res["checks"] = dict(token_over_half_of_M=E_["M"]["token"] > 0.5, token_M_over_2x_pc33=E_["M"]["token"] > 2 * E_["pc33_40"]["token"], position_beats_token=E_["M"]["position"] > E_["M"]["token"],
                     token_part_carries_2_3=P_["token"]["0.0"]["dL"] > (2 / 3) * P_["full"]["0.0"]["dL"], token_part_knee=P_["token"]["knee"] > 2, rest_no_knee=P_["within_token"]["knee"] < 1.5)
summ = (f"{name} L{L}: share of variance explained (token/position/both/block-0 state): " + " ; ".join(f"{sn} {E_[sn]['token']:.2f}/{E_[sn]['position']:.2f}/{E_[sn]['token_plus_position']:.2f}/{E_[sn]['block0_state']:.2f}" for sn in subs)
        + " | M parts (energy share, removal dL, knee, kept by next block): " + " ; ".join(f"{pn} {P_[pn]['energy_share_of_M']:.2f} {P_[pn]['0.0']['dL']:+.3f} {P_[pn]['knee']:.1f} {P_[pn]['kept_next_block']:.2f}" for pn in parts)
        + f" | known tokens {res['known_token_share']:.2f} | checks {json.dumps(res['checks'])}")
log(summ); record(f"e433_identity_{name}", res, summ)
