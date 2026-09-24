"""e438: is the knee of M the attention softmax? e432 found that removing M costs 2-4x its local quadratic prediction
(M: the top-8 principal directions at typical positions, middle depth), while a random displacement of the same energy
stays quadratic. e433 found that M's token part and within-token part both show the knee. Phase 2 (e372, e374)
traced the knee of circuit ablations mostly to the softmax.
Here M_ns is scaled by a in {0, 0.75, 1.25} at typical positions (sinks exact), with every later block's attention
patterns either live or frozen at their clean values. Frozen means the value path stays live: each head's output is
(clean pattern) x (values of the perturbed state).
Also recorded: how far the live patterns move (mean total-variation distance per query row, averaged over heads and
later layers). Control: a random 8-dimensional displacement with M's energy (e432's rand8_matched).
Pre-registered: with frozen patterns the knee ratio of M falls below 1.5 in at least four of the five models (the knee
is the softmax), and removing M moves the live patterns over three times as much as the matched random displacement."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from p3_common import attn_mod, load_eager
from lr_common import eval_ids
name = sys.argv[1]; model, tok, fam = load_eager(name); arch = Arch(model, fam); L = arch.NB // 2; D, NH, HD = arch.D, arch.NH, arch.HD
for p in model.parameters(): p.requires_grad_(False)
E = eval_ids(name); ev = E[:4].to(DEV); fit = E[6:12].to(DEV); B, T = ev.shape; LATER = list(range(L + 1, arch.NB)); CH = 1
xf = block_states(model, arch, fit, [L], chunk=3)[L]; U_M, evs = pcs(xf[~sinkmask(xf)], 8)
xe = block_states(model, arch, ev, [L], chunk=2)[L]; ne_ = ~sinkmask(xe); mu = xe[ne_].mean(0); PM = U_M @ U_M.T
disp = dict(M=(xe - mu) @ PM)
sig2 = (disp["M"][ne_]).pow(2).sum(-1).mean() / 8; U_R = torch.linalg.qr(torch.randn(D, 8, generator=torch.Generator().manual_seed(0)))[0].to(DEV)
disp["rand8_matched"] = (torch.randn(B, T - 1, 8, generator=torch.Generator().manual_seed(1)).to(DEV) * sig2.sqrt()) @ U_R.T

def values(l, xn):
    a = attn_mod(arch, l); b_, t_ = xn.shape[:2]
    if fam == "gpt2": v = a.c_attn(xn)[..., 2 * D:].reshape(b_, t_, NH, HD)
    elif fam == "neox": v = a.query_key_value(xn).view(b_, t_, NH, 3 * HD)[..., 2 * HD:]
    else:
        v = a.v_proj(xn); cq = getattr(model.config, "clip_qkv", None)
        if cq is not None: v = v.clamp(-cq, cq)                                  # OLMo clips q, k and v
        nkv = v.shape[-1] // HD; v = v.view(b_, t_, nkv, HD).repeat_interleave(NH // nkv, dim=2)
    return v
CLEANP, CLEANL = {}, {}
def run(Xnew, frozen=False, measure=False):
    """per-position loss [B, T-1]; optionally the mean TV distance of the live patterns from the clean ones per layer"""
    losses, tv = [], {l: 0.0 for l in LATER}
    for s0 in range(0, B, CH):
        ids = ev[s0:s0 + CH]; hs = []; pats = {}
        if Xnew is not None: hs.append(arch.layers[L].register_forward_hook(replace_hook(Xnew[s0:s0 + CH])))
        if frozen:
            for l in LATER:
                def hk(m, args, kwargs, out, l=l):
                    xn = args[0] if len(args) > 0 else kwargs["hidden_states"]; P = CLEANP[(s0, l)].to(xn.dtype)
                    z = torch.einsum("bhts,bshd->bthd", P, values(l, xn)).reshape(xn.shape[0], xn.shape[1], NH * HD); o = arch.attn_lin(l)(z)
                    return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o
                hs.append(attn_mod(arch, l).register_forward_hook(hk, with_kwargs=True))
        try:
            with torch.no_grad(): o = model(ids, output_attentions=(Xnew is None or measure))
        finally: [h.remove() for h in hs]
        lg = o.logits.float(); lp = torch.log_softmax(lg[:, :-1], -1); losses.append(-lp.gather(2, ids[:, 1:, None])[..., 0])
        if Xnew is None:
            for l in LATER: CLEANP[(s0, l)] = o.attentions[l].detach()
        elif measure:
            for l in LATER: tv[l] += 0.5 * (o.attentions[l] - CLEANP[(s0, l)]).abs().sum(-1).mean().item() / (B // CH)
        del o, lg, lp
    return torch.cat(losses), tv
LC, _ = run(None); lm = torch.zeros(B, T - 1, dtype=torch.bool, device=DEV); lm[:, 1:] = ne_[:, :-1]
res = dict(model=name, level=L, M_var_share=(evs[:8].sum() / evs.sum()).item(), runs={})
for dn, d in disp.items():
    for fz in (False, True):
        out = {}
        for a in (0.0, 0.75, 1.25):
            X = xe.clone(); X[ne_] = (xe + (a - 1) * d)[ne_]; l_, tv = run(X, frozen=fz, measure=(a == 0.0 and not fz))
            out[str(a)] = (l_ - LC)[lm].mean().item()
            if a == 0.0 and not fz: out["tv_by_layer"] = tv; out["tv_mean"] = sum(tv.values()) / len(tv)
        up, dn_ = out["1.25"], out["0.75"]; g_, h_ = (up - dn_) / 0.5, (up + dn_) / 0.125; out["quad_pred"] = -g_ + h_; out["knee"] = out["0.0"] / max(abs(out["quad_pred"]), 1e-6)
        res["runs"][f"{dn}_{'frozen' if fz else 'live'}"] = out
        log(f"{name} {dn} {'frozen' if fz else 'live'}: dL a=0 {out['0.0']:+.3f}, quadratic {out['quad_pred']:+.3f}, knee {out['knee']:.2f}" + (f" | pattern TV {out['tv_mean']:.4f}" if "tv_mean" in out else ""))
R = res["runs"]
res["checks"] = dict(frozen_knee_under_1_5=R["M_frozen"]["knee"] < 1.5, tv_M_over_3x_random=R["M_live"]["tv_mean"] > 3 * R["rand8_matched_live"]["tv_mean"])
summ = (f"{name} L{L}: removing M live dL {R['M_live']['0.0']:+.3f} (knee {R['M_live']['knee']:.2f}) vs patterns frozen {R['M_frozen']['0.0']:+.3f} (knee {R['M_frozen']['knee']:.2f}) | matched random live {R['rand8_matched_live']['0.0']:+.3f} (knee {R['rand8_matched_live']['knee']:.2f}) "
        f"frozen {R['rand8_matched_frozen']['0.0']:+.3f} (knee {R['rand8_matched_frozen']['knee']:.2f}) | pattern TV per row: M {R['M_live']['tv_mean']:.4f}, matched random {R['rand8_matched_live']['tv_mean']:.4f} | checks {json.dumps(res['checks'])}")
log(summ); record(f"e438_kneesoftmax_{name}", res, summ)
