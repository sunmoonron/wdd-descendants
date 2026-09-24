"""e441: where does the knee of M come from? e432: removing M (top-8 principal directions at typical positions,
middle depth) costs 2-4x its local quadratic prediction. e438: freezing every later attention pattern halves the cost
but leaves the knee (1.9-2.5). The remaining nonlinearities between block L and the logits are the MLP activations and
the norms' scales. Here they are removed one at a time, cumulatively:
 live;
 P     attention patterns frozen at their clean values (value path live);
 P+M   and every later MLP linearised around its clean pre-activations (first-order Taylor of the activation; for
       gated MLPs, of act(gate) x up);
 P+M+N and every later norm's scale frozen at its clean value (centring kept), so the map from block L to the logits
       is linear;
 M     linear MLPs only.
For each: removing M_ns at typical positions (a = 0, sinks exact), its quadratic prediction from a = 0.75/1.25 and the
knee ratio. Control: a random 8-dimensional displacement with M's energy, live and P+M+N.
Pre-registered: the knee falls below 1.3 at P+M (MLP gating is the knee) in at least four of five models."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from p3_common import attn_mod, load_eager
from lr_common import eval_ids
name = sys.argv[1]; model, tok, fam = load_eager(name); arch = Arch(model, fam); L = arch.NB // 2; D, NH, HD = arch.D, arch.NH, arch.HD
for p in model.parameters(): p.requires_grad_(False)
E = eval_ids(name); ev = E[:4].to(DEV); fit = E[6:12].to(DEV); B, T = ev.shape; LATER = list(range(L + 1, arch.NB)); CH = 1
xf = block_states(model, arch, fit, [L], chunk=3)[L]; U_M, evs = pcs(xf[~sinkmask(xf)], 8)
xe = block_states(model, arch, ev, [L], chunk=2)[L]; ne_ = ~sinkmask(xe); mu = xe[ne_].mean(0)
disp = dict(M=(xe - mu) @ (U_M @ U_M.T))
sig2 = (disp["M"][ne_]).pow(2).sum(-1).mean() / 8; U_R = torch.linalg.qr(torch.randn(D, 8, generator=torch.Generator().manual_seed(0)))[0].to(DEV)
disp["rand8_matched"] = (torch.randn(B, T - 1, 8, generator=torch.Generator().manual_seed(1)).to(DEV) * sig2.sqrt()) @ U_R.T
CQ = getattr(model.config, "clip_qkv", None)
def values(l, xn):
    a = attn_mod(arch, l); b_, t_ = xn.shape[:2]
    if fam == "gpt2": return a.c_attn(xn)[..., 2 * D:].reshape(b_, t_, NH, HD)
    if fam == "neox": return a.query_key_value(xn).view(b_, t_, NH, 3 * HD)[..., 2 * HD:]
    v = a.v_proj(xn)
    if CQ is not None: v = v.clamp(-CQ, CQ)
    nkv = v.shape[-1] // HD; return v.view(b_, t_, nkv, HD).repeat_interleave(NH // nkv, dim=2)
def mlp_parts(l):
    m = arch.layers[l].mlp
    if fam == "gpt2": return m, (m.c_fc,), m.act, m.c_proj
    if fam == "neox": return m, (m.dense_h_to_4h,), m.act, m.dense_4h_to_h
    return m, (m.gate_proj, m.up_proj), m.act_fn, m.down_proj
def dact(f, p, h=1e-3): return (f(p + h) - f(p - h)) / (2 * h)
def norms(l):
    y = arch.layers[l]; return [y.ln_1, y.ln_2] if fam == "gpt2" else [y.input_layernorm, y.post_attention_layernorm]
final_norm = model.transformer.ln_f if fam == "gpt2" else (model.gpt_neox.final_layer_norm if fam == "neox" else model.model.norm)
NORMS = [n for l in LATER for n in norms(l)] + [final_norm]
def is_rms(n): return "rms" in type(n).__name__.lower()
def eps_of(n):
    e = getattr(n, "eps", getattr(n, "variance_epsilon", 1e-5)); return e[0] if isinstance(e, (tuple, list)) else e
def scale(n, x):
    x = x.float(); xc = x if is_rms(n) else x - x.mean(-1, keepdim=True); return (xc.pow(2).mean(-1, keepdim=True) + eps_of(n)).rsqrt()
CP, CMLP, CS = {}, {}, {}
def run(Xnew, cond=()):
    losses = []
    for s0 in range(0, B, CH):
        ids = ev[s0:s0 + CH]; hs = []
        if Xnew is not None: hs.append(arch.layers[L].register_forward_hook(replace_hook(Xnew[s0:s0 + CH])))
        else:
            for l in LATER: hs.append(arch.layers[l].mlp.register_forward_pre_hook(lambda m, a, l=l: CMLP.__setitem__((s0, l), a[0].detach())))
            for i, n in enumerate(NORMS): hs.append(n.register_forward_pre_hook(lambda m, a, i=i: CS.__setitem__((s0, i), scale(m, a[0]).detach())))
        if "P" in cond:
            for l in LATER:
                def hk(m, args, kwargs, out, l=l):
                    xn = args[0] if len(args) > 0 else kwargs["hidden_states"]
                    z = torch.einsum("bhts,bshd->bthd", CP[(s0, l)].to(xn.dtype), values(l, xn)).reshape(xn.shape[0], xn.shape[1], NH * HD)
                    o = arch.attn_lin(l)(z); return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o
                hs.append(attn_mod(arch, l).register_forward_hook(hk, with_kwargs=True))
        if "M" in cond:
            for l in LATER:
                def mk(m, args, out, l=l):
                    x = args[0]; x0 = CMLP[(s0, l)]; _, ins, f, down = mlp_parts(l)
                    if len(ins) == 1:
                        p, p0 = ins[0](x), ins[0](x0); h = f(p0) + dact(f, p0) * (p - p0)
                    else:
                        g, u, g0, u0 = ins[0](x), ins[1](x), ins[0](x0), ins[1](x0); h = f(g0) * u0 + dact(f, g0) * u0 * (g - g0) + f(g0) * (u - u0)
                    return down(h)
                hs.append(arch.layers[l].mlp.register_forward_hook(mk))
        if "N" in cond:
            for i, n in enumerate(NORMS):
                def nk(m, args, out, i=i):
                    x = args[0].float(); xc = x if is_rms(m) else x - x.mean(-1, keepdim=True); y = xc * CS[(s0, i)]
                    w, b = getattr(m, "weight", None), getattr(m, "bias", None)
                    if w is not None: y = y * w.float()
                    if b is not None: y = y + b.float()
                    return y.to(out.dtype)
                hs.append(n.register_forward_hook(nk))
        try:
            with torch.no_grad(): o = model(ids, output_attentions=(Xnew is None))
        finally: [h.remove() for h in hs]
        lg = o.logits.float(); lp = torch.log_softmax(lg[:, :-1], -1); losses.append(-lp.gather(2, ids[:, 1:, None])[..., 0])
        if Xnew is None:
            for l in LATER: CP[(s0, l)] = o.attentions[l].detach()
        del o, lg, lp
    return torch.cat(losses)
LC = run(None); lm = torch.zeros(B, T - 1, dtype=torch.bool, device=DEV); lm[:, 1:] = ne_[:, :-1]
chk = run(xe.clone(), cond=("P", "M", "N")); res = dict(model=name, level=L, identity_check=(chk - LC)[lm].abs().mean().item(), runs={})
log(f"{name}: clean state through the linearised network: mean |dL| {res['identity_check']:.5f} (should be ~0)")
CONDS = dict(live=(), P=("P",), P_M=("P", "M"), P_M_N=("P", "M", "N"), M=("M",))
for dn, d in disp.items():
    for cn, cond in CONDS.items():
        if dn != "M" and cn not in ("live", "P_M_N"): continue
        out = {}
        for a in (0.0, 0.75, 1.25):
            X = xe.clone(); X[ne_] = (xe + (a - 1) * d)[ne_]; out[str(a)] = (run(X, cond) - LC)[lm].mean().item()
        up, dn_ = out["1.25"], out["0.75"]; g_, h_ = (up - dn_) / 0.5, (up + dn_) / 0.125; out["quad_pred"] = -g_ + h_; out["knee"] = out["0.0"] / max(abs(out["quad_pred"]), 1e-6)
        res["runs"][f"{dn}_{cn}"] = out; log(f"{name} {dn} {cn}: dL a=0 {out['0.0']:+.3f}, quadratic {out['quad_pred']:+.3f}, knee {out['knee']:.2f}")
R = res["runs"]; res["checks"] = dict(knee_gone_at_P_M=R["M_P_M"]["knee"] < 1.3)
summ = (f"{name} L{L}: removing M, dL (knee): " + " ; ".join(f"{cn} {R['M_' + cn]['0.0']:+.3f} ({R['M_' + cn]['knee']:.2f})" for cn in CONDS)
        + f" | matched random live {R['rand8_matched_live']['0.0']:+.3f} ({R['rand8_matched_live']['knee']:.2f}), linear {R['rand8_matched_P_M_N']['0.0']:+.3f} ({R['rand8_matched_P_M_N']['knee']:.2f}) | identity check {res['identity_check']:.5f} | checks {json.dumps(res['checks'])}")
log(summ); record(f"e441_kneedecomp_{name}", res, summ)
