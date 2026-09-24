"""e429: where does the forward-backward duality hold? e426 found states better described by the writers' words and
loss gradients by the readers' words in Pythia, OLMo and SmolLM2, a tie in Qwen-0.5B and the reverse in GPT-2, at the
middle depth with MLP input rows as the readers. Here, at a quarter, half and three quarters of the depth, two reader
sets: the MLP input rows of the later blocks (as e426) and all their read directions (attention query/key/value input
rows as well), gain-scaled and centred; writers are the MLP write rows of blocks 0..L. Unit per-position gradients and
centred states, 16 OMP words, advantage over the same set rotated (fraction unexplained). Pre-registered: with the
attention readers included the duality (readers ahead on errors, writers ahead on states) holds at some depth in all
five models; no prediction for GPT-2's middle depth."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sd_common import *
from lr_common import eval_ids
name = sys.argv[1]; model, tok, fam = load_model(name); arch = Arch(model, fam); E = eval_ids(name); ev = E[:4].to(DEV)
for p in model.parameters(): p.requires_grad_(False)
def attn_reader_rows(blocks):
    rows = []
    for b in blocks:
        l = arch.layers[b]
        if fam == "gpt2": W, ln = l.attn.c_attn.weight.T, l.ln_1
        elif fam == "neox": W, ln = l.attention.query_key_value.weight, l.input_layernorm
        else: a = l.self_attn; W, ln = torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight]), l.input_layernorm
        w = getattr(ln, "weight", None); g_ = w.detach().float() if w is not None else torch.ones(arch.D, device=DEV)
        W = W.detach().float() * g_[None]
        if "rms" not in type(ln).__name__.lower(): W = W - W.mean(-1, keepdim=True)
        rows.append(W)
    return unitr(torch.cat(rows))
def grads_states(L):
    gs, xs = [], []; torch.set_grad_enabled(True)
    try:
        for s0 in range(0, ev.shape[0], 2):
            leaf = {}
            def hk(m, i, o):
                xo = o[0] if isinstance(o, tuple) else o; y = xo.detach().requires_grad_(True); leaf["x"] = y
                return (y,) + tuple(o[1:]) if isinstance(o, tuple) else y
            h = arch.layers[L].register_forward_hook(hk)
            try: lg = model(ev[s0:s0 + 2]).logits.float()
            finally: h.remove()
            token_loss(lg, ev[s0:s0 + 2]).sum().backward()
            gs.append(leaf["x"].grad[:, 1:-1].reshape(-1, arch.D).detach()); xs.append(leaf["x"][:, 1:-1].reshape(-1, arch.D).detach())
    finally: torch.set_grad_enabled(False)
    G, X = torch.cat(gs), torch.cat(xs); keep = G.norm(dim=-1) > 0
    return unitr(G[keep]), unitr(X[keep] - X[keep].mean(0, keepdim=True))
def fvu16(X, V):
    V = unitr(V); sel, _, _ = omp(X, V, 16, batch=256, record_err=False); _, err = refit(X, V, sel); return (err / X.pow(2).sum(-1)).mean().item()
res = dict(model=name, depths={})
for L in [arch.NB // 4, arch.NB // 2, (3 * arch.NB) // 4]:
    Gd, Sd = grads_states(L); later = list(range(L + 1, arch.NB))
    A, lab = build_dictionary(arch, blocks=list(range(L + 1))); W = A[lab["type"].to(DEV) == T_MLP].clone(); del A
    Rm = reader_rows(arch, later); Ra = torch.cat([Rm, attn_reader_rows(later)])
    out = {}
    for tg, X in [("grad", Gd), ("state", Sd)]:
        for nm, V in [("W", W), ("Rmlp", Rm), ("Rall", Ra)]: out[f"{tg}:{nm}"] = fvu16(X, rotate(V, seed=7)) - fvu16(X, V)
    res["depths"][str(L)] = out
    log(f"{name} L{L}: advantage over rotation, states W {out['state:W']:.3f} Rmlp {out['state:Rmlp']:.3f} Rall {out['state:Rall']:.3f} | errors W {out['grad:W']:.3f} Rmlp {out['grad:Rmlp']:.3f} Rall {out['grad:Rall']:.3f}")
    del W, Rm, Ra, Gd, Sd
ok = lambda L, R: res["depths"][L][f"grad:{R}"] > res["depths"][L]["grad:W"] and res["depths"][L]["state:W"] > res["depths"][L][f"state:{R}"]
summ = f"{name}: duality (readers ahead on errors, writers on states) with MLP readers at depths " + ",".join(L for L in res["depths"] if ok(L, "Rmlp")) + "; with all readers at " + ",".join(L for L in res["depths"] if ok(L, "Rall"))
log(summ); record(f"e429_duality_{name}", res, summ)
