"""e228: is the descendant transported by MLPs, by attention, or by alternation? For births b = 2, 3: the full
footprint F (clean minus ablated), the footprint with every later attention output frozen to its clean value
(MLP-only transport, F_M), and with every later MLP output frozen (attention-only transport, F_A). Additive paths
predict F = F_M + F_A - write. At b+2, b+4 and L: the norms relative to the write and the interaction residual
||F - F_M - F_A + write|| / ||F||. Split: additive single-path transport (residual < 0.3) vs alternating transport."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 8; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; NB = c.NB; lab = c.d["lab"]; A = c.d["A"]
def rows(bb): return A[(lab["type"] == T_MLP) & (lab["block"] == bb)].float().to(DEV)
def mlp_module(j):
    lin = arch.mlp_lin(j)
    for name, mod in arch.layers[j].named_modules():
        if any(ch is lin for ch in mod.children()): return mod
def run(b=None, neuron=None, freeze=None, clean=None):
    st = {"att": {}, "mlp": {}}; hs = []
    for j in range(NB):
        hs.append(arch.layers[j].register_forward_hook((lambda j_: lambda m, i, o: st.__setitem__(j_, (o[0] if isinstance(o, tuple) else o).detach().float().reshape(-1, c.D)))(j)))
        hs.append(arch.attn_lin(j).register_forward_hook((lambda j_: lambda m, i, o: st["att"].__setitem__(j_, o.detach().clone()))(j)))
        hs.append(mlp_module(j).register_forward_hook((lambda j_: lambda m, i, o: st["mlp"].__setitem__(j_, o.detach().clone()))(j)))
    if b is not None:
        def pre(m, inp):
            x = inp[0].clone(); flat = x.reshape(-1, c.DFF); flat[torch.arange(NT, device=DEV), neuron] = 0; return (flat.reshape(x.shape),)
        hs.append(arch.mlp_lin(b).register_forward_pre_hook(pre))
    if freeze == "att":
        for j in range(b + 1, NB): hs.append(arch.attn_lin(j).register_forward_hook((lambda j_: lambda m, i, o: clean["att"][j_])(j)))
    if freeze == "mlp":
        for j in range(b + 1, NB): hs.append(mlp_module(j).register_forward_hook((lambda j_: lambda m, i, o: clean["mlp"][j_])(j)))
    model(ids_seq); [h.remove() for h in hs]; return st
S0 = run(); res = {}
for b in (2, 3):
    st_a = {}; h = arch.mlp_lin(b).register_forward_pre_hook(lambda m, inp: st_a.__setitem__("a", inp[0].detach().float().reshape(-1, c.DFF))); model(ids_seq); h.remove()
    led = st_a["a"] * c.d["WN"][b].to(DEV)[None]; tn = led.abs().argmax(1); tc = torch.gather(led, 1, tn[:, None])[:, 0]; d = rows(b)[tn]; big = tc.abs() >= tc.abs().quantile(0.5); w = tc[:, None] * d
    S1 = run(b, tn); SM = run(b, tn, "att", S0); SA = run(b, tn, "mlp", S0); rec = {}
    for lv in sorted({b + 2, b + 4, L}):
        if lv >= NB: continue
        typ = typical_mask(S0[lv]) & big; F = S0[lv] - S1[lv]; FM = S0[lv] - SM[lv]; FA = S0[lv] - SA[lv]; inter = (F - FM - FA + w).norm(dim=1) / F.norm(dim=1).clamp_min(1e-6)
        rec[lv] = dict(full=(F.norm(dim=1) / tc.abs())[typ].median().item(), mlp_only=(FM.norm(dim=1) / tc.abs())[typ].median().item(), att_only=(FA.norm(dim=1) / tc.abs())[typ].median().item(), interaction=inter[typ].median().item(), cos_full_mlp=((F * FM).sum(1) / (F.norm(dim=1) * FM.norm(dim=1)).clamp_min(1e-9))[typ].median().item(), cos_full_att=((F * FA).sum(1) / (F.norm(dim=1) * FA.norm(dim=1)).clamp_min(1e-9))[typ].median().item())
        log(f"{tag} born b{b} level {lv}: |F|/|w| full {rec[lv]['full']:.2f}, MLP-only transport {rec[lv]['mlp_only']:.2f}, attention-only {rec[lv]['att_only']:.2f} | interaction residual {rec[lv]['interaction']:.2f} | cos(full, MLP-only) {rec[lv]['cos_full_mlp']:.2f}, cos(full, attention-only) {rec[lv]['cos_full_att']:.2f}")
    res[b] = rec
import numpy as np
agg = lambda k, which: float(np.mean([res[b][lv][k] for b in res for lv in res[b] if (lv == L if which == "L" else lv == b + 2)]))
record(f"e228_paths_{tag}", dict(model=tag, L=L, per_birth={str(k): {str(kk): vv for kk, vv in v.items()} for k, v in res.items()}), f"at +2 / L: full {agg('full', 2):.2f} / {agg('full', 'L'):.2f}, MLP-only transport {agg('mlp_only', 2):.2f} / {agg('mlp_only', 'L'):.2f}, attention-only {agg('att_only', 2):.2f} / {agg('att_only', 'L'):.2f}, interaction residual {agg('interaction', 2):.2f} / {agg('interaction', 'L'):.2f}, cos(full, MLP-only) {agg('cos_full_mlp', 2):.2f} / {agg('cos_full_mlp', 'L'):.2f}, cos(full, attention-only) {agg('cos_full_att', 2):.2f} / {agg('cos_full_att', 'L'):.2f}")
