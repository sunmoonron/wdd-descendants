"""e225: do scattered writes land in a shared low-dimensional subspace? Across tokens, the footprints delta(lv) of
the ablated dominant block-b writes (b = 2) form a matrix; its participation ratio (effective dimension) at levels
b+1, b+2, b+4, L is compared with the effective dimension of the injected writes themselves, of the footprints of
random directions of the same size, and of the state itself at that level. Funnelling: real-write footprints have
a much lower effective dimension than random-direction footprints. Also the overlap of the footprint subspace with
the state's top principal subspace."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 8; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; NB = c.NB; lab = c.d["lab"]; A = c.d["A"]; b = 2
def rows(bb): return A[(lab["type"] == T_MLP) & (lab["block"] == bb)].float().to(DEV)
def run(neuron=None, inject=None):
    st = {}; hs = [arch.layers[j].register_forward_hook((lambda j_: lambda m, i, o: st.__setitem__(j_, (o[0] if isinstance(o, tuple) else o).detach().float().reshape(-1, c.D)))(j)) for j in range(NB)]
    if neuron is not None:
        def pre(m, inp):
            x = inp[0].clone(); flat = x.reshape(-1, c.DFF); flat[torch.arange(NT, device=DEV), neuron] = 0; return (flat.reshape(x.shape),)
        hs.append(arch.mlp_lin(b).register_forward_pre_hook(pre))
    if inject is not None:
        def pre2(m, args, kwargs):
            if len(args) > 0: return (args[0] + inject.to(args[0].dtype).reshape(args[0].shape),) + tuple(args[1:]), kwargs
            kwargs = dict(kwargs); kwargs["hidden_states"] = kwargs["hidden_states"] + inject.to(kwargs["hidden_states"].dtype).reshape(kwargs["hidden_states"].shape); return args, kwargs
        hs.append(arch.layers[b + 1].register_forward_pre_hook(pre2, with_kwargs=True))
    model(ids_seq); [h.remove() for h in hs]; return st
st_a = {}; h = arch.mlp_lin(b).register_forward_pre_hook(lambda m, inp: st_a.__setitem__("a", inp[0].detach().float().reshape(-1, c.DFF))); model(ids_seq); h.remove()
led = st_a["a"] * c.d["WN"][b].to(DEV)[None]; tn = led.abs().argmax(1); tc = torch.gather(led, 1, tn[:, None])[:, 0]; d = rows(b)[tn]; big = tc.abs() >= tc.abs().quantile(0.5)
S0 = run(); S1 = run(neuron=tn); torch.manual_seed(0); U = torch.randn(NT, c.D, device=DEV); U = U / U.norm(dim=1, keepdim=True); S2 = run(inject=-tc.abs()[:, None] * U)
def pr(M):
    M = M - M.mean(0, keepdim=True); s = torch.linalg.svdvals(M); e = s ** 2; return ((e.sum() ** 2) / (e ** 2).sum()).item()
out = {}
for lv in sorted({b + 1, b + 2, b + 4, L}):
    if lv >= NB: continue
    typ = typical_mask(S0[lv]) & big; Dw = S1[lv][typ] - S0[lv][typ]; Dr = S2[lv][typ] - S0[lv][typ]; W = (tc[:, None] * d)[typ]; X = S0[lv][typ]
    Xs = c.X(lv)[sub(c.NT, 8192, seed=3)]; Sig = Xs.T @ Xs / len(Xs); ev, V = torch.linalg.eigh(Sig); top = V[:, -c.D // 10:]
    frac = lambda M: ((M @ top) ** 2).sum() / (M ** 2).sum()
    out[lv] = dict(dim_write_footprint=pr(Dw), dim_random_footprint=pr(Dr), dim_writes=pr(W), dim_state=pr(X), top10pc_share_write_fp=frac(Dw).item(), top10pc_share_random_fp=frac(Dr).item(), top10pc_share_writes=frac(W).item())
    log(f"{tag} level {lv}: effective dimension: write footprints {out[lv]['dim_write_footprint']:.0f}, random-direction footprints {out[lv]['dim_random_footprint']:.0f}, the writes themselves {out[lv]['dim_writes']:.0f}, the state {out[lv]['dim_state']:.0f} | energy in the state's top-10% PCs: write footprints {out[lv]['top10pc_share_write_fp']:.2f}, random footprints {out[lv]['top10pc_share_random_fp']:.2f}, writes {out[lv]['top10pc_share_writes']:.2f}")
record(f"e225_fpdim_{tag}", dict(model=tag, b=b, L=L, per_level={str(k): v for k, v in out.items()}), " | ".join(f"lv{lv}: dim write-fp {v['dim_write_footprint']:.0f} vs random-fp {v['dim_random_footprint']:.0f} (writes {v['dim_writes']:.0f}, state {v['dim_state']:.0f}); top-10%-PC share write-fp {v['top10pc_share_write_fp']:.2f} vs random-fp {v['top10pc_share_random_fp']:.2f}" for lv, v in out.items()))
