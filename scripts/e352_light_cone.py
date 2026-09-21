"""e352: the forward light cone of a perturbation, with and without attention. A write and a random vector injected
at one token per sequence (position 32) at the natural amplitude; the response energy at every later level by
token offset 0..64, normal and with all attention outputs of the later blocks frozen at their unperturbed values
(no cross-token route). Reported per level: the fraction of energy at other tokens, the offset containing 90% of the
off-source energy (light-cone width), and the same from the final token and from position 1."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quot_common import *
tag = sys.argv[1]; S0_ = Setup(tag, levels=[0]); L = S0_.L; NB = S0_.NB; b = 2; S = Setup(tag, levels=[b + 2, L, NB - 2]); torch.manual_seed(0); wvec = S.W2[int(S.keep[0])]; rvec = unit(torch.randn(S.D, device=DEV)); amp = S.s_inj; mods = {lv: S.arch.attn_lin(lv) for lv in range(b + 1, NB)}; cache = {}
hs = [m.register_forward_hook((lambda key: lambda mod, i, o: cache.__setitem__(key, o.detach().clone()))(key)) for key, m in mods.items()]; S.model(S.ids_seq); [h.remove() for h in hs]
def resp(vec, p0, freeze):
    tokens = torch.arange(S.NS, device=DEV) * CTX + p0; inj = torch.zeros(S.NT, S.D, device=DEV); inj[tokens] = amp * vec[None]; hs = [mods[lv].register_forward_hook((lambda key: lambda mod, i, o: cache[key])(lv)) for lv in mods] if freeze else []; r = S.run(inject=inj, inject_block=b + 1); [h.remove() for h in hs]; out = {}
    for lv in S.levels:
        R = (r[lv] - S.S0[lv]).reshape(S.NS, CTX, S.D); e = (R ** 2).sum(2).mean(0); src = e[p0]; after = e[p0 + 1:]; cum = after.cumsum(0) / after.sum().clamp_min(1e-9); out[lv] = dict(other_fraction=(after.sum() / (src + after.sum()).clamp_min(1e-9)).item(), width90=int((cum >= 0.9).nonzero()[0].item()) + 1 if (cum >= 0.9).any() else int(len(after)), before_fraction=(e[:p0].sum() / (src + after.sum() + e[:p0].sum()).clamp_min(1e-9)).item())
    return out
res = {}
for nm, vec in (("write", wvec), ("random", rvec)):
    for p0 in (1, CTX // 4, CTX - 2):
        for fr in (False, True):
            res[f"{nm}_pos{p0}_{'attn_frozen' if fr else 'normal'}"] = resp(vec, p0, fr)
log(f"{tag}: source -> level: fraction of energy at later tokens / 90%-width in tokens :: " + " ; ".join(f"{k}: " + " ".join(f"lv{lv} {v['other_fraction']:.2f}/{v['width90']}" for lv, v in rec.items()) for k, rec in res.items()))
record(f"e352_lightcone_{tag}", dict(model=tag, L=L, per_source={k: {str(lv): v for lv, v in rec.items()} for k, rec in res.items()}), " ; ".join(f"{k}: " + " ".join(f"{lv}:{v['other_fraction']:.2f}/{v['width90']}" for lv, v in rec.items()) for k, rec in res.items() if "pos%d" % (CTX // 4) in k))
