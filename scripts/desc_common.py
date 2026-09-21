"""Shared helpers for the descendant-signature attacks (e231-e235)."""
from wdd_common import *
def unit(v): return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
def make_runner(model, arch, c, ids_seq, levels, NT):
    def run(b=None, neuron=None, inject=None, inject_block=None):
        st = {}; hs = [arch.layers[lv].register_forward_hook((lambda lv_: lambda m, i, o: st.__setitem__(lv_, (o[0] if isinstance(o, tuple) else o).detach().float().reshape(-1, c.D)))(lv)) for lv in levels]
        if b is not None:
            def pre(m, inp):
                x = inp[0].clone(); flat = x.reshape(-1, c.DFF); flat[torch.arange(NT, device=DEV), neuron] = 0; return (flat.reshape(x.shape),)
            hs.append(arch.mlp_lin(b).register_forward_pre_hook(pre))
        if inject is not None:
            def pre2(m, args, kwargs):
                if len(args) > 0: return (args[0] + inject.to(args[0].dtype).reshape(args[0].shape),) + tuple(args[1:]), kwargs
                kwargs = dict(kwargs); kwargs["hidden_states"] = kwargs["hidden_states"] + inject.to(kwargs["hidden_states"].dtype).reshape(kwargs["hidden_states"].shape); return args, kwargs
            hs.append(arch.layers[inject_block].register_forward_pre_hook(pre2, with_kwargs=True))
        model(ids_seq); [h.remove() for h in hs]; return st
    return run
def dominant(model, arch, c, ids_seq, b):
    st = {}; h = arch.mlp_lin(b).register_forward_pre_hook(lambda m, inp: st.__setitem__("a", inp[0].detach().float().reshape(-1, c.DFF))); model(ids_seq); h.remove()
    led = st["a"] * c.d["WN"][b].to(DEV)[None]; tn = led.abs().argmax(1); tc = torch.gather(led, 1, tn[:, None])[:, 0]; return led, tn, tc
def classes(tn, mask, min_tokens):
    idx = torch.nonzero(mask)[:, 0]; neur = tn[idx]; u, cnt = torch.unique(neur, return_counts=True); keep = u[cnt >= min_tokens]; m = torch.isin(neur, keep); idx = idx[m]; neur = neur[m]; lab_i = (neur[:, None] == keep[None, :]).float().argmax(1); return idx, lab_i, keep
def centroids(F, lab_i, K): return unit(torch.stack([F[lab_i == k].mean(0) if (lab_i == k).any() else torch.zeros(F.shape[1], device=DEV) for k in range(K)]))
def accuracy(F, cents, lab_i): return ((unit(F) @ cents.T).argmax(1) == lab_i).float().mean().item()
def mutual_info_bits(F, cents, lab_i, K):
    pred = (unit(F) @ cents.T).argmax(1); M = torch.zeros(K, K, device=DEV); M.index_put_((lab_i, pred), torch.ones_like(lab_i, dtype=torch.float), accumulate=True); P = M / M.sum(); ps = P.sum(1, keepdim=True); pp = P.sum(0, keepdim=True); nz = P > 0
    return (P[nz] * (P[nz] / (ps @ pp)[nz]).log2()).sum().item()
