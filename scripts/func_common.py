"""Shared helpers for the function-side experiments (e255-e258): natural ablation with logits at chosen positions."""
from desc_common import *
def make_runner_logits(model, arch, c, ids_seq, levels, NT, b):
    def run(neuron=None, positions=None, inject=None, inject_block=None, ablate_positions=None):
        st = {}; hs = [arch.layers[lv].register_forward_hook((lambda lv_: lambda m, i, o: st.__setitem__(lv_, (o[0] if isinstance(o, tuple) else o).detach().float().reshape(-1, c.D)))(lv)) for lv in levels]
        if neuron is not None:
            def pre(m, inp):
                x = inp[0].clone(); flat = x.reshape(-1, c.DFF); rows_ = torch.arange(NT, device=DEV) if ablate_positions is None else ablate_positions; flat[rows_, neuron[rows_] if neuron.numel() == NT else neuron] = 0; return (flat.reshape(x.shape),)
            hs.append(arch.mlp_lin(b).register_forward_pre_hook(pre))
        if inject is not None:
            def pre2(m, args, kwargs):
                if len(args) > 0: return (args[0] + inject.to(args[0].dtype).reshape(args[0].shape),) + tuple(args[1:]), kwargs
                kwargs = dict(kwargs); kwargs["hidden_states"] = kwargs["hidden_states"] + inject.to(kwargs["hidden_states"].dtype).reshape(kwargs["hidden_states"].shape); return args, kwargs
            hs.append(arch.layers[inject_block].register_forward_pre_hook(pre2, with_kwargs=True))
        out = model(ids_seq); [h.remove() for h in hs]; lg = out.logits.reshape(NT, -1).float(); st["lg"] = (lg[positions] if positions is not None else None); del out, lg; return st
    return run
def gram_spearman(G1, G2):
    K = G1.shape[0]; iu = torch.triu_indices(K, K, 1, device=DEV); a_, b_ = G1[iu[0], iu[1]], G2[iu[0], iu[1]]; ra = a_.argsort().argsort().float(); rb = b_.argsort().argsort().float(); return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
