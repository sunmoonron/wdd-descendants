"""e392: can the model's own writers carry the whole forward pass? The output of every block (all positions but the first)
is replaced, as the forward pass runs, by its k-sparse WDD reconstruction (OMP over the model's own atoms up to that
block, centring mean added back), so every layer computes on a sparse re-description of its input and the errors
compound. k = 16, 32, 64, 128. Reported: the loss against the clean loss, and the same with the dictionary randomly
rotated (same Gram matrix, no provenance). If the loss stays near clean at small k, a sparse replacement model built on
the model's own neuron write directions, with no training, is possible."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p3_common import *
tag = sys.argv[1]; c = Cache(tag); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NB = arch.NB; torch.manual_seed(0)
ev = c.s["eval_ids"][:2].to(DEV); B, T = ev.shape
with torch.no_grad(): Lc = token_loss(model(ev).logits.float(), ev).mean().item()
dicts = {l: c.dictionary(l)[0] for l in range(NB - 1)}; mus = {l: c.s["mu"][l + 1].to(DEV).float() for l in range(NB - 1)}
def run(k, rotated):
    hs = []
    for l in range(NB - 1):
        Dct = rotate(dicts[l]) if rotated else dicts[l]
        def hk(m, i, o, l=l, Dct=Dct):
            xo = o[0] if isinstance(o, tuple) else o; xp = xo[:, 1:].reshape(-1, xo.shape[-1]).float() - mus[l][None]
            sel, cof, _ = omp(xp, Dct, k, batch=512, record_err=False); xh = mus[l][None] + torch.einsum("nk,nkd->nd", cof, Dct[sel])
            y = xo.clone(); y[:, 1:] = xh.to(xo.dtype).view(xo.shape[0], xo.shape[1] - 1, -1)
            return (y,) + tuple(o[1:]) if isinstance(o, tuple) else y
        hs.append(arch.layers[l].register_forward_hook(hk))
    try:
        with torch.no_grad(): return token_loss(model(ev).logits.float(), ev).mean().item()
    finally: [h.remove() for h in hs]
res = dict(model=tag, clean=Lc, n_layers_replaced=NB - 1, runs={})
for k in (16, 32, 64, 128):
    lw = run(k, False); lr = run(k, True); res["runs"][str(k)] = dict(wdd=lw, rotated=lr, wdd_excess=lw - Lc, rotated_excess=lr - Lc)
log(f"{tag}: every block output replaced by its k-sparse reconstruction ({NB - 1} blocks), clean loss {Lc:.3f}: " + " | ".join(f"k{k}: WDD {v['wdd']:.3f} (+{v['wdd_excess']:.3f}), rotated {v['rotated']:.3f} (+{v['rotated_excess']:.3f})" for k, v in res["runs"].items()))
record(f"e392_replacement_{tag}", res, " | ".join(f"k{k}: WDD +{v['wdd_excess']:.2f} rot +{v['rotated_excess']:.2f}" for k, v in res["runs"].items()))
