"""e166: does the increment reading preserve function? Reconstruct the level-L state as the sum of per-block
increment reconstructions (each increment: OMP@k over its block atoms, k = 8 and 16) plus the exact embedding
(H[0]), splice into the forward pass, and compare the cross-entropy cost with global OMP@64 on the state and with
the e44 pipeline refit. Also the same with k=32 per block (a larger total budget) to see whether function can be
bought back at the increment level."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 8; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; ids = torch.arange(NT)
Xraw = c.X(L, center=False)[ids]; mu = c.s["mu"][L + 1].to(DEV); X = Xraw - mu; A, lab = c.dictionary(L); typ = typical_mask(Xraw)
def ce_with(rep):
    def hook(mod, inp, out):
        o = out[0] if isinstance(out, tuple) else out; o = o.clone(); o[:] = rep.view(NS, CTX, c.D).to(o.dtype); return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o
    h = arch.layers[L].register_forward_hook(hook); loss = model(ids_seq, labels=ids_seq).loss.item(); h.remove(); return loss
base = model(ids_seq, labels=ids_seq).loss.item(); res = dict(model=tag, L=L, base_ce=base, variants={})
sg, cg, eg = omp(X, A, 64); res["variants"]["global_omp64"] = dict(dce=ce_with(torch.einsum("nk,nkd->nd", cg, A[sg]) + mu) - base, fvu=fvu(eg[:, -1], X, typ))
E0 = c.s["H"][0][ids].float().to(DEV)
for k in (8, 16, 32):
    rec = E0.clone(); tot_atoms = 0
    for b in range(L + 1):
        D_ = c.s["H"][b + 1][ids].float().to(DEV) - c.s["H"][b][ids].float().to(DEV); Ab, labb = c.dictionary(b, blocks=[b], types=(T_MLP, T_ATT, T_BIAS))
        sb, cb, eb = omp(D_, Ab, k); rec = rec + torch.einsum("nk,nkd->nd", cb, Ab[sb]); tot_atoms += k
    res["variants"][f"stream_k{k}"] = dict(dce=ce_with(rec) - base, fvu=fvu(((Xraw - rec) ** 2).sum(1), X, typ), atoms=tot_atoms); log(f"{tag} stream k={k}/block ({tot_atoms} atoms): dCE {res['variants'][f'stream_k{k}']['dce']:+.3f} fvu {res['variants'][f'stream_k{k}']['fvu']:.3f}")
res["variants"]["embedding_only"] = dict(dce=ce_with(E0) - base, fvu=fvu(((Xraw - E0) ** 2).sum(1), X, typ))
record(f"e166_stream_{tag}", res, " | ".join(f"{k}: dCE {v['dce']:+.3f} fvu {v['fvu']:.3f}" + (f" ({v['atoms']} atoms)" if 'atoms' in v else "") for k, v in res["variants"].items()))
