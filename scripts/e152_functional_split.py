"""e152: the functional role of the identified vs unidentified writes, at the state level. For 8 sequences at
level L: remove from the state (a) the part explained by WDD's REAL atoms (identified true writes, their refit
coefficients), (b) the part explained by the spurious atoms, (c) the true top-3 writes exactly (ledger), (d) an
equal-energy set of random true writes, (e) the whole OMP reconstruction; splice and measure the cross-entropy
increase. Which part of what WDD reads carries the function?"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 8; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; ids = torch.arange(NT)
Xraw = c.X(L, center=False)[ids]; mu = c.s["mu"][L + 1].to(DEV); X = Xraw - mu; A, lab = c.dictionary(L); typA, blkA, idxA = lab["type"].to(DEV), lab["block"].to(DEV), lab["index"].to(DEV)
led = c.ledger(L); C = torch.cat([led[b][ids] for b in range(L + 1)], 1).to(DEV); thr = 0.05 * C.abs().max(1, keepdim=True).values
sel, cof, err = omp(X, A, 64); ismlp = typA[sel] == T_MLP; key = torch.where(ismlp, blkA[sel] * c.DFF + idxA[sel], torch.zeros_like(sel)); real = ismlp & (C.gather(1, key).abs() >= thr)
part = lambda mask: torch.einsum("nk,nkd->nd", cof * mask.float(), A[sel])
recon = part(torch.ones_like(real)); real_part = part(real); spur_part = part(~real)
tb, tn, tc = c.top_writes(L, 3); rows3 = torch.stack([c.atom_index(L, tb[ids, j], tn[ids, j]) for j in range(3)], 1).to(DEV); true3 = torch.einsum("nk,nkd->nd", tc[ids].to(DEV), A[rows3])
g = torch.Generator().manual_seed(0); mlp_rows = torch.cat([c.atom_index(L, torch.full((c.DFF,), b), torch.arange(c.DFF)) for b in range(L + 1)]).to(DEV)
big = C.abs() >= thr; rnd = torch.zeros_like(Xraw)
for i in range(NT):
    cand = torch.nonzero(big[i])[:, 0]; pick = cand[torch.randperm(len(cand), generator=g)[:3].to(DEV)] if len(cand) >= 3 else cand
    v = (C[i, pick][:, None] * A[mlp_rows[pick]]).sum(0); rnd[i] = v * (true3[i].norm() / v.norm().clamp_min(1e-6))
def ce_with(rep):
    def hook(mod, inp, out):
        o = out[0] if isinstance(out, tuple) else out; o = o.clone(); o[:] = rep.view(NS, CTX, c.D).to(o.dtype); return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o
    h = arch.layers[L].register_forward_hook(hook); loss = model(ids_seq, labels=ids_seq).loss.item(); h.remove(); return loss
base = model(ids_seq, labels=ids_seq).loss.item(); en = lambda v: ((v ** 2).sum(1) / (X ** 2).sum(1)).median().item()
res = dict(model=tag, L=L, base_ce=base, real_atoms_per_token=real.float().sum(1).mean().item(),
           remove=dict(real_part=dict(dce=ce_with(Xraw - real_part) - base, energy=en(real_part)), spurious_part=dict(dce=ce_with(Xraw - spur_part) - base, energy=en(spur_part)), whole_recon=dict(dce=ce_with(Xraw - recon) - base, energy=en(recon)),
                       true_top3=dict(dce=ce_with(Xraw - true3) - base, energy=en(true3)), random_true3_matched=dict(dce=ce_with(Xraw - rnd) - base, energy=en(rnd))))
record(f"e152_funcsplit_{tag}", res, f"base {base:.3f}; real atoms/token {res['real_atoms_per_token']:.1f} | remove real part: dCE {res['remove']['real_part']['dce']:+.3f} (energy {res['remove']['real_part']['energy']:.2f}) | remove spurious part: {res['remove']['spurious_part']['dce']:+.3f} ({res['remove']['spurious_part']['energy']:.2f}) | remove whole recon: {res['remove']['whole_recon']['dce']:+.3f} ({res['remove']['whole_recon']['energy']:.2f}) | remove true top-3: {res['remove']['true_top3']['dce']:+.3f} ({res['remove']['true_top3']['energy']:.2f}) | remove energy-matched random true writes: {res['remove']['random_true3_matched']['dce']:+.3f}")
