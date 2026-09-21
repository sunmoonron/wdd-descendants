"""e105: functional fidelity. Replace the level-L residual state with each decomposition's reconstruction and run
the rest of the model: cross-entropy increase for OMP@64, one-shot@64 (refit), dual one-shot@64, the 64 largest
TRUE writes (least-squares refit), the per-block increment pipeline support (refit), and a random-64-atom refit.
Which reading preserves the model's function, and does function track reconstruction or attribution?"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam)
NSEQ = 8; ids_seq = c.s["eval_ids"][:NSEQ].to(DEV); ids = torch.arange(NSEQ * CTX); X = c.X(L)[ids]; mu = c.s["mu"][L + 1].to(DEV); A, lab = c.dictionary(L)
S = A.T @ A; ev, V = torch.linalg.eigh(S); Winv = V @ torch.diag(1 / (ev + 1e-2 * ev[-1])) @ V.T
recons = {}
sel, cof, _ = omp(X, A, 64); recons["omp64"] = torch.einsum("nk,nkd->nd", cof, A[sel])
s1, c1, _ = oneshot(X, A, 64); recons["oneshot64"] = torch.einsum("nk,nkd->nd", c1, A[s1])
sd, cd, _ = oneshot(X, A, 64, whiten=Winv); recons["dual64"] = torch.einsum("nk,nkd->nd", cd, A[sd])
led = c.ledger(L); C = torch.cat([led[b][ids] for b in range(L + 1)], 1).to(DEV); top = C.abs().topk(64, dim=1).indices
rows_t = torch.stack([c.atom_index(L, top[:, j].cpu() // c.DFF, top[:, j].cpu() % c.DFF) for j in range(64)], 1).to(DEV); ct_, _ = refit(X, A, rows_t); recons["true64_refit"] = torch.einsum("nk,nkd->nd", ct_, A[rows_t])
g = torch.Generator().manual_seed(0); rnd = torch.randint(0, A.shape[0], (X.shape[0], 64), generator=g).to(DEV); cr, _ = refit(X, A, rnd); recons["random64_refit"] = torch.einsum("nk,nkd->nd", cr, A[rnd])
# increment pipeline support: per block top-8 on the increment, union, refit on the state
parts = []
for b in range(L + 1):
    D_ = c.s["H"][b + 1][ids].float().to(DEV) - c.s["H"][b][ids].float().to(DEV); Ab, labb = c.dictionary(b, blocks=[b], types=(T_MLP, T_ATT, T_BIAS)); rows_b = torch.nonzero((lab["block"] == b) & (lab["type"] >= T_MLP))[:, 0].to(DEV)
    sb, _, _ = omp(D_, Ab, 9); parts.append(rows_b[sb])
selp = torch.cat(parts, 1)[:, :64]; cp, _ = refit(X, A, selp); recons["pipeline64"] = torch.einsum("nk,nkd->nd", cp, A[selp])
res = dict(model=tag, L=L, n_tokens=X.shape[0], results={})
def ce_with(rep):
    def hook(mod, inp, out):
        o = out[0] if isinstance(out, tuple) else out; o = o.clone(); o[:] = rep.view(NSEQ, CTX, c.D).to(o.dtype); return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o
    h = arch.layers[L].register_forward_hook(hook); loss = model(ids_seq, labels=ids_seq).loss.item(); h.remove(); return loss
base = model(ids_seq, labels=ids_seq).loss.item(); res["base_ce"] = base
raw = c.X(L, center=False)[ids]; res["results"]["identity_check"] = dict(dce=ce_with(raw) - base)
for nm, R in recons.items():
    full = R + mu; res["results"][nm] = dict(dce=ce_with(full) - base, fvu=fvu(((X - R) ** 2).sum(1), X)); log(f"{tag} {nm}: dCE {res['results'][nm]['dce']:+.3f} fvu {res['results'][nm]['fvu']:.3f}")
res["results"]["mean_only"] = dict(dce=ce_with(mu.expand_as(X)) - base, fvu=1.0)
record(f"e105_splice_{tag}", res, f"base CE {base:.3f} identity {res['results']['identity_check']['dce']:+.4f} | " + " | ".join(f"{k}: dCE {v['dce']:+.3f} fvu {v['fvu']:.2f}" for k, v in res["results"].items() if k != "identity_check"))
