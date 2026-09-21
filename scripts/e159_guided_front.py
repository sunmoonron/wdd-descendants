"""e159: the guided decoder's Pareto front. Candidates = atoms nominated by per-block increment OMP with n_b in
(4, 8, 16, 32) per block, plus embeddings; OMP@64 on the full state restricted to the candidates. Real-write share,
top-1/top-3 recall, FVU, and spliced cross-entropy (8 sequences) per setting, next to global OMP."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); K = 64; NS = 8; ids = torch.arange(NS * CTX); N = len(ids)
X = c.X(L)[ids]; typ = typical_mask(c.X(L, center=False)[ids]); A, lab = c.dictionary(L); typA, blkA, idxA = lab["type"].to(DEV), lab["block"].to(DEV), lab["index"].to(DEV)
led = c.ledger(L); C = torch.cat([led[b][ids] for b in range(L + 1)], 1).to(DEV); thr = 0.05 * C.abs().max(1, keepdim=True).values
tb, tn, tc = c.top_writes(L, 3); rows3 = torch.stack([c.atom_index(L, tb[ids, j], tn[ids, j]) for j in range(3)], 1).to(DEV)
inc_sel = {}
for b in range(L + 1):
    D_ = c.s["H"][b + 1][ids].float().to(DEV) - c.s["H"][b][ids].float().to(DEV); Ab, labb = c.dictionary(b, blocks=[b], types=(T_MLP, T_ATT, T_BIAS)); rows_b = torch.nonzero((blkA == b) & (typA >= T_MLP))[:, 0]
    sb, _, _ = omp(D_, Ab, 32); inc_sel[b] = rows_b[sb]
def omp_masked(X, A, k, allowed, batch=512):
    N_ = X.shape[0]; sel = torch.zeros(N_, k, dtype=torch.long, device=DEV); cof = torch.zeros(N_, k, device=DEV); err = torch.zeros(N_, device=DEV); eye = torch.eye(k, device=DEV)
    for s in range(0, N_, batch):
        x = X[s:s + batch]; n = x.shape[0]; r = x.clone(); S = torch.zeros(n, 0, dtype=torch.long, device=DEV); taken = ~allowed[s:s + batch].clone()
        for step in range(k):
            pick = (r @ A.T).abs_().masked_fill_(taken, -1.0).argmax(-1, keepdim=True); taken.scatter_(1, pick, True); S = torch.cat([S, pick], 1)
            As = A[S]; G = As @ As.transpose(1, 2) + 1e-5 * eye[:step + 1, :step + 1]; cc = torch.cholesky_solve(As @ x[:, :, None], torch.linalg.cholesky(G)); r = x - (cc.transpose(1, 2) @ As)[:, 0]
        sel[s:s + n] = S; cof[s:s + n] = cc[:, :, 0]; err[s:s + n] = (r ** 2).sum(-1)
    return sel, cof, err
model, tok, fam = load_model(c.name); arch = Arch(model, fam); ids_seq = c.s["eval_ids"][:NS].to(DEV); mu = c.s["mu"][L + 1].to(DEV)
def ce_with(rep):
    def hook(mod, inp, out):
        o = out[0] if isinstance(out, tuple) else out; o = o.clone(); o[:] = rep.view(NS, CTX, c.D).to(o.dtype); return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o
    h = arch.layers[L].register_forward_hook(hook); loss = model(ids_seq, labels=ids_seq).loss.item(); h.remove(); return loss
base = model(ids_seq, labels=ids_seq).loss.item()
def score(sel, cof, e):
    ismlp = typA[sel] == T_MLP; key = torch.where(ismlp, blkA[sel] * c.DFF + idxA[sel], torch.zeros_like(sel)); real = ismlp & (C.gather(1, key).abs() >= thr)
    return dict(recall1=(sel == rows3[:, :1]).any(1)[typ].float().mean().item(), recall3=(sel[:, :, None] == rows3[:, None, :]).any(1).all(1)[typ].float().mean().item(), real_share=(real[typ].sum() / ismlp[typ].sum()).item(), fvu=fvu(e, X, typ), dce=ce_with(torch.einsum("nk,nkd->nd", cof, A[sel]) + mu) - base)
res = dict(model=tag, L=L, settings={}); sg, cg, eg = omp(X, A, K); res["settings"]["global"] = score(sg, cg, eg[:, -1])
for nb in (4, 8, 16, 32):
    allowed = torch.zeros(N, A.shape[0], dtype=torch.bool, device=DEV); allowed[:, typA <= T_POS] = True
    for b in range(L + 1): allowed[torch.arange(N, device=DEV)[:, None], inc_sel[b][:, :nb]] = True
    sm, cm, em = omp_masked(X, A, K, allowed); res["settings"][f"guided_{nb}"] = score(sm, cm, em); log(f"{tag} guided n_b={nb}: " + " ".join(f"{k} {v:.3f}" for k, v in res["settings"][f"guided_{nb}"].items()))
allowed = torch.zeros(N, A.shape[0], dtype=torch.bool, device=DEV)
for b in range(L + 1): allowed[torch.arange(N, device=DEV)[:, None], inc_sel[b][:, :16]] = True
sm, cm, em = omp_masked(X, A, K, allowed); res["settings"]["guided_16_no_emb"] = score(sm, cm, em)
record(f"e159_front_{tag}", res, " | ".join(f"{k}: r1 {v['recall1']:.2f} r3 {v['recall3']:.2f} real {v['real_share']:.2f} fvu {v['fvu']:.3f} dCE {v['dce']:+.3f}" for k, v in res["settings"].items()))
