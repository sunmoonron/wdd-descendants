"""e180: a self-certificate at inference. Without the ledger, WDD reports for each token whether its top recovered
MLP atom is 'trustworthy': its prominence (|x_c . a| / ||x_c||) exceeds the covariance-matched competitor level
(computed once per model from held-out states). Against the ledger: precision and recall of the flag for 'the top
recovered atom is a real write' and for 'it is the dominant write', for OMP and the dual reading."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); X = c.X(L); typ = typical_mask(c.X(L, center=False)); A, lab = c.dictionary(L); typA, blkA, idxA = lab["type"].to(DEV), lab["block"].to(DEV), lab["index"].to(DEV)
sel, cof, err = get_omp(c, L, A=A, X=X); S = A.T @ A; ev, V = torch.linalg.eigh(S); Winv = V @ torch.diag(1 / (ev + 1e-2 * ev[-1])) @ V.T; sd, cd, _ = oneshot(X, A, 64, whiten=Winv)
led = c.ledger(L); C = torch.cat([led[b] for b in range(L + 1)], 1).to(DEV); thr = 0.05 * C.abs().max(1, keepdim=True).values; tb, tn, tc = c.top_writes(L, 1); row = c.atom_index(L, tb[:, 0], tn[:, 0]).to(DEV)
seq = (torch.arange(c.NT) // CTX).to(DEV); fit = typ & (seq % 2 == 1); ev_ = typ & (seq % 2 == 0); g = torch.Generator(device=DEV).manual_seed(0)
Xt = X[fit]; Sig = (Xt.T @ Xt) / Xt.shape[0]; e2, V2 = torch.linalg.eigh(Sig); half = V2 @ torch.diag(e2.clamp_min(0).sqrt()) @ V2.T; z = torch.randn(1024, c.D, generator=g, device=DEV) @ half; z = z / z.norm(dim=1, keepdim=True)
lvl = torch.cat([(z[s:s + 256] @ A.T).abs().topk(64, dim=1).values[:, -1] for s in range(0, 1024, 256)]).mean().item()
res = dict(model=tag, L=L, level=lvl, decoders={})
for nm, s_, c_ in (("omp", sel, cof), ("dual", sd, cd)):
    ismlp = typA[s_] == T_MLP; cm = c_.abs() * ismlp.float(); j = cm.argmax(1); top_atom = s_.gather(1, j[:, None])[:, 0]; has = ismlp.any(1)
    key = blkA[top_atom] * c.DFF + idxA[top_atom]; real = C.gather(1, key.clamp(min=0)[:, None])[:, 0].abs() >= thr[:, 0]; dominant = top_atom == row
    prom = (X * A[top_atom]).sum(1).abs() / X.norm(dim=1); flag = prom > lvl; m = ev_ & has
    prec = lambda y: (flag & y)[m].float().sum().item() / max(1, flag[m].float().sum().item()); rec = lambda y: (flag & y)[m].float().sum().item() / max(1, y[m].float().sum().item())
    res["decoders"][nm] = dict(flag_rate=flag[m].float().mean().item(), base_real=real[m].float().mean().item(), base_dominant=dominant[m].float().mean().item(), precision_real=prec(real), recall_real=rec(real), precision_dominant=prec(dominant), recall_dominant=rec(dominant), real_rate_when_unflagged=real[m & ~flag].float().mean().item() if (m & ~flag).any() else None)
record(f"e180_cert_{tag}", res, " | ".join(f"{k}: flagged {v['flag_rate']:.2f}; top atom real base {v['base_real']:.2f} -> precision {v['precision_real']:.2f} (unflagged real {v['real_rate_when_unflagged']:.2f}); dominant base {v['base_dominant']:.2f} -> precision {v['precision_dominant']:.2f} recall {v['recall_dominant']:.2f}" for k, v in res["decoders"].items()))
