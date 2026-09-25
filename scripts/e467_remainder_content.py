"""e467: what does a 16-word native description keep, and what does it leave out? The external review called the
remainder possibly "unverbalized computation"; this probes it directly, following e465.
At the middle depth the centred state is split into the 16-word native description (OMP) and the remainder:
x - mu = d + r. For comparison, the same split is made with 16 rotated words and with the top 16 principal components.
Ridge probes (closed form, trained on 12 sequences, tested on 4 others, 256 tokens each, typical positions) read four
things from d, from r, and from the whole state:
- the current token, the previous token and the next token (each among the 200 most frequent; accuracy over positions
  whose token is among them);
- the position (R^2 of log position).
The question is which kind of information the native description keeps, compared with generic descriptions of the same
size.
Models (argument): gpt2, qwen05.
Pre-registered (honest guesses):
- the remainder carries the current token better than the native description (0.6) (e439: descriptions rarely name
  the current token);
- the native description carries the next token at least as well as its remainder (0.5)."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; model, tok, fam = load_model(name); arch = Arch(model, fam); L = arch.NB // 2; K = 16; D = arch.D
ids = eval_ids(name)[:16, :256].to(DEV); Bn, T = ids.shape
X = block_states(model, arch, ids, [L], chunk=4)[L]                            # positions 1..T-1
flat = X.reshape(-1, D); keep = ~sinkmask(flat); mu = flat[keep].mean(0); Xc = flat - mu
seq = torch.arange(Bn, device=DEV)[:, None].expand(Bn, T - 1).reshape(-1); pos = torch.arange(1, T, device=DEV)[None].expand(Bn, T - 1).reshape(-1)
cur = ids[:, 1:].reshape(-1); prev = ids[:, :-1].reshape(-1); nxt = torch.cat([ids[:, 2:], torch.full((Bn, 1), -1, device=DEV)], 1).reshape(-1)
A, lab = build_dictionary(arch, blocks=list(range(L + 1))); Au = unitr(A); Ar = unitr(rotate(A, seed=7)); del A
def desc(Dct):
    out = torch.zeros_like(Xc); idx = torch.nonzero(keep)[:, 0]
    sel, _, _ = omp(Xc[idx], Dct, K, batch=1024, record_err=False); cof, _ = refit(Xc[idx], Dct, sel); out[idx] = torch.einsum("nk,nkd->nd", cof, Dct[sel]); return out
ev_, U_ = torch.linalg.eigh(torch.cov(Xc[keep].T.double(), correction=0)); P = U_.flip(-1)[:, :K].float()
parts = {}
for nm, d in (("native", desc(Au)), ("rotated", desc(Ar)), ("pca", Xc @ P @ P.T)):
    parts[f"{nm}_desc"] = d; parts[f"{nm}_rem"] = Xc - d
parts["whole"] = Xc
train = keep & (seq < 12); test = keep & (seq >= 12)
def ridge(F, Y, lam=1.0):
    Ft = F[train]; mf = Ft.mean(0); Ft = Ft - mf; G = Ft.T @ Ft; lam_ = lam * G.trace() / G.shape[0]
    W = torch.linalg.solve(G + lam_ * torch.eye(G.shape[0], device=DEV), Ft.T @ (Y[train] - Y[train].mean(0)))
    return (F[test] - mf) @ W + Y[train].mean(0)
def tok_acc(F, target):
    top = torch.bincount(target[train & (target >= 0)], minlength=model.config.vocab_size).topk(200).indices; lut = torch.full((model.config.vocab_size + 1,), -1, device=DEV); lut[top] = torch.arange(200, device=DEV)
    y = lut[target.clamp_min(0)]; y[target < 0] = -1; Y = torch.zeros(y.numel(), 200, device=DEV); ok = y >= 0; Y[ok, y[ok]] = 1.0
    pred = ridge(F, Y).argmax(-1); yt = y[test]; m = yt >= 0
    return float((pred[m] == yt[m]).float().mean())
def pos_r2(F):
    y = torch.log(pos.float())[:, None]; pr = ridge(F, y)[:, 0]; yt = y[test][:, 0]
    return float(1 - ((pr - yt) ** 2).sum() / ((yt - yt.mean()) ** 2).sum())
res = dict(model=name, level=L, k=K, probes={})
for nm, F in parts.items():
    res["probes"][nm] = dict(current=tok_acc(F, cur), previous=tok_acc(F, prev), next=tok_acc(F, nxt), position_r2=pos_r2(F),
                             energy_share=float(F[keep].pow(2).sum() / Xc[keep].pow(2).sum()))
    p_ = res["probes"][nm]; log(f"{name} {nm}: energy {p_['energy_share']:.2f} | current token {p_['current']:.2f}, previous {p_['previous']:.2f}, next {p_['next']:.2f}, position R2 {p_['position_r2']:.2f}")
Pr_ = res["probes"]
res["checks"] = dict(remainder_carries_current_more=Pr_["native_rem"]["current"] > Pr_["native_desc"]["current"], desc_carries_next_as_well=Pr_["native_desc"]["next"] >= Pr_["native_rem"]["next"])
f_ = lambda k: f"{k}: cur {Pr_[k]['current']:.2f} prev {Pr_[k]['previous']:.2f} next {Pr_[k]['next']:.2f} pos {Pr_[k]['position_r2']:.2f} (energy {Pr_[k]['energy_share']:.2f})"
summ = f"{name} L{L}, ridge probes (accuracy among the 200 most frequent tokens; position R2): " + " | ".join(f_(k) for k in ("whole", "native_desc", "native_rem", "rotated_desc", "rotated_rem", "pca_desc", "pca_rem")) + f" | checks {_json.dumps(res['checks'])}"
log(summ); record(f"e467_remainder_{name}", res, summ)
