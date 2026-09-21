"""e227: does a neuron keep a recognisable descendant signature after its direction is gone? For births b = 1, 2, 3
(dominant write per token, one ablation run per birth), footprints at b+2 and L. For neurons dominant at >= 20
tokens: mean cosine between footprints of the same neuron across tokens vs between different neurons; leave-one-
out nearest-centroid identification of the source neuron from its footprint (vs chance); nearest-centroid
identification of the source BLOCK (three births pooled, chance 1/3). Logit-space provenance: per-token Spearman
of the write's |coefficient|, its readability at L and its along-direction footprint with the KL of the removal.
Split: neuron-specific descendant identity (identification far above chance, within > between) vs none."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; NB = c.NB; lab = c.d["lab"]; A = c.d["A"]
def rows(bb): return A[(lab["type"] == T_MLP) & (lab["block"] == bb)].float().to(DEV)
AL, labL = c.dictionary(L); mu = c.s["mu"][L + 1].to(DEV); S = AL.T @ AL; ev, V = torch.linalg.eigh(S); Winv = V @ torch.diag(1 / (ev + 1e-2 * ev[-1])) @ V.T
def run(b=None, neuron=None):
    st = {}; hs = [arch.layers[j].register_forward_hook((lambda j_: lambda m, i, o: st.__setitem__(j_, (o[0] if isinstance(o, tuple) else o).detach().float().reshape(-1, c.D)))(j)) for j in (L, ) + tuple(range(NB))]
    if b is not None:
        def pre(m, inp):
            x = inp[0].clone(); flat = x.reshape(-1, c.DFF); flat[torch.arange(NT, device=DEV), neuron] = 0; return (flat.reshape(x.shape),)
        hs.append(arch.mlp_lin(b).register_forward_pre_hook(pre))
    out = model(ids_seq); [h.remove() for h in hs]; lg = out.logits[:, :-1].reshape(-1, out.logits.shape[-1]).float(); return st, torch.log_softmax(lg, -1)
S0, lp0 = run(); typL = typical_mask(S0[L]); X = S0[L] - mu; sel = oneshot(X, AL, 64, whiten=Winv)[0]
def unit(v): return v / v.norm(dim=1, keepdim=True).clamp_min(1e-9)
def spearman(a_, b_):
    ra = a_.argsort().argsort().float(); rb = b_.argsort().argsort().float(); return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
res = {}; pool = {"F": [], "blk": []}
for b in (1, 2, 3):
    st_a = {}; h = arch.mlp_lin(b).register_forward_pre_hook(lambda m, inp: st_a.__setitem__("a", inp[0].detach().float().reshape(-1, c.DFF))); model(ids_seq); h.remove()
    led = st_a["a"] * c.d["WN"][b].to(DEV)[None]; tn = led.abs().argmax(1); tc = torch.gather(led, 1, tn[:, None])[:, 0]; d = rows(b)[tn]; big = tc.abs() >= tc.abs().quantile(0.5); S1, lp1 = run(b, tn); rec = {}
    for lv in sorted({b + 2, L}):
        if lv >= NB: continue
        typ = typical_mask(S0[lv]) & big; F = unit(S1[lv] - S0[lv]); idx = torch.nonzero(typ)[:, 0]; neur = tn[idx]; u, cnt = torch.unique(neur, return_counts=True); keep = u[cnt >= 20]
        if len(keep) < 3: rec[lv] = dict(n_neurons=int(len(keep))); continue
        m = torch.isin(neur, keep); idx = idx[m]; neur = neur[m]; Fm = F[idx]; G = Fm @ Fm.T; same = neur[:, None] == neur[None, :]; eye = torch.eye(len(idx), device=DEV, dtype=torch.bool)
        within = G[same & ~eye].mean().item(); between = G[~same].mean().item()
        # leave-one-out nearest centroid over neurons
        cents = torch.stack([Fm[neur == k].sum(0) for k in keep]); lab_i = (neur[:, None] == keep[None, :]).float().argmax(1); own = cents[lab_i] - Fm; cents_loo = cents[None].expand(len(idx), -1, -1).clone(); cents_loo[torch.arange(len(idx)), lab_i] = own
        sims = torch.einsum("nd,nkd->nk", Fm, unit(cents_loo.reshape(-1, c.D)).reshape(len(idx), len(keep), c.D)); acc = (sims.argmax(1) == lab_i).float().mean().item()
        rec[lv] = dict(n_neurons=int(len(keep)), n_tokens=int(len(idx)), within=within, between=between, loo_accuracy=acc, chance=1.0 / len(keep))
        if lv == L: pool["F"].append(Fm); pool["blk"].append(torch.full((len(idx),), b, device=DEV))
    kl = (lp0.exp() * (lp0 - lp1)).sum(1); bigm = big.reshape(NS, CTX)[:, :-1].reshape(-1); typm = typL.reshape(NS, CTX)[:, :-1].reshape(-1) & bigm
    row = c.atom_index(L, torch.full_like(tn.cpu(), b), tn.cpu()).to(DEV); read = (sel == row[:, None]).any(1).float(); along = -((S1[L] - S0[L]) * d).sum(1) / tc
    tcm = tc.reshape(NS, CTX)[:, :-1].reshape(-1); readm = read.reshape(NS, CTX)[:, :-1].reshape(-1); alongm = along.reshape(NS, CTX)[:, :-1].reshape(-1)
    rec["logit"] = dict(rho_coef_kl=spearman(tcm.abs()[typm], kl[typm]), rho_read_kl=spearman(readm[typm], kl[typm]), rho_along_kl=spearman(alongm[typm], kl[typm]))
    res[b] = rec
    log(f"{tag} born b{b}: " + " | ".join(f"lv{lv}: {v['n_neurons']} neurons, within {v.get('within', float('nan')):.3f} vs between {v.get('between', float('nan')):.3f}, source-neuron LOO accuracy {v.get('loo_accuracy', float('nan')):.2f} (chance {v.get('chance', float('nan')):.2f})" for lv, v in rec.items() if lv != "logit") + f" | Spearman with KL: |coef| {rec['logit']['rho_coef_kl']:+.2f}, read at L {rec['logit']['rho_read_kl']:+.2f}, along-footprint {rec['logit']['rho_along_kl']:+.2f}")
blk_acc = float("nan")
if len(pool["F"]) >= 2:
    Fp = torch.cat(pool["F"]); bl = torch.cat(pool["blk"]); ub = bl.unique(); cents = torch.stack([Fp[bl == k].mean(0) for k in ub]); lab_i = (bl[:, None] == ub[None, :]).float().argmax(1); own = cents[lab_i] * (bl[:, None] == bl[None, :]).float().sum(1, keepdim=True) - Fp
    pred = torch.einsum("nd,kd->nk", Fp, unit(cents)).argmax(1); blk_acc = (pred == lab_i).float().mean().item()
import numpy as np
w = [res[b][L]["within"] for b in res if L in res[b] and "within" in res[b][L]]; bt = [res[b][L]["between"] for b in res if L in res[b] and "between" in res[b][L]]; ac = [res[b][L]["loo_accuracy"] for b in res if L in res[b] and "loo_accuracy" in res[b][L]]; ch = [res[b][L]["chance"] for b in res if L in res[b] and "chance" in res[b][L]]
record(f"e227_identity_{tag}", dict(model=tag, L=L, per_birth={str(k): {str(kk): vv for kk, vv in v.items()} for k, v in res.items()}, block_id_accuracy=blk_acc), f"at L: within-neuron footprint cosine {np.mean(w):.3f} vs between {np.mean(bt):.3f}; source-neuron LOO accuracy {np.mean(ac):.2f} (chance {np.mean(ch):.2f}); source-block accuracy {blk_acc:.2f} (chance 0.33) | Spearman(|coef|, KL) " + " ".join(f"{res[b]['logit']['rho_coef_kl']:+.2f}" for b in res) + ", (read at L, KL) " + " ".join(f"{res[b]['logit']['rho_read_kl']:+.2f}" for b in res))
