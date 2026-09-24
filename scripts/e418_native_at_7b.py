"""e418: does the native vocabulary's self-description hold at 7B? Everything in sessions 36-39 was measured on models
up to 1.2B parameters. Qwen2.5-7B (bf16), the state after the middle block (14 of 28), 4 sequences of 511 tokens: every
position but the first replaced by its k-word description (k 4-64), loss recovered in the model's own forward pass
(as e388, e399). Vocabularies: the native vocabulary (token embeddings, MLP write rows and head output bases of blocks
0-14, about 490k words), a random rotation of it (by rotating the states instead of copying the 7 GB dictionary), and
Gaussian words with the states' covariance (from 4 other sequences; data-derived, as e399 covX); words chosen under the
Euclidean metric and under the Fisher metric (as e400). Also the share of variance and of the Fisher trace on the
states' top-8 principal directions. Pre-registered: the own words beat their rotation at k 16 under both metrics (the
effect survives scale); the huge-direction split (variance share far above Fisher share) is present."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ws_common import *
from sd_common import Level, fisher_gram, metric_sqrt, gauss_like, unitr
name = sys.argv[1] if len(sys.argv) > 1 else "qwen7"; KS = [4, 8, 16, 32, 64]
model, tok, fam = load_bf16(name); arch = Arch(model, fam); L = arch.NB // 2
E = torch.load("/workspace/wdd/eval_ids_all.pt", weights_only=False)["qwen05"]["eval_ids"]; ev = E[:4].to(DEV); fit = E[4:8].to(DEV)
A, blk, typ, ends = lean_dictionary(arch, L); A = A[:ends[L]]
lv = Level(model, arch, ev, L); fl = Level(model, arch, fit, L); SX = (fl.Xc.T @ fl.Xc) / fl.Xc.shape[0]; del fl
evX, UX = torch.linalg.eigh(SX.double()); UM = UX[:, -8:].float()
GF = fisher_gram(model, arch, fit, L); SF = metric_sqrt(GF)
g = torch.Generator().manual_seed(7); R = torch.linalg.qr(torch.randn(arch.D, arch.D, generator=g))[0].to(DEV)
res = dict(model=name, level=L, k=KS, gap=lv.gap, n_words=A.shape[0], var_share_top8=(evX[-8:].sum() / evX.sum()).item(),
           fisher_share_top8=((UM.T @ GF @ UM).trace() / GF.trace()).item(), cells={})
log(f"{name} L{L}: gap {lv.gap:.2f} nats; {A.shape[0]} words; top-8 PCs hold {res['var_share_top8']:.2f} of the variance and {res['fisher_share_top8']:.3f} of the Fisher trace")
def describe_lean(Xc, V, S=None, post=None):
    """OMP over unit rows V (no copy); S: metric square root (states and words mapped by x -> x S, words renormalised);
    post: matrix applied to the reconstruction (the rotation back)"""
    if S is None: Xs = Xc; Vs = V; nrm = None
    else:
        Xs = Xc @ S; nrm = torch.cat([(V[s:s + 65536] @ S).norm(dim=-1) for s in range(0, V.shape[0], 65536)]).clamp_min(1e-8)
    out = {}
    if S is None: sel, _, _ = omp(Xs, V, max(KS), batch=128, record_err=False)
    else:
        # OMP on metric-mapped words without materialising V S: correlations <x S, v S>/|v S| = (x S S^T) v / |v S|
        M = S @ S.T; sel = torch.zeros(Xs.shape[0], max(KS), dtype=torch.long, device=DEV)
        for s0 in range(0, Xc.shape[0], 128):
            x = Xc[s0:s0 + 128]; r = x.clone(); chosen = []
            for step in range(max(KS)):
                corr = ((r @ M) @ V.T) / nrm[None]
                if chosen: corr.scatter_(1, torch.stack(chosen, 1), 0.0)
                j = corr.abs().argmax(1); chosen.append(j)
                As = V[torch.stack(chosen, 1)] @ S; Gm = As @ As.transpose(1, 2) + 1e-5 * torch.eye(len(chosen), device=DEV)
                c = torch.cholesky_solve(As @ (x @ S)[:, :, None], torch.linalg.cholesky(Gm))[:, :, 0]
                r = x - torch.einsum("nk,nkd->nd", c, V[torch.stack(chosen, 1)])   # x S ~ sum c (v S)  =>  x ~ sum c v
            sel[s0:s0 + 128] = torch.stack(chosen, 1)
    tot = Xc.pow(2).sum().item()
    for k in KS:
        s = sel[:, :k]
        if S is None:
            cof, _ = refit(Xs, V, s); Xh = torch.einsum("nk,nkd->nd", cof, V[s])
        else:
            As = V[s] @ S; Gm = As @ As.transpose(1, 2) + 1e-5 * torch.eye(k, device=DEV)
            cof = torch.cholesky_solve(As @ Xs[:, :, None], torch.linalg.cholesky(Gm))[:, :, 0]; Xh = torch.einsum("nk,nkd->nd", cof, V[s])
        if post is not None: Xh = Xh @ post
        Xd = lv.Xc - Xh; r_, lo, hi = lv.recovered(lv.splice(lv.mu + Xh)); out[str(k)] = dict(rec=r_, lo=lo, hi=hi, fvu=Xd.pow(2).sum().item() / tot)
    return out
cX = gauss_like(A.shape[0] // 2, SX, seed=2)   # half the native count (memory); a weaker control than e399's equal-size one
for mt, S in [("euc", None), ("fisher", SF)]:
    res["cells"][f"own:{mt}"] = describe_lean(lv.Xc, A, S)
    res["cells"][f"rot:{mt}"] = describe_lean(lv.Xc @ R.T, A, None if S is None else R @ S @ R.T, post=R) if S is not None else describe_lean(lv.Xc @ R.T, A, None, post=R)
    res["cells"][f"covX:{mt}"] = describe_lean(lv.Xc, cX, S)
    for v in ("own", "rot", "covX"):
        c_ = res["cells"][f"{v}:{mt}"]; log(f"{name} {v} {mt}: rec k4..64 " + " ".join(f"{c_[str(k)]['rec']:.2f}" for k in KS) + " | fvu " + " ".join(f"{c_[str(k)]['fvu']:.2f}" for k in KS))
g_ = lambda n, k=16: res["cells"][n][str(k)]["rec"]
summ = (f"{name} L{L} k16 euc/fisher: own {g_('own:euc'):.2f}/{g_('own:fisher'):.2f} rot {g_('rot:euc'):.2f}/{g_('rot:fisher'):.2f} covX {g_('covX:euc'):.2f}/{g_('covX:fisher'):.2f} | "
        f"top-8: variance {res['var_share_top8']:.2f}, Fisher {res['fisher_share_top8']:.3f}")
log(summ); record(f"e418_native7b_{name}", res, summ)
