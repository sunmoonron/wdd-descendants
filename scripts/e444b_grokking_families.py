"""e444b: e444 re-run (grok, frozenW, frozenR) with the family-by-family test at the end: is each family of writers (MLP rows,
head bases, embeddings) a vocabulary on its own, against its own rotation? e449 showed that with frozen MLP rows the
description can move to other trained writers, so the frozen rows themselves must be tested. Snapshots every 1250 steps.
Original description follows.
e444: does the native vocabulary announce generalisation, and is it written or spoken? (new testbed: grokking)
All 400+ earlier experiments observed pretrained checkpoints. Here the network is trained from scratch on a task whose
algorithm is known: (a + b) mod 113 with a one-layer transformer. The setup follows Nanda et al. 2023:
- d_model 128, 4 heads, 512 ReLU neurons, no norms;
- full-batch AdamW, weight decay 1, 30% of the pairs for training.
Its generalising solution uses a handful of Fourier frequencies, so the right words are known.
Variants (argument):
- grok (weight decay 1), grok_s1 (seed 1);
- nowd (weight decay 0: memorises);
- randlab (training labels permuted: pure memorisation);
- frozenW (MLP write rows fixed at their random initialisation), frozenR (MLP read rows fixed).
Every 250 steps, on the final residual state at the '=' position for all 12769 inputs:
 - train and test accuracy and loss;
 - self-description of the state (centred on training inputs) with k = 2, 4, 8 words from four sources:
   - the model's own words: embedding rows, position rows, head output bases, MLP write rows and the MLP bias;
   - the same words rotated;
   - Gaussian words with their second moment (covA);
   - the top-k principal directions of the training states (PCA, a dictionary-free baseline).
   Scored by loss recovered through the unembedding (training and test inputs) and by the label-free fraction of
   variance unexplained on training inputs;
 - ground truth:
   - the share of the embedding's norm in its top five Fourier frequencies;
   - at the end, the frequency purity of the most used MLP words (the share of their output Fourier power at one
     frequency) against all MLP rows.
Pre-registered (probabilities are honest guesses):
 G1 (0.5) in the grokking runs the own-over-rotation advantage on training inputs rises before test accuracy passes 50%;
 G2 (0.5) memorising runs (nowd, randlab) have a smaller advantage than grokking runs at equal training accuracy;
 G3 (0.6) after grokking the most used MLP words are frequency-pure;
 G4 (0.4) with frozen random write rows the states still become sparse in them (own over rotation): a spoken vocabulary;
 G5 (0.5) after grokking, k own words beat the top-k principal subspace (input-dependent sparsity beyond low rank)."""
import sys, os, math, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn as nn, torch.nn.functional as F
from wdd_common import log, record, DEV, omp, refit
torch.set_grad_enabled(True)
variant = sys.argv[1]
cfg = dict(seed=0, wd=1.0, randlab=False, freeze=None, steps=25000, every=1250, lr=1e-3, frac=0.3)
cfg.update({"grok": {}, "grok_s1": dict(seed=1), "nowd": dict(wd=0.0), "randlab": dict(randlab=True), "frozenW": dict(freeze="W"), "frozenR": dict(freeze="R")}[variant])
p, d, NH, HD, DM = 113, 128, 4, 32, 512; V = p + 1
torch.manual_seed(cfg["seed"])
A_, B_ = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij"); ALL = torch.stack([A_.flatten(), B_.flatten(), torch.full((p * p,), p)], 1).to(DEV)
LAB = ((ALL[:, 0] + ALL[:, 1]) % p).clone()
perm = torch.randperm(p * p, generator=torch.Generator().manual_seed(cfg["seed"] + 1)).to(DEV); ntr = int(cfg["frac"] * p * p)
TR = torch.zeros(p * p, dtype=torch.bool, device=DEV); TR[perm[:ntr]] = True; TE = ~TR
if cfg["randlab"]:
    tri = torch.nonzero(TR)[:, 0]; LAB[tri] = LAB[tri][torch.randperm(tri.numel(), generator=torch.Generator().manual_seed(7)).to(DEV)]
class Grok(nn.Module):
    def __init__(s):
        super().__init__(); r = lambda *sh, fan: nn.Parameter(torch.randn(*sh) / math.sqrt(fan))
        s.WE, s.WP = r(V, d, fan=d), r(3, d, fan=d); s.WQ, s.WK, s.WV = r(NH, d, HD, fan=d), r(NH, d, HD, fan=d), r(NH, d, HD, fan=d); s.WO = r(NH, HD, d, fan=HD)
        s.Win, s.bin = r(d, DM, fan=d), nn.Parameter(torch.zeros(DM)); s.Wout, s.bout = r(DM, d, fan=DM), nn.Parameter(torch.zeros(d)); s.WU = r(d, p, fan=d)
    def forward(s, ids):
        x = s.WE[ids] + s.WP[None]
        q, k, v = (torch.einsum("btd,hde->bhte", x, W) for W in (s.WQ, s.WK, s.WV))
        att = (q @ k.transpose(-1, -2)) / math.sqrt(HD); att = att.masked_fill(torch.triu(torch.ones(3, 3, dtype=torch.bool, device=ids.device), 1), -1e9).softmax(-1)
        x = x + torch.einsum("bhte,hed->btd", att @ v, s.WO); h = F.relu(x @ s.Win + s.bin); x = x + h @ s.Wout + s.bout
        return x[:, -1] @ s.WU, x[:, -1], h[:, -1]
model = Grok().to(DEV)
if cfg["freeze"] == "W": model.Wout.requires_grad_(False); model.bout.requires_grad_(False)
if cfg["freeze"] == "R": model.Win.requires_grad_(False); model.bin.requires_grad_(False)
opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=cfg["lr"], weight_decay=cfg["wd"], betas=(0.9, 0.98))
Q = torch.linalg.qr(torch.randn(d, d, generator=torch.Generator().manual_seed(11)))[0].to(DEV)
unit = lambda M: M / M.norm(dim=-1, keepdim=True).clamp_min(1e-8)
fm = lambda x: "n/a" if x is None else f"{x:.2f}"
def own_words():
    heads = torch.cat([torch.linalg.svd(model.WO[h].detach(), full_matrices=False).Vh for h in range(NH)])
    parts = [model.WE.detach(), model.WP.detach(), heads, model.Wout.detach(), model.bout.detach()[None]]
    labels = ["E"] * V + ["P"] * 3 + ["H"] * (NH * HD) + ["M"] * DM + ["b"]
    return unit(torch.cat(parts)), labels
def fourier_power(M, per_col=False):
    """power of M [p, n] along its p axis per frequency 1..p//2 (DC dropped): summed over columns, or per column [p//2, n]"""
    Fm = torch.fft.rfft(M.float(), dim=0).abs().pow(2)[1:]; return Fm if per_col else Fm.sum(-1)
@torch.no_grad()
def measure(step):
    lg, xf, h = model(ALL); L = F.cross_entropy(lg, LAB, reduction="none"); acc = (lg.argmax(-1) == LAB).float()
    WU = model.WU.detach(); mu = xf[TR].mean(0); Xc = xf - mu; Lm = F.cross_entropy((mu[None] @ WU).expand(xf.shape[0], -1), LAB, reduction="none")
    A, _ = own_words(); n = A.shape[0]
    ev, U = torch.linalg.eigh((A.T @ A / n).double()); Rm = ((U * ev.clamp_min(0).sqrt()) @ U.T).float()
    dicts = dict(own=A, rot=A @ Q, covA=unit(torch.randn(n, d, generator=torch.Generator().manual_seed(13)).to(DEV) @ Rm))
    out = dict(step=step, train_acc=acc[TR].mean().item(), test_acc=acc[TE].mean().item(), train_loss=L[TR].mean().item(), test_loss=L[TE].mean().item(),
               wnorm=sum(q.detach().pow(2).sum() for q in model.parameters()).sqrt().item())
    def rec(Xh, mask):
        """loss recovered on the inputs in mask; None while the model is within 0.5 nats of the mean state there (undefined)"""
        gap = (Lm[mask] - L[mask]).mean()
        if gap < 0.5: return None
        Ld = F.cross_entropy((mu[None] + Xh) @ WU, LAB, reduction="none"); return ((Lm[mask] - Ld[mask]).mean() / gap).item()
    tot = Xc[TR].pow(2).sum()
    for vn, D in dicts.items():
        sel, _, _ = omp(Xc, D, 8, batch=4096, record_err=False)
        for k in (2, 4, 8):
            cof, err = refit(Xc, D, sel[:, :k]); Xh = torch.einsum("nk,nkd->nd", cof, D[sel[:, :k]])
            out[f"{vn}_k{k}_train"], out[f"{vn}_k{k}_test"], out[f"{vn}_k{k}_fvu"] = rec(Xh, TR), rec(Xh, TE), (err[TR].sum() / tot).item()
        if vn == "own": out["_sel_own"] = sel
    evs, Us = torch.linalg.eigh(torch.cov(Xc[TR].T.double(), correction=0)); Us = Us.flip(-1).float()
    for k in (2, 4, 8):
        P = Us[:, :k] @ Us[:, :k].T; Xh = Xc @ P; out[f"pca_k{k}_train"], out[f"pca_k{k}_test"], out[f"pca_k{k}_fvu"] = rec(Xh, TR), rec(Xh, TE), ((Xc[TR] - Xh[TR]).pow(2).sum() / tot).item()
    out["pr_states"] = (evs.sum() ** 2 / evs.pow(2).sum()).item()
    fp = fourier_power(model.WE.detach()[:p]); out["emb_top5_fourier"] = (fp.sort(descending=True).values[:5].sum() / fp.sum()).item()
    return out
log_rows = []; t0 = time.time()
for step in range(cfg["steps"] + 1):
    if step % cfg["every"] == 0:
        m = measure(step); m.pop("_sel_own"); log_rows.append(m)
        if step % 2500 == 0: log(f"{variant} step {step}: train/test acc {m['train_acc']:.2f}/{m['test_acc']:.2f} | k4 loss recovered own/rot/covA/pca (train) " + "/".join(fm(m[f'{v}_k4_train']) for v in ('own', 'rot', 'covA', 'pca'))
                                 + " | k4 FVU own/rot/covA/pca " + "/".join(fm(m[f'{v}_k4_fvu']) for v in ('own', 'rot', 'covA', 'pca')) + f" | emb top-5 Fourier {m['emb_top5_fourier']:.2f} | {time.time() - t0:.0f}s")
    if step == cfg["steps"]: break
    for gr in opt.param_groups: gr["lr"] = cfg["lr"] * min(1.0, (step + 1) / 10)
    lg, _, _ = model(ALL[TR]); loss = F.cross_entropy(lg, LAB[TR]); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
# end of training: which words the description uses, and are they frequency-pure?
fin = measure(cfg["steps"]); sel = fin.pop("_sel_own"); A, labels = own_words(); isM = torch.tensor([l == "M" for l in labels], device=DEV)
cnt = torch.bincount(sel[TR].flatten(), minlength=A.shape[0]).float(); mlp_idx = torch.nonzero(isM)[:, 0]; top = mlp_idx[cnt[mlp_idx].topk(20).indices]
with torch.no_grad():
    WU = model.WU.detach(); outF = fourier_power((A[mlp_idx] @ WU).T, per_col=True); purity_all = outF.max(0).values / outF.sum(0)       # [n_mlp]
    outT = fourier_power((A[top] @ WU).T, per_col=True); purity_top = outT.max(0).values / outT.sum(0)
    fpE = fourier_power(model.WE.detach()[:p]); key = (fpE.sort(descending=True).indices[:5] + 1).tolist()
    dom_top = (outT.argmax(0) + 1).tolist()
fam = {}
with torch.no_grad():
    lg_, xf_, _ = model(ALL); mu_ = xf_[TR].mean(0); Xc_ = (xf_ - mu_)[TR]; tot_ = Xc_.pow(2).sum()
    for fname, t in (("mlp_rows", "M"), ("head_bases", "H"), ("embeddings", "E")):
        Df = A[torch.tensor([l == t for l in labels], device=DEV)]; o = {}
        for vn, Dd in (("own", Df), ("rot", Df @ Q)):
            s_, _, _ = omp(Xc_, Dd, 4, batch=4096, record_err=False); _, e_ = refit(Xc_, Dd, s_); o[vn] = (e_.sum() / tot_).item()
        fam[fname] = o
res = dict(variant=variant, cfg=cfg, log=log_rows, final=fin, key_freqs=key, families_k4=fam, top_mlp_words=top.tolist(), top_word_uses=cnt[top].tolist(), top_word_purity=purity_top.tolist(),
           top_word_dominant_freq=dom_top, mean_purity_top=purity_top.mean().item(), mean_purity_all_mlp=purity_all.mean().item(), top_in_key=sum(f in key for f in dom_top) / len(dom_top),
           usage_share_by_type={t: (cnt[torch.tensor([l == t for l in labels], device=DEV)].sum() / cnt.sum()).item() for t in "EPHMb"})
gstep = next((r["step"] for r in log_rows if r["test_acc"] > 0.5), None)
first = lambda key, frac: (lambda v: next((r["step"] for r, a in zip(log_rows, v) if a > frac * max(v)), None) if max(v) > 0 else None)([r[key] if not callable(key) else key(r) for r in log_rows])
adv = [r["rot_k4_fvu"] - r["own_k4_fvu"] for r in log_rows]; wadv = [r["covA_k4_fvu"] - r["own_k4_fvu"] for r in log_rows]
res["grok_step"] = gstep; res["adv_fvu_half_step"] = first(lambda r: r["rot_k4_fvu"] - r["own_k4_fvu"], 0.5); res["wordlevel_fvu_half_step"] = first(lambda r: r["covA_k4_fvu"] - r["own_k4_fvu"], 0.5)
res["fourier_half_step"] = first(lambda r: r["emb_top5_fourier"] - log_rows[0]["emb_top5_fourier"], 0.5); res["adv_fvu_max"], res["wordlevel_fvu_max"] = max(adv), max(wadv)
fit = next((r for r in log_rows if r["train_acc"] >= 0.99), None); res["at_train_fit"] = {k: fit[k] for k in ("step", "test_acc", "own_k4_fvu", "rot_k4_fvu", "covA_k4_fvu", "pca_k4_fvu", "own_k4_train", "rot_k4_train", "covA_k4_train")} if fit else None
half = res["adv_fvu_half_step"]
af = res["at_train_fit"]
summ = (f"{variant}: final train/test acc {fin['train_acc']:.2f}/{fin['test_acc']:.2f}; test acc >0.5 from step {gstep}; label-free own-over-rotation FVU advantage (k4, train) reaches half its max ({max(adv):.2f}) at step {half}, "
        f"own-over-covA (word-level) at step {res['wordlevel_fvu_half_step']} (max {max(wadv):.2f}); embedding Fourier concentration half-way at step {res['fourier_half_step']} | "
        + (f"at train fit (step {af['step']}, test acc {af['test_acc']:.2f}): k4 FVU own/rot/covA/pca {af['own_k4_fvu']:.2f}/{af['rot_k4_fvu']:.2f}/{af['covA_k4_fvu']:.2f}/{af['pca_k4_fvu']:.2f} | " if af else "")
        + "final k2/k4/k8 loss recovered (train) " + " ".join(f"{v} " + "/".join(fm(fin[f'{v}_k{k}_train']) for k in (2, 4, 8)) for v in ("own", "rot", "covA", "pca"))
        + " | final k4 FVU " + " ".join(f"{v} {fin[f'{v}_k4_fvu']:.2f}" for v in ("own", "rot", "covA", "pca")) + f" | test k4 own/pca {fm(fin['own_k4_test'])}/{fm(fin['pca_k4_test'])} | "
        f"usage by type {json.dumps({k: round(v, 2) for k, v in res['usage_share_by_type'].items()})} | top-20 MLP words frequency purity {res['mean_purity_top']:.2f} vs all MLP rows {res['mean_purity_all_mlp']:.2f}, dominant frequency among the embedding's key frequencies {res['top_in_key']:.2f} (key {key}) | "
        f"emb top-5 Fourier share {fin['emb_top5_fourier']:.2f} | by family k4 unexplained own/rot: " + " ".join(f"{k} {v['own']:.2f}/{v['rot']:.2f}" for k, v in fam.items()) + f" | {time.time() - t0:.0f}s")
log(summ); record(f"e444b_grok_{variant}", res, summ)
