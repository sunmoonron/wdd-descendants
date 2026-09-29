"""e580 (session 105): the generalisation detector, tested brutally simply. Twenty toy trajectories (e567's trainer: five
knobs, base / decay 0.3 / a fifth of the data / lr 3e-3 / decay 3, times four seeds; 10000 steps), every 250 steps the
WDD statistics (the native excess: the 32 most used MLP rows' S over the rotated rows' S; the own-over-rotation FVU
advantage; the top-32 usage share) beside standard label-free baselines (training loss; the total and MLP weight norms;
the training states' participation ratio and top-8 principal share; the ReLU activation density; the embedding's
top-5 Fourier share, the task-specific classic) and the labels (test loss and accuracy). e580b asks which statistic at
time t predicts generalisation at t + delta. Argument: variant_seed, e.g. base_0."""
import sys, os, math, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn as nn, torch.nn.functional as F
from wdd_common import log, record, DEV, omp, refit
from ex_common import gabs, unitr, rotate, med, mean
torch.set_grad_enabled(True); variant, seed = sys.argv[1].rsplit("_", 1); seed = int(seed)
cfg = dict(seed=seed, wd=1.0, randlab=False, steps=10000, every=250, lr=1e-3, frac=0.3, DM=512, init=1.0)
cfg.update({"base": {}, "wd0.3": dict(wd=0.3), "frac0.2": dict(frac=0.2), "lr3e-3": dict(lr=3e-3), "wd3": dict(wd=3.0)}[variant])
p, d, NH, HD = 113, 128, 4, 32; DM = cfg["DM"]; V = p + 1; NW = 32
torch.manual_seed(cfg["seed"])
A_, B_ = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij"); ALL = torch.stack([A_.flatten(), B_.flatten(), torch.full((p * p,), p)], 1).to(DEV); LAB = ((ALL[:, 0] + ALL[:, 1]) % p).clone()
perm = torch.randperm(p * p, generator=torch.Generator().manual_seed(cfg["seed"] + 1)).to(DEV); ntr = int(cfg["frac"] * p * p); TR = torch.zeros(p * p, dtype=torch.bool, device=DEV); TR[perm[:ntr]] = True; TE = ~TR
class Grok(nn.Module):
    def __init__(s):
        super().__init__(); r = lambda *sh, fan: nn.Parameter(cfg["init"] * torch.randn(*sh) / math.sqrt(fan))
        s.WE, s.WP = r(V, d, fan=d), r(3, d, fan=d); s.WQ, s.WK, s.WV = r(NH, d, HD, fan=d), r(NH, d, HD, fan=d), r(NH, d, HD, fan=d); s.WO = r(NH, HD, d, fan=HD)
        s.Win, s.bin = r(d, DM, fan=d), nn.Parameter(torch.zeros(DM)); s.Wout, s.bout = r(DM, d, fan=DM), nn.Parameter(torch.zeros(d)); s.WU = r(d, p, fan=d)
    def forward(s, ids):
        x = s.WE[ids] + s.WP[None]; q, k, v = (torch.einsum("btd,hde->bhte", x, W) for W in (s.WQ, s.WK, s.WV))
        att = (q @ k.transpose(-1, -2)) / math.sqrt(HD); att = att.masked_fill(torch.triu(torch.ones(3, 3, dtype=torch.bool, device=ids.device), 1), -1e9).softmax(-1)
        x = x + torch.einsum("bhte,hed->btd", att @ v, s.WO); h = F.relu(x @ s.Win + s.bin); x = x + h @ s.Wout + s.bout
        return x[:, -1] @ s.WU, x[:, -1], h[:, -1]
model = Grok().to(DEV); opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"], betas=(0.9, 0.98))
Q = torch.linalg.qr(torch.randn(d, d, generator=torch.Generator().manual_seed(11)))[0].to(DEV); unit = lambda M: M / M.norm(dim=-1, keepdim=True).clamp_min(1e-8); gm = gabs(DM)
def own_words():
    heads = torch.cat([torch.linalg.svd(model.WO[h].detach(), full_matrices=False).Vh for h in range(NH)])
    return unit(torch.cat([model.WE.detach(), model.WP.detach(), heads, model.Wout.detach(), model.bout.detach()[None]])), V + 3 + NH * HD
def fourier_top5(M):
    Fm = torch.fft.rfft(M.float(), dim=0).abs().pow(2)[1:].sum(-1); return float(Fm.sort(descending=True).values[:5].sum() / Fm.sum())
@torch.no_grad()
def measure(step):
    lg, xf, h = model(ALL); L = F.cross_entropy(lg, LAB, reduction="none"); acc = (lg.argmax(-1) == LAB).float(); A, m0 = own_words(); n = A.shape[0]; Xc = xf[TR] - xf[TR].mean(0); U = unit(Xc); tot = Xc.pow(2).sum()
    out = dict(step=step, train_loss=L[TR].mean().item(), test_loss=L[TE].mean().item(), train_acc=acc[TR].mean().item(), test_acc=acc[TE].mean().item())
    for vn, Dd in (("own", A), ("rot", A @ Q)):
        sel, _, _ = omp(Xc, Dd, 8, batch=4096, record_err=False); cof, err = refit(Xc, Dd, sel[:, :4]); out[f"{vn}_k4_fvu"] = (err.sum() / tot).item()
        if vn == "own": usage = torch.bincount(sel.flatten(), minlength=n).float()[m0:m0 + DM]
    Am = A[m0:m0 + DM]; Ar = unit(rotate(Am, seed=7)); CA = Am.T @ Am / DM; CR = Ar.T @ Ar / DM; s2 = ((U @ CA) * U).sum(1); s2r = ((U @ CR) * U).sum(1); Mr = (U @ Ar.T).abs().max(1).values
    r_cal = float(Mr.mean() / (s2r.clamp_min(1e-12).sqrt() * gm).mean()); Lf = s2.clamp_min(1e-12).sqrt() * gm * r_cal; S = ((U @ Am.T).abs() / Lf[:, None]).max(0).values; Sr = ((U @ Ar.T).abs() / Lf[:, None]).max(0).values; words = usage.topk(NW).indices
    out.update(wdd_excess=float(S[words].median() / Sr.topk(NW).values.median().clamp_min(1e-6)), wdd_advantage=out["rot_k4_fvu"] - out["own_k4_fvu"], wdd_top32_share=float(usage[words].sum() / usage.sum().clamp_min(1)), wdd_rows_over_floor=int((S >= 1).sum()) - int((Sr >= 1).sum()))
    ev = torch.linalg.eigvalsh(torch.cov(Xc.T.double(), correction=0)).flip(0).clamp_min(0); out.update(weight_norm=float(sum(q.pow(2).sum() for q in model.parameters()).sqrt()), mlp_weight_norm=float((model.Win.pow(2).sum() + model.Wout.pow(2).sum()).sqrt()),
               state_pr=float(ev.sum() ** 2 / ev.pow(2).sum()), top8_pc_share=float(ev[:8].sum() / ev.sum()), relu_density=float((h[TR] > 0).float().mean()), emb_fourier_top5=fourier_top5(model.WE.detach()[:p]))
    return out
rows, t0 = [], time.time()
for step in range(cfg["steps"] + 1):
    if step % cfg["every"] == 0:
        rows.append(measure(step))
        if step % 2500 == 0: r = rows[-1]; log(f"{variant}_{seed} step {step}: acc {r['train_acc']:.2f}/{r['test_acc']:.2f}, excess {r['wdd_excess']:.2f}, advantage {r['wdd_advantage']:.2f}, weight norm {r['weight_norm']:.1f}, PR {r['state_pr']:.1f}, Fourier {r['emb_fourier_top5']:.2f} | {time.time() - t0:.0f}s")
    if step == cfg["steps"]: break
    for gr in opt.param_groups: gr["lr"] = cfg["lr"] * min(1.0, (step + 1) / 10)
    lg, _, _ = model(ALL[TR]); loss = F.cross_entropy(lg, LAB[TR]); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
gs = next((r["step"] for r in rows if r["test_acc"] > 0.5), None)
summ = f"{variant}_{seed}: final acc {rows[-1]['train_acc']:.2f}/{rows[-1]['test_acc']:.2f}, test > 0.5 from step {gs}; final excess {rows[-1]['wdd_excess']:.2f}, advantage {rows[-1]['wdd_advantage']:.2f} | {time.time() - t0:.0f}s"
log(summ); record(f"e580_det_{variant}_{seed}", dict(variant=variant, seed=seed, cfg=cfg, log=rows, grok_step=gs), summ)
