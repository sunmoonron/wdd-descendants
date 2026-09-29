"""e576 (session 103): can trained rows attract a fresh network's cloud? The loop's direction in the language models is
data -> neuron (e571). The toy allows the reverse test: the grokked base run's MLP write rows are saved; then a fresh
network (new embeddings, attention, read weights, biases, unembedding) is trained with the write rows frozen at
 (a) the trained rows, (b) the trained rows permuted among the neurons (the same directions, other neurons), and
 (c) fresh random rows (e444's frozenW). At the end: the words (32 most used MLP rows), their overlap with the base
run's words, and for every base word the correlation of its projection profile over the training inputs in the new run
with its profile in the base run (the same class of inputs?), against random row pairs; the own-over-rotation
advantage and the accuracies. Pre-registered (probabilities are honest guesses):
 Z1 (0.5) with the trained rows frozen the fresh network re-uses the same rows as words for the same classes (profile
    correlation over 0.5 for most base words): the rows can attract the cloud when they already point the right way;
 Z2 (0.6) with the rows permuted the classes follow the directions, not the neuron indices;
 Z3 (0.7) with fresh random rows frozen the vocabulary barely forms (e444, e449)."""
import sys, os, math, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn as nn, torch.nn.functional as F
from wdd_common import log, record, DEV, omp, refit
from ex_common import gabs, unitr, rotate, med, mean
torch.set_grad_enabled(True); variant = sys.argv[1]; ROWS = "/workspace/wdd/results/e576_base_rows.pt"
p, d, NH, HD, DM, V = 113, 128, 4, 32, 512, 114; STEPS, EVERY, NW = 10000, 500, 32; lr, wd = 1e-3, 1.0
torch.manual_seed({"base": 0, "trained": 5, "permuted": 6, "random": 7}[variant])
A_, B_ = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij"); ALL = torch.stack([A_.flatten(), B_.flatten(), torch.full((p * p,), p)], 1).to(DEV); LAB = ((ALL[:, 0] + ALL[:, 1]) % p).clone()
perm = torch.randperm(p * p, generator=torch.Generator().manual_seed(1)).to(DEV); ntr = int(0.3 * p * p); TR = torch.zeros(p * p, dtype=torch.bool, device=DEV); TR[perm[:ntr]] = True; TE = ~TR
class Grok(nn.Module):
    def __init__(s):
        super().__init__(); r = lambda *sh, fan: nn.Parameter(torch.randn(*sh) / math.sqrt(fan))
        s.WE, s.WP = r(V, d, fan=d), r(3, d, fan=d); s.WQ, s.WK, s.WV = r(NH, d, HD, fan=d), r(NH, d, HD, fan=d), r(NH, d, HD, fan=d); s.WO = r(NH, HD, d, fan=HD)
        s.Win, s.bin = r(d, DM, fan=d), nn.Parameter(torch.zeros(DM)); s.Wout, s.bout = r(DM, d, fan=DM), nn.Parameter(torch.zeros(d)); s.WU = r(d, p, fan=d)
    def forward(s, ids):
        x = s.WE[ids] + s.WP[None]; q, k, v = (torch.einsum("btd,hde->bhte", x, W) for W in (s.WQ, s.WK, s.WV))
        att = (q @ k.transpose(-1, -2)) / math.sqrt(HD); att = att.masked_fill(torch.triu(torch.ones(3, 3, dtype=torch.bool, device=ids.device), 1), -1e9).softmax(-1)
        x = x + torch.einsum("bhte,hed->btd", att @ v, s.WO); h = F.relu(x @ s.Win + s.bin); x = x + h @ s.Wout + s.bout
        return x[:, -1] @ s.WU, x[:, -1], h[:, -1]
model = Grok().to(DEV)
if variant != "base":
    base = torch.load(ROWS, map_location=DEV); Wb = base["Wout"]
    with torch.no_grad():
        if variant == "trained": model.Wout.copy_(Wb)
        elif variant == "permuted": model.Wout.copy_(Wb[torch.randperm(DM, generator=torch.Generator().manual_seed(9)).to(DEV)])
        elif variant == "random": model.Wout.mul_(1.0)   # fresh random rows at their initial scale
        model.Wout.requires_grad_(False)
opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=lr, weight_decay=wd, betas=(0.9, 0.98))
Q = torch.linalg.qr(torch.randn(d, d, generator=torch.Generator().manual_seed(11)))[0].to(DEV); unit = lambda M: M / M.norm(dim=-1, keepdim=True).clamp_min(1e-8); gm = gabs(DM)
def own_words():
    heads = torch.cat([torch.linalg.svd(model.WO[h].detach(), full_matrices=False).Vh for h in range(NH)])
    return unit(torch.cat([model.WE.detach(), model.WP.detach(), heads, model.Wout.detach(), model.bout.detach()[None]])), V + 3 + NH * HD
@torch.no_grad()
def measure():
    lg, xf, h = model(ALL); acc = (lg.argmax(-1) == LAB).float(); A, m0 = own_words(); n = A.shape[0]; Xc = xf[TR] - xf[TR].mean(0); U = unit(Xc); tot = Xc.pow(2).sum(); out = dict(train_acc=acc[TR].mean().item(), test_acc=acc[TE].mean().item())
    for vn, Dd in (("own", A), ("rot", A @ Q)):
        sel, _, _ = omp(Xc, Dd, 8, batch=4096, record_err=False); cof, err = refit(Xc, Dd, sel[:, :4]); out[f"{vn}_k4_fvu"] = (err.sum() / tot).item()
        if vn == "own": usage = torch.bincount(sel.flatten(), minlength=n).float()[m0:m0 + DM]
    Am = A[m0:m0 + DM]; Ar = unit(rotate(Am, seed=7)); CA = Am.T @ Am / DM; CR = Ar.T @ Ar / DM; s2 = ((U @ CA) * U).sum(1); s2r = ((U @ CR) * U).sum(1); Mr = (U @ Ar.T).abs().max(1).values
    r_cal = float(Mr.mean() / (s2r.clamp_min(1e-12).sqrt() * gm).mean()); Lf = s2.clamp_min(1e-12).sqrt() * gm * r_cal; ratio = (U @ Am.T).abs() / Lf[:, None]; S = ratio.max(0).values; words = usage.topk(NW).indices
    out.update(advantage=out["rot_k4_fvu"] - out["own_k4_fvu"], words_S=S[words].median().item(), rotated_S=((U @ Ar.T).abs() / Lf[:, None]).max(0).values.topk(NW).values.median().item(), top32_share=(usage[words].sum() / usage.sum().clamp_min(1)).item())
    return out, dict(words=words.cpu(), ratio=ratio.cpu(), usage=usage.cpu())
t0 = time.time(); rows = []
for step in range(STEPS + 1):
    if step % EVERY == 0:
        o, cur = measure(); rows.append(dict(step=step, **o))
        if step % 2500 == 0: log(f"{variant} step {step}: acc {o['train_acc']:.2f}/{o['test_acc']:.2f}, advantage {o['advantage']:.2f}, words' S {o['words_S']:.2f} (rotated {o['rotated_S']:.2f}), top-32 share {o['top32_share']:.2f} | {time.time() - t0:.0f}s")
    if step == STEPS: break
    for gr in opt.param_groups: gr["lr"] = lr * min(1.0, (step + 1) / 10)
    lg, _, _ = model(ALL[TR]); loss = F.cross_entropy(lg, LAB[TR]); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
res = dict(variant=variant, log=rows, final=rows[-1])
if variant == "base":
    torch.save(dict(Wout=model.Wout.detach().cpu(), words=cur["words"], ratio=cur["ratio"], usage=cur["usage"]), ROWS); summ = f"base: acc {rows[-1]['train_acc']:.2f}/{rows[-1]['test_acc']:.2f}, advantage {rows[-1]['advantage']:.2f}, words' S {rows[-1]['words_S']:.2f} (rotated {rows[-1]['rotated_S']:.2f}); rows saved | {time.time() - t0:.0f}s"
else:
    bw = base["words"].cpu(); br = base["ratio"].cpu(); bu = base["usage"].cpu()
    if variant == "permuted": pm = torch.randperm(DM, generator=torch.Generator().manual_seed(9)); inv = torch.empty_like(pm); inv[pm] = torch.arange(DM); nw_of = lambda r: int(inv[r])   # base neuron r's row now sits at neuron inv[r]
    else: nw_of = lambda r: int(r)
    cw = set(cur["words"].tolist()); overlap = mean([float(nw_of(int(r)) in cw) for r in bw]); corr = [float(torch.corrcoef(torch.stack([br[:, int(r)], cur["ratio"][:, nw_of(int(r))]]))[0, 1]) for r in bw]
    g = torch.Generator().manual_seed(2); rc = [float(torch.corrcoef(torch.stack([br[:, int(r)], cur["ratio"][:, int(j)]]))[0, 1]) for r, j in zip(bw, torch.randint(0, DM, (NW,), generator=g))]
    ucorr = float(torch.corrcoef(torch.stack([bu, cur["usage"][torch.tensor([nw_of(i) for i in range(DM)])]]))[0, 1])
    res.update(base_words_reused=overlap, profile_corr_median=med(corr), profile_corr_share_over_05=mean([float(c > 0.5) for c in corr]), random_pair_corr=med(rc), usage_corr_with_base=ucorr)
    summ = (f"{variant} rows frozen, fresh network: acc {rows[-1]['train_acc']:.2f}/{rows[-1]['test_acc']:.2f}, advantage {rows[-1]['advantage']:.2f}, words' S {rows[-1]['words_S']:.2f} (rotated {rows[-1]['rotated_S']:.2f}), top-32 share {rows[-1]['top32_share']:.2f}; base words re-used as words {overlap:.2f}, base words' profile correlation (same class of inputs) {med(corr):.2f} (share > 0.5 {mean([float(c > 0.5) for c in corr]):.2f}; random pairs {med(rc):.2f}), usage correlation with the base {ucorr:.2f} | {time.time() - t0:.0f}s")
log(summ); record(f"e576_frozen_{variant}", res, summ)
