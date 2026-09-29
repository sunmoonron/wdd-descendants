"""e586 (session 107): block the implementation, watch the object re-implement. In the grokking toy (e567's base run) the
32 eventual words are known; at step 1500, before grokking, training resumes from a saved state with those 32 neurons
frozen (read weights, bias and write row held at their step-1500 values), against a control with 32 random non-word
neurons frozen and the unfrozen run. At the end: test accuracy (does the network still grok?), the words (are they
other neurons, substitutes?), and for each base word the best-matching final word by the correlation of their
projection profiles over the training inputs (does the class re-implement itself in another row?), against random
pairs. Pre-registered (probabilities are honest guesses):
 S1 (0.6) the network still groks with its eventual words frozen (the object is not the rows');
 S2 (0.6) most base classes re-appear on substitute rows (profile correlation over 0.5 with a new word for most);
 S3 (0.7) the frozen eventual rows are not words at the end (a neuron cannot be recruited without moving)."""
import sys, os, math, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn as nn, torch.nn.functional as F
from wdd_common import log, record, DEV, omp, refit
from ex_common import gabs, unitr, rotate, med, mean
torch.set_grad_enabled(True); variant = sys.argv[1]; SAVE = "/workspace/wdd/results/e586_state.pt"; STEPS, FREEZE_AT, NW = 10000, 1500, 32
p, d, NH, HD, DM, V = 113, 128, 4, 32, 512, 114; lr, wd = 1e-3, 1.0
torch.manual_seed(0)
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
model = Grok().to(DEV); opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.98))
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
    out.update(advantage=out["rot_k4_fvu"] - out["own_k4_fvu"], words_S=S[words].median().item(), rotated_S=((U @ Ar.T).abs() / Lf[:, None]).max(0).values.topk(NW).values.median().item()); return out, dict(words=words.cpu(), ratio=ratio.cpu(), usage=usage.cpu())
def step_train():
    lg, _, _ = model(ALL[TR]); loss = F.cross_entropy(lg, LAB[TR]); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
t0 = time.time()
if variant == "base":
    for step in range(STEPS):
        if step == FREEZE_AT: torch.save(dict(model=model.state_dict(), opt=opt.state_dict()), SAVE)
        for gr in opt.param_groups: gr["lr"] = lr * min(1.0, (step + 1) / 10)
        step_train()
    o, cur = measure(); torch.save(dict(words=cur["words"], ratio=cur["ratio"], usage=cur["usage"]), SAVE.replace("state", "basewords")); summ = f"base: acc {o['train_acc']:.2f}/{o['test_acc']:.2f}, advantage {o['advantage']:.2f}, words' S {o['words_S']:.2f} (rotated {o['rotated_S']:.2f}); state at {FREEZE_AT} and the eventual words saved | {time.time() - t0:.0f}s"
    log(summ); record("e586_block_base", dict(variant=variant, final=o), summ)
else:
    sd = torch.load(SAVE, map_location=DEV); model.load_state_dict(sd["model"]); opt.load_state_dict(sd["opt"]); bw = torch.load(SAVE.replace("state", "basewords"), map_location="cpu"); words_b = bw["words"]
    if variant == "freeze_words": frozen = words_b.to(DEV)
    elif variant == "freeze_random": g = torch.Generator().manual_seed(3); nonw = torch.tensor([i for i in range(DM) if i not in set(words_b.tolist())]); frozen = nonw[torch.randperm(nonw.numel(), generator=g)[:NW]].to(DEV)
    else: frozen = torch.tensor([], dtype=torch.long, device=DEV)
    Win0, bin0, Wout0 = model.Win.detach()[:, frozen].clone(), model.bin.detach()[frozen].clone(), model.Wout.detach()[frozen].clone()
    def refreeze():
        with torch.no_grad(): model.Win[:, frozen] = Win0; model.bin[frozen] = bin0; model.Wout[frozen] = Wout0
    hooks = []
    if frozen.numel():
        def mask_grad(param, idx, dim):
            def hk(grad):
                g2 = grad.clone()
                if dim == 1: g2[:, idx] = 0
                else: g2[idx] = 0
                return g2
            return param.register_hook(hk)
        hooks = [mask_grad(model.Win, frozen, 1), mask_grad(model.bin, frozen, 0), mask_grad(model.Wout, frozen, 0)]
    for step in range(FREEZE_AT, STEPS):
        for gr in opt.param_groups: gr["lr"] = lr
        step_train(); refreeze()
    o, cur = measure(); words_f = cur["words"]; br, cr = bw["ratio"], cur["ratio"]
    frozen_are_words = mean([float(int(i) in set(words_f.tolist())) for i in frozen.tolist()]) if frozen.numel() else None; base_words_still_words = mean([float(int(i) in set(words_f.tolist())) for i in words_b.tolist()])
    # each base class's best substitute among the final words (excluding the base row itself)
    best = []
    for w in words_b.tolist():
        cands = [int(v) for v in words_f.tolist() if int(v) != w]; corr = [float(torch.corrcoef(torch.stack([br[:, w], cr[:, v]]))[0, 1]) for v in cands]; best.append(max(corr) if corr else 0.0)
    g2 = torch.Generator().manual_seed(4); rnd = [float(torch.corrcoef(torch.stack([br[:, int(w)], cr[:, int(v)]]))[0, 1]) for w, v in zip(words_b, torch.randint(0, DM, (NW,), generator=g2))]
    same = [float(torch.corrcoef(torch.stack([br[:, w], cr[:, w]]))[0, 1]) for w in words_b.tolist()]
    res = dict(variant=variant, final=o, frozen_are_words=frozen_are_words, base_words_still_words=base_words_still_words, substitute_corr_median=med(best), substitute_share_over_05=mean([float(x > 0.5) for x in best]), random_pair_corr=med(rnd), same_row_corr_median=med(same), n_frozen=int(frozen.numel()))
    summ = (f"{variant}: acc {o['train_acc']:.2f}/{o['test_acc']:.2f}, advantage {o['advantage']:.2f}, words' S {o['words_S']:.2f} (rotated {o['rotated_S']:.2f}); frozen neurons that are words at the end {frozen_are_words if frozen_are_words is None else round(frozen_are_words, 2)}, base words still words {base_words_still_words:.2f}; each base class's best substitute among the other final words: profile correlation {med(best):.2f} (over 0.5 for {mean([float(x > 0.5) for x in best]):.2f}; random pairs {med(rnd):.2f}; the same row {med(same):.2f}) | {time.time() - t0:.0f}s")
    log(summ); record(f"e586_block_{variant}", res, summ)
