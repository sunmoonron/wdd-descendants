"""e593 (session 110): recruitment-guided training. e587 and e588 showed that which rows will be recruited is readable
from the states long before recruitment. If so, the rows that will never be recruited are capacity the network is not
going to use, and pruning them early should cost nothing, while pruning the future recruits should cost grokking. The
toy of e586 (its base run saves the state at step 1500 and its eventual words); training resumes at 1500 with a set of
MLP neurons zeroed and frozen: the 256 with the lowest S at 1500 (the predicted never-recruits), the 256 with the
highest S (the predicted recruits among them), 256 at random, the 384 lowest (a quarter of the width kept), and none.
Four seeds of the resumed run per condition. Measures: test accuracy every 250 steps, the step at which it passes one
half, the final accuracy, the own-over-rotation advantage, and how many of the base run's 32 eventual words are words at
the end. Argument: condition_seed, e.g. low_0. Pre-registered (probabilities are honest guesses):
 T1 (0.6) pruning the lowest-S half does not delay grokking (within 500 steps of the unpruned run) while pruning the
    highest-S half delays it by 1000 steps or more or prevents it;
 T2 (0.5) keeping only the 128 highest-S neurons (the lowest 384 pruned) still groks;
 T3 (0.6) the base run's eventual words are words at the end for most of the low-pruned runs and few of the high-pruned."""
import sys, os, math, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn as nn, torch.nn.functional as F
from wdd_common import log, record, DEV, omp, refit
from ex_common import gabs, unitr, rotate, med, mean
torch.set_grad_enabled(True); cond, seed = sys.argv[1].rsplit("_", 1); seed = int(seed); SAVE = "/workspace/wdd/results/e586_state.pt"; STEPS, FREEZE_AT, NW = 10000, 1500, 32
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
sd = torch.load(SAVE, map_location=DEV); model.load_state_dict(sd["model"]); opt.load_state_dict(sd["opt"]); bw = torch.load(SAVE.replace("state", "basewords"), map_location="cpu"); words_b = bw["words"]
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
    r_cal = float(Mr.mean() / (s2r.clamp_min(1e-12).sqrt() * gm).mean()); Lf = s2.clamp_min(1e-12).sqrt() * gm * r_cal; S = ((U @ Am.T).abs() / Lf[:, None]).max(0).values; words = usage.topk(NW).indices
    out.update(advantage=out["rot_k4_fvu"] - out["own_k4_fvu"], words_S=S[words].median().item()); return out, S, words
o0, S0, _ = measure(); torch.manual_seed(seed); g = torch.Generator().manual_seed(seed + 100)
order = S0.argsort()   # ascending S at step 1500
frozen = {"low": order[:256], "high": order[256:], "random": torch.randperm(DM, generator=g)[:256].to(DEV), "low384": order[:384], "none": torch.tensor([], dtype=torch.long, device=DEV)}[cond]
base_in_frozen = float(sum(int(w) in set(frozen.tolist()) for w in words_b.tolist()) / NW) if frozen.numel() else 0.0
with torch.no_grad():
    if frozen.numel(): model.Win[:, frozen] = 0; model.bin[frozen] = 0; model.Wout[frozen] = 0
hooks = []
if frozen.numel():
    def mk(dim):
        def hk(grad):
            g2 = grad.clone()
            if dim == 1: g2[:, frozen] = 0
            else: g2[frozen] = 0
            return g2
        return hk
    hooks = [model.Win.register_hook(mk(1)), model.bin.register_hook(mk(0)), model.Wout.register_hook(mk(0))]
def rezero():
    with torch.no_grad():
        if frozen.numel(): model.Win[:, frozen] = 0; model.bin[frozen] = 0; model.Wout[frozen] = 0
t0 = time.time(); rows = [dict(step=FREEZE_AT, **o0)]
for step in range(FREEZE_AT, STEPS):
    for gr in opt.param_groups: gr["lr"] = lr
    lg, _, _ = model(ALL[TR]); loss = F.cross_entropy(lg, LAB[TR]); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); rezero()
    if (step + 1) % 250 == 0:
        o, S, words = measure(); rows.append(dict(step=step + 1, **o))
        if (step + 1) % 2500 == 0: log(f"{cond}_{seed} step {step + 1}: acc {o['train_acc']:.2f}/{o['test_acc']:.2f}, advantage {o['advantage']:.2f} | {time.time() - t0:.0f}s")
o, S, words = measure(); gs = next((r["step"] for r in rows if r["test_acc"] > 0.5), None); base_words_final = float(sum(int(w) in set(words.tolist()) for w in words_b.tolist()) / NW)
res = dict(condition=cond, seed=seed, n_frozen=int(frozen.numel()), base_words_in_frozen=base_in_frozen, log=rows, grok_step=gs, final=o, base_words_final=base_words_final)
summ = f"{cond}_{seed}: {int(frozen.numel())} neurons pruned at 1500 ({base_in_frozen:.2f} of the base words among them); test > 0.5 from step {gs}; final acc {o['train_acc']:.2f}/{o['test_acc']:.2f}, advantage {o['advantage']:.2f}, words' S {o['words_S']:.2f}; base words that are words at the end {base_words_final:.2f} | {time.time() - t0:.0f}s"
log(summ); record(f"e593_prune_{cond}_{seed}", res, summ)
