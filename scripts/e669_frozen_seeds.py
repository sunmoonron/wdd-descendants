"""e669 (session 151; Gemini 4's reading of e576): e576 froze a fresh grokking network's MLP write rows at the grokked
run's rows, at those rows permuted among the neurons, or at random rows, one run each for 10000 steps, and recorded that
none had grokked. Gemini: the permuted run was mid-grok (test accuracy 0.144 at 5000 steps, 0.335 at 10000) while the
random one was flat (0.048), so trained write directions accelerate generalization, and a run with random directions at
the trained rows' norms (e569b: trained rows are heavy-tailed in norm) would separate direction from norm. Missing from
that reading: e576 gave each condition its own initialization seed, and in a fresh network the trained and permuted
conditions are the same experiment in distribution (every other weight is new), yet they ended at 0.08 and 0.34, so one
run per condition cannot separate a condition from its seed. Here e576's network and training exactly (p = 113, a third
of the pairs for training, AdamW, weight decay 1), five conditions with the write rows frozen (trained; permuted; random
at the initial scale, e576's; random directions with the trained rows' norms; random directions at the trained rows'
mean norm) and the unfrozen network as the reference, five seeds each, 20000 steps, test accuracy every 250 steps.
Arguments: condition seed [--smoke]."""
import sys, os, math, time, json
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_grad_enabled(True); DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
cond, seed = sys.argv[1], int(sys.argv[2]); SMOKE = "--smoke" in sys.argv; t0 = time.time()
p, d, NH, HD, DM, V = 113, 128, 4, 32, 512, 114; STEPS, EVERY = (2000 if SMOKE else 20000), 250; lr, wd = 1e-3, 1.0
torch.manual_seed(1000 * seed + {"unfrozen": 0, "trained": 1, "permuted": 2, "random": 3, "random_normmatched": 4, "random_meanmatched": 5}[cond])
A_, B_ = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij"); ALL = torch.stack([A_.flatten(), B_.flatten(), torch.full((p * p,), p)], 1).to(DEV); LAB = ((ALL[:, 0] + ALL[:, 1]) % p).clone()
perm = torch.randperm(p * p, generator=torch.Generator().manual_seed(1)).to(DEV); ntr = int(0.3 * p * p); TR = torch.zeros(p * p, dtype=torch.bool, device=DEV); TR[perm[:ntr]] = True; TE = ~TR   # e576's split
class Grok(nn.Module):
    def __init__(s):
        super().__init__(); r = lambda *sh, fan: nn.Parameter(torch.randn(*sh) / math.sqrt(fan))
        s.WE, s.WP = r(V, d, fan=d), r(3, d, fan=d); s.WQ, s.WK, s.WV = r(NH, d, HD, fan=d), r(NH, d, HD, fan=d), r(NH, d, HD, fan=d); s.WO = r(NH, HD, d, fan=HD)
        s.Win, s.bin = r(d, DM, fan=d), nn.Parameter(torch.zeros(DM)); s.Wout, s.bout = r(DM, d, fan=DM), nn.Parameter(torch.zeros(d)); s.WU = r(d, p, fan=d)
    def forward(s, ids):
        x = s.WE[ids] + s.WP[None]; q, k, v = (torch.einsum("btd,hde->bhte", x, W) for W in (s.WQ, s.WK, s.WV))
        att = (q @ k.transpose(-1, -2)) / math.sqrt(HD); att = att.masked_fill(torch.triu(torch.ones(3, 3, dtype=torch.bool, device=ids.device), 1), -1e9).softmax(-1)
        x = x + torch.einsum("bhte,hed->btd", att @ v, s.WO); h = F.relu(x @ s.Win + s.bin); x = x + h @ s.Wout + s.bout
        return x[:, -1] @ s.WU
model = Grok().to(DEV); Wb = torch.load("e576_base_rows.pt", map_location=DEV)["Wout"].float(); nb = Wb.norm(dim=1)
if cond != "unfrozen":
    with torch.no_grad():
        g = torch.Generator().manual_seed(9000 + seed)
        if cond == "trained": model.Wout.copy_(Wb)
        elif cond == "permuted": model.Wout.copy_(Wb[torch.randperm(DM, generator=g).to(DEV)])
        elif cond == "random": pass   # e576's random rows: the fresh initialization, randn / sqrt(512)
        else:
            Rd = torch.randn(DM, d, generator=g).to(DEV); Rd = Rd / Rd.norm(dim=1, keepdim=True)
            model.Wout.copy_(Rd * (nb[:, None] if cond == "random_normmatched" else nb.mean()))
    model.Wout.requires_grad_(False)
opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=lr, weight_decay=wd, betas=(0.9, 0.98)); curve = []
for step in range(STEPS + 1):
    if step % EVERY == 0:
        with torch.no_grad(): acc = (model(ALL).argmax(-1) == LAB).float(); curve.append((step, float(acc[TR].mean()), float(acc[TE].mean())))
    if step == STEPS: break
    for gr in opt.param_groups: gr["lr"] = lr * min(1.0, (step + 1) / 10)
    loss = F.cross_entropy(model(ALL[TR]), LAB[TR]); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
first = lambda thr: next((s for s, _, te in curve if te >= thr), None)
res = dict(cond=cond, seed=seed, steps=STEPS, wout_norm_mean=float(model.Wout.norm(dim=1).mean()), wb_norm_mean=float(nb.mean()), wb_norm_cv=float(nb.std() / nb.mean()), curve=curve,
           test_at={str(s): te for s, _, te in curve if s in (5000, 10000, 15000, 20000)}, first_test_05=first(0.5), first_test_09=first(0.9), final_train=curve[-1][1], final_test=curve[-1][2], seconds=time.time() - t0)
os.makedirs("results/e669", exist_ok=True); json.dump(res, open(f"results/e669/{cond}_s{seed}{'_smoke' if SMOKE else ''}.json", "w"))
print(f"{cond} seed {seed}: test {res['test_at']}, first 0.5 at {res['first_test_05']}, 0.9 at {res['first_test_09']}, final {res['final_train']:.2f}/{res['final_test']:.2f} ({res['seconds']:.0f}s)", flush=True)
