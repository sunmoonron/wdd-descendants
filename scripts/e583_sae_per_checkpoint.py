"""e583 (session 106): a small TopK sparse autoencoder trained on every checkpoint's block-12 states (4096 latents, 32
active, 12k centred and scaled states, 4000 Adam steps at batch 2048): the per-checkpoint learned dictionary that has
no parameter identity across checkpoints, the control for e584. Saved with its reconstruction FVU and the number of
latents alive (active at ten positions or more)."""
from s101_common import *
t0 = time.time(); CD = "/workspace/wdd/cache/e582_pythia410"; F_, TOPK, STEPS, BS = 4096, 32, 4000, 2048; res = {}
class TopKSAE(torch.nn.Module):
    def __init__(s, D):
        super().__init__(); s.We = torch.nn.Parameter(torch.randn(F_, D) / D ** 0.5); s.be = torch.nn.Parameter(torch.zeros(F_)); s.Wd = torch.nn.Parameter(s.We.detach().clone()); s.bd = torch.nn.Parameter(torch.zeros(D))
    def encode(s, x):
        z = (x - s.bd) @ s.We.T + s.be; top = z.topk(TOPK, dim=1); out = torch.zeros_like(z); out.scatter_(1, top.indices, torch.relu(top.values)); return out
    def forward(s, x): z = s.encode(x); return z @ s.Wd + s.bd, z
for n in range(1000, 16001, 1000):
    out = f"{CD}/sae_step{n}.pt"
    if os.path.exists(out): res[n] = torch.load(out, map_location="cpu")["meta"]; continue
    c = torch.load(f"{CD}/step{n}.pt", map_location="cpu"); X = c["X"][c["keep"]].float().to(DEV); mu = X.mean(0); Xc = X - mu; scale = float(Xc.norm(dim=1).mean()); Xs = Xc / scale; N = Xs.shape[0]
    torch.manual_seed(n); sae = TopKSAE(Xs.shape[1]).to(DEV); opt = torch.optim.Adam(sae.parameters(), lr=1e-3); g = torch.Generator(device=DEV).manual_seed(n)
    with torch.enable_grad():
        for step in range(STEPS):
            idx = torch.randint(0, N, (BS,), device=DEV, generator=g); xb = Xs[idx]; xh, z = sae(xb); loss = (xh - xb).pow(2).sum(1).mean(); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            with torch.no_grad(): sae.Wd.data = sae.Wd.data / sae.Wd.data.norm(dim=1, keepdim=True).clamp_min(1e-6)
    with torch.no_grad(): xh, z = sae(Xs); fvu = float((xh - Xs).pow(2).sum(1).mean() / Xs.pow(2).sum(1).mean()); alive = int(((z > 0).sum(0) >= 10).sum())
    meta = dict(step=n, fvu=fvu, alive=alive, F=F_, k=TOPK, scale=scale); res[n] = meta
    torch.save(dict(We=sae.We.detach().cpu(), be=sae.be.detach().cpu(), Wd=sae.Wd.detach().cpu(), bd=sae.bd.detach().cpu(), mu=mu.cpu(), scale=scale, meta=meta), out); log(f"step {n}: SAE fvu {fvu:.3f}, alive {alive} of {F_} | {time.time() - t0:.0f}s"); del sae, X, Xs; torch.cuda.empty_cache()
summ = "per-checkpoint TopK autoencoders (4096 latents, 32 active): fvu " + ", ".join(f"{n}: {m['fvu']:.2f} ({m['alive']} alive)" for n, m in sorted(res.items())) + f" | {time.time() - t0:.0f}s"; log(summ); record("e583_sae_per_checkpoint", {str(k): v for k, v in res.items()}, summ)
