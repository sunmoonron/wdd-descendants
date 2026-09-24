"""e393: why does WDD's sparse code beat the model's own largest writes (e391)? Support or coefficients. At the middle
depth, for k = 8, 16, 32, 64, 128: (a) the k largest actual component deviations kept with their actual coefficients and
the rest at their means (e391); (b) the same k components' directions, but coefficients refitted by least squares to the
whole state (so the small writes left out can be absorbed along the kept directions); (c) WDD's OMP support over the
model's own atoms, refitted (e388). Also: the share of the dropped tail's norm that lies in the span of the kept
directions, and how many of WDD's selected MLP atoms are among the k largest actual neuron writes."""

import sys, os, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p3_common import *
tag = sys.argv[1]; c = Cache(tag); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NB, NH, HD = arch.NB, arch.NH, arch.HD; L = NB // 2; torch.manual_seed(0)
ev = c.s["eval_ids"][:3].to(DEV); B, T = ev.shape; KS = [8, 16, 32, 64, 128, 256, 1024]
acts, Z, X = {}, {}, {}
hs = [arch.mlp_lin(b).register_forward_pre_hook((lambda b_: lambda m, a: acts.__setitem__(b_, a[0].detach().float()))(b)) for b in range(L + 1)]
hs += [arch.attn_lin(b).register_forward_pre_hook((lambda b_: lambda m, a: Z.__setitem__(b_, a[0].detach().float()))(b)) for b in range(L + 1)]
def pre0(m, args, kwargs):
    h = args[0] if len(args) > 0 else kwargs["hidden_states"]; X["emb"] = h.detach().float(); return None
hs.append(arch.layers[0].register_forward_pre_hook(pre0, with_kwargs=True)); hs.append(arch.layers[L].register_forward_hook(lambda m, i, o: X.__setitem__("x", (o[0] if isinstance(o, tuple) else o).detach().float())))
with torch.no_grad(): lg = model(ev).logits.float()
[h.remove() for h in hs]; Lc = token_loss(lg, ev).mean().item()
sl = lambda t: t[:, 1:].reshape(-1, t.shape[-1]); x = sl(X["x"]); P, D = x.shape
W = {b: arch.wdir(b).to(DEV) for b in range(L + 1)}; Wo = {b: arch.wo(b).to(DEV) for b in range(L + 1)}
A = torch.cat([sl(acts[b]) for b in range(L + 1)], 1); Abar = A.mean(0, keepdim=True); Wall = torch.cat([W[b] for b in range(L + 1)], 0); wn = Wall.norm(dim=-1)   # neurons [P, N], rows [N, D]
H = torch.cat([(sl(Z[b]).view(P, NH, HD)[:, :, None, :] @ Wo[b].view(NH, HD, -1)[None]).squeeze(2) for b in range(L + 1)], 1); Hbar = H.mean(0, keepdim=True)    # heads [P, G, D]
E = sl(X["emb"]); Ebar = E.mean(0, keepdim=True)
dev_n = (A - Abar).abs() * wn[None]; dev_h = (H - Hbar).norm(dim=-1); dev_e = (E - Ebar).norm(dim=-1, keepdim=True)
base = x - ((A - Abar) @ Wall) - (H - Hbar).sum(1) - (E - Ebar)            # the state with every component at its mean (plus biases)
N, G = A.shape[1], H.shape[1]
def splice(Xh):
    def hk(m, i, o):
        xo = o[0] if isinstance(o, tuple) else o; y = xo.clone(); y[:, 1:] = Xh.to(xo.dtype).view(B, T - 1, -1)
        return (y,) + tuple(o[1:]) if isinstance(o, tuple) else y
    h = arch.layers[L].register_forward_hook(hk)
    try:
        with torch.no_grad(): return token_loss(model(ev).logits.float(), ev).mean().item()
    finally: h.remove()
Lm = splice(base); res = dict(model=tag, level=L, clean=Lc, all_at_mean=Lm, n_neurons=N, n_heads=G, variants={})
def recon(keep_n, keep_h, keep_e):
    """keep masks [P,N], [P,G], [P,1] -> reconstructed state"""
    return base + (((A - Abar) * keep_n) @ Wall) + ((H - Hbar) * keep_h[:, :, None]).sum(1) + (E - Ebar) * keep_e
allscore = torch.cat([dev_n, dev_h, dev_e], 1); U_n = Wall / wn[:, None].clamp_min(1e-9)
res = dict(model=tag, level=L, clean=Lc, all_at_mean=Lm, variants={"actual": {}, "refit": {}}, tail_in_span={}, wdd_overlap={})
target = x - base
Adict, lab = c.dictionary(L); mu = c.s["mu"][L + 1].to(DEV).float(); selW, _, _ = omp(x - mu[None], Adict, 128, batch=256, record_err=False)
typ, blk, idxl = lab["type"].to(DEV), lab["block"].to(DEV), lab["index"].to(DEV); offs = torch.tensor([0] + [W[b].shape[0] for b in range(L)], device=DEV).cumsum(0)
for k in (8, 16, 32, 64, 128):
    top = allscore.topk(k, dim=1).indices; keep = torch.zeros(P, N + G + 1, device=DEV); keep.scatter_(1, top, 1.0); kn, kh, ke = keep[:, :N], keep[:, N:N + G], keep[:, N + G:]
    Xa = recon(kn, kh, ke); La = splice(Xa)
    # component deviation vectors for the kept set, per position: [P, k, D]
    isn = top < N; ish = (top >= N) & (top < N + G); ise = top == N + G
    vec = torch.zeros(P, k, D, device=DEV)
    vec[isn] = U_n[top[isn]]
    hi = (top - N).clamp(0, G - 1); hv = (H - Hbar)[torch.arange(P, device=DEV)[:, None].expand(-1, k), hi]; hv = hv / hv.norm(dim=-1, keepdim=True).clamp_min(1e-9); vec[ish] = hv[ish]
    ev_ = ((E - Ebar) / (E - Ebar).norm(dim=-1, keepdim=True).clamp_min(1e-9))[:, None, :].expand(-1, k, -1); vec[ise] = ev_[ise]
    Gm = vec @ vec.transpose(1, 2) + 1e-5 * torch.eye(k, device=DEV); beta = torch.cholesky_solve(vec @ target[:, :, None], torch.linalg.cholesky(Gm))[:, :, 0]; Xr = base + torch.einsum("pk,pkd->pd", beta, vec); Lr = splice(Xr)
    tail = x - Xa; proj = torch.einsum("pk,pkd->pd", torch.cholesky_solve(vec @ tail[:, :, None], torch.linalg.cholesky(Gm))[:, :, 0], vec)
    res["tail_in_span"][str(k)] = (proj.norm(dim=-1) / tail.norm(dim=-1).clamp_min(1e-9)).median().item()
    res["variants"]["actual"][str(k)] = (Lm - La) / max(Lm - Lc, 1e-9); res["variants"]["refit"][str(k)] = (Lm - Lr) / max(Lm - Lc, 1e-9)
    s_ = selW[:, :k]; tsel, bsel, isel = typ[s_], blk[s_], idxl[s_]; mlp_m = tsel == T_MLP; flat = torch.where(mlp_m, offs[bsel.clamp(0, L)] + isel, torch.full_like(isel, -1))
    topn = dev_n.topk(k, dim=1).indices; hit = (flat[:, :, None] == topn[:, None, :]).any(-1) & mlp_m
    res["wdd_overlap"][str(k)] = dict(share_mlp_atoms=mlp_m.float().mean().item(), share_of_mlp_atoms_in_true_topk=(hit.sum() / mlp_m.sum().clamp_min(1)).item())
ef = os.path.join(RESULTS, f"e388_wddsae_{tag}.json")
if os.path.exists(ef):
    e = json.load(open(ef)); e = e.get("result", e); lv = e["levels"].get(str(L))
    if lv: res["wdd_e388"] = {k.split("_")[1]: v["recovered"] for k, v in lv.items() if k.startswith("wdd_")}
log(f"{tag} L{L}: loss recovered by k: actual top-k writes " + " ".join(f"k{k} {v:.2f}" for k, v in res["variants"]["actual"].items()) + " | same directions refitted " + " ".join(f"k{k} {v:.2f}" for k, v in res["variants"]["refit"].items()) + (" | WDD (e388) " + " ".join(f"k{k} {v:.2f}" for k, v in res["wdd_e388"].items()) if "wdd_e388" in res else "") + " | dropped tail inside the kept span (median) " + " ".join(f"k{k} {v:.2f}" for k, v in res["tail_in_span"].items()) + " | WDD's MLP atoms among the true top-k neuron writes " + " ".join(f"k{k} {v['share_of_mlp_atoms_in_true_topk']:.2f} (of {v['share_mlp_atoms']:.2f} MLP)" for k, v in res["wdd_overlap"].items()))
record(f"e393_supportcoef_{tag}", res, "actual " + " ".join(f"k{k} {v:.2f}" for k, v in res["variants"]["actual"].items()) + " | refit " + " ".join(f"k{k} {v:.2f}" for k, v in res["variants"]["refit"].items()))
