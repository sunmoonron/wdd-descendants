"""e391: does a residual state need only its few largest actual writes? At the middle depth the state is an exact sum of
components: the embedding, every head's write and every MLP neuron's write (activation times its write row) in the
blocks so far, plus biases. Keep, at each position, the k components that deviate most from their own mean (over
positions) and replace every other component by its mean; splice and run on (loss recovered as in e388, the reference
being every component at its mean). Variants: largest deviations over all components; over MLP neurons only (heads and
embedding at their means); k random components. Compared with WDD's OMP reconstruction from the state alone (e388)."""
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
allscore = torch.cat([dev_n, dev_h, dev_e], 1); g = torch.Generator(device=DEV).manual_seed(1)
for nm in ("largest_all", "largest_neurons", "random_all"):
    row = {}
    for k in KS:
        if nm == "largest_all": idx = allscore.topk(k, dim=1).indices
        elif nm == "largest_neurons": idx = dev_n.topk(k, dim=1).indices
        else: idx = torch.rand(P, N + G + 1, generator=g, device=DEV).topk(k, dim=1).indices
        keep = torch.zeros(P, N + G + 1, device=DEV); keep.scatter_(1, idx, 1.0)
        kn, kh, ke = keep[:, :N], keep[:, N:N + G], keep[:, N + G:]
        if nm == "largest_neurons": kh, ke = torch.zeros_like(kh), torch.zeros_like(ke)
        Ls = splice(recon(kn, kh, ke)); row[str(k)] = dict(recovered=(Lm - Ls) / max(Lm - Lc, 1e-9), loss=Ls, share_kept_from_heads=kh.sum().item() / max(keep.sum().item(), 1), share_kept_from_emb=ke.sum().item() / max(keep.sum().item(), 1))
    res["variants"][nm] = row
ef = os.path.join(RESULTS, f"e388_wddsae_{tag}.json")
if os.path.exists(ef):
    e = json.load(open(ef)); e = e.get("result", e); lv = e["levels"].get(str(L))
    if lv: res["wdd_e388"] = {k.split("_")[1]: v["recovered"] for k, v in lv.items() if k.startswith("wdd_")}
log(f"{tag} L{L} (clean {Lc:.3f}, every component at its mean {Lm:.3f}; {N} neuron writes, {G} head writes): loss recovered by keeping the k largest actual deviations: " + " | ".join(f"{nm}: " + " ".join(f"k{k} {v['recovered']:.2f}" for k, v in row.items()) for nm, row in res["variants"].items()) + (" | WDD OMP from the state (e388): " + " ".join(f"k{k} {v:.2f}" for k, v in res["wdd_e388"].items()) if "wdd_e388" in res else ""))
record(f"e391_truewrites_{tag}", res, " | ".join(f"{nm}: k8 {row['8']['recovered']:.2f} k32 {row['32']['recovered']:.2f} k256 {row['256']['recovered']:.2f}" for nm, row in res["variants"].items()))
