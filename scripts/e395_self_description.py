"""e395: self-description length. Every network carries its own vocabulary (its token embeddings, MLP write rows, head
output bases). How many of its own words does it need to describe its own state, and does learning create that
self-describability? For one Pythia-410m checkpoint (argument), at the middle depth, with the checkpoint's OWN vocabulary
and states: the fraction of variance unexplained and the loss recovered (state replaced by the reconstruction, as e388)
for k = 4 to 256 own atoms (OMP), the same with the vocabulary randomly rotated, and the same keeping the k largest
ACTUAL writes (every other component at its mean, as e391). Self-description length k90 = the k at which own-word
reconstruction recovers 90% of the loss; actual-write length = the k at which the largest actual writes do; their ratio
is the redundancy of the model's own writing."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p3_common import *
rev = sys.argv[1]; c = Cache("pythia410"); model, tok, fam = load_model("pythia410", revision=None if rev == "final" else rev); arch = Arch(model, fam); NB, NH, HD = arch.NB, arch.NH, arch.HD; L = NB // 2; torch.manual_seed(0)
ev = c.s["eval_ids"][:3].to(DEV); B, T = ev.shape; KS = [4, 8, 16, 32, 64, 128, 256]
acts, Z, X = {}, {}, {}
hs = [arch.mlp_lin(b).register_forward_pre_hook((lambda b_: lambda m, a: acts.__setitem__(b_, a[0].detach().float()))(b)) for b in range(L + 1)]
hs += [arch.attn_lin(b).register_forward_pre_hook((lambda b_: lambda m, a: Z.__setitem__(b_, a[0].detach().float()))(b)) for b in range(L + 1)]
def pre0(m, args, kwargs):
    h = args[0] if len(args) > 0 else kwargs["hidden_states"]; X["emb"] = h.detach().float(); return None
hs.append(arch.layers[0].register_forward_pre_hook(pre0, with_kwargs=True)); hs.append(arch.layers[L].register_forward_hook(lambda m, i, o: X.__setitem__("x", (o[0] if isinstance(o, tuple) else o).detach().float())))
with torch.no_grad(): lg = model(ev).logits.float()
[h.remove() for h in hs]; Lc = token_loss(lg, ev).mean().item()
sl = lambda t: t[:, 1:].reshape(-1, t.shape[-1]); x = sl(X["x"]); P, D = x.shape; mu = x.mean(0, keepdim=True); Xc = x - mu
def splice(Xh):
    def hk(m, i, o):
        xo = o[0] if isinstance(o, tuple) else o; y = xo.clone(); y[:, 1:] = Xh.to(xo.dtype).view(B, T - 1, -1)
        return (y,) + tuple(o[1:]) if isinstance(o, tuple) else y
    h = arch.layers[L].register_forward_hook(hk)
    try:
        with torch.no_grad(): return token_loss(model(ev).logits.float(), ev).mean().item()
    finally: h.remove()
Lm = splice(mu.expand(P, -1)); gap = Lm - Lc
A, lab = build_dictionary(arch, blocks=list(range(L + 1))); res = dict(rev=rev, level=L, clean=Lc, mean_state=Lm, gap=gap, n_atoms=int(A.shape[0]), own={}, rotated={}, actual={})
for nm, Dct in (("own", A), ("rotated", rotate(A))):
    sel, _, _ = omp(Xc, Dct, max(KS), batch=256, record_err=False)
    for k in KS:
        cof, err = refit(Xc, Dct, sel[:, :k]); Ls = splice(mu + torch.einsum("nk,nkd->nd", cof, Dct[sel[:, :k]])); res[nm][str(k)] = dict(fvu=(err.sum() / Xc.pow(2).sum()).item(), recovered=(Lm - Ls) / gap if abs(gap) > 1e-6 else float("nan"))
    del sel
W = torch.cat([arch.wdir(b).to(DEV) for b in range(L + 1)], 0); Aact = torch.cat([sl(acts[b]) for b in range(L + 1)], 1); Abar = Aact.mean(0, keepdim=True); wn = W.norm(dim=-1)
H = torch.cat([(sl(Z[b]).view(P, NH, HD)[:, :, None, :] @ arch.wo(b).to(DEV).view(NH, HD, -1)[None]).squeeze(2) for b in range(L + 1)], 1); Hbar = H.mean(0, keepdim=True); E = sl(X["emb"]); Ebar = E.mean(0, keepdim=True)
base = x - ((Aact - Abar) @ W) - (H - Hbar).sum(1) - (E - Ebar); N, G = Aact.shape[1], H.shape[1]; Lb = splice(base)
score = torch.cat([(Aact - Abar).abs() * wn[None], (H - Hbar).norm(dim=-1), (E - Ebar).norm(dim=-1, keepdim=True)], 1)
for k in KS + [1024, 4096]:
    idx = score.topk(k, dim=1).indices; keep = torch.zeros(P, N + G + 1, device=DEV); keep.scatter_(1, idx, 1.0)
    Xh = base + (((Aact - Abar) * keep[:, :N]) @ W) + ((H - Hbar) * keep[:, N:N + G, None]).sum(1) + (E - Ebar) * keep[:, N + G:]
    Ls = splice(Xh); res["actual"][str(k)] = dict(recovered=(Lb - Ls) / (Lb - Lc) if abs(Lb - Lc) > 1e-6 else float("nan"))
def k_at(curve, thr=0.9):
    ks = sorted(int(k) for k in curve); prev = None
    for k in ks:
        v = curve[str(k)]["recovered"]
        if v == v and v >= thr: return k
    return None
res["self_description_k90"] = k_at(res["own"]); res["rotated_k90"] = k_at(res["rotated"]); res["actual_write_k90"] = k_at(res["actual"]); res["writes_at_mean"] = Lb
log(f"pythia410 {rev} L{L}: clean {Lc:.3f}, mean state {Lm:.3f} (gap {gap:.3f}) | own words: " + " ".join(f"k{k} {v['recovered']:.2f}/{v['fvu']:.2f}" for k, v in res["own"].items()) + " | rotated: " + " ".join(f"k{k} {v['recovered']:.2f}/{v['fvu']:.2f}" for k, v in res["rotated"].items()) + " | largest actual writes: " + " ".join(f"k{k} {v['recovered']:.2f}" for k, v in res["actual"].items()) + f" || k90: own {res['self_description_k90']}, rotated {res['rotated_k90']}, actual writes {res['actual_write_k90']}")
record(f"e395_selfdesc_pythia410_{rev}", res, f"k90 own {res['self_description_k90']} rotated {res['rotated_k90']} actual {res['actual_write_k90']} | gap {gap:.2f}")
