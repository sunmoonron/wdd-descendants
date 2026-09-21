"""e351: attention-only and MLP-only transport of the quotient. A random family of 30 directions injected at the
block-3 input; the images at L read with (i) the normal network, (ii) the attention sublayers of blocks 3..L
frozen at their unperturbed outputs at every token (transport through MLPs only), (iii) the MLP sublayers frozen
(transport through attention only). For each: the function quotient's dimension and own score, the gain, the
overlap of the quotient with the normal one, and the cosine of the family's image centroids with the normal ones."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quot_common import *
tag = sys.argv[1]; S = Setup(tag, levels=[0]); L = S.L; b = S.b; S = Setup(tag, levels=[L]); Kf = 30; d = 16; torch.manual_seed(0); V = unit(torch.randn(Kf, S.D, device=DEV)); pool = S.pool; mods = {("attn", lv): S.arch.attn_lin(lv) for lv in range(b + 1, L + 1)}; mods.update({("mlp", lv): S.arch.mlp_lin(lv) for lv in range(b + 1, L + 1)}); cache = {}
hs = [m.register_forward_hook((lambda key: lambda mod, i, o: cache.__setitem__(key, o.detach().clone()))(key)) for key, m in mods.items()]; S.model(S.ids_seq); [h.remove() for h in hs]
def run_frozen(keys, inj):
    def mk(key):
        def hook(mod, i, o): return cache[key]
        return hook
    hs = [mods[key].register_forward_hook(mk(key)) for key in keys]; r = S.run(positions=pool, inject=inj, inject_block=b + 1); [h.remove() for h in hs]; return r
a = torch.randint(0, Kf, (len(pool),), device=DEV); inj = torch.zeros(S.NT, S.D, device=DEV); inj[pool] = S.s_inj * V[a]; base = S.run(positions=pool); out = {}; Qn = None; Cn = None
for nm, keys in (("normal", []), ("attention_frozen", [k for k in mods if k[0] == "attn"]), ("mlp_frozen", [k for k in mods if k[0] == "mlp"]), ("both_frozen", list(mods))):
    r = run_frozen(keys, inj) if keys else S.run(positions=pool, inject=inj, inject_block=b + 1); F = (r[L] - S.S0[L])[pool]; dl = r["lg"] - base["lg"]; dl = dl - dl.mean(1, keepdim=True); tr, te = S.halves(len(pool), seed=2); Fc = F - F[tr].mean(0, keepdim=True); Z, _ = logit_scores(dl, tr); dln = unit(dl); Q = pls(Fc[tr], Z[tr], 64); cv = {q: knn_cos(Fc @ Q[:, :q], dln, tr, te) for q in (1, 2, 4, 8, 16, 32, 64)}; cent = torch.stack([F[a == k].mean(0) for k in range(Kf)])
    if Qn is None: Qn, Cn = Q[:, :d], cent
    out[nm] = dict(dim=next((q for q in cv if cv[q] >= 0.9 * cv[64]), 64), own=cv[16], gain=(F.norm(dim=1) / S.s_inj).median().item(), overlap_with_normal=inside(Q[:, :d], Qn), centroid_cos_with_normal=((unit(cent) * unit(Cn)).sum(1)).median().item(), identity=accuracy((Fc @ Q[:, :d])[te], centroids((Fc @ Q[:, :d])[tr], a[tr], Kf), a[te]), logit_response=dl.norm(dim=1).median().item())
log(f"{tag}: transport variant -> function dim | own-16 | identity | gain | logit response | quotient overlap with normal | image cos with normal :: " + " ; ".join(f"{nm}: {v['dim']} | {v['own']:.2f} | {v['identity']:.2f} | {v['gain']:.2f} | {v['logit_response']:.1f} | {v['overlap_with_normal']:.2f} | {v['centroid_cos_with_normal']:.2f}" for nm, v in out.items()))
record(f"e351_comptransport_{tag}", dict(model=tag, L=L, per_variant=out), " ; ".join(f"{nm}: dim {v['dim']} own {v['own']:.2f} gain {v['gain']:.2f} ov {v['overlap_with_normal']:.2f} cos {v['centroid_cos_with_normal']:.2f}" for nm, v in out.items()))
