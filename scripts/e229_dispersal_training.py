"""e229: is dispersal architectural or learned? Pythia-410m at a training revision (argv[1]; 'final' = trained):
birth block 2, the two largest writes a and c per token, two random directions of the same size. At levels 4 and
6: the total and along-direction footprint of a (relative), its Mahalanobis footprint (whitened by that revision's
own state covariance at the level), the footprint cosine between a and c (real-real) vs the two random directions,
and the footprint cloud's effective dimension for real writes vs random directions. Split: dispersal present at
initialisation (architectural) vs appearing with training (learned), and whether the no-collision structure is
learned. Downloaded snapshots are deleted afterwards."""
import sys, os, glob, shutil; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
rev = sys.argv[1]; c = Cache("pythia410"); hub = os.path.join(os.environ.get("HF_HOME", "/workspace/.hf_home"), "hub", "models--EleutherAI--pythia-410m"); before = set(glob.glob(os.path.join(hub, "snapshots", "*")))
model, tok, fam = load_model(c.name, revision=None if rev == "final" else rev); arch = Arch(model, fam); NS = 8; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; NB = arch.NB; D = arch.D; b = 2; W = arch.wdir(b).to(DEV); WN = W.norm(dim=1); R = W / WN[:, None].clamp_min(1e-9)
def run(neuron=None, inject=None):
    st = {}; hs = [arch.layers[j].register_forward_hook((lambda j_: lambda m, i, o: st.__setitem__(j_, (o[0] if isinstance(o, tuple) else o).detach().float().reshape(-1, D)))(j)) for j in range(NB)]
    if neuron is not None:
        def pre(m, inp):
            x = inp[0].clone(); flat = x.reshape(-1, arch.DFF); flat[torch.arange(NT, device=DEV), neuron] = 0; return (flat.reshape(x.shape),)
        hs.append(arch.mlp_lin(b).register_forward_pre_hook(pre))
    if inject is not None:
        def pre2(m, args, kwargs):
            if len(args) > 0: return (args[0] + inject.to(args[0].dtype).reshape(args[0].shape),) + tuple(args[1:]), kwargs
            kwargs = dict(kwargs); kwargs["hidden_states"] = kwargs["hidden_states"] + inject.to(kwargs["hidden_states"].dtype).reshape(kwargs["hidden_states"].shape); return args, kwargs
        hs.append(arch.layers[b + 1].register_forward_pre_hook(pre2, with_kwargs=True))
    model(ids_seq); [h.remove() for h in hs]; return st
st_a = {}; h = arch.mlp_lin(b).register_forward_pre_hook(lambda m, inp: st_a.__setitem__("a", inp[0].detach().float().reshape(-1, arch.DFF))); model(ids_seq); h.remove()
led = st_a["a"] * WN[None]; top2 = led.abs().topk(2, dim=1).indices; na, nc = top2[:, 0], top2[:, 1]; ca = torch.gather(led, 1, na[:, None])[:, 0]; d = R[na]; big = ca.abs() >= ca.abs().quantile(0.5)
S0 = run(); Sa = run(neuron=na); Sc = run(neuron=nc); torch.manual_seed(0); U1 = torch.randn(NT, D, device=DEV); U1 = U1 / U1.norm(dim=1, keepdim=True); U2 = torch.randn(NT, D, device=DEV); U2 = U2 / U2.norm(dim=1, keepdim=True); S1 = run(inject=-ca.abs()[:, None] * U1); S2 = run(inject=-ca.abs()[:, None] * U2)
def cosv(x, y): return (x * y).sum(1) / (x.norm(dim=1) * y.norm(dim=1)).clamp_min(1e-9)
def pr(M):
    M = M - M.mean(0, keepdim=True); s = torch.linalg.svdvals(M); e = s ** 2; return ((e.sum() ** 2) / (e ** 2).sum()).item()
out = {}
for lv in (4, 6):
    typ = typical_mask(S0[lv]) & big; Fa = S0[lv] - Sa[lv]; Fc = S0[lv] - Sc[lv]; F1 = S0[lv] - S1[lv]; F2 = S0[lv] - S2[lv]; X = S0[lv][typical_mask(S0[lv])]; Xc = X - X.mean(0); Sig = Xc.T @ Xc / len(Xc); Si = torch.linalg.inv(Sig + 1e-3 * Sig.diagonal().mean() * torch.eye(D, device=DEV))
    wv = ca[:, None] * d; mh = ((Fa @ Si * Fa).sum(1) / (wv @ Si * wv).sum(1).clamp_min(1e-9)).sqrt()
    out[lv] = dict(total=(Fa.norm(dim=1) / ca.abs())[typ].median().item(), along=((Fa * d).sum(1) / ca)[typ].median().item(), mahalanobis=mh[typ].median().item(), cos_real_real=cosv(Fa, Fc)[typ].median().item(), cos_random_random=cosv(F1, F2)[typ].median().item(), dim_real=pr(Fa[typ]), dim_random=pr(F1[typ]), dim_state=pr(S0[lv][typ]))
    log(f"{rev} level {lv}: footprint of write a: total {out[lv]['total']:.2f}, along {out[lv]['along']:.2f}, Mahalanobis {out[lv]['mahalanobis']:.2f} | footprint cosine real-real {out[lv]['cos_real_real']:+.2f} vs random-random {out[lv]['cos_random_random']:+.2f} | cloud dimension real {out[lv]['dim_real']:.0f} vs random {out[lv]['dim_random']:.0f} (state {out[lv]['dim_state']:.0f})")
record(f"e229_dispersal_{rev}", dict(revision=rev, per_level={str(k): v for k, v in out.items()}), " | ".join(f"lv{lv}: total {v['total']:.2f} along {v['along']:.2f} Mahalanobis {v['mahalanobis']:.2f}; cos real-real {v['cos_real_real']:+.2f} random-random {v['cos_random_random']:+.2f}; dim real {v['dim_real']:.0f} random {v['dim_random']:.0f} state {v['dim_state']:.0f}" for lv, v in out.items()))
del model; torch.cuda.empty_cache()
for p in set(glob.glob(os.path.join(hub, "snapshots", "*"))) - before: shutil.rmtree(p, ignore_errors=True)
rf = os.path.join(hub, "refs", rev)
if rev != "final" and os.path.exists(rf) and rev not in ("step0", "step1000", "step4000", "step16000", "step32000", "step64000"): os.remove(rf)
