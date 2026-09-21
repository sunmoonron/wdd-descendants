"""e226: are descendants additive? For each token the two largest block-2 writes (distinct neurons a, c): footprints
F_a, F_c (zero one), F_ac (zero both), F_half (halve a). At levels 3, 4, 6, L: eps_add = ||F_ac - F_a - F_c|| /
||F_a + F_c|| and eps_hom = ||F_half - F_a/2|| / ||F_a/2||. Split: linear transport (eps < 0.2) vs nonlinear
mixing. Also eps_add for pairs whose write directions reinforce (cos > 0.1) vs oppose (cos < -0.1)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 8; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; NB = c.NB; lab = c.d["lab"]; A = c.d["A"]; b = 2
def rows(bb): return A[(lab["type"] == T_MLP) & (lab["block"] == bb)].float().to(DEV)
def run(scale=None):
    st = {}; hs = [arch.layers[j].register_forward_hook((lambda j_: lambda m, i, o: st.__setitem__(j_, (o[0] if isinstance(o, tuple) else o).detach().float().reshape(-1, c.D)))(j)) for j in range(NB)]
    if scale is not None:
        def pre(m, inp):
            x = inp[0].clone(); flat = x.reshape(-1, c.DFF)
            for neuron, s in scale: flat[torch.arange(NT, device=DEV), neuron] *= s
            return (flat.reshape(x.shape),)
        hs.append(arch.mlp_lin(b).register_forward_pre_hook(pre))
    model(ids_seq); [h.remove() for h in hs]; return st
st_a = {}; h = arch.mlp_lin(b).register_forward_pre_hook(lambda m, inp: st_a.__setitem__("a", inp[0].detach().float().reshape(-1, c.DFF))); model(ids_seq); h.remove()
led = st_a["a"] * c.d["WN"][b].to(DEV)[None]; top2 = led.abs().topk(2, dim=1).indices; na, nc = top2[:, 0], top2[:, 1]; coefs = torch.gather(led, 1, top2); big = coefs[:, 0].abs() >= coefs[:, 0].abs().quantile(0.5); R = rows(b); cosd = ((R[na] * R[nc]).sum(1)) * torch.sign(coefs[:, 0] * coefs[:, 1])
S0 = run(); Sa = run([(na, 0.0)]); Sc = run([(nc, 0.0)]); Sac = run([(na, 0.0), (nc, 0.0)]); Sh = run([(na, 0.5)]); out = {}
for lv in sorted({b + 1, b + 2, b + 4, L}):
    if lv >= NB: continue
    typ = typical_mask(S0[lv]) & big; Fa = S0[lv] - Sa[lv]; Fc = S0[lv] - Sc[lv]; Fac = S0[lv] - Sac[lv]; Fh = S0[lv] - Sh[lv]
    e_add = (Fac - Fa - Fc).norm(dim=1) / (Fa + Fc).norm(dim=1).clamp_min(1e-6); e_hom = (Fh - 0.5 * Fa).norm(dim=1) / (0.5 * Fa).norm(dim=1).clamp_min(1e-6)
    out[lv] = dict(eps_add=e_add[typ].median().item(), eps_hom=e_hom[typ].median().item(), eps_add_reinforcing=e_add[typ & (cosd > 0.1)].median().item() if (typ & (cosd > 0.1)).sum() > 20 else float("nan"), eps_add_opposing=e_add[typ & (cosd < -0.1)].median().item() if (typ & (cosd < -0.1)).sum() > 20 else float("nan"), n_reinf=int((typ & (cosd > 0.1)).sum()), n_opp=int((typ & (cosd < -0.1)).sum()))
    log(f"{tag} level {lv}: additivity error {out[lv]['eps_add']:.2f}, homogeneity error {out[lv]['eps_hom']:.2f} | reinforcing pairs {out[lv]['eps_add_reinforcing']:.2f} (n {out[lv]['n_reinf']}), opposing pairs {out[lv]['eps_add_opposing']:.2f} (n {out[lv]['n_opp']})")
record(f"e226_additivity_{tag}", dict(model=tag, b=b, L=L, per_level={str(k): v for k, v in out.items()}), "additivity error by level " + " ".join(f"{lv}:{v['eps_add']:.2f}" for lv, v in out.items()) + " | homogeneity error " + " ".join(f"{lv}:{v['eps_hom']:.2f}" for lv, v in out.items()) + " | reinforcing vs opposing pairs at L: " + f"{out[max(out)]['eps_add_reinforcing']:.2f} vs {out[max(out)]['eps_add_opposing']:.2f}")
