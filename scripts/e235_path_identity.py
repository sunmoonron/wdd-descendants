"""e235: which path carries the identity? Block b = 2, level L. Natural footprints F (full), F_M (later attention
frozen: MLP-only transport) and F_A (later MLPs frozen: attention-only transport) for the same ablations.
Centroids from full natural footprints (train half); held-out accuracy for F, F_M and F_A; plus self-consistent
accuracy (centroids from F_M classify F_M, from F_A classify F_A). Split: identity carried by the MLP chain, by
attention, or only by their combination."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from desc_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; NB = c.NB; b = 2
def mlp_module(j):
    lin = arch.mlp_lin(j)
    for name, mod in arch.layers[j].named_modules():
        if any(ch is lin for ch in mod.children()): return mod
def run(neuron=None, freeze=None, clean=None):
    st = {"att": {}, "mlp": {}}; hs = [arch.layers[L].register_forward_hook(lambda m, i, o: st.__setitem__("H", (o[0] if isinstance(o, tuple) else o).detach().float().reshape(-1, c.D)))]
    for j in range(b + 1, NB):
        hs.append(arch.attn_lin(j).register_forward_hook((lambda j_: lambda m, i, o: st["att"].__setitem__(j_, o.detach().clone()))(j))); hs.append(mlp_module(j).register_forward_hook((lambda j_: lambda m, i, o: st["mlp"].__setitem__(j_, o.detach().clone()))(j)))
    if neuron is not None:
        def pre(m, inp):
            x = inp[0].clone(); flat = x.reshape(-1, c.DFF); flat[torch.arange(NT, device=DEV), neuron] = 0; return (flat.reshape(x.shape),)
        hs.append(arch.mlp_lin(b).register_forward_pre_hook(pre))
    if freeze == "att":
        for j in range(b + 1, NB): hs.append(arch.attn_lin(j).register_forward_hook((lambda j_: lambda m, i, o: clean["att"][j_])(j)))
    if freeze == "mlp":
        for j in range(b + 1, NB): hs.append(mlp_module(j).register_forward_hook((lambda j_: lambda m, i, o: clean["mlp"][j_])(j)))
    model(ids_seq); [h.remove() for h in hs]; return st
led, tn, tc = dominant(model, arch, c, ids_seq, b); S0 = run(); S1 = run(tn); SM = run(tn, "att", S0); SA = run(tn, "mlp", S0); typ = typical_mask(S0["H"]); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep)
torch.manual_seed(0); split = torch.rand(len(idx), device=DEV) < 0.5; tr, te = idx[split], idx[~split]; ltr, lte = lab_i[split], lab_i[~split]; F = S0["H"] - S1["H"]; FM = S0["H"] - SM["H"]; FA = S0["H"] - SA["H"]; cents = centroids(F[tr], ltr, K)
res = dict(K=K, chance=1.0 / K, full=accuracy(F[te], cents, lte), mlp_only_vs_full_centroids=accuracy(FM[te], cents, lte), att_only_vs_full_centroids=accuracy(FA[te], cents, lte), mlp_only_self=accuracy(FM[te], centroids(FM[tr], ltr, K), lte), att_only_self=accuracy(FA[te], centroids(FA[tr], ltr, K), lte), cos_full_mlp=((unit(F[te]) * unit(FM[te])).sum(1)).median().item(), cos_full_att=((unit(F[te]) * unit(FA[te])).sum(1)).median().item())
log(f"{tag} (K {K}, chance {1 / K:.2f}): full {res['full']:.2f} | MLP-only transport: vs full centroids {res['mlp_only_vs_full_centroids']:.2f}, self {res['mlp_only_self']:.2f} | attention-only transport: vs full centroids {res['att_only_vs_full_centroids']:.2f}, self {res['att_only_self']:.2f} | cos(full, MLP-only) {res['cos_full_mlp']:.2f}, cos(full, attention-only) {res['cos_full_att']:.2f}")
record(f"e235_pathid_{tag}", dict(model=tag, b=b, L=L, **res), f"K {K} (chance {res['chance']:.2f}): full {res['full']:.2f}; MLP-only transport {res['mlp_only_vs_full_centroids']:.2f} (self {res['mlp_only_self']:.2f}); attention-only {res['att_only_vs_full_centroids']:.2f} (self {res['att_only_self']:.2f})")
