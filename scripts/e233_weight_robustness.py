"""e233: is the descendant signature robust to weight perturbation? Centroids from the unperturbed model's natural
footprints (train half, block b = 2, level L). Then every parameter is perturbed by eps x its RMS x Gaussian noise
(eps = 1e-3, 1e-2, 3e-2, 1e-1), footprints are regenerated for the same neurons and classified against the
UNPERTURBED centroids; the clean state's relative change and the loss change calibrate the perturbation.
Split: robust dynamical structure (identity survives until the model itself degrades) vs brittle coincidence."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from desc_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; b = 2; run = make_runner(model, arch, c, ids_seq, [L], NT)
led, tn, tc = dominant(model, arch, c, ids_seq, b); S0 = run(); S1 = run(b, tn); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep)
torch.manual_seed(0); split = torch.rand(len(idx), device=DEV) < 0.5; tr, te = idx[split], idx[~split]; ltr, lte = lab_i[split], lab_i[~split]; F0 = S0[L] - S1[L]; cents = centroids(F0[tr], ltr, K)
def ce():
    out = model(ids_seq); lg = out.logits[:, :-1].reshape(-1, out.logits.shape[-1]).float(); return torch.nn.functional.cross_entropy(lg, ids_seq[:, 1:].reshape(-1)).item()
ce0 = ce(); params = [p for p in model.parameters()]; orig = [p.detach().clone() for p in params]; out = {0.0: dict(accuracy=accuracy(F0[te], cents, lte), state_change=0.0, dce=0.0)}
for eps in (1e-3, 1e-2, 3e-2, 1e-1):
    torch.manual_seed(1)
    with torch.no_grad():
        for p, o in zip(params, orig): p.copy_(o + eps * o.pow(2).mean().sqrt() * torch.randn_like(o))
    Sp0 = run(); Sp1 = run(b, tn); Fp = Sp0[L] - Sp1[L]; out[eps] = dict(accuracy=accuracy(Fp[te], cents, lte), state_change=((Sp0[L] - S0[L]).norm(dim=1) / S0[L].norm(dim=1))[typ].median().item(), dce=ce() - ce0, footprint_cos_to_unperturbed=((unit(Fp[te]) * unit(F0[te])).sum(1)).median().item())
    log(f"{tag} eps {eps:g}: source identification {out[eps]['accuracy']:.2f} (unperturbed {out[0.0]['accuracy']:.2f}, chance {1 / K:.2f}) | clean state change {out[eps]['state_change']:.3f}, dCE {out[eps]['dce']:+.3f}, footprint cosine to unperturbed {out[eps]['footprint_cos_to_unperturbed']:.2f}")
with torch.no_grad():
    for p, o in zip(params, orig): p.copy_(o)
record(f"e233_robust_{tag}", dict(model=tag, b=b, L=L, K=K, per_eps={str(k): v for k, v in out.items()}), f"K {K} (chance {1 / K:.2f}); accuracy at eps 0/1e-3/1e-2/3e-2/1e-1: " + " ".join(f"{out[e]['accuracy']:.2f}" for e in (0.0, 1e-3, 1e-2, 3e-2, 1e-1)) + " | state change " + " ".join(f"{out[e]['state_change']:.3f}" for e in (1e-3, 1e-2, 3e-2, 1e-1)) + " | dCE " + " ".join(f"{out[e]['dce']:+.2f}" for e in (1e-3, 1e-2, 3e-2, 1e-1)))
