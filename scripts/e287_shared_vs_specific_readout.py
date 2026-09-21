"""e287: is the low-rank causal read-out the shared transport term, or does the token-specific term carry causal
information too? At mid depth the per-token descendant is split into its component in the span of the K class
centroids (the shared, between-neuron part, K directions, estimated on the train half) and the orthogonal
remainder (the token-specific part, D - K directions). Each observable (source identity, logit footprint, removal
KL, future descendant two blocks later) is decoded from the full descendant, from the span part, from the
remainder, from a random K-dimensional projection (dimension control) and from the top-K principal components of
the remainder, with the decoders of e286 on held-out tokens. If the span part reaches the full score, the causal
read-out is the shared transport; if the remainder carries a large share, the token-specific term is read too."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from func_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 16; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; NB = c.NB; b = 2; D = c.D; lf = L + 2 if L + 2 < NB - 1 else NB - 2
led, tn, tc = dominant(model, arch, c, ids_seq, b); run = make_runner_logits(model, arch, c, ids_seq, [L, lf], NT, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep)
S0 = run(positions=idx); S1 = run(tn, positions=idx); dl = S0["lg"] - S1["lg"]; lp0, lp1 = torch.log_softmax(S0["lg"], -1), torch.log_softmax(S1["lg"], -1); kl = (lp0.exp() * (lp0 - lp1)).sum(1); dl = dl - dl.mean(1, keepdim=True); dln = unit(dl); F = (S0[L] - S1[L])[idx]; Ff = unit((S0[lf] - S1[lf])[idx])
torch.manual_seed(0); split = torch.rand(len(idx), device=DEV) < 0.5; tr, te = torch.nonzero(split)[:, 0], torch.nonzero(~split)[:, 0]
Cd = centroids(F[tr], lab_i[tr], K); Q = torch.linalg.qr(Cd.T)[0]; Fspan = F @ Q @ Q.T; Frem = F - Fspan; Urem = torch.linalg.svd(Frem[tr] - Frem[tr].mean(0, keepdim=True), full_matrices=False)[2][:K]; Rq = torch.linalg.qr(torch.randn(D, K, device=DEV))[0]
inputs = {"full": F, "span_K": F @ Q, "remainder": Frem, "remainder_topK_pcs": Frem @ Urem.T, "random_K": F @ Rq}
def knn(Ptr, Pte, target, k=5):
    nn = torch.cdist(Pte, Ptr).topk(k, dim=1, largest=False).indices; return target[tr][nn].mean(1)
def scores(P):
    Ptr, Pte = P[tr], P[te]; a_ = knn(Ptr, Pte, kl); ra = a_.argsort().argsort().float(); rb = kl[te].argsort().argsort().float()
    return dict(identity=accuracy(Pte, centroids(Ptr, lab_i[tr], K), lab_i[te]), function=((unit(knn(Ptr, Pte, dln)) * dln[te]).sum(1)).median().item(), behaviour=torch.corrcoef(torch.stack([ra, rb]))[0, 1].item(), future=((unit(knn(Ptr, Pte, Ff)) * Ff[te]).sum(1)).median().item())
out = {nm: scores(P) for nm, P in inputs.items()}; energy_span = ((Fspan[te] ** 2).sum() / (F[te] ** 2).sum()).item()
log(f"{tag} (K {K}, D {D}, L {L}; span holds {energy_span:.2f} of descendant energy): " + " | ".join(f"{nm}: identity {v['identity']:.2f}, function {v['function']:.2f}, behaviour {v['behaviour']:.2f}, future {v['future']:.2f}" for nm, v in out.items()))
record(f"e287_readout_{tag}", dict(model=tag, b=b, L=L, future=lf, K=K, D=D, span_energy=energy_span, per_input=out), f"span energy {energy_span:.2f} | " + " | ".join(f"{nm}: id {v['identity']:.2f} fn {v['function']:.2f} beh {v['behaviour']:.2f} fut {v['future']:.2f}" for nm, v in out.items()))
