"""e308: forward causal core versus backward sensitivity. The backward sensitivity subspace at L: gradients of the
logits projected on the top-16 footprint directions with respect to the level-L residual state, over the typical
tokens, covariance, top-16 eigenvectors. Energy overlaps with the forward function core, the identity core, the
descendant PCA and the activation PCA (chance 16/D). The direct first-order check f = G d with the true Jacobian:
correlation between the actual footprint projections and the gradient-predicted ones over tokens and directions.
The forward x backward intersection (top-8 directions of the forward core most aligned with the backward space)
used alone to decode function, identity and KL against the forward core's top-8, the backward top-8 and random 8."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from func_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; NB = c.NB; b = 2; D = c.D; d = 16
led, tn, tc = dominant(model, arch, c, ids_seq, b); run = make_runner_logits(model, arch, c, ids_seq, [L], NT, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep)
S0i = run(positions=idx); S1i = run(tn, positions=idx); dl = S0i["lg"] - S1i["lg"]; lp0, lp1 = torch.log_softmax(S0i["lg"], -1), torch.log_softmax(S1i["lg"], -1); kl = (lp0.exp() * (lp0 - lp1)).sum(1); dl = dl - dl.mean(1, keepdim=True); dln = unit(dl); F = (S0i[L] - S1i[L])[idx]; Fc = F - F.mean(0, keepdim=True); Ud, Sd, Vd = torch.linalg.svd(dl, full_matrices=False); B = Vd[:256].T; Zs = dl @ B; Zs = Zs - Zs.mean(0, keepdim=True)
S_fn = torch.linalg.svd(Fc.T @ Zs, full_matrices=False)[0][:, :d]; cents = torch.stack([Fc[lab_i == k].mean(0) for k in range(K)]); w = torch.bincount(lab_i, minlength=K).float(); Sb = (cents * w[:, None]).T @ cents / w.sum(); S_id = torch.linalg.eigh(Sb)[1].flip(1)[:, :d]; S_pca = torch.linalg.svd(Fc, full_matrices=False)[2][:d].T; S_act = torch.linalg.svd(S0[L][typ] - S0[L][typ].mean(0, keepdim=True), full_matrices=False)[2][:d].T
leaf = {}
def pre(m, args, kwargs):
    x = args[0] if len(args) > 0 else kwargs["hidden_states"]; xl = x.detach().clone().requires_grad_(True); leaf["x"] = xl
    if len(args) > 0: return (xl,) + tuple(args[1:]), kwargs
    kwargs = dict(kwargs); kwargs["hidden_states"] = xl; return args, kwargs
h = arch.layers[L + 1].register_forward_pre_hook(pre, with_kwargs=True); grads = []
with torch.enable_grad():
    out = model(ids_seq); h.remove(); logits = out.logits.reshape(NT, -1).float(); Bd = B[:, :d]; proj = logits @ Bd
    for j in range(d):
        g = torch.autograd.grad(proj[:, j].sum(), leaf["x"], retain_graph=(j < d - 1))[0].reshape(NT, D).detach().float(); grads.append(g)
Gt = torch.stack(grads, 1); del out, logits; Gpool = Gt[typ].reshape(-1, D); Gc = Gpool - Gpool.mean(0, keepdim=True); S_back = torch.linalg.eigh(Gc.T @ Gc)[1].flip(1)[:, :d]
inside = lambda A_, B_: ((B_.T @ A_) ** 2).sum().item() / A_.shape[1]; torch.manual_seed(0); S_rand = torch.linalg.qr(torch.randn(D, d, device=DEV))[0]
ov = dict(forward_core_vs_backward=inside(S_fn, S_back), identity_core_vs_backward=inside(S_id, S_back), descendant_pca_vs_backward=inside(S_pca, S_back), activation_pca_vs_backward=inside(S_act, S_back), random_vs_backward=inside(S_rand, S_back), chance=d / D)
pred = (Gt[idx] * F[:, None, :]).sum(2); actual = (dl @ Bd); first_order = dict(correlation_over_tokens_and_directions=torch.corrcoef(torch.stack([pred.flatten(), actual.flatten()]))[0, 1].item(), median_per_token_cos=((unit(pred) * unit(actual)).sum(1)).median().item())
Mi = S_fn @ (S_fn.T @ S_back) @ (S_back.T @ S_fn) @ S_fn.T; S_int = torch.linalg.eigh(Mi)[1].flip(1)[:, :8]
torch.manual_seed(1); split = torch.rand(len(idx), device=DEV) < 0.5; tr, te = torch.nonzero(split)[:, 0], torch.nonzero(~split)[:, 0]
def knn(Ptr, Pte, target, k=5):
    nn = torch.cdist(Pte, Ptr).topk(k, dim=1, largest=False).indices; return target[tr][nn].mean(1)
def scores(P):
    Ptr, Pte = P[tr], P[te]; a_ = knn(Ptr, Pte, kl); ra = a_.argsort().argsort().float(); rb = kl[te].argsort().argsort().float(); return dict(identity=accuracy(Pte, centroids(Ptr, lab_i[tr], K), lab_i[te]), function=((unit(knn(Ptr, Pte, dln)) * dln[te]).sum(1)).median().item(), kl=torch.corrcoef(torch.stack([ra, rb]))[0, 1].item())
dec = {"intersection8": scores(Fc @ S_int), "forward_core8": scores(Fc @ S_fn[:, :8]), "backward8": scores(Fc @ S_back[:, :8]), "random8": scores(Fc @ S_rand[:, :8]), "full": scores(Fc)}
log(f"{tag} (K {K}, chance {d / D:.3f}): backward sensitivity subspace vs forward core {ov['forward_core_vs_backward']:.2f}, vs identity core {ov['identity_core_vs_backward']:.2f}, vs descendant PCA {ov['descendant_pca_vs_backward']:.2f}, vs activation PCA {ov['activation_pca_vs_backward']:.2f}, vs random {ov['random_vs_backward']:.3f} | first-order check f = G d with the true Jacobian: correlation {first_order['correlation_over_tokens_and_directions']:.2f}, median per-token cosine {first_order['median_per_token_cos']:.2f} | decoding from 8 directions (identity/function/KL): intersection " + "/".join(f"{dec['intersection8'][o]:.2f}" for o in ("identity", "function", "kl")) + ", forward core " + "/".join(f"{dec['forward_core8'][o]:.2f}" for o in ("identity", "function", "kl")) + ", backward " + "/".join(f"{dec['backward8'][o]:.2f}" for o in ("identity", "function", "kl")) + ", random " + "/".join(f"{dec['random8'][o]:.2f}" for o in ("identity", "function", "kl")) + ", full " + "/".join(f"{dec['full'][o]:.2f}" for o in ("identity", "function", "kl")))
record(f"e308_fwdbwd_{tag}", dict(model=tag, b=b, L=L, K=K, overlaps=ov, first_order=first_order, decoding=dec), f"back vs fwd core {ov['forward_core_vs_backward']:.2f} id {ov['identity_core_vs_backward']:.2f} pca {ov['descendant_pca_vs_backward']:.2f} act {ov['activation_pca_vs_backward']:.2f} rand {ov['random_vs_backward']:.3f} | f=Gd corr {first_order['correlation_over_tokens_and_directions']:.2f} cos {first_order['median_per_token_cos']:.2f} | decode8 int " + "/".join(f"{dec['intersection8'][o]:.2f}" for o in ("identity", "function", "kl")) + " fwd " + "/".join(f"{dec['forward_core8'][o]:.2f}" for o in ("identity", "function", "kl")) + " bwd " + "/".join(f"{dec['backward8'][o]:.2f}" for o in ("identity", "function", "kl")) + " rand " + "/".join(f"{dec['random8'][o]:.2f}" for o in ("identity", "function", "kl")))
