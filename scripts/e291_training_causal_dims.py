"""e291: when does the low-dimensional causal coordinate appear in training? The compression ladder (e286) and the
shared-versus-token-specific split (e287) on Pythia-410m at one revision per run (block-2 writers, level 12,
future 14): supervised dimensions for 90% of the full score on identity, function, future descendant and removal
KL, the full scores, the reconstruction dimension, and the fraction of each score reached by the centroid span."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from func_common import *
rev = sys.argv[1]; c = Cache("pythia410"); model, tok, fam = load_model(c.name, revision=None if rev == "final" else rev); arch = Arch(model, fam); NS = 16; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; NB = arch.NB; D = arch.D; b = 2; L = NB // 2; lf = L + 2
led, tn, tc = dominant(model, arch, c, ids_seq, b); run = make_runner_logits(model, arch, c, ids_seq, [L, lf], NT, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep)
S0i = run(positions=idx); S1i = run(tn, positions=idx); dl = S0i["lg"] - S1i["lg"]; lp0, lp1 = torch.log_softmax(S0i["lg"], -1), torch.log_softmax(S1i["lg"], -1); kl = (lp0.exp() * (lp0 - lp1)).sum(1); dl = dl - dl.mean(1, keepdim=True); dln = unit(dl); F = (S0i[L] - S1i[L])[idx]; Ff = unit((S0i[lf] - S1i[lf])[idx])
torch.manual_seed(0); split = torch.rand(len(idx), device=DEV) < 0.5; tr, te = torch.nonzero(split)[:, 0], torch.nonzero(~split)[:, 0]; mu = F[tr].mean(0, keepdim=True); Fc = F - mu
G = dl[tr] @ dl[tr].T; evg, Vg = torch.linalg.eigh(G); Zt = Vg.flip(1)[:, :256] * evg.flip(0)[:256].clamp_min(0).sqrt()[None]; Zt = Zt - Zt.mean(0, keepdim=True)
U_pca = torch.linalg.svd(Fc[tr], full_matrices=False)[2]; cents = torch.stack([Fc[tr][lab_i[tr] == k].mean(0) for k in range(K)]); w = torch.bincount(lab_i[tr], minlength=K).float(); Sb = (cents * w[:, None]).T @ cents / w.sum(); U_id = torch.linalg.eigh(Sb)[1].flip(1).T; U_fn = torch.linalg.svd(Fc[tr].T @ Zt, full_matrices=False)[0].T; Fft = Ff[tr] - Ff[tr].mean(0, keepdim=True); U_fu = torch.linalg.svd(Fc[tr].T @ Fft, full_matrices=False)[0].T
def knn(Ptr, Pte, target, k=5):
    nn = torch.cdist(Pte, Ptr).topk(k, dim=1, largest=False).indices; return target[tr][nn].mean(1)
def score(P, obs):
    Ptr, Pte = P[tr], P[te]
    if obs == "identity": return accuracy(Pte, centroids(Ptr, lab_i[tr], K), lab_i[te])
    if obs == "function": return ((unit(knn(Ptr, Pte, dln)) * dln[te]).sum(1)).median().item()
    if obs == "future": return ((unit(knn(Ptr, Pte, Ff)) * Ff[te]).sum(1)).median().item()
    a_ = knn(Ptr, Pte, kl); ra = a_.argsort().argsort().float(); rb = kl[te].argsort().argsort().float(); return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
dims = [dd for dd in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, D) if dd <= D]; subs = {"identity": U_id, "function": U_fn, "future": U_fu, "kl": U_fn}; out = {}
for obs, U in subs.items():
    curve = {r: score(Fc @ U[:min(r, U.shape[0])].T, obs) for r in dims}; full = curve[dims[-1]]; chance = 1 / K if obs == "identity" else 0.0; out[obs] = dict(required_dim=next((r for r in dims if curve[r] - chance >= 0.9 * (full - chance)), dims[-1]), full=full, curve={str(r): v for r, v in curve.items()})
rec_curve = {r: 1 - ((Fc[te] - (Fc[te] @ U_pca[:r].T) @ U_pca[:r]) ** 2).sum().item() / (Fc[te] ** 2).sum().item() for r in dims}; rec_dim = next((r for r in dims if rec_curve[r] >= 0.9 * rec_curve[dims[-1]]), dims[-1])
Cd = centroids(F[tr], lab_i[tr], K); Q = torch.linalg.qr(Cd.T)[0]; span = {obs: score(F @ Q, obs) for obs in subs}; span_energy = ((F[te] @ Q) ** 2).sum().item() / (F[te] ** 2).sum().item(); rem = {obs: score(F - (F @ Q) @ Q.T, obs) for obs in subs}
log(f"{rev} (K {K}): required dims identity {out['identity']['required_dim']} function {out['function']['required_dim']} future {out['future']['required_dim']} kl {out['kl']['required_dim']} reconstruction {rec_dim} | full scores id {out['identity']['full']:.2f} fn {out['function']['full']:.2f} fut {out['future']['full']:.2f} kl {out['kl']['full']:.2f} | centroid span ({span_energy:.2f} of energy) scores id {span['identity']:.2f} fn {span['function']:.2f} fut {span['future']:.2f} kl {span['kl']:.2f}; remainder id {rem['identity']:.2f} fn {rem['function']:.2f} fut {rem['future']:.2f} kl {rem['kl']:.2f}")
record(f"e291_traindims_{rev}", dict(revision=rev, K=K, L=L, future=lf, per_observable=out, reconstruction_dim=rec_dim, reconstruction_curve={str(r): v for r, v in rec_curve.items()}, span=span, span_energy=span_energy, remainder=rem), f"dims id {out['identity']['required_dim']} fn {out['function']['required_dim']} fut {out['future']['required_dim']} kl {out['kl']['required_dim']} rec {rec_dim} | full id {out['identity']['full']:.2f} fn {out['function']['full']:.2f} fut {out['future']['full']:.2f} kl {out['kl']['full']:.2f} | span ({span_energy:.2f}) id {span['identity']:.2f} fn {span['function']:.2f} fut {span['future']:.2f} kl {span['kl']:.2f} | remainder id {rem['identity']:.2f} fn {rem['function']:.2f} fut {rem['future']:.2f} kl {rem['kl']:.2f}")
