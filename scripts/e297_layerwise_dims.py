"""e297: birth and death of causal dimensions within one forward pass. At every block output after the write: the
supervised dimensions for 90% of the full score on identity, function, future (two blocks on) and removal KL, and
the PCA dimension for 90% of the reconstruction, with the full scores; the layerwise analogue of e291."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from func_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; NB = c.NB; b = 2; D = c.D; levels = list(range(b + 1, NB - 1))
led, tn, tc = dominant(model, arch, c, ids_seq, b); run = make_runner_logits(model, arch, c, ids_seq, levels, NT, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep)
S0i = run(positions=idx); S1i = run(tn, positions=idx); dl = S0i["lg"] - S1i["lg"]; lp0, lp1 = torch.log_softmax(S0i["lg"], -1), torch.log_softmax(S1i["lg"], -1); kl = (lp0.exp() * (lp0 - lp1)).sum(1); dl = dl - dl.mean(1, keepdim=True); dln = unit(dl)
torch.manual_seed(0); split = torch.rand(len(idx), device=DEV) < 0.5; tr, te = torch.nonzero(split)[:, 0], torch.nonzero(~split)[:, 0]; G = dl[tr] @ dl[tr].T; evg, Vg = torch.linalg.eigh(G); Zt = Vg.flip(1)[:, :256] * evg.flip(0)[:256].clamp_min(0).sqrt()[None]; Zt = Zt - Zt.mean(0, keepdim=True)
def knn(Ptr, Pte, target, k=5):
    nn = torch.cdist(Pte, Ptr).topk(k, dim=1, largest=False).indices; return target[tr][nn].mean(1)
dims = [dd for dd in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, D) if dd <= D]; out = {}
for lv in levels:
    F = (S0i[lv] - S1i[lv])[idx]; Fc = F - F[tr].mean(0, keepdim=True); Ff = unit((S0i[lv + 2] - S1i[lv + 2])[idx]) if lv + 2 in S0i else None
    cents = torch.stack([Fc[tr][lab_i[tr] == k].mean(0) for k in range(K)]); w = torch.bincount(lab_i[tr], minlength=K).float(); Sb = (cents * w[:, None]).T @ cents / w.sum(); U_id = torch.linalg.eigh(Sb)[1].flip(1).T; U_fn = torch.linalg.svd(Fc[tr].T @ Zt, full_matrices=False)[0].T; U_pca = torch.linalg.svd(Fc[tr], full_matrices=False)[2]
    def score(U, r, obs):
        P = U[:min(r, U.shape[0])].T; Ptr, Pte = Fc[tr] @ P, Fc[te] @ P
        if obs == "identity": return accuracy(Pte, centroids(Ptr, lab_i[tr], K), lab_i[te])
        if obs == "function": return ((unit(knn(Ptr, Pte, dln)) * dln[te]).sum(1)).median().item()
        if obs == "future": return ((unit(knn(Ptr, Pte, Ff)) * Ff[te]).sum(1)).median().item()
        a_ = knn(Ptr, Pte, kl); ra = a_.argsort().argsort().float(); rb = kl[te].argsort().argsort().float(); return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
    rec = {}
    for obs, U in (("identity", U_id), ("function", U_fn), ("kl", U_fn)) + ((("future", torch.linalg.svd(Fc[tr].T @ (Ff[tr] - Ff[tr].mean(0, keepdim=True)), full_matrices=False)[0].T),) if Ff is not None else ()):
        curve = {r: score(U, r, obs) for r in dims}; full = curve[dims[-1]]; ch = 1 / K if obs == "identity" else 0.0; rec[obs] = dict(dim=next((r for r in dims if curve[r] - ch >= 0.9 * (full - ch)), dims[-1]), full=full)
    recon = {r: 1 - ((Fc[te] - (Fc[te] @ U_pca[:r].T) @ U_pca[:r]) ** 2).sum().item() / (Fc[te] ** 2).sum().item() for r in dims}; rec["reconstruction"] = dict(dim=next((r for r in dims if recon[r] >= 0.9 * recon[dims[-1]]), dims[-1]), full=recon[dims[-1]]); out[lv] = rec
log(f"{tag} (K {K}): block: id-dim/fn-dim/fut-dim/kl-dim/rec-dim (full scores) :: " + " ; ".join(f"{lv}: {v['identity']['dim']}/{v['function']['dim']}/{v['future']['dim'] if 'future' in v else '-'}/{v['kl']['dim']}/{v['reconstruction']['dim']} ({v['identity']['full']:.2f}/{v['function']['full']:.2f}/{v['future']['full'] if 'future' in v else float('nan'):.2f}/{v['kl']['full']:.2f})" for lv, v in out.items()))
record(f"e297_layerdims_{tag}", dict(model=tag, b=b, L=L, K=K, per_block={str(k): v for k, v in out.items()}), " ; ".join(f"{lv}: {v['identity']['dim']}/{v['function']['dim']}/{v['future']['dim'] if 'future' in v else '-'}/{v['kl']['dim']}/{v['reconstruction']['dim']}" for lv, v in out.items()))
