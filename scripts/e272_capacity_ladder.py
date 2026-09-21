"""e272: the four-capacity ladder per depth. For the descendant at each of six levels, the number of top principal
components needed for 90% of the full gain in: reconstruction of the descendant itself (FVU), source identity,
function (logit-footprint prediction) and behaviour (removal KL), on held-out tokens."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from func_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; NB = c.NB; b = 2; D = c.D; levels = [lv for lv in sorted({b + 1, b + 2, b + 4, L, (L + NB - 1) // 2, NB - 2}) if lv < NB]
led, tn, tc = dominant(model, arch, c, ids_seq, b); run = make_runner_logits(model, arch, c, ids_seq, levels, NT, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep)
S0 = run(positions=idx); S1 = run(tn, positions=idx); dl = S0["lg"] - S1["lg"]; lp0, lp1 = torch.log_softmax(S0["lg"], -1), torch.log_softmax(S1["lg"], -1); kl = (lp0.exp() * (lp0 - lp1)).sum(1); dl = dl - dl.mean(1, keepdim=True); dln = unit(dl); torch.manual_seed(0); split = torch.rand(len(idx), device=DEV) < 0.5; tr, te = torch.nonzero(split)[:, 0], torch.nonzero(~split)[:, 0]
def knn_cos(Ptr, Pte, k=5):
    nn = torch.cdist(Pte, Ptr).topk(k, dim=1, largest=False).indices; return ((unit(dln[tr][nn].mean(1)) * dln[te]).sum(1)).median().item()
def knn_kl(Ptr, Pte, k=5):
    nn = torch.cdist(Pte, Ptr).topk(k, dim=1, largest=False).indices; a_, b_ = kl[tr][nn].mean(1), kl[te]; ra = a_.argsort().argsort().float(); rb = b_.argsort().argsort().float(); return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
out = {}; dims = [dd for dd in (1, 2, 4, 8, 16, 32, 64, 128, 256, D) if dd <= D]
for lv in levels:
    F = (S0[lv] - S1[lv])[idx]; mu_ = F[tr].mean(0, keepdim=True); Uf = torch.linalg.svd(F[tr] - mu_, full_matrices=False)[2]; curves = {}
    for d in dims: P = Uf[:d].T; curves[d] = dict(reconstruction=1 - (((F[te] - mu_) @ P) ** 2).sum().item() / ((F[te] - mu_) ** 2).sum().item(), identity=accuracy(F[te] @ P, centroids(F[tr] @ P, lab_i[tr], K), lab_i[te]), function=knn_cos(F[tr] @ P, F[te] @ P), behaviour=knn_kl(F[tr] @ P, F[te] @ P))
    def first_d(key, higher_better=True):
        vals = {d: (curves[d][key] if higher_better else -curves[d][key]) for d in dims}; full = vals[dims[-1]]; base = min(vals.values())
        for d in dims:
            if vals[d] - base >= 0.9 * (full - base): return d
        return dims[-1]
    out[lv] = dict(reconstruction=first_d("reconstruction", False), identity=first_d("identity"), function=first_d("function"), behaviour=first_d("behaviour"), full=curves[dims[-1]], fvu_at_32=curves[32]["reconstruction"] if 32 in curves else float("nan"))
    log(f"{tag} level {lv}: dimensions for 90% of the full gain: reconstruction {out[lv]['reconstruction']} (FVU at 32 dims {out[lv]['fvu_at_32']:.2f}), identity {out[lv]['identity']}, function {out[lv]['function']}, behaviour {out[lv]['behaviour']}")
record(f"e272_ladder_{tag}", dict(model=tag, b=b, L=L, K=K, per_level={str(k): v for k, v in out.items()}), "dims (recon / identity / function / behaviour) by level: " + " ".join(f"{lv}:{v['reconstruction']}/{v['identity']}/{v['function']}/{v['behaviour']}" for lv, v in out.items()))
