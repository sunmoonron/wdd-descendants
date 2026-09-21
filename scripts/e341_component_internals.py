"""e341: does the core predict what attention and MLPs do with the perturbation? For the natural ablation, at the
source tokens in blocks L+1 to L+3: the change of each head's output norm, the change of the attention-score
pre-activations (query and key projections read through the o-projection input is unavailable, so the change of the
query and key vectors' norms per head from hooks on the q/k projections where the architecture exposes them), the
change of every MLP neuron's activation (top-64 PCs of the DFF-dimensional change), the change of activation
sparsity, and the identity of the most-recruited neurons. Each is predicted on held-out tokens from the 16-dim core,
from the descendant's top-64 PCs, and from a random 16-dim projection (ridge R2 or kNN overlap)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quot_common import *
tag = sys.argv[1]; S0_ = Setup(tag, levels=[0]); L = S0_.L; NB = S0_.NB; S = Setup(tag, levels=[L]); d = 16; blocks = [lv for lv in (L + 1, L + 2, L + 3) if lv < NB]; cap = {}
def hooks():
    hs = []
    for lv in blocks:
        hs.append(S.arch.attn_lin(lv).register_forward_pre_hook((lambda lv_: lambda m, inp: cap.__setitem__(("head", lv_), inp[0].detach().float().reshape(S.NT, S.arch.NH, -1).norm(dim=2)))(lv))); hs.append(S.arch.mlp_lin(lv).register_forward_pre_hook((lambda lv_: lambda m, inp: cap.__setitem__(("act", lv_), inp[0].detach().float().reshape(S.NT, -1)))(lv)))
    return hs
hs = hooks(); r0 = S.run(positions=S.idx); [h.remove() for h in hs]; c0 = {k: v.clone() for k, v in cap.items()}; hs = hooks(); r1 = S.run(S.tn, positions=S.idx); [h.remove() for h in hs]; c1 = {k: v.clone() for k, v in cap.items()}
tr, te = S.halves(len(S.idx)); F = (r0[L] - r1[L])[S.idx]; Fc = F - F[tr].mean(0, keepdim=True); dl = r0["lg"] - r1["lg"]; dl = dl - dl.mean(1, keepdim=True); Z, _ = logit_scores(dl, tr); P = pls(Fc[tr], Z[tr], d); U64 = torch.linalg.svd(Fc[tr], full_matrices=False)[2][:64].T; torch.manual_seed(0); Rn = torch.linalg.qr(torch.randn(S.D, d, device=DEV))[0]
heads = torch.cat([(c0[("head", lv)] - c1[("head", lv)])[S.idx] for lv in blocks], 1); acts = torch.cat([(c0[("act", lv)] - c1[("act", lv)])[S.idx] for lv in blocks], 1); actpc = (acts - acts[tr].mean(0, keepdim=True)) @ torch.linalg.svd((acts - acts[tr].mean(0, keepdim=True))[tr], full_matrices=False)[2][:64].T; sparsity = torch.cat([((c0[("act", lv)].abs() > 1e-3).float().mean(1) - (c1[("act", lv)].abs() > 1e-3).float().mean(1))[S.idx][:, None] for lv in blocks], 1); top = acts.abs().topk(32, dim=1).indices
def r2(X, Y):
    mx, my = X[tr].mean(0, keepdim=True), Y[tr].mean(0, keepdim=True); W = ridge(X[tr] - mx, Y[tr] - my, 1e-1); pred = (X[te] - mx) @ W + my; return 1 - ((Y[te] - pred) ** 2).sum().item() / ((Y[te] - my) ** 2).sum().item()
def recruit(X):
    nn = torch.cdist(X[te], X[tr]).topk(5, dim=1, largest=False).indices; return float(torch.tensor([len(set(top[te[i]].tolist()) & set(top[tr[nn[i, 0]]].tolist())) / 32 for i in range(len(te))]).mean())
inputs = {"core16": Fc @ P, "pca64": Fc @ U64, "random16": Fc @ Rn}; out = {k: dict(head_norms=r2(X, heads), mlp_activation_pcs=r2(X, actpc), sparsity=r2(X, sparsity), neuron_recruitment_overlap=recruit(X)) for k, X in inputs.items()}; base_rec = float(torch.tensor([len(set(top[te[i]].tolist()) & set(top[tr[int(torch.randint(0, len(tr), (1,)))]].tolist())) / 32 for i in range(len(te))]).mean())
log(f"{tag} (K {S.K}, blocks {blocks}): held-out R2 for head-output norm changes / MLP activation-change PCs / activation-sparsity change, and top-32 neuron recruitment overlap (random-token baseline {base_rec:.2f}): " + " | ".join(f"{k}: {v['head_norms']:+.2f} / {v['mlp_activation_pcs']:+.2f} / {v['sparsity']:+.2f} / {v['neuron_recruitment_overlap']:.2f}" for k, v in out.items()))
record(f"e341_internals_{tag}", dict(model=tag, L=L, K=S.K, per_input=out, recruitment_baseline=base_rec), " | ".join(f"{k} {v['head_norms']:+.2f}/{v['mlp_activation_pcs']:+.2f}/{v['sparsity']:+.2f}/{v['neuron_recruitment_overlap']:.2f}" for k, v in out.items()) + f" (base {base_rec:.2f})")
