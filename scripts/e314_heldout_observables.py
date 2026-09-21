"""e314: does the causal coordinate predict observables that were not used to build it? The 16-dimensional core is
built from the logit footprints only. New per-token observables of the same natural ablation: the change of the
next-token entropy, the change of the log-probability of the actual next token, the change of the top-1 probability,
the state change at the following token (t+1) two blocks later (direction and norm), and the change of the last
block's attention-output norm at t. Each is decoded on held-out tokens from the core, from the descendant's top-64
PCs, from a random 16-dimensional projection and from the descendant norm alone (kNN; Spearman for scalars,
cosine for the vector)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from func_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; NB = c.NB; b = 2; D = c.D; lf = L + 2 if L + 2 < NB - 1 else NB - 2
led, tn, tc = dominant(model, arch, c, ids_seq, b); run = make_runner_logits(model, arch, c, ids_seq, [L, lf], NT, b); comp = {}
def capture(ablate):
    hs = [arch.attn_lin(NB - 1).register_forward_hook(lambda m, i, o: comp.__setitem__("attn", o.detach().float().reshape(NT, -1).norm(dim=1)))]
    if ablate:
        def pre(m, inp):
            x = inp[0].clone(); flat = x.reshape(-1, c.DFF); flat[torch.arange(NT, device=DEV), tn] = 0; return (flat.reshape(x.shape),)
        hs.append(arch.mlp_lin(b).register_forward_pre_hook(pre))
    r = run(positions=torch.arange(NT, device=DEV)); [h.remove() for h in hs]; return r, comp["attn"].clone()
r0, attn0 = capture(False); r1, attn1 = capture(True); typ = typical_mask(r0[L]); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep); idx = idx[(idx % CTX) < CTX - 1]; lab_i = lab_i[:len(idx)] if False else torch.tensor([int((keep == tn[t]).nonzero()[0]) for t in idx.tolist()], device=DEV)
lg0, lg1 = r0["lg"][idx], r1["lg"][idx]; lp0, lp1 = torch.log_softmax(lg0, -1), torch.log_softmax(lg1, -1); dl = lg0 - lg1; dl = dl - dl.mean(1, keepdim=True); F = (r0[L] - r1[L])[idx]; Fc = F - F.mean(0, keepdim=True)
nxt = ids_seq.reshape(-1)[idx + 1]; obs = {"entropy_change": (-(lp0.exp() * lp0).sum(1)) - (-(lp1.exp() * lp1).sum(1)), "next_token_logprob_change": lp0[torch.arange(len(idx), device=DEV), nxt] - lp1[torch.arange(len(idx), device=DEV), nxt], "top1_prob_change": lp0.exp().max(1).values - lp1.exp().max(1).values, "attention_norm_change_last_block": attn0[idx] - attn1[idx], "next_token_state_change_norm": (r0[lf] - r1[lf])[idx + 1].norm(dim=1)}; vec_obs = unit((r0[lf] - r1[lf])[idx + 1])
torch.manual_seed(0); split = torch.rand(len(idx), device=DEV) < 0.5; tr, te = torch.nonzero(split)[:, 0], torch.nonzero(~split)[:, 0]; G = dl[tr] @ dl[tr].T; evg, Vg = torch.linalg.eigh(G); Zt = Vg.flip(1)[:, :256] * evg.flip(0)[:256].clamp_min(0).sqrt()[None]; Zt = Zt - Zt.mean(0, keepdim=True); P = torch.linalg.svd(Fc[tr].T @ Zt, full_matrices=False)[0][:, :16]; U64 = torch.linalg.svd(Fc[tr], full_matrices=False)[2][:64].T; Rn = torch.linalg.qr(torch.randn(D, 16, device=DEV))[0]
inputs = {"core16": Fc @ P, "pca64": Fc @ U64, "random16": Fc @ Rn, "norm_only": F.norm(dim=1, keepdim=True)}
def knn(Ptr, Pte, target, k=5):
    nn = torch.cdist(Pte, Ptr).topk(k, dim=1, largest=False).indices; return target[tr][nn].mean(1)
def spearman(a_, b_):
    ra = a_.argsort().argsort().float(); rb = b_.argsort().argsort().float(); return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
out = {}
for inm, X in inputs.items():
    out[inm] = {onm: spearman(knn(X[tr], X[te], v), v[te]) for onm, v in obs.items()}; out[inm]["next_token_state_direction"] = ((unit(knn(X[tr], X[te], vec_obs)) * vec_obs[te]).sum(1)).median().item()
names = list(obs) + ["next_token_state_direction"]
log(f"{tag} (K {K}, {len(idx)} tokens): held-out prediction of new observables (rows: input; columns: " + ", ".join(names) + "): " + " | ".join(f"{inm}: " + "/".join(f"{out[inm][o]:.2f}" for o in names) for inm in inputs))
record(f"e314_heldout_{tag}", dict(model=tag, b=b, L=L, K=K, observables=names, per_input=out), " | ".join(f"{inm} " + "/".join(f"{out[inm][o]:.2f}" for o in names) for inm in inputs))
