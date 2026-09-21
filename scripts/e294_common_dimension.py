"""e294: the dimension of the causal core independent of any single observable. Reduced-rank ridge regression from
the per-token descendant at L to four standardised target blocks (logit-footprint scores, the future descendant two
blocks later, the source one-hot, the removal KL), each alone and all jointly: held-out R2 per block as a function
of the shared rank, the rank at which each block reaches 90% of its full-rank R2, and the joint rank against the
largest and the sum of the individual ranks. One shared latent shows as a joint rank near the largest individual
rank; separate latents as a joint rank near the sum."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from func_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 16; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; NB = c.NB; b = 2; D = c.D; lf = L + 2 if L + 2 < NB - 1 else NB - 2
led, tn, tc = dominant(model, arch, c, ids_seq, b); run = make_runner_logits(model, arch, c, ids_seq, [L, lf], NT, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep)
S0i = run(positions=idx); S1i = run(tn, positions=idx); dl = S0i["lg"] - S1i["lg"]; lp0, lp1 = torch.log_softmax(S0i["lg"], -1), torch.log_softmax(S1i["lg"], -1); kl = (lp0.exp() * (lp0 - lp1)).sum(1); dl = dl - dl.mean(1, keepdim=True); F = (S0i[L] - S1i[L])[idx]; Ff = unit((S0i[lf] - S1i[lf])[idx])
torch.manual_seed(0); split = torch.rand(len(idx), device=DEV) < 0.5; tr, te = torch.nonzero(split)[:, 0], torch.nonzero(~split)[:, 0]; Fc = F - F[tr].mean(0, keepdim=True)
G = dl[tr] @ dl[tr].T; evg, Vg = torch.linalg.eigh(G); Zs = torch.zeros(len(idx), 256, device=DEV); Bv = dl[tr].T @ (Vg.flip(1)[:, :256] / evg.flip(0)[:256].clamp_min(1e-6).sqrt()[None]); Zs = dl @ Bv
blocks = {"function": Zs, "future": Ff, "identity": torch.nn.functional.one_hot(lab_i, K).float(), "kl": kl[:, None]}
def std(Y): Yc = Y - Y[tr].mean(0, keepdim=True); return Yc / Yc[tr].pow(2).sum().sqrt()
Y = {nm: std(v) for nm, v in blocks.items()}; ranks = [1, 2, 4, 8, 16, 32, 64, 128]
Gx = Fc[tr].T @ Fc[tr]; Winv = torch.linalg.inv(Gx + 1e-1 * Gx.diagonal().mean() * torch.eye(D, device=DEV))
def rrr(Yt):
    Wf = Winv @ (Fc[tr].T @ Yt[tr]); fitted = Fc[tr] @ Wf; V = torch.linalg.svd(fitted, full_matrices=False)[2]; return Wf, V
def r2(Yt, pred, cols): return 1 - ((Yt[te][:, cols] - pred[:, cols]) ** 2).sum().item() / (Yt[te][:, cols] ** 2).sum().item()
res = {}; Yall = torch.cat([Y[nm] for nm in blocks], 1); offs = {}; o = 0
for nm in blocks: offs[nm] = slice(o, o + Y[nm].shape[1]); o += Y[nm].shape[1]
for nm in blocks:
    Wf, V = rrr(Y[nm]); curve = {r: r2(Y[nm], (Fc[te] @ Wf) @ V[:r].T @ V[:r], slice(None)) for r in ranks}; full = r2(Y[nm], Fc[te] @ Wf, slice(None)); res[nm] = dict(full_r2=full, curve={str(k): v for k, v in curve.items()}, rank90=next((r for r in ranks if curve[r] >= 0.9 * full), ranks[-1]))
Wj, Vj = rrr(Yall); joint = {}
for nm in blocks:
    curve = {r: r2(Yall, (Fc[te] @ Wj) @ Vj[:r].T @ Vj[:r], offs[nm]) for r in ranks}; full = r2(Yall, Fc[te] @ Wj, offs[nm]); joint[nm] = dict(full_r2=full, curve={str(k): v for k, v in curve.items()}, rank90=next((r for r in ranks if curve[r] >= 0.9 * full), ranks[-1]))
joint_rank = max(v["rank90"] for v in joint.values()); ind = {nm: v["rank90"] for nm, v in res.items()}
log(f"{tag} (K {K}): individual ranks for 90% of full R2: " + ", ".join(f"{nm} {v} (R2 {res[nm]['full_r2']:.2f})" for nm, v in ind.items()) + f" | joint ranks: " + ", ".join(f"{nm} {v['rank90']} (R2 {v['full_r2']:.2f})" for nm, v in joint.items()) + f" | joint rank {joint_rank} vs largest individual {max(ind.values())} vs sum {sum(ind.values())}")
record(f"e294_commondim_{tag}", dict(model=tag, b=b, L=L, K=K, individual=res, joint=joint, joint_rank=joint_rank), "individual " + " ".join(f"{nm} {v}" for nm, v in ind.items()) + f" | joint " + " ".join(f"{nm} {v['rank90']}" for nm, v in joint.items()) + f" | joint {joint_rank} vs max {max(ind.values())} vs sum {sum(ind.values())}")
