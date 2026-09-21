"""e273b: what are the bottom principal components of the descendant cloud? For each direction class of e273, the
energy fraction on the three largest-magnitude channels of the block-L output (the massive-activation channels),
and the cosine between the top such channel and the state mean. If GPT-2's 2.6x functional response along its
bottom PCs comes from the massive-activation channel, the bottom class will carry most of its energy there."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from func_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; b = 2; D = c.D
led, tn, tc = dominant(model, arch, c, ids_seq, b); run = make_runner_logits(model, arch, c, ids_seq, [L], NT, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep)
S1 = run(tn, positions=idx); F = (S0[L] - S1[L])[idx]; Fc = F - F.mean(0, keepdim=True); U = torch.linalg.svd(Fc, full_matrices=False)[2]; r = U.shape[0]
mean_state = S0[L][typ].mean(0); ch = mean_state.abs().argsort(descending=True)[:3]; torch.manual_seed(0)
Cd = unit(centroids(F, lab_i, K)); classes_u = {"pc_top8": U[:8], "pc_mid8": U[r // 2: r // 2 + 8], "pc_bottom8": U[-8:], "random8": unit(torch.randn(8, D, device=DEV)), "desc_diff8": unit(Cd[torch.randperm(K, device=DEV)[:8]] - Cd[0][None])}
share = {nm: (Us[:, ch] ** 2).sum(1).mean().item() for nm, Us in classes_u.items()}; sv = torch.linalg.svdvals(Fc); var_bottom = (sv[-8:] ** 2).sum().item() / (sv ** 2).sum().item()
res = dict(model=tag, L=L, K=K, massive_channels=ch.tolist(), massive_channel_share_of_state_energy=((S0[L][typ][:, ch] ** 2).sum() / (S0[L][typ] ** 2).sum()).item(), energy_on_massive_channels_by_class=share, bottom8_variance_fraction=var_bottom)
log(f"{tag}: channels {ch.tolist()} hold {res['massive_channel_share_of_state_energy']:.2f} of state energy at L; energy of each direction class on those channels: " + " ".join(f"{nm} {v:.3f}" for nm, v in share.items()) + f" (chance {3 / D:.4f}); bottom-8 PCs hold {var_bottom:.1e} of descendant variance")
record(f"e273b_bottompc_{tag}", res, f"massive channels {ch.tolist()} ({res['massive_channel_share_of_state_energy']:.2f} of state energy); class energy on them: " + " ".join(f"{nm} {v:.3f}" for nm, v in share.items()))
