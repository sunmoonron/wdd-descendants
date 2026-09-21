"""e281: does the transport law compose functionally? Linear operators T(3->L), T(3->M), T(M->L) (M the midpoint block)
are fitted on random injections; for held-out random vectors and for the block-2 candidate writes the direct
prediction x T(3->L) and the composed prediction x T(3->M) T(M->L) are compared with the actual image (state
level), and then each of the three (actual image, direct, composed) is injected at the block-(L+1) input and its
logit effect compared (function level). If composed and direct are functionally indistinguishable from the actual
image, the law is a state-space dynamics rather than a set of level-specific observations."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from func_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; b = 2; D = c.D; M = (b + 1 + L) // 2; lab = c.d["lab"]; A = c.d["A"]; R = A[(lab["type"] == T_MLP) & (lab["block"] == b)].float().to(DEV)
led, tn, tc = dominant(model, arch, c, ids_seq, b); run = make_runner_logits(model, arch, c, ids_seq, [M, L], NT, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep); S1 = run(tn); F = (S0[L] - S1[L])[idx]; Cd = centroids(F, lab_i, K); smed = torch.stack([tc[idx][lab_i == k].median() for k in range(K)]); sgn = torch.sign(smed)
pool = torch.nonzero(typ)[:, 0]; s_inj = tc.abs().median(); torch.manual_seed(0)
def fit_T(blk, lv, seed):
    torch.manual_seed(seed); Vr = unit(torch.randn(1024, D, device=DEV)); X = []; Y = []
    for p in range(2):
        a = torch.randint(0, 1024, (len(pool),), device=DEV); inj = torch.zeros(NT, D, device=DEV); inj[pool] = s_inj * Vr[a]; S2 = run(inject=inj, inject_block=blk); X.append(Vr[a]); Y.append((S2[lv] - S0[lv])[pool] / s_inj)
    X, Y = torch.cat(X), torch.cat(Y); G = X.T @ X; return torch.linalg.solve(G + 1e-2 * G.diagonal().mean() * torch.eye(D, device=DEV), X.T @ Y)
T3L = fit_T(b + 1, L, 1); T3M = fit_T(b + 1, M, 2); TML = fit_T(M + 1, L, 3)
torch.manual_seed(7); NX = 24; Xh = unit(torch.randn(NX, D, device=DEV)); a = torch.randint(0, NX, (len(pool),), device=DEV); inj = torch.zeros(NT, D, device=DEV); inj[pool] = s_inj * Xh[a]; S2 = run(inject=inj, inject_block=b + 1); Yact = torch.stack([(S2[L] - S0[L])[pool][a == q].mean(0) for q in range(NX)]) / s_inj
Pd = Xh @ T3L; Pc = (Xh @ T3M) @ TML; cs = lambda P, Q: ((unit(P) * unit(Q)).sum(1)).median().item()
state = dict(random_direct_vs_actual=cs(Pd, Yact), random_composed_vs_actual=cs(Pc, Yact), random_composed_vs_direct=cs(Pc, Pd), write_direct_vs_natural=cs(sgn[:, None] * (R[keep] @ T3L), Cd), write_composed_vs_natural=cs(sgn[:, None] * ((R[keep] @ T3M) @ TML), Cd), write_composed_vs_direct=cs((R[keep] @ T3M) @ TML, R[keep] @ T3L))
foreign = torch.nonzero(typ & ~torch.isin(tn, keep))[:, 0][:1536]; NF = len(foreign); base = run(positions=foreign); lg0 = base["lg"]; ynorm = Yact.norm(dim=1).median() * s_inj; asg = torch.randint(0, NX, (NF,), device=DEV)
def effect(V):
    inj = torch.zeros(NT, D, device=DEV); inj[foreign] = ynorm * unit(V)[asg]; r = run(positions=foreign, inject=inj, inject_block=L + 1); dl = r["lg"] - lg0; return dl - dl.mean(1, keepdim=True)
Ea, Ed, Ec = effect(Yact), effect(Pd), effect(Pc); pc = lambda P, Q: ((P * Q).sum(1) / (P.norm(dim=1) * Q.norm(dim=1)).clamp_min(1e-9)).median().item()
func = dict(direct_vs_actual=pc(Ed, Ea), composed_vs_actual=pc(Ec, Ea), composed_vs_direct=pc(Ec, Ed))
torch.manual_seed(11); Er = effect(unit(torch.randn(NX, D, device=DEV))); func["random_vs_actual"] = pc(Er, Ea)
log(f"{tag} (M {M}, L {L}): state level, random vectors: direct vs actual {state['random_direct_vs_actual']:.2f}, composed vs actual {state['random_composed_vs_actual']:.2f}, composed vs direct {state['random_composed_vs_direct']:.2f}; writes: direct vs natural {state['write_direct_vs_natural']:.2f}, composed vs natural {state['write_composed_vs_natural']:.2f}, composed vs direct {state['write_composed_vs_direct']:.2f} | function level (logit effect of the injected prediction vs of the actual image): direct {func['direct_vs_actual']:.2f}, composed {func['composed_vs_actual']:.2f}, composed vs direct {func['composed_vs_direct']:.2f}, random reference {func['random_vs_actual']:.2f}")
record(f"e281_composition_{tag}", dict(model=tag, b=b, M=M, L=L, K=K, state=state, function=func), f"state: direct {state['random_direct_vs_actual']:.2f} composed {state['random_composed_vs_actual']:.2f} (writes vs natural: direct {state['write_direct_vs_natural']:.2f} composed {state['write_composed_vs_natural']:.2f}) | function: direct {func['direct_vs_actual']:.2f} composed {func['composed_vs_actual']:.2f} composed-vs-direct {func['composed_vs_direct']:.2f} random {func['random_vs_actual']:.2f}")
