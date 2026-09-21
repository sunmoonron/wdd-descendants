"""e336: the empirical null space and the coset test. Directions of minimal logit response at the block-(L+1) input,
found from the read-out fitted on random injections (bottom singular directions), are injected and their actual
response measured against top and random directions (the null space's emptiness at the natural norm). Coset test:
for equivalent atom pairs (a ~ a'), the difference a - a' is injected; if equivalence is a coset of a null space,
the difference has a small functional response relative to a itself; compared with random differences."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quot_common import *
tag = sys.argv[1]; S = Setup(tag, levels=[0]); L = S.L; S = Setup(tag, levels=[L]); d = 16; nat = S.natural(); F = nat["F"][L]; tr, te = S.halves(len(S.idx)); Fc = F - F[tr].mean(0, keepdim=True); Z, Bv = logit_scores(nat["dl"], tr); P = pls(Fc[tr], Z[tr], d); fnorm = F.norm(dim=1).median()
torch.manual_seed(0); Vr = unit(torch.randn(1024, S.D, device=DEV)); X = []; Y = []; sub = S.foreign[:1536]
for p in range(2):
    r = S.inject_family(Vr, L + 1, amp=fnorm, positions=sub, seed=p); X.append(Vr[r["a"]]); Y.append((r["dl"] @ Bv) / fnorm)
X, Y = torch.cat(X), torch.cat(Y); Wg = ridge(X, Y, 1e-2); Ug, Sg, _ = torch.linalg.svd(Wg, full_matrices=False); classes_u = {"top8": Ug[:, :8].T, "bottom8": Ug[:, -8:].T, "random8": unit(torch.randn(8, S.D, device=DEV))}; resp = {}
for nm, dirs in classes_u.items():
    r = S.inject_family(dirs, L + 1, amp=fnorm, positions=sub, seed=5); resp[nm] = r["dl"].norm(dim=1).median().item()
atoms = torch.randperm(S.DFF, device=DEV)[:256]; V = S.W2[atoms]; img = torch.zeros(256, S.D, device=DEV); cnt = torch.zeros(256, device=DEV)
for p in range(3):
    r = S.inject_family(V, S.b + 1, seed=p); img.index_add_(0, r["a"], r["F"][L]); cnt.index_add_(0, r["a"], torch.ones(len(r["a"]), device=DEV))
img = img / cnt[:, None].clamp_min(1); z = img @ P; zc = unit(z) @ unit(z).T; pc = (V @ V.T).abs(); zc.fill_diagonal_(-1); pairs = [(i, int(zc[i].argmax())) for i in range(256) if zc[i].max() >= zc.max(1).values.quantile(0.9) and pc[i, int(zc[i].argmax())] < 0.3][:12]; rnd = [(i, int(torch.randint(0, 256, (1,)).item())) for i, _ in pairs]
def resp_of(v): r = S.inject_family(unit(v)[None], S.b + 1, positions=sub, seed=9); return r["dl"].norm(dim=1).median().item(), (r["F"][L].mean(0) @ P).norm().item()
diff_eq = [resp_of(V[i] - V[j]) for i, j in pairs]; diff_rn = [resp_of(V[i] - V[j]) for i, j in rnd]; single = [resp_of(V[i]) for i, _ in pairs]
res = dict(K=S.K, response_top8=resp["top8"], response_bottom8=resp["bottom8"], response_random8=resp["random8"], bottom_over_random=resp["bottom8"] / resp["random8"], n_pairs=len(pairs), diff_equivalent_logit=float(torch.tensor([x[0] for x in diff_eq]).median()), diff_random_logit=float(torch.tensor([x[0] for x in diff_rn]).median()), single_logit=float(torch.tensor([x[0] for x in single]).median()), diff_equivalent_core=float(torch.tensor([x[1] for x in diff_eq]).median()), diff_random_core=float(torch.tensor([x[1] for x in diff_rn]).median()), single_core=float(torch.tensor([x[1] for x in single]).median()))
log(f"{tag} (K {S.K}): logit response at the natural footprint norm along the read-out's top-8 / bottom-8 / random directions {res['response_top8']:.2f} / {res['response_bottom8']:.2f} / {res['response_random8']:.2f} (bottom/random {res['bottom_over_random']:.2f}) | coset test over {len(pairs)} equivalent atom pairs: logit response of a - a' {res['diff_equivalent_logit']:.2f} vs random differences {res['diff_random_logit']:.2f} vs a alone {res['single_logit']:.2f}; core-coordinate norm of a - a' {res['diff_equivalent_core']:.2f} vs random {res['diff_random_core']:.2f} vs a alone {res['single_core']:.2f}")
record(f"e336_nullspace_{tag}", dict(model=tag, L=L, **res), f"top/bottom/random {res['response_top8']:.2f}/{res['response_bottom8']:.2f}/{res['response_random8']:.2f} | coset logit eq/rand/single {res['diff_equivalent_logit']:.2f}/{res['diff_random_logit']:.2f}/{res['single_logit']:.2f} core {res['diff_equivalent_core']:.2f}/{res['diff_random_core']:.2f}/{res['single_core']:.2f}")
