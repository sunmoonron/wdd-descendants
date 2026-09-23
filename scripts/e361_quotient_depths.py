"""e361: the quotient's core results at other birth depths, removing the "mostly block 2" caveat. Writers of block b
(b = 2, a quarter, a half and three quarters of the depth), read a quarter of the depth later (Lr) and two blocks
after that (Lf). Battery, each as in the original experiment: (1) function dimension (PLS curve to 90% of the
full-descendant kNN score) and the random-16 score (e286); (2) universality: pairwise overlap of the logit, KL,
future and identity quotients against the split-half reliability of the logit quotient (e317); (3) the
shuffled-target audit (e340); (4) the discarded complement against the core (e346); (5) PCA-16 against PLS-16 (e347);
(6) the WDD ledger predicting the functional coordinate of joint ablations of six candidate atoms (e313)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quot_common import *
tag, b = sys.argv[1], int(sys.argv[2]); c = Cache(tag); NB = c.NB; Lr = min(b + max(2, NB // 4), NB - 2); Lf = min(Lr + 2, NB - 1); d = 16; torch.manual_seed(0)
S = Setup(tag, levels=[Lr, Lf], b=b); nat = S.natural(); F = nat["F"][Lr]; tr, te = S.halves(len(S.idx)); Fc = F - F[tr].mean(0, keepdim=True); Z, Bv = logit_scores(nat["dl"], tr); dln = unit(nat["dl"]); Ff = unit(nat["F"][Lf]); D = S.D
Q64 = pls(Fc[tr], Z[tr], 64); curve = {q: knn_cos(Fc @ Q64[:, :q], dln, tr, te) for q in (1, 2, 4, 8, 16, 32, 64)}; full = knn_cos(Fc, dln, tr, te); dim = next((q for q in curve if curve[q] >= 0.9 * full), 64); rand16 = knn_cos(Fc @ torch.linalg.qr(torch.randn(D, d, device=DEV))[0], dln, tr, te)
kc = nat["kl"][tr] - nat["kl"][tr].mean(); Qs = {"logits": Q64[:, :d], "kl": torch.linalg.eigh(Fc[tr].T @ (kc[:, None] * Fc[tr]))[1].flip(1)[:, :d], "future": pls(Fc[tr], Ff[tr], d), "identity": scatter_basis(Fc[tr], S.lab_i[tr], S.K, d)}
names = list(Qs); cross = [inside(Qs[a], Qs[bb]) for i, a in enumerate(names) for bb in names[i + 1:]]; h1, h2 = tr[: len(tr) // 2], tr[len(tr) // 2:]; rel = inside(pls(Fc[h1] - Fc[h1].mean(0, keepdim=True), Z[h1], d), pls(Fc[h2] - Fc[h2].mean(0, keepdim=True), Z[h2], d))
perm = torch.randperm(len(S.idx), device=DEV); shuffled = knn_cos(Fc @ pls(Fc[tr], Z[perm][tr], d), dln, tr, te)
Q16 = Q64[:, :d]; core = Fc @ Q16; rem = Fc - core @ Q16.T; U64 = torch.linalg.svd(rem[tr], full_matrices=False)[2][:64].T; comp = rem @ U64
acc = lambda X: accuracy(X[te], centroids(X[tr], S.lab_i[tr], S.K), S.lab_i[te]); Upca = torch.linalg.svd(Fc[tr], full_matrices=False)[2][:d].T
res = dict(model=tag, b=b, Lr=Lr, Lf=Lf, K=S.K, chance=d / D, function_dim=dim, full_score=full, curve={str(k): v for k, v in curve.items()}, random16=rand16, pls16=curve[16], pca16=knn_cos(Fc @ Upca, dln, tr, te), shuffled_target16=shuffled, cross_observable_overlap=sum(cross) / len(cross), logit_quotient_reliability=rel, core_function=curve[16], complement_function=knn_cos(comp, dln, tr, te), core_identity=acc(core), complement_identity=acc(comp), core_future=knn_cos(core, Ff, tr, te), complement_future=knn_cos(comp, Ff, tr, te))
Kc = S.K; img = torch.zeros(Kc, D, device=DEV); cnt = torch.zeros(Kc, device=DEV)
for p in range(3):
    r = S.inject_family(S.W2[S.keep], b + 1, positions=S.foreign, seed=p); img.index_add_(0, r["a"], r["F"][Lr]); cnt.index_add_(0, r["a"], torch.ones(len(r["a"]), device=DEV))
Zat = (img / cnt[:, None].clamp_min(1) / S.s_inj) @ Q16; cs, csr = [], []
for t in range(8):
    sub = torch.randperm(Kc, device=DEV)[:min(6, Kc)]; neur = S.keep[sub]
    def pre(m, a, neur=neur): x = a[0].clone(); x.reshape(-1, S.DFF)[:, neur] = 0; return (x,) + tuple(a[1:])
    h = S.arch.mlp_lin(b).register_forward_pre_hook(pre); rr = S.run(); h.remove(); dS = S.S0[Lr] - rr[Lr]; coef = S.led[:, neur]; wgt = coef.abs().sum(1); rows = torch.nonzero(S.typ & (wgt >= wgt[S.typ].quantile(0.9)))[:, 0]
    act = (dS[rows] - F[tr].mean(0, keepdim=True)) @ Q16 if False else dS[rows] @ Q16; pred = coef[rows] @ Zat[sub]; ctrl = coef[rows] @ Zat[torch.randperm(Kc, device=DEV)[:len(sub)]]
    cs.append(((unit(act) * unit(pred)).sum(1)).median().item()); csr.append(((unit(act) * unit(ctrl)).sum(1)).median().item())
res.update(ledger_to_core_cos=float(torch.tensor(cs).median()), ledger_random_control=float(torch.tensor(csr).median()))
log(f"{tag} b{b} -> L{Lr} (K {S.K}, chance {d / D:.3f}): function dim {dim} (full {full:.2f}, PLS-16 {curve[16]:.2f}, PCA-16 {res['pca16']:.2f}, random-16 {rand16:.2f}, shuffled-target-16 {shuffled:.2f}) | observable overlap {res['cross_observable_overlap']:.2f} vs reliability {rel:.2f} | complement vs core: function {res['complement_function']:.2f}/{res['core_function']:.2f}, identity {res['complement_identity']:.2f}/{res['core_identity']:.2f}, future {res['complement_future']:.2f}/{res['core_future']:.2f} | ledger->core {res['ledger_to_core_cos']:.2f} (random {res['ledger_random_control']:.2f})")
record(f"e361_depth_{tag}_{b}", res, f"b{b}->L{Lr}: dim {dim} full {full:.2f} pls16 {curve[16]:.2f} pca16 {res['pca16']:.2f} rand16 {rand16:.2f} shuf16 {shuffled:.2f} | overlap {res['cross_observable_overlap']:.2f}/{rel:.2f} | comp/core fn {res['complement_function']:.2f}/{res['core_function']:.2f} id {res['complement_identity']:.2f}/{res['core_identity']:.2f} | ledger {res['ledger_to_core_cos']:.2f}/{res['ledger_random_control']:.2f}")
