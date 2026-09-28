"""e559 (session 101): is the ledger useful for detecting a corrupted context, and does it identify a document? (a) In
each of 32 Pile sequences one token between positions 40 and 200 is replaced by a random token; the block-12 states of
Pythia-410m are read on the clean and the corrupted run. A detector sees one state and must say whether a corruption
lies upstream (positions after the corrupted token in the corrupted run, label 1, against the same positions in the
clean run, label 0; the token at the position is the same in both). Features: the raw state (1024), its top-256
principal components, the native codes (the OMP coefficients of the 256 words, 256-dim, zero where the word is not
selected), the same codes over the rotated dictionary's 256 most used atoms, and three ledger scalars (the number of
words among the 16 selected, S at the position, the cutoff). L2 logistic regression trained on sequences 0-15, AUC on
16-31, overall and by distance from the corruption (1-2, 3-8, 9-32, 33+). The codes are a function of the state, so
the raw state is the ceiling; the question is how much of the ceiling a 256-number ledger keeps against an equally
sized generic code. (b) Document identity: each clean sequence split in halves; the ledger histogram of the first half
(word usage, 256), the rotated-atom usage histogram, the mean unit state and the token histogram each retrieve the
matching second half among 32 by cosine (top-1 accuracy, chance 0.03). Pre-registered (probabilities are honest guesses):
 D1 (0.75) AUC order raw >= PCA-256 >= native codes >= rotated codes at distances up to 8;
 D2 (0.5) the native codes keep at least 80% of the raw state's AUC over chance at distances up to 8;
 D3 (0.6) the ledger histogram identifies the document above the rotated-atom histogram and below the token histogram."""
from s101_common import *
t0 = time.time(); B = 12; NS = 32; T = 256
model, tok, fam = load_model("pythia410"); arch = Arch(model, fam); ids = eval_ids("pythia410")[:NS, :T].to(DEV); g = torch.Generator().manual_seed(3)
t0s = torch.randint(40, 200, (NS,), generator=g); V = model.config.vocab_size; cids = ids.clone()
for s in range(NS):
    new = int(torch.randint(0, V, (1,), generator=g));
    while new == int(ids[s, t0s[s]]): new = int(torch.randint(0, V, (1,), generator=g))
    cids[s, t0s[s]] = new
X0 = block_states(model, arch, ids, [B])[B]; X1 = block_states(model, arch, cids, [B])[B]     # [NS, T-1, D], position p of the tensor is token position p+1
A, norms = rows_of(arch, B); m = A.shape[0]; keep0 = ~sinkmask(X0.reshape(-1, arch.D)); mu = X0.reshape(-1, arch.D)[keep0].mean(0)
U0 = unitr(X0.reshape(-1, arch.D) - mu); U1 = unitr(X1.reshape(-1, arch.D) - mu)
# native words and floor from the clean run; codes for both runs
st0 = stats(U0[keep0], A, K); words = torch.nonzero(wordset(st0["usage"]))[:, 0]; cal = floor_calibration(U0[keep0], A)
Ar = unitr(rotate(A, seed=7)); str0 = stats(U0[keep0], Ar, K); wr = torch.nonzero(wordset(str0["usage"]))[:, 0]
def codes(U, Dd, wsel):
    sel, cof, err = omp(U, Dd, K, batch=1024, record_err=True); C = torch.zeros(U.shape[0], wsel.numel(), device=DEV)
    pos = torch.full((Dd.shape[0],), -1, dtype=torch.long, device=DEV); pos[wsel] = torch.arange(wsel.numel(), device=DEV); p = pos[sel]; ok = p >= 0
    C[torch.nonzero(ok)[:, 0], p[ok]] = cof[ok]; nw = ok.sum(1).float()
    L = floor_of(U, cal); ratio = torch.cat([(U[s:s + 512] @ Dd.T).abs() / L[s:s + 512, None] for s in range(0, U.shape[0], 512)]); S = ratio.max(1).values; cut = ratio.topk(K, dim=1).values[:, -1]
    return C, torch.stack([nw, S, cut], 1)
C0, sc0 = codes(U0, A, words); C1, sc1 = codes(U1, A, words); R0, _ = codes(U0, Ar, wr); R1, _ = codes(U1, Ar, wr)
Xc = U0.reshape(NS, T - 1, -1); evs, Vp = torch.linalg.eigh(torch.cov(U0[keep0][:8000].T.double())); Vp = Vp.flip(-1)[:, :256].float()
# samples: positions after the corruption, both runs
rows = []; 
for s in range(NS):
    for p in range(int(t0s[s]), T - 1):    # tensor index p = token position p+1 > t0
        d = p + 1 - int(t0s[s]); rows.append((s, p, d))
S_ = torch.tensor([r[0] for r in rows]); P_ = torch.tensor([r[1] for r in rows]); D_ = torch.tensor([r[2] for r in rows]); idx = S_ * (T - 1) + P_
feats = {"raw": (U0, U1), "pca256": (U0 @ Vp, U1 @ Vp), "native_codes": (C0, C1), "rotated_codes": (R0, R1), "ledger_scalars": (sc0, sc1), "native_codes_plus_scalars": (torch.cat([C0, sc0], 1), torch.cat([C1, sc1], 1))}
tr = S_ < NS // 2; te = ~tr; res = dict(n_train=int(tr.sum()) * 2, n_test=int(te.sum()) * 2, auc={}, auc_by_distance={}); bins = [(1, 2), (3, 8), (9, 32), (33, 10 ** 6)]
for name, (F0, F1) in feats.items():
    Xtr = torch.cat([F0[idx[tr]], F1[idx[tr]]]); ytr = torch.cat([torch.zeros(int(tr.sum())), torch.ones(int(tr.sum()))]).to(DEV)
    Xte = torch.cat([F0[idx[te]], F1[idx[te]]]); yte = torch.cat([torch.zeros(int(te.sum())), torch.ones(int(te.sum()))]); dte = torch.cat([D_[te], D_[te]])
    l2 = 1e-2 if F0.shape[1] > 16 else 1e-4; sc = logreg(Xtr, ytr, Xte, l2=l2).cpu()
    res["auc"][name] = auc(sc, yte); res["auc_by_distance"][name] = {f"{a}-{b}": auc(sc[(dte >= a) & (dte <= b)], yte[(dte >= a) & (dte <= b)]) for a, b in bins}
    log(f"{name} ({F0.shape[1]}): AUC {res['auc'][name]:.3f} | by distance " + ", ".join(f"{k} {v:.3f}" for k, v in res["auc_by_distance"][name].items()))
# (b) document identity from halves
h1 = slice(0, (T - 1) // 2); h2 = slice((T - 1) // 2, T - 1)
def hist(C): return (C != 0).float().reshape(NS, T - 1, -1)
def retrieve(F):
    a = unitr(F[:, h1].mean(1)); b = unitr(F[:, h2].mean(1)); sim = a @ b.T; return float((sim.argmax(1) == torch.arange(NS, device=sim.device)).float().mean()), float(torch.diagonal(sim).mean() - (sim.sum() - torch.diagonal(sim).sum()) / (NS * (NS - 1)))
tokh = torch.zeros(NS, T - 1, V, device=DEV); tokh.scatter_(2, ids[:, 1:, None], 1.0)
res["document"] = {name: dict(zip(("top1", "margin"), retrieve(F))) for name, F in (("ledger_histogram", hist(C0)), ("rotated_histogram", hist(R0)), ("ledger_weighted", C0.abs().reshape(NS, T - 1, -1)), ("mean_state", U0.reshape(NS, T - 1, -1)), ("token_histogram", tokh))}
del tokh
for k, v in res["document"].items(): log(f"document identity by {k}: top-1 {v['top1']:.2f} (chance {1 / NS:.2f}), cosine margin {v['margin']:.3f}")
a = res["auc"]; ad = res["auc_by_distance"]
summ = (f"corruption detection from one block-12 state (Pythia-410m, {res['n_test']} test states): AUC raw/PCA-256/native codes/rotated codes/ledger scalars {a['raw']:.3f}/{a['pca256']:.3f}/{a['native_codes']:.3f}/{a['rotated_codes']:.3f}/{a['ledger_scalars']:.3f}; "
        f"at distance 1-2: {ad['raw']['1-2']:.3f}/{ad['pca256']['1-2']:.3f}/{ad['native_codes']['1-2']:.3f}/{ad['rotated_codes']['1-2']:.3f}; 3-8: {ad['raw']['3-8']:.3f}/{ad['pca256']['3-8']:.3f}/{ad['native_codes']['3-8']:.3f}/{ad['rotated_codes']['3-8']:.3f}; 33+: {ad['raw']['33-1000000']:.3f}/{ad['pca256']['33-1000000']:.3f}/{ad['native_codes']['33-1000000']:.3f}/{ad['rotated_codes']['33-1000000']:.3f}; "
        f"document identity top-1 (chance 0.03): ledger histogram {res['document']['ledger_histogram']['top1']:.2f}, rotated-atom histogram {res['document']['rotated_histogram']['top1']:.2f}, mean state {res['document']['mean_state']['top1']:.2f}, token histogram {res['document']['token_histogram']['top1']:.2f} | {time.time() - t0:.0f}s")
log(summ); record("e559_corruption", res, summ)
