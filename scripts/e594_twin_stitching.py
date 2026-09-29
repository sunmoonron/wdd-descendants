"""e594 (session 110): twin-based stitching. Context-set twins give class correspondences between models without any
feature matching; are they useful anchors for stitching one model into another? Pythia-160m's block-6 states are
mapped into Pythia-410m's block-12 states by a ridge map fit on: all training positions (the standard stitch, 16
sequences); only the positions inside the 160m words' classes that have a 410m twin (Jaccard 0.25 or more); random
positions of the same count; and the twins' class centroids alone (about a hundred pairs). Evaluated on eight held-out
sequences by the cosine between mapped and true 410m states and by the stitched loss: 410m with its block-12 states
replaced by the mapped 160m states, against 410m's own loss and 160m's. Pre-registered (probabilities are honest
guesses):
 S1 (0.6) the twinned-position fit stitches no better than random positions of the same count (the classes are not
    privileged anchors);
 S2 (0.5) the centroid-only fit recovers more than half of the full stitch's improvement over the unstitched baseline."""
from s101_common import *
t0 = time.time(); ids = eval_ids("pythia410")[:24, :512].to(DEV); TRN, TST = ids[:16], ids[16:]
def get(name, B, ids_):
    m, _, fam = load_model(name); arch = Arch(m, fam); X = block_states(m, arch, ids_, [B], chunk=4)[B]; return m, arch, X
m1, a1, X1 = get("pythia160", 6, ids); m4, a4, X4 = get("pythia410", 12, ids); D1, D4 = a1.D, a4.D; T = ids.shape[1] - 1
Xtr1, Xtr4 = X1[:16].reshape(-1, D1), X4[:16].reshape(-1, D4); Xte1, Xte4 = X1[16:].reshape(-1, D1), X4[16:].reshape(-1, D4)
keep1, keep4 = ~sinkmask(Xtr1), ~sinkmask(Xtr4); keepc = keep1 & keep4
A1, _ = rows_of(a1, 6); A4, _ = rows_of(a4, 12); U1 = unitr(Xtr1[keepc] - Xtr1[keepc].mean(0)); U4 = unitr(Xtr4[keepc] - Xtr4[keepc].mean(0))
st1, st4 = stats(U1, A1, K), stats(U4, A4, K); w1, w4 = torch.nonzero(wordset(st1["usage"]))[:, 0], torch.nonzero(wordset(st4["usage"]))[:, 0]; C1, C4 = st1["ratio"][:, w1].float() > 1, st4["ratio"][:, w4].float() > 1
def jac(P, Q):
    P = P.float(); Q = Q.float(); inter = P.T @ Q; return inter / (P.sum(0)[:, None] + Q.sum(0)[None] - inter).clamp_min(1)
J = jac(C1, C4); best, arg = J.max(1); tw = torch.nonzero(best >= 0.25)[:, 0]; kidx = torch.nonzero(keepc)[:, 0]
twin_pos = torch.unique(torch.cat([kidx[torch.nonzero(C1[:, i] | C4[:, arg[i]])[:, 0]] for i in tw.tolist()])); log(f"{tw.numel()} twinned classes of {w1.numel()} 160m words, {twin_pos.numel()} twinned positions of {int(keepc.sum())} kept")
cent1 = torch.stack([Xtr1[kidx[torch.nonzero(C1[:, i])[:, 0]]].mean(0) for i in tw.tolist()]); cent4 = torch.stack([Xtr4[kidx[torch.nonzero(C4[:, arg[i]])[:, 0]]].mean(0) for i in tw.tolist()])
def ridge(Xa, Xb, lam):
    Xa1 = torch.cat([Xa, torch.ones(Xa.shape[0], 1, device=DEV)], 1); G = Xa1.T @ Xa1 + lam * torch.eye(Xa1.shape[1], device=DEV); return torch.linalg.solve(G, Xa1.T @ Xb)
def apply(W, Xa): return torch.cat([Xa, torch.ones(Xa.shape[0], 1, device=DEV)], 1) @ W
def stitched_loss(Xmap):
    """410m with its block-12 output at positions 1: replaced by Xmap on the test sequences"""
    def hk(m, i, o):
        x = o[0] if isinstance(o, tuple) else o; y = x.clone(); y[:, 1:] = Xmap.reshape(TST.shape[0], T, D4).to(y.dtype); return (y,) + tuple(o[1:]) if isinstance(o, tuple) else y
    h = a4.layers[12].register_forward_hook(hk)
    try:
        with torch.no_grad(): lg = m4(TST).logits.float()
    finally: h.remove()
    return float(torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), TST[:, 1:].reshape(-1)))
with torch.no_grad(): L4 = float(torch.nn.functional.cross_entropy(m4(TST).logits.float()[:, :-1].reshape(-1, m4.config.vocab_size), TST[:, 1:].reshape(-1))); L1 = float(torch.nn.functional.cross_entropy(m1(TST).logits.float()[:, :-1].reshape(-1, m1.config.vocab_size), TST[:, 1:].reshape(-1)))
g = torch.Generator().manual_seed(0); kall = torch.nonzero(keepc)[:, 0]; rnd = kall[torch.randperm(kall.numel(), generator=g)[:twin_pos.numel()]]
res = dict(n_twins=int(tw.numel()), n_twin_positions=int(twin_pos.numel()), n_train_positions=int(keepc.sum()), loss_410m=L4, loss_160m=L1, fits={})
for tag, (Xa, Xb, lam) in {"all_positions": (Xtr1[kall], Xtr4[kall], 1.0), "twinned_positions": (Xtr1[twin_pos], Xtr4[twin_pos], 1.0), "random_positions_same_count": (Xtr1[rnd], Xtr4[rnd], 1.0), "twin_centroids": (cent1, cent4, 10.0), "twin_centroids_strong_ridge": (cent1, cent4, 1000.0)}.items():
    W = ridge(Xa, Xb, lam); Xm = apply(W, Xte1); cos = float(torch.nn.functional.cosine_similarity(Xm, Xte4, dim=1).median()); Ls = stitched_loss(Xm); res["fits"][tag] = dict(n=int(Xa.shape[0]), cosine=cos, stitched_loss=Ls); log(f"{tag} ({Xa.shape[0]} pairs): mapped-vs-true cosine {cos:.3f}, stitched loss {Ls:.3f} (410m {L4:.3f}, 160m {L1:.3f})")
F_ = res["fits"]
summ = (f"twin stitching 160m block 6 -> 410m block 12 ({tw.numel()} twinned classes, {twin_pos.numel()} twinned positions of {int(keepc.sum())}): stitched loss all positions {F_['all_positions']['stitched_loss']:.3f}, twinned positions {F_['twinned_positions']['stitched_loss']:.3f}, random positions of the same count {F_['random_positions_same_count']['stitched_loss']:.3f}, twin centroids {F_['twin_centroids']['stitched_loss']:.3f} (strong ridge {F_['twin_centroids_strong_ridge']['stitched_loss']:.3f}); 410m's own {L4:.3f}, 160m's own {L1:.3f}; mapped-state cosine {F_['all_positions']['cosine']:.3f} / {F_['twinned_positions']['cosine']:.3f} / {F_['random_positions_same_count']['cosine']:.3f} / {F_['twin_centroids']['cosine']:.3f} | {time.time() - t0:.0f}s")
log(summ); record("e594_twin_stitching", res, summ)
