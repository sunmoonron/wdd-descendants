"""e653 (session 142; Gemini 4's idea 1, generalising e481): a zero-pass lookahead from the carrier word. e481 found that
one native word at Qwen2.5-7B's final position can carry both tokens of a two-token country name (the second ordered
right in 24 of 25 items) while the lens shows the second token 0.04 of the time. Generalised to running text: at a
position p where the model emits x_{p+1} (greedy choice equal to the corpus token), the final-block state is decomposed
into its 16 native words (OMP over every block's MLP write rows), the word contributing most to x_{p+1}'s logit is the
carrier, and its other top tokens are a one-token draft for the token after next, which the model itself chooses at p+1
(the acceptance test of greedy speculative decoding; the target is the model's own argmax at p+1). Readers: the carrier;
the three top carriers pooled; the carrier from the rotated dictionary; the runner-up of the model's own logits at p;
the embedding bigram W_U norm(W_E[x_{p+1}]); the empirical bigram of wikitext-2 train (a static table, zero passes);
the unigram; and a trained linear future lens z = W_U norm(L h_p), L initialised at the identity (prior art: Pal et al.
2023, Future Lens; the honest ceiling of what h_p holds about the token after next). Readers that read the current token
(carrier, pooled, rotated, runner-up, embedding bigram) exclude x_{p+1}. Scored by top-1 and top-5 acceptance overall and
by the target's kind (a subword continuation, a new word, other). Arguments: model [--smoke]."""
from bx_common import *
import torch.nn.functional as F
name = sys.argv[1]; SMOKE = "--smoke" in sys.argv; t0 = time.time()
M = load(name); tok = M["tok"]; WU = M["WU"]; V = WU.shape[0]; D = M["D"]; T = 128
NTR = 24 if SMOKE else 400; NTE = 6 if SMOKE else 48
A, blk = dictionary(M); Ar = unitr(rotate(A)); print(f"[{name}] dictionary {A.shape[0]} rows x {D} ({time.time() - t0:.0f}s)", flush=True)
TR = wikitext(M, "train", NTR, T); TE = wikitext(M, "test", NTE, T)


def collect(W, eligible_only):
    """per position p in 1..T-2: final raw state h_p, x_{p+1}, the model's argmax at p+1 (y), the runner-up top-5 at p excluding x_{p+1}; eligible = argmax at p equals x_{p+1}, not a sink"""
    H, X1, Y, RU = [], [], [], []
    for s0 in range(0, W.shape[0], 4):
        ids = W[s0:s0 + 4]; h, lg = final_states(M, ids); pred = lg.argmax(-1)
        assert float((readout(M, h[:1, :8]) - lg[:1, :8]).abs().max()) < 1e-2 * float(lg[:1, :8].abs().max()), "readout mismatch"
        ps = torch.arange(1, T - 1, device=DEV); hp = h[:, ps]; x1 = ids[:, ps + 1]; y = pred[:, ps + 1]
        ok = ~sinkmask(hp.reshape(-1, D)).reshape(hp.shape[:2])
        if eligible_only: ok &= pred[:, ps] == x1
        lgm = lg[:, ps].clone(); lgm.scatter_(-1, x1[..., None], -1e9); ru = lgm.topk(5, -1).indices; del lgm, lg
        H.append(hp[ok]); X1.append(x1[ok]); Y.append(y[ok]); RU.append(ru[ok])
    return torch.cat(H), torch.cat(X1), torch.cat(Y), torch.cat(RU)


Htr, X1tr, Ytr, _ = collect(TR, False); Hte, X1, Y, RU = collect(TE, True); N = Hte.shape[0]
mu = Htr.mean(0); print(f"[{name}] train positions {Htr.shape[0]}, eligible test positions {N} ({time.time() - t0:.0f}s)", flush=True)
# static tables from wikitext-2 train: unigram and bigram (top-5 continuations of each token, and the top-1 share)
tid = train_token_ids(M, 3000 if SMOKE else None); a = np.concatenate([np.array(x[:-1]) for x in tid]); b = np.concatenate([np.array(x[1:]) for x in tid])
uni = np.bincount(np.concatenate([np.array(x) for x in tid]), minlength=V); UG5 = torch.tensor(np.argsort(-uni)[:5], device=DEV)
key, cnt = np.unique(a.astype(np.int64) * V + b, return_counts=True); ka, kb = key // V, key % V; o = np.lexsort((-cnt, ka)); ka, kb, cnt = ka[o], kb[o], cnt[o]
starts = np.r_[0, np.nonzero(np.diff(ka))[0] + 1]; sizes = np.diff(np.r_[starts, len(ka)]); tot = np.add.reduceat(cnt, starts)
BG5 = np.full((V, 5), -1, dtype=np.int64); P1 = np.zeros(V)
for j in range(5):
    m_ = sizes > j; BG5[ka[starts[m_]], j] = kb[starts[m_] + j]
P1[ka[starts]] = cnt[starts] / tot; BG5 = torch.tensor(BG5, device=DEV); P1 = torch.tensor(P1, device=DEV)
print(f"[{name}] bigram table from {len(a)} train pairs ({time.time() - t0:.0f}s)", flush=True)


def top5_excl(z, x1): z = z.clone(); z.scatter_(-1, x1[:, None], -1e9); return z.topk(5, -1).indices


def carriers(Adict, H, X1):
    """the carrier (word contributing most to x_{p+1}'s logit), its draft, the three top carriers pooled, and the naming carrier
    (e481's reading: the word in whose own signed signature x_{p+1} ranks highest, ties to the larger contribution)"""
    c1, c3, cn = [], [], []
    for s in range(0, H.shape[0], 256):
        h = H[s:s + 256]; x1 = X1[s:s + 256]; sel, cof = omp(unitr(h - mu), Adict); ga = gain(M, Adict[sel])  # [n,16,D]
        sc = cof * (ga @ WU[x1][:, :, None])[..., 0]; top3 = sc.topk(3, 1).indices; c = top3[:, 0]
        n_ = torch.arange(h.shape[0], device=DEV); v1 = cof[n_, c][:, None] * ga[n_, c]; v3 = (cof.gather(1, top3)[..., None] * ga.gather(1, top3[..., None].expand(-1, -1, D))).sum(1)
        c1.append(top5_excl(v1 @ WU.T, x1)); c3.append(top5_excl(v3 @ WU.T, x1))
        for t in range(0, h.shape[0], 32):
            zz = (cof[t:t + 32, :, None] * ga[t:t + 32]) @ WU.T; xs = x1[t:t + 32]; zx = zz.gather(2, xs[:, None, None].expand(-1, K, 1)); rk = (zz > zx).sum(2).float() - 1e-6 * sc[t:t + 32]
            w_ = rk.argmin(1); cn.append(top5_excl(zz[torch.arange(zz.shape[0], device=DEV), w_], xs)); del zz
    return torch.cat(c1), torch.cat(c3), torch.cat(cn)


CA, C3, CN = carriers(A, Hte, X1); RC, _, _ = carriers(Ar, Hte, X1); print(f"[{name}] carriers done ({time.time() - t0:.0f}s)", flush=True)
EB = torch.cat([top5_excl(readout(M, M["WE"][X1[s:s + 256]]), X1[s:s + 256]) for s in range(0, N, 256)])
BG = BG5[X1]; UG = UG5[None].expand(N, -1)
# the trained future lens (prior art), on the train positions' model-chosen token after next
L = torch.nn.Linear(D, D).to(DEV)
with torch.no_grad(): L.weight.copy_(torch.eye(D)); L.bias.zero_()
opt = torch.optim.Adam(L.parameters(), lr=1e-3); NT = Htr.shape[0]; EP = 4 if SMOKE else 8; BS = 256 if V > 100000 else 512
with torch.enable_grad():
    for ep in range(EP):
        perm = torch.randperm(NT, device=DEV); tl = []
        for s in range(0, NT, BS):
            i = perm[s:s + BS]; loss = F.cross_entropy(readout(M, L(Htr[i])), Ytr[i]); opt.zero_grad(); loss.backward(); opt.step(); tl.append(float(loss))
        print(f"[{name}] future lens epoch {ep}: train loss {np.mean(tl):.3f}", flush=True)
FL = torch.cat([readout(M, L(Hte[s:s + 256])).topk(5, -1).indices for s in range(0, N, 256)])
# the target's kind
strs = [tok.decode([int(t)]) for t in Y.tolist()]
kind = np.array(["cont" if s[:1].isalpha() else ("new" if s[:1] == " " and s[1:2].isalpha() else "other") for s in strs])
amb = (P1[X1] < 0.3).cpu().numpy() & (kind == "cont")
if SMOKE:
    dec = lambda ids: [tok.decode([int(t)]) for t in ids]
    for i in range(0, N, max(N // 8, 1)):
        print(f"  x1={tok.decode([int(X1[i])])!r} y={tok.decode([int(Y[i])])!r} kind={kind[i]} | carrier {dec(CA[i])} | naming {dec(CN[i])} | top3 {dec(C3[i])} | emb-bigram {dec(EB[i])} | bigram {dec([t for t in BG[i].tolist() if t >= 0])} | runner-up {dec(RU[i])}", flush=True)
    print("  n_by_kind", {k: int((kind == k).sum()) for k in ("cont", "new", "other")}, flush=True)
R = {"carrier": CA, "naming_carrier": CN, "carriers_top3": C3, "rotated_carrier": RC, "runner_up": RU, "embedding_bigram": EB, "empirical_bigram": BG, "unigram": UG, "future_lens": FL}
res = dict(model=name, n=N, n_by_kind={k: int((kind == k).sum()) for k in ("cont", "new", "other")}, n_ambiguous_cont=int(amb.sum()), readers={})
for rn, P in R.items():
    hit1 = (P[:, 0] == Y).cpu().numpy(); hit5 = (P == Y[:, None]).any(1).cpu().numpy()
    res["readers"][rn] = dict(top1=float(hit1.mean()), top5=float(hit5.mean()), **{f"top1_{k}": float(hit1[kind == k].mean()) if (kind == k).any() else None for k in ("cont", "new", "other")},
                              **{f"top5_{k}": float(hit5[kind == k].mean()) if (kind == k).any() else None for k in ("cont", "new", "other")}, top5_ambiguous_cont=float(hit5[amb].mean()) if amb.any() else None)
    r = res["readers"][rn]; print(f"[{name}] {rn:17s} top-1 {r['top1']:.3f} top-5 {r['top5']:.3f} | continuation top-1 {r['top1_cont']} | new word top-1 {r['top1_new']}", flush=True)
rd = res["readers"]; zp = ("carrier", "naming_carrier", "carriers_top3", "rotated_carrier", "runner_up", "embedding_bigram", "empirical_bigram", "unigram")
best = lambda key: max(rd[k][key] or 0 for k in ("carrier", "naming_carrier", "carriers_top3"))
res["verdicts"] = dict(A1=rd["carrier"]["top1"] > rd["runner_up"]["top1"], A2=best("top1") > rd["empirical_bigram"]["top1"], A3=best("top1_cont") > (rd["empirical_bigram"]["top1_cont"] or 0),
                       A4=rd["carrier"]["top1"] > rd["rotated_carrier"]["top1"], A5=all(rd["future_lens"]["top1"] > rd[k]["top1"] for k in zp), A6=best("top5_ambiguous_cont") > (rd["empirical_bigram"]["top5_ambiguous_cont"] or 0))
summ = (f"[{name}] lookahead from the carrier word, {N} positions where the model emits the corpus token: top-1 acceptance of the token after next: " + ", ".join(f"{k} {rd[k]['top1']:.3f}" for k in R)
        + f"; verdicts {res['verdicts']} ({time.time() - t0:.0f}s)")
save(f"e653_lookahead_{name}" + ("_smoke" if SMOKE else ""), res, summ); print("done", flush=True)
