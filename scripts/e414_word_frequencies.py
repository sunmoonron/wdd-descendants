"""e414: word frequencies in the native language. How often is each word of the own vocabulary used across the k-sparse
descriptions (OMP, k 16 and 64) of the middle-depth states of 8 sequences? Arguments: model and optional revision
(Pythia-410m at steps 1000, 16000 and 143000; GPT-2). Own vocabulary and a rotation of it: the rank-frequency slope
(log frequency against log rank over ranks 10-1000, the Zipf exponent), the share of all uses taken by the top 10 and
top 100 words, the number of distinct words used, and what the top words are (family, block, and the share of their
squared norm in the states' top-8 principal subspace M, the huge directions). Also the functional weight of the most
frequent words: loss recovered when the 16-word descriptions drop the top-10 most frequent words, against dropping 10
random used words. The analogy under test is function words: extremely frequent, little meaning of their own (e400:
the huge directions carry 1.5% of the Fisher trace), structurally necessary (e404: setting them to their mean costs
0.17 of the loss). Pre-registered: own-word usage is more concentrated than rotated-word usage (a steeper slope and a
larger top-10 share); at the end of Pythia's training the most used words are the writers of the huge directions."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sd_common import *
name = sys.argv[1]; rev = sys.argv[2] if len(sys.argv) > 2 else None
c = Cache(name); ev = c.s["eval_ids"][:8].to(DEV); model, tok, fam = load_model(name, revision=rev); arch = Arch(model, fam); L = mid(c) if name == "gpt2" else arch.NB // 2
A, lab = build_dictionary(arch, blocks=list(range(L + 1))); typ, blk = lab["type"].to(DEV), lab["block"].to(DEV)
lv = Level(model, arch, ev, L); ev_, UX = torch.linalg.eigh(((lv.Xc.T @ lv.Xc) / lv.Xc.shape[0]).double()); UM = UX[:, -8:].float()
mshare = (unitr(A) @ UM).pow(2).sum(-1); tag = f"{name}_{rev or 'final'}"; res = dict(model=name, rev=rev, level=L, top8_var=(ev_[-8:].sum() / ev_.sum()).item(), vocab={})
def usage(V, k):
    sel, _, _ = omp(lv.Xc, unitr(V), k, batch=256, record_err=False); cnt = torch.bincount(sel.flatten(), minlength=V.shape[0]).float(); return cnt, sel
def zipf(cnt):
    f = cnt.sort(descending=True).values; f = f[f > 0]; r = torch.arange(1, f.numel() + 1, device=f.device).float(); lo, hi = 10, min(1000, f.numel())
    x, y = r[lo - 1:hi].log(), f[lo - 1:hi].log(); slope = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
    tot = f.sum(); return dict(slope=slope.item(), top10=(f[:10].sum() / tot).item(), top100=(f[:100].sum() / tot).item(), distinct=int(f.numel()))
for vn, V in [("own", A), ("rot", rotate(A, seed=7))]:
    out = {}
    for k in (16, 64):
        cnt, sel = usage(V, k); z = zipf(cnt); top = cnt.topk(10).indices
        z["top_words"] = [dict(type=int(typ[i]), block=int(blk[i]), uses=int(cnt[i]), mshare=float(mshare[i]) if vn == "own" else None) for i in top.tolist()]
        z["top10_mean_mshare"] = float(mshare[top].mean()) if vn == "own" else None
        if k == 16:   # functional weight of the most frequent words: drop them from the 16-word descriptions and refit
            Vu = unitr(V); keep_top = ~torch.isin(sel, top); used = torch.unique(sel); g = torch.Generator(device=DEV).manual_seed(0)
            rnd = used[torch.randperm(used.numel(), device=DEV, generator=g)[:10]]; keep_rnd = ~torch.isin(sel, rnd)
            for lbl, keep in [("full", torch.ones_like(sel, dtype=torch.bool)), ("drop_top10", keep_top), ("drop_rand10", keep_rnd)]:
                As = Vu[sel] * keep[..., None].float(); Gm = As @ As.transpose(1, 2) + 1e-5 * torch.eye(sel.shape[1], device=DEV)
                cof = torch.cholesky_solve(As @ lv.Xc[:, :, None], torch.linalg.cholesky(Gm))[:, :, 0]   # dropped words are zero vectors: coefficient 0
                Xh = torch.einsum("nk,nkd->nd", cof, As); z[f"rec_{lbl}"] = lv.recovered(lv.splice(lv.mu + Xh))[0]; del As, Gm
            z["share_of_positions_using_top10"] = (~keep_top).any(-1).float().mean().item()
        out[str(k)] = z; del sel
        log(f"{tag} {vn} k{k}: Zipf slope {z['slope']:.2f} top10 {z['top10']:.3f} top100 {z['top100']:.3f} distinct {z['distinct']}"
            + (f" | top10 mean M-share {z['top10_mean_mshare']:.2f} types {[w['type'] for w in z['top_words']]}" if vn == "own" else "")
            + (f" | rec full {z['rec_full']:.2f} drop top10 {z['rec_drop_top10']:.2f} drop rand10 {z['rec_drop_rand10']:.2f} (positions using a top-10 word {z['share_of_positions_using_top10']:.2f})" if k == 16 else ""))
    res["vocab"][vn] = out
o, r_ = res["vocab"]["own"], res["vocab"]["rot"]
summ = (f"{tag} k16 own/rot: slope {o['16']['slope']:.2f}/{r_['16']['slope']:.2f} top10 {o['16']['top10']:.3f}/{r_['16']['top10']:.3f} distinct {o['16']['distinct']}/{r_['16']['distinct']} | "
        f"own top-10 words' mean M-share {o['16']['top10_mean_mshare']:.2f} (chance {8 / arch.D:.3f}); dropping them: {o['16']['rec_full']:.2f} -> {o['16']['rec_drop_top10']:.2f} (10 random used words: {o['16']['rec_drop_rand10']:.2f}) | top-8 var {res['top8_var']:.2f}")
log(summ); record(f"e414_zipf_{tag}", res, summ)
