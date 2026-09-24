"""e435: is the self-description advantage a property of the states' sparse structure or only of their covariance?
Every control so far changed the vocabulary (rotation, covA, covX, mix8). Here the states change and the vocabulary
stays. Per model, middle depth, typical positions of 6 sequences (sinks excluded), four state sets:
 real:     the centred states;
 gauss:    Gaussian states with the same covariance;
 shuffled: each principal coordinate permuted independently across positions (same covariance and the same marginal
           distribution along every principal direction, joint structure destroyed);
 lexical:  the token's mean state (from 64 other sequences) plus Gaussian within-token noise with the pooled
           within-token covariance (keeps only the token-identity structure; e433 asks whether M is that channel).
Vocabularies: own words (blocks 0..L), rotation, covA (Gaussian words with the own vocabulary's second moment), mix8.
Measured: fraction unexplained at k 4, 16 and 32 (OMP); for each state set, the own-over-rotation advantage and the
own-over-covA (word-level) advantage; the overlap of the 200 most used own words with the real states' (Jaccard); and
the Zipf slope of own-word usage.
Pre-registered:
- the own-over-rotation advantage at k 16 on gauss is under half of that on real, in all five (the advantage is sparse
  structure, not alignment with the covariance);
- own over covA on gauss is near zero (under a quarter of real);
- lexical keeps over half of real's advantage."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
name = sys.argv[1]; model, tok, fam = load_model(name); arch = Arch(model, fam); L = arch.NB // 2; D = arch.D
EA = torch.load("/workspace/wdd/eval_ids_all.pt", weights_only=False)[name]; ev = EA["eval_ids"][:6].to(DEV); cen = EA["cen_ids"].to(DEV); del EA
xe = block_states(model, arch, ev, [L], chunk=3)[L]; xc = block_states(model, arch, cen, [L], chunk=4)[L]
ne_, nc = ~sinkmask(xe), ~sinkmask(xc)
X = xe[ne_]; mu = X.mean(0); Xr = X - mu; N = Xr.shape[0]; C = torch.cov(Xr.T.double(), correction=0)
evl, U = torch.linalg.eigh(C); R = ((U * evl.clamp_min(0).sqrt()) @ U.T).float(); g = torch.Generator(device=DEV).manual_seed(0)
Xg = torch.randn(N, D, device=DEV, generator=g) @ R
Y = Xr @ U.float(); Ys = torch.stack([Y[torch.randperm(N, device=DEV, generator=g), j] for j in range(D)], 1); Xs = Ys @ U.float().T; del Y, Ys
Xcc = xc[nc]; vc = cen[:, 1:][nc]; uniq, inv, cnt = torch.unique(vc, return_inverse=True, return_counts=True)
tmean = torch.zeros(len(uniq), D, device=DEV).index_add_(0, inv, Xcc) / cnt[:, None]; mu_c = Xcc.mean(0)
ve = ev[:, 1:][ne_]; ix = torch.searchsorted(uniq, ve).clamp_max(len(uniq) - 1); known = (uniq[ix] == ve) & (cnt[ix] >= 3)
Mt = torch.where(known[:, None], tmean[ix], mu_c[None]); Wc = torch.cov((X - Mt).T.double(), correction=0); ew, Uw = torch.linalg.eigh(Wc)
Rw = ((Uw * ew.clamp_min(0).sqrt()) @ Uw.T).float(); Xl = (Mt - Mt.mean(0)) + torch.randn(N, D, device=DEV, generator=g) @ Rw
del xc, Xcc, tmean
sets = dict(real=Xr, gauss=Xg, shuffled=Xs, lexical=Xl)
A, lab = build_dictionary(arch, blocks=list(range(L + 1))); typ = lab["type"].to(DEV)
groups = [torch.nonzero(typ == t)[:, 0] for t in (T_TOK, T_POS, T_MLP, T_ATT, T_BIAS) if (typ == t).any()]
vocabs = dict(own=A, rot=rotate(A, seed=7), covA=gauss_like(A.shape[0], (A.T @ A) / A.shape[0], seed=1), mix8=mixtures(A, groups, m=8, seed=4))
KS = [4, 16, 32]; res = dict(model=name, level=L, n=N, known_token_share=known.float().mean().item(), fvu={}, usage={})
def zipf(cnt_):
    f = cnt_.sort(descending=True).values; f = f[f > 0]; r = torch.arange(1, f.numel() + 1, device=DEV).float(); hi = min(1000, f.numel())
    x, y = r[9:hi].log(), f[9:hi].log(); return (((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()).item()
tops = {}
for sn, S_ in sets.items():
    tot = S_.pow(2).sum(); res["fvu"][sn] = {}
    for vn, V in vocabs.items():
        sel, _, err = omp(S_, unitr(V), max(KS), batch=256, record_err=True)
        res["fvu"][sn][vn] = {str(k): (err[:, k - 1].sum() / tot).item() for k in KS}
        if vn == "own":
            cnt_ = torch.bincount(sel[:, :16].flatten(), minlength=V.shape[0]).float(); tops[sn] = set(cnt_.topk(200).indices.tolist()); res["usage"][sn] = dict(zipf=zipf(cnt_), distinct=int((cnt_ > 0).sum()))
        del sel, err
    f = res["fvu"][sn]
    log(f"{name} {sn}: FVU k16 " + " ".join(f"{vn} {f[vn]['16']:.3f}" for vn in vocabs) + f" | own over rot {f['rot']['16'] - f['own']['16']:+.3f}, over covA {f['covA']['16'] - f['own']['16']:+.3f}")
for sn in sets: res["usage"][sn]["jaccard_top200_with_real"] = len(tops[sn] & tops["real"]) / len(tops[sn] | tops["real"])
adv = {sn: {k: dict(rot=res["fvu"][sn]["rot"][k] - res["fvu"][sn]["own"][k], covA=res["fvu"][sn]["covA"][k] - res["fvu"][sn]["own"][k], mix8=res["fvu"][sn]["mix8"][k] - res["fvu"][sn]["own"][k]) for k in map(str, KS)} for sn in sets}
res["advantage"] = adv; a16 = {sn: adv[sn]["16"] for sn in sets}
res["checks"] = dict(gauss_under_half=a16["gauss"]["rot"] < 0.5 * a16["real"]["rot"], gauss_wordlevel_under_quarter=a16["gauss"]["covA"] < 0.25 * a16["real"]["covA"], lexical_over_half=a16["lexical"]["rot"] > 0.5 * a16["real"]["rot"])
summ = (f"{name} L{L}: k16 FVU advantage of own words over rotation / over covA / over mix8: " + " ; ".join(f"{sn} {a16[sn]['rot']:+.3f}/{a16[sn]['covA']:+.3f}/{a16[sn]['mix8']:+.3f}" for sn in sets)
        + " | k4 over rotation: " + " ".join(f"{sn} {adv[sn]['4']['rot']:+.3f}" for sn in sets) + " | own FVU k16: " + " ".join(f"{sn} {res['fvu'][sn]['own']['16']:.3f}" for sn in sets)
        + " | Zipf slope " + " ".join(f"{sn} {res['usage'][sn]['zipf']:.2f}" for sn in sets) + " | top-200 overlap with real " + " ".join(f"{sn} {res['usage'][sn]['jaccard_top200_with_real']:.2f}" for sn in sets if sn != "real")
        + f" | checks {json.dumps(res['checks'])}")
log(summ); record(f"e435_surrogates_{name}", res, summ)
