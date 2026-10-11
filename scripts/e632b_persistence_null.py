"""e632b (session 128): is the ticket at initialisation only persistence? A row keeps part of its initial direction, and a
class's direction is correlated with its speaker's final direction, so a random initial direction can look aligned with
the class merely by surviving into the final row. For every eventual word of e632 (classes from e632_classes): the
persistence c = cos(initial row, final row); the alignment of the initial row with the class direction; the alignment
of the initial row's component orthogonal to the final row (the residual) with the class direction, ranked among all
rows' residuals; and a persistence-matched null, in which every row's initial direction is replaced by a synthetic one
with the same cosine to its own final row and a random orthogonal remainder, ranked the same way (five draws). If the
residual rank is at chance and the null reproduces the observed rank, the ticket is persistence and nothing else.
Arguments: tag."""
import sys, os, time, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s101_common import *
import wdd_common
wdd_common.MODELS.update({"pythia160s1": ("EleutherAI/pythia-160m-seed1", "neox"), "pythia160s2": ("EleutherAI/pythia-160m-seed2", "neox")}); MID.update({"pythia160s1": 6, "pythia160s2": 6})
tag = sys.argv[1]; t0 = time.time(); torch.set_grad_enabled(False); B = 6; last = 16000
ck = torch.load(f"/workspace/wdd/cache/e629_{tag}_B6/step{last}.pt", map_location="cpu"); ck1 = torch.load(f"/workspace/wdd/cache/e629_{tag}_B6/step1000.pt", map_location="cpu")
keepc = ck["keep"] & ck1["keep"]
for n in range(2000, 16000, 1000): keepc &= torch.load(f"/workspace/wdd/cache/e629_{tag}_B6/step{n}.pt", map_location="cpu")["keep"]
kidx = torch.nonzero(keepc)[:, 0]; N = int(keepc.sum()); X = ck["X"][keepc].float().to(DEV); U16 = unitr(X - X.mean(0)); A16 = ck["rows"].float().to(DEV); m = A16.shape[0]
cl = torch.load(f"/workspace/wdd/results/e632_classes/{tag}.pt"); pos_now = torch.full((keepc.numel(),), -1, dtype=torch.long); pos_now[kidx] = torch.arange(N)
model, _, fam = load_model(tag, revision="step0"); arch = Arch(model, fam); A0, _ = rows_of(arch, B); del model
c_all = (A0 * A16).sum(1); RES = unitr(A0 - c_all[:, None] * A16)   # every row's initial component orthogonal to its own final direction
g = torch.Generator(device=DEV).manual_seed(0); NULLS = []
for k in range(5):
    Z = torch.randn(m, A0.shape[1], device=DEV, generator=g); Z = Z - (Z * A16).sum(1, keepdim=True) * A16; Z = unitr(Z); NULLS.append(c_all[:, None] * A16 + (1 - c_all.pow(2)).clamp_min(0).sqrt()[:, None] * Z)
rk = lambda v, w: int((v > v[w]).sum()) + 1; out = []
for w, full in cl["classes"].items():
    idx = pos_now[full]; idx = idx[idx >= 0]
    if idx.numel() < 10: continue
    d = unitr(U16[idx.to(DEV)].mean(0, keepdim=True))[0]; w = int(w)
    cos0 = A0 @ d; cosF = A16 @ d; cosR = RES @ d; nulls = [(Nk @ d) for Nk in NULLS]
    out.append(dict(w=w, persistence=float(c_all[w]), cos0=float(cos0[w]), cosF=float(cosF[w]), path=float(c_all[w] * cosF[w]), cos_res=float(cosR[w]), rank0=rk(cos0, w) / m, rankF=rk(cosF, w) / m, rank_res=rk(cosR, w) / m, rank_null=mean([rk(nk, w) / m for nk in nulls]), top100_res=float(rk(cosR, w) <= 100), top1000_res=float(rk(cosR, w) <= 1000), top1000_null=mean([float(rk(nk, w) <= 1000) for nk in nulls]), top1000_0=float(rk(cos0, w) <= 1000)))
res = dict(model=tag, n=len(out), m=m, persistence_tracked=med([o["persistence"] for o in out]), persistence_all=float(c_all.median()), cos0_median=med([o["cos0"] for o in out]), path_median=med([o["path"] for o in out]), cos_res_median=med([o["cos_res"] for o in out]), cosF_median=med([o["cosF"] for o in out]),
           rank0=med([o["rank0"] for o in out]), rankF=med([o["rankF"] for o in out]), rank_res=med([o["rank_res"] for o in out]), rank_null=med([o["rank_null"] for o in out]), top1000_0=mean([o["top1000_0"] for o in out]), top1000_res=mean([o["top1000_res"] for o in out]), top1000_null=mean([o["top1000_null"] for o in out]), top100_res=mean([o["top100_res"] for o in out]), per=out)
summ = (f"{tag} ({len(out)} eventual words, {m} rows): persistence of the recruits' rows from step 0 to 16000 {res['persistence_tracked']:.2f} (all rows {res['persistence_all']:.2f}); the initial row's cosine with the class direction {res['cos0_median']:.3f}, of which the path through the final row explains {res['path_median']:.3f} (final row's cosine with the class direction {res['cosF_median']:.2f}); the residual initial component's cosine {res['cos_res_median']:.3f}; normalised rank of the initial row {res['rank0']:.2f}, of the residual {res['rank_res']:.2f}, under the persistence-matched null {res['rank_null']:.2f}, of the final row {res['rankF']:.2f}; top 1000 initial {res['top1000_0']:.2f}, residual {res['top1000_res']:.2f}, null {res['top1000_null']:.2f}; top 100 residual {res['top100_res']:.2f} | {time.time() - t0:.0f}s")
log(summ); record(f"e632b_persistence_{tag}", res, summ)
