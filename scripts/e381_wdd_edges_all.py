"""e381: WDD as a general circuit-edge finder. For every head of layers 1 and up and each of its query, key and value
inputs, on natural text (4 x 256 tokens): the exact share of that input contributed by every upstream head (from the
actual per-head writes) against the WDD share (OMP, k = 64, over the model's own write atoms, of the centred state at
that layer, each atom's coefficient times its share of the input; one WDD reading per layer serves every reader in it).
Reported over all readers: how often WDD's top upstream head is the exact top one, the overlap of their top-3 sets,
the rank correlation over upstream heads, and the same for a plain WDD ranking that ignores the reader; restricted to
the edges that matter (the reader's exact top upstream head carries at least 10% of the input)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p3_common import *
tag = sys.argv[1]; c = Cache(tag); model, tok, fam = load_eager(c.name); arch = Arch(model, fam); NB, NH, HD = arch.NB, arch.NH, arch.HD; torch.manual_seed(0)
nat = c.s["eval_ids"][:4, :256].to(DEV); cap = Capture(model, arch); X, Z, M, out = cap(nat); P = nat.shape[0] * (nat.shape[1] - 1)
keep = torch.ones(nat.shape[0], nat.shape[1], dtype=torch.bool, device=DEV); keep[:, 0] = False   # drop the first position (attention sink)
rows = lambda T: T[keep].reshape(-1, T.shape[-1])
def spear(a, b): return torch.corrcoef(torch.stack([a.argsort().argsort().float(), b.argsort().argsort().float()]))[0, 1].item()
recs = []
for l in range(1, NB):
    Xr = rows(X[l]); A, lab = c.dictionary(l - 1, blocks=list(range(l))); mu = c.s["mu"][l].to(DEV).float(); sel, cof, _ = omp(Xr - mu[None], A, 64, batch=512, record_err=False)
    typ, blk, idx = lab["type"].to(DEV)[sel], lab["block"].to(DEV)[sel], lab["index"].to(DEV)[sel]; ups = [(b, h) for b in range(l) for h in range(NH)]
    gid = torch.where(typ == T_ATT, blk * NH + idx, torch.full_like(blk, -1)); plain = torch.zeros(len(ups), device=DEV)
    for g in range(len(ups)): plain[g] = (cof.abs() * (gid == g)).sum(-1).mean()
    Zr = {b: rows(Z[b]) for b in range(l)}; Wo = {b: arch.wo(b).to(DEV) for b in range(l)}
    for h in range(NH):
        for w in ("Q", "K", "V"):
            r, u = reader_dirs(arch, l, h, w, Xr)
            ex = torch.cat([(Zr[b] * (r @ Wo[b].T)).view(-1, NH, HD).sum(-1).mean(0) for b in range(l)])
            proj = (A[sel] * r[:, None, :]).sum(-1) * cof; wd = torch.zeros(len(ups), device=DEV)
            for g in range(len(ups)): wd[g] = (proj * (gid == g)).sum(-1).mean()
            et, wt, pt = ex.argsort(descending=True)[:3].tolist(), wd.argsort(descending=True)[:3].tolist(), plain.argsort(descending=True)[:3].tolist()
            recs.append(dict(reader=[l, h], which=w, exact_top=ups[et[0]], exact_top_share=ex[et[0]].item(), top1_match=et[0] == wt[0], top3_overlap=len(set(et) & set(wt)) / 3, spearman=spear(wd.cpu(), ex.cpu()) if len(ups) > 2 else float("nan"), plain_top1_match=et[0] == pt[0], plain_top3_overlap=len(set(et) & set(pt)) / 3, plain_spearman=spear(plain.cpu(), ex.cpu()) if len(ups) > 2 else float("nan"), n_up=len(ups)))
def agg(rr):
    n = len(rr); m = lambda k: sum(r[k] for r in rr) / max(n, 1)
    return dict(n=n, top1=m("top1_match"), top3=m("top3_overlap"), spearman=m("spearman"), plain_top1=m("plain_top1_match"), plain_top3=m("plain_top3_overlap"), plain_spearman=m("plain_spearman"), chance_top1=sum(1 / r["n_up"] for r in rr) / max(n, 1))
res = dict(model=tag, all=agg(recs), strong=agg([r for r in recs if r["exact_top_share"] >= 0.1]), by_input={w: agg([r for r in recs if r["which"] == w]) for w in ("Q", "K", "V")}, records=recs)
f = lambda d: f"n {d['n']}: WDD top-1 match {d['top1']:.2f} (plain {d['plain_top1']:.2f}, chance {d['chance_top1']:.3f}), top-3 overlap {d['top3']:.2f} (plain {d['plain_top3']:.2f}), Spearman {d['spearman']:+.2f} (plain {d['plain_spearman']:+.2f})"
log(f"{tag}: ALL readers {f(res['all'])} | STRONG edges {f(res['strong'])} | " + " | ".join(f"{w}: {f(d)}" for w, d in res["by_input"].items()))
record(f"e381_wddedges_{tag}", res, f"strong edges: top1 {res['strong']['top1']:.2f} (plain {res['strong']['plain_top1']:.2f}) top3 {res['strong']['top3']:.2f} rho {res['strong']['spearman']:+.2f}")
