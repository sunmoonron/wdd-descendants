"""e375: WDD read through a circuit's reader. The known mechanism of induction: a previous-token head writes, at each
position, information about the preceding token; an induction head's keys read it (K-composition). For each
attention-defined induction head R (reader) of a model, at the key positions its induction attention uses, the key
vector is decomposed additively over the residual state (frozen-scale input norm), giving every upstream component's
share of R's key: (1) EXACT shares from the actual per-head outputs, MLP outputs and embeddings; (2) WDD shares: the
original WDD reading (OMP, k = 64, over the model's own write atoms: token and position embeddings, MLP rows and the
per-head write bases of every block below R) of the centred state, each selected atom's coefficient times its share
of the key; (3) PLAIN WDD: which heads carry the most coefficient mass in the state, ignoring the reader. Ground truth
labels: previous-token heads by attention pattern. Controls: a non-induction head of the same layer as reader. For
GPT-2 also the IOI edge: name movers' queries at the final position against the S-inhibition heads."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p3_common import *
tag = sys.argv[1]; c = Cache(tag); model, tok, fam = load_eager(c.name); arch = Arch(model, fam); NB, NH, HD = arch.NB, arch.NH, arch.HD; torch.manual_seed(0); rng = random.Random(0)
ind, off, half = induction_batch(tok, c, n=8, half=128, seed=0); pref, prev = attention_scores(model, ind, off, half, NB, NH); pf, pv = pref.cpu(), prev.cpu()
cand = [(l, h) for l in range(1, NB) for h in range(NH) if pf[l, h] >= 0.2 and pf[l, h] >= pv[l, h]]; readers = sorted(cand, key=lambda x: -pf[x].item())[:4]
PREV = set((l, h) for l in range(NB) for h in range(NH) if pv[l, h] >= 0.3)
cap = Capture(model, arch); X, Z, M, out = cap(ind)
kpos = torch.arange(off + 2, off + half, device=DEV)   # key positions the induction attention uses (off + j + 1, j = 1 .. half - 2)
def rows(T): return T[:, kpos].reshape(-1, T.shape[-1])
def groups_exact(l_R, r):
    """exact share of every upstream component: dict group -> mean share over positions"""
    sh = {}
    E = rows(X[0]); sh["emb"] = (E * r).sum(-1)
    for b in range(l_R):
        Wo = arch.wo(b).to(DEV); Rb = r @ Wo.T; z = rows(Z[b]); s = (z * Rb).view(-1, NH, HD).sum(-1)
        for h in range(NH): sh[(b, h)] = s[:, h]
        ab = arch.attn_bias(b)
        if ab is not None: sh[("abias", b)] = (ab.to(DEV)[None] * r).sum(-1)
        sh[("mlp", b)] = (rows(M[b]) * r).sum(-1)
    return sh
def wdd_shares(l_R, r, Xr):
    A, lab = c.dictionary(l_R - 1, blocks=list(range(l_R))); mu = c.s["mu"][l_R].to(DEV).float(); Xc = Xr - mu[None]
    sel, cof, err = omp(Xc, A, 64, batch=512, record_err=False); proj = (A[sel] * r[:, None, :]).sum(-1) * cof  # [P, 64] per-atom share
    typ, blk, idx = lab["type"].to(DEV)[sel], lab["block"].to(DEV)[sel], lab["index"].to(DEV)[sel]; sh, mass = {}, {}
    for b in range(l_R):
        for h in range(NH):
            m_ = (typ == T_ATT) & (blk == b) & (idx == h); sh[(b, h)] = (proj * m_).sum(-1); mass[(b, h)] = (cof.abs() * m_).sum(-1)
        m_ = (typ == T_MLP) & (blk == b); sh[("mlp", b)] = (proj * m_).sum(-1)
    m_ = (typ == T_TOK) | (typ == T_POS); sh["emb"] = (proj * m_).sum(-1)
    fve = 1 - ((Xc - (cof[:, :, None] * A[sel]).sum(1)) ** 2).sum(-1) / (Xc ** 2).sum(-1).clamp_min(1e-9)
    return sh, mass, fve.mean().item()
def summarise(sh_ex, sh_w, mass, l_R, truth):
    heads = [(b, h) for b in range(l_R) for h in range(NH)]; ex = torch.tensor([sh_ex[x].mean().item() for x in heads]); wd = torch.tensor([sh_w[x].mean().item() for x in heads]); ms = torch.tensor([mass[x].mean().item() for x in heads])
    def top(v, k=3): return [list(heads[i]) for i in v.argsort(descending=True)[:k].tolist()]
    def rank_truth(v): o = v.argsort(descending=True).tolist(); rk = [o.index(heads.index(t)) for t in truth if t in heads]; return min(rk) + 1 if rk else None
    tmask = torch.tensor([x in truth for x in heads])
    rs = lambda a, b: torch.corrcoef(torch.stack([a.argsort().argsort().float(), b.argsort().argsort().float()]))[0, 1].item()
    return dict(exact_top3=top(ex), wdd_reader_top3=top(wd), plain_wdd_top3=top(ms), exact_truth_share=ex[tmask].sum().item() if tmask.any() else None, wdd_truth_share=wd[tmask].sum().item() if tmask.any() else None,
                exact_rank_best_truth=rank_truth(ex), wdd_reader_rank_best_truth=rank_truth(wd), plain_wdd_rank_best_truth=rank_truth(ms), spearman_wdd_vs_exact=rs(wd, ex), n_upstream_heads=len(heads),
                exact_mlp_share=sum(sh_ex[("mlp", b)].mean().item() for b in range(l_R)), exact_emb_share=sh_ex["emb"].mean().item(), wdd_mlp_share=sum(sh_w[("mlp", b)].mean().item() for b in range(l_R)), wdd_emb_share=sh_w["emb"].mean().item())
res = dict(model=tag, readers=[list(x) for x in readers], previous_token_heads=sorted([list(x) for x in PREV]), induction={}, control={})
for R in readers:
    l_R, h_R = R; Xr = rows(X[l_R]); truth = [t for t in PREV if t[0] < l_R]
    r, u = reader_dirs(arch, l_R, h_R, "K", Xr); ex = groups_exact(l_R, r); tot = sum(v.mean().item() for v in ex.values())
    w, mass, fve = wdd_shares(l_R, r, Xr); s = summarise(ex, w, mass, l_R, truth); s.update(exact_total_share=tot, wdd_fve=fve, n_truth_upstream=len(truth)); res["induction"][f"{l_R}.{h_R}"] = s
    others = [h for h in range(NH) if (l_R, h) not in cand and h != h_R]
    if others:
        hc = rng.choice(others); rc, _ = reader_dirs(arch, l_R, hc, "K", Xr); exc = groups_exact(l_R, rc); wc, massc, _ = wdd_shares(l_R, rc, Xr); res["control"][f"{l_R}.{hc}"] = summarise(exc, wc, massc, l_R, truth)
if tag == "gpt2":
    groups, _ = ioi_groups(tok, n_per=16); SIN = [(7, 3), (7, 9), (8, 6), (8, 10)]; res["ioi"] = {}
    for R in [(9, 9), (9, 6), (10, 0)]:
        acc = None
        for ids, io, s_ in groups:
            Xg, Zg, Mg, _ = cap(ids); last = torch.tensor([ids.shape[1] - 1], device=DEV)
            Xr = Xg[R[0]][:, -1]; r, u = reader_dirs(arch, R[0], R[1], "Q", Xr)
            sh = {"emb": (Xg[0][:, -1] * r).sum(-1)}
            for b in range(R[0]):
                Wo = arch.wo(b).to(DEV); s2 = (Zg[b][:, -1] * (r @ Wo.T)).view(-1, NH, HD).sum(-1)
                for h in range(NH): sh[(b, h)] = s2[:, h]
                sh[("mlp", b)] = (Mg[b][:, -1] * r).sum(-1)
            A, lab = c.dictionary(R[0] - 1, blocks=list(range(R[0]))); mu = c.s["mu"][R[0]].to(DEV).float(); sel, cof, _ = omp(Xr - mu[None], A, 64, batch=512, record_err=False); proj = (A[sel] * r[:, None, :]).sum(-1) * cof
            typ, blk, idx = lab["type"].to(DEV)[sel], lab["block"].to(DEV)[sel], lab["index"].to(DEV)[sel]; wsh = {}; ms = {}
            for b in range(R[0]):
                for h in range(NH): m_ = (typ == T_ATT) & (blk == b) & (idx == h); wsh[(b, h)] = (proj * m_).sum(-1); ms[(b, h)] = (cof.abs() * m_).sum(-1)
                wsh[("mlp", b)] = (proj * ((typ == T_MLP) & (blk == b))).sum(-1)
            wsh["emb"] = (proj * ((typ == T_TOK) | (typ == T_POS))).sum(-1)
            acc = (sh, wsh, ms) if acc is None else tuple({k: torch.cat([a[k], b_[k]]) for k in a} for a, b_ in zip(acc, (sh, wsh, ms)))
        res["ioi"][f"{R[0]}.{R[1]}"] = summarise(acc[0], acc[1], acc[2], R[0], SIN)
def fmt(d): return f"exact top {d['exact_top3']} | WDD-reader top {d['wdd_reader_top3']} | plain WDD top {d['plain_wdd_top3']} | best truth rank exact {d['exact_rank_best_truth']} / WDD {d['wdd_reader_rank_best_truth']} / plain {d['plain_wdd_rank_best_truth']} of {d['n_upstream_heads']} | truth share exact {d['exact_truth_share'] if d['exact_truth_share'] is None else round(d['exact_truth_share'], 3)} WDD {d['wdd_truth_share'] if d['wdd_truth_share'] is None else round(d['wdd_truth_share'], 3)} | Spearman WDD vs exact {d['spearman_wdd_vs_exact']:+.2f}"
log(f"{tag}: readers {res['readers']}, previous-token heads {res['previous_token_heads']}")
for k, d in res["induction"].items(): log(f"  INDUCTION reader {k} (fve {d['wdd_fve']:.2f}, exact shares sum {d['exact_total_share']:.2f}): " + fmt(d))
for k, d in res["control"].items(): log(f"  CONTROL reader {k}: " + fmt(d))
for k, d in res.get("ioi", {}).items(): log(f"  IOI name-mover query {k}: " + fmt(d))
agg = lambda key, part="induction": [d[key] for d in res[part].values() if d[key] is not None]
m = lambda v: sum(v) / len(v) if v else float("nan")
res["summary"] = dict(induction_exact_best_truth_rank_mean=m(agg("exact_rank_best_truth")), induction_wdd_best_truth_rank_mean=m(agg("wdd_reader_rank_best_truth")), induction_plain_best_truth_rank_mean=m(agg("plain_wdd_rank_best_truth")), induction_spearman_mean=m(agg("spearman_wdd_vs_exact")), control_exact_best_truth_rank_mean=m(agg("exact_rank_best_truth", "control")), induction_exact_truth_share_mean=m(agg("exact_truth_share")), control_exact_truth_share_mean=m(agg("exact_truth_share", "control")))
record(f"e375_readerwdd_{tag}", res, " | ".join(f"{k} {v:.2f}" for k, v in res["summary"].items()))
