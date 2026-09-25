"""e476: audit of the native-only profile signal of e475 (H299). At the deepest depth tested, the shape of a native
16-word description's coefficient profile (its sorted, normalised magnitudes, with no word identities) predicted the
similarity of two positions' next-token distributions beyond activation distance and the entropy gap, at partial rank
correlation +0.13 to +0.14 in three models, while rotated words' profiles predicted nothing. Before that number is built
on, this run asks whether it is a property of the description or a proxy for something plainer.
Setup as e475 (8 x 256 evaluation tokens, about 2040 typical positions, all pairs; behaviour distance d_B = squared
Hellinger distance of the next-token distributions; activation distance d_X = 1 - cosine of the centred states), at the
middle and the three-quarter depths.
Per position: the native description at k = 8, 16 and 32 (nested OMP supports, refitted); the rotated description; the
top-16 principal components (fitted on other sequences). Profile statistics: the sorted L2-normalised magnitudes (e475's
metric), the sorted L1-normalised magnitudes, the top word's share, and the effective number of words.
Candidate causes, per position: the state's norm; its energy share in the top 8 principal directions (M); the next-token
entropy and top probability; the position in the sequence; the log frequency of the current and next token; whether the
next token copies an earlier one; the description's unexplained fraction (FVU); the prominence of the top word (its
projection over the state's norm); the top word's type (token embedding, MLP row, head basis), and whether it is the
current token's own embedding.
Analysis: (a) per position, Spearman of the native top-word share with each candidate, against the rotated share;
(b) pairwise, the partial rank correlation of each profile distance with d_B given d_X, then given d_X and each
candidate's gap, then given d_X and all gaps at once (with indicators for the same current token, the same top word and
the same top-word type).
Models (argument): gpt2, qwen05, smollm2.
Pre-registered (honest guesses):
- the signal is a proxy: given d_X and all candidate gaps, the deepest depth's partial falls below 0.05 (0.5);
- the native top-word share tracks whether the top word is the current token's embedding (Spearman above 0.3) (0.5);
- it survives L1 normalisation and the top-share-only distance (partials at least 0.10) (0.6);
- it survives k = 8 and k = 32 (partials at least 0.10) (0.5)."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; model, tok, fam = load_model(name); arch = Arch(model, fam); NB = arch.NB
E = eval_ids(name); ids = E[:8, :256].to(DEV); fit = E[8:16, :256].to(DEV); B_, T_ = ids.shape
DEPTHS = sorted({NB // 2, (3 * NB) // 4})
S_ = block_states(model, arch, ids, DEPTHS, chunk=4); SF = block_states(model, arch, fit, DEPTHS, chunk=4)
keep = torch.ones(B_ * (T_ - 1), dtype=torch.bool, device=DEV)
for b in DEPTHS: keep &= ~sinkmask(S_[b].reshape(-1, arch.D))
N = int(keep.sum()); rows = []
for s0 in range(0, B_, 2):
    with torch.no_grad(): lg = model(ids[s0:s0 + 2]).logits[:, 1:].float()
    k2 = keep.view(B_, T_ - 1)[s0:s0 + 2].reshape(-1); rows.append(torch.softmax(lg, -1).reshape(-1, lg.shape[-1])[k2].sqrt()); del lg
Sq = torch.cat(rows); dB = (1 - Sq @ Sq.T).clamp_min(0); Pk = Sq.pow(2); Hn = -(Pk * Pk.clamp_min(1e-30).log()).sum(-1); top_p = Pk.max(1).values; del Sq, rows, Pk
# per-position covariates that do not depend on depth
bi = torch.arange(B_, device=DEV)[:, None].expand(B_, T_ - 1).reshape(-1)[keep]; tj = torch.arange(1, T_, device=DEV)[None].expand(B_, T_ - 1).reshape(-1)[keep]
cur = ids[bi, tj]; has_next = tj + 1 < T_; nxt = torch.where(has_next, ids[bi, (tj + 1).clamp_max(T_ - 1)], torch.full_like(cur, -1))
cnt = torch.bincount(E.reshape(-1).to(DEV), minlength=int(E.max()) + 1).float()
logf_cur = (cnt[cur] + 1).log(); logf_nxt = torch.where(has_next, (cnt[nxt.clamp_min(0)] + 1).log(), logf_cur.median().expand_as(logf_cur))
copy = ((ids[bi] == nxt[:, None]) & (torch.arange(T_, device=DEV)[None] <= tj[:, None])).any(1) & has_next
iu = torch.triu_indices(N, N, 1, device=DEV); pick = lambda M: M[iu[0], iu[1]]
def rank(v):
    u, inv, c_ = torch.unique(v, return_inverse=True, return_counts=True); e_ = c_.cumsum(0).double(); return ((2 * e_ - c_.double() - 1) / 2)[inv].float()
def corr(a, b): a = a - a.mean(); b = b - b.mean(); return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-9))
def resid_on(a, cols):                                                 # least squares on [1, cols] by normal equations with a pseudo-inverse (constant columns are harmless)
    cols = [c - c.mean() for c in cols if float(c.std()) > 0]; Xd = torch.stack([torch.ones_like(a)] + cols, 1).double(); y = a.double()
    beta = torch.linalg.pinv(Xd.T @ Xd) @ (Xd.T @ y); return (y - Xd @ beta).float()
def partial(rv, rB, cols): return corr(resid_on(rv, cols), resid_on(rB, cols))
def cosd(M): U = unitr(M); return 1 - U @ U.T
def gap(v): return rank(pick((v[:, None] - v[None, :]).abs().float()))
def same(v): return rank(pick((v[:, None] == v[None, :]).float()))
def prof(cof):
    a = cof.abs(); l2 = torch.sort(a / a.norm(dim=-1, keepdim=True).clamp_min(1e-9), 1, descending=True).values; l1 = torch.sort(a / a.sum(-1, keepdim=True).clamp_min(1e-9), 1, descending=True).values
    return dict(sorted_l2=l2, sorted_l1=l1, top1=l2[:, 0], neff=a.sum(-1).pow(2) / a.pow(2).sum(-1).clamp_min(1e-9))
pB = pick(dB); rB = rank(pB); rH = gap(Hn)
res = dict(model=name, depths=DEPTHS, n_positions=N, n_pairs=int(pB.numel()), by_depth={})
for b in DEPTHS:
    X = S_[b].reshape(-1, arch.D)[keep]; mu = X.mean(0); Xc = X - mu; nrm = Xc.norm(dim=-1)
    Ff = SF[b].reshape(-1, arch.D); Ff = Ff[~sinkmask(Ff)]; P16, _ = pcs(Ff, 16); Xf = Xc - (Ff.mean(0) - mu)
    mshare = (Xf @ P16[:, :8]).pow(2).sum(1) / Xf.pow(2).sum(1).clamp_min(1e-9)
    A, lab = build_dictionary(arch, blocks=list(range(b + 1))); Au = unitr(A); Ar = unitr(rotate(A, seed=7)); del A
    typ = lab["type"].to(DEV); idx = lab["index"].to(DEV)
    desc = {}
    for kind, Dct in (("native", Au), ("rotated", Ar)):
        sel, _, _ = omp(Xc, Dct, 32, batch=1024, record_err=False)
        for k in (8, 16, 32):
            cof, _ = refit(Xc, Dct, sel[:, :k]); rec = torch.einsum("nk,nkd->nd", cof, Dct[sel[:, :k]])
            desc[(kind, k)] = dict(sel=sel[:, :k], cof=cof, fvu=(Xc - rec).pow(2).sum(1) / Xc.pow(2).sum(1).clamp_min(1e-9), **prof(cof))
        desc[(kind, "prom")] = (Xc * Dct[sel[:, 0]]).sum(1).abs() / nrm.clamp_min(1e-9); desc[(kind, "top")] = sel[:, 0]
    pc = Xf @ P16; desc[("pca", 16)] = dict(cof=pc, fvu=1 - pc.pow(2).sum(1) / Xf.pow(2).sum(1).clamp_min(1e-9), **prof(pc))
    del Au, Ar; torch.cuda.empty_cache()
    top = desc[("native", "top")]; top_typ = typ[top]; is_tok_top = ((top_typ == T_TOK) & (idx[top] == cur)).float()
    cov = dict(norm=nrm, m_share=mshare, entropy=Hn, top_prob=top_p, position=tj.float(), logf_cur=logf_cur, logf_next=logf_nxt, copy=copy.float(),
               fvu_native=desc[("native", 16)]["fvu"], fvu_rotated=desc[("rotated", 16)]["fvu"], prominence=desc[("native", "prom")], top_is_token=is_tok_top, top_is_mlp=(top_typ == T_MLP).float())
    out = dict(top_word_types=dict(token=float((top_typ == T_TOK).float().mean()), mlp=float((top_typ == T_MLP).float().mean()), head=float((top_typ == T_ATT).float().mean()), current_token_embedding=float(is_tok_top.mean())),
               per_position={}, pairwise={})
    r_top = {kind: rank(desc[(kind, 16)]["top1"]) for kind in ("native", "rotated", "pca")}
    for cn, cv in cov.items(): out["per_position"][cn] = {kind: corr(r_top[kind], rank(cv)) for kind in ("native", "rotated", "pca")}
    rX = rank(pick(cosd(Xc))); gaps = {cn: gap(cv) for cn, cv in cov.items() if cn not in ("top_is_token", "top_is_mlp")}
    gaps.update(same_current_token=same(cur), same_top_word=same(top), same_top_type=same(top_typ), same_top_is_token=same(is_tok_top))
    allg = list(gaps.values())
    variants = {"M_native_l2_16": torch.cdist(desc[("native", 16)]["sorted_l2"], desc[("native", 16)]["sorted_l2"]), "M_native_l1_16": torch.cdist(desc[("native", 16)]["sorted_l1"], desc[("native", 16)]["sorted_l1"]),
                "top1_native_16": (desc[("native", 16)]["top1"][:, None] - desc[("native", 16)]["top1"][None, :]).abs(), "neff_native_16": (desc[("native", 16)]["neff"][:, None] - desc[("native", 16)]["neff"][None, :]).abs(),
                "M_native_l2_8": torch.cdist(desc[("native", 8)]["sorted_l2"], desc[("native", 8)]["sorted_l2"]), "M_native_l2_32": torch.cdist(desc[("native", 32)]["sorted_l2"], desc[("native", 32)]["sorted_l2"]),
                "M_rotated_l2_16": torch.cdist(desc[("rotated", 16)]["sorted_l2"], desc[("rotated", 16)]["sorted_l2"]), "M_pca_l2_16": torch.cdist(desc[("pca", 16)]["sorted_l2"], desc[("pca", 16)]["sorted_l2"]),
                "fvu_native_gap": (desc[("native", 16)]["fvu"][:, None] - desc[("native", 16)]["fvu"][None, :]).abs(), "prominence_gap": (desc[("native", "prom")][:, None] - desc[("native", "prom")][None, :]).abs()}
    for vn, M in variants.items():
        rv = rank(pick(M)); row = dict(spearman=corr(rv, rB), partial_X=partial(rv, rB, [rX]), partial_X_entropy=partial(rv, rB, [rX, rH]), partial_X_all=partial(rv, rB, [rX] + allg), partial_X_each={})
        if vn in ("M_native_l2_16", "M_rotated_l2_16", "M_pca_l2_16", "top1_native_16"):
            for cn, gv in gaps.items(): row["partial_X_each"][cn] = partial(rv, rB, [rX, gv])
        out["pairwise"][vn] = row
    res["by_depth"][b] = out
    m_ = out["pairwise"]["M_native_l2_16"]
    log(f"{name} depth {b}: top word types {out['top_word_types']} | native top-share vs covariates: " + ", ".join(f"{cn} {v['native']:+.2f} (rot {v['rotated']:+.2f})" for cn, v in out["per_position"].items())
        + f" | M_native partial given X {m_['partial_X']:+.3f}, given X+entropy {m_['partial_X_entropy']:+.3f}, given X+all {m_['partial_X_all']:+.3f}; given X+each: " + ", ".join(f"{cn} {v:+.3f}" for cn, v in m_["partial_X_each"].items())
        + " | other variants partial given X / X+all: " + ", ".join(f"{vn} {v['partial_X']:+.3f}/{v['partial_X_all']:+.3f}" for vn, v in out["pairwise"].items() if vn != "M_native_l2_16"))
    del variants, gaps; torch.cuda.empty_cache()
deep = res["by_depth"][DEPTHS[-1]]; pw = deep["pairwise"]
res["checks"] = dict(signal_is_proxy=pw["M_native_l2_16"]["partial_X_all"] < 0.05, top_share_tracks_token_embedding=deep["per_position"]["top_is_token"]["native"] > 0.3,
                     robust_to_normalisation=min(pw["M_native_l1_16"]["partial_X"], pw["top1_native_16"]["partial_X"]) >= 0.10, robust_to_k=min(pw["M_native_l2_8"]["partial_X"], pw["M_native_l2_32"]["partial_X"]) >= 0.10)
best = {b: min(v["pairwise"]["M_native_l2_16"]["partial_X_each"].items(), key=lambda kv: kv[1]) for b, v in res["by_depth"].items()}      # the candidate whose gap leaves the least
summ = (f"{name}, {N} positions: " + " || ".join(f"depth {b}: native top word is a token embedding {v['top_word_types']['token']:.2f} (the current token's {v['top_word_types']['current_token_embedding']:.2f}), MLP row {v['top_word_types']['mlp']:.2f}; "
        f"native top-share correlates with prominence {v['per_position']['prominence']['native']:+.2f}, FVU {v['per_position']['fvu_native']['native']:+.2f}, norm {v['per_position']['norm']['native']:+.2f}, position {v['per_position']['position']['native']:+.2f}, entropy {v['per_position']['entropy']['native']:+.2f}, current-token embedding on top {v['per_position']['top_is_token']['native']:+.2f}; "
        f"profile distance partial given X {v['pairwise']['M_native_l2_16']['partial_X']:+.3f} -> given X and all candidates {v['pairwise']['M_native_l2_16']['partial_X_all']:+.3f}; the single candidate that removes most: {best[b][0]} (leaves {best[b][1]:+.3f}); "
        f"L1 {v['pairwise']['M_native_l1_16']['partial_X']:+.3f}, top share only {v['pairwise']['top1_native_16']['partial_X']:+.3f}, k=8 {v['pairwise']['M_native_l2_8']['partial_X']:+.3f}, k=32 {v['pairwise']['M_native_l2_32']['partial_X']:+.3f}, rotated {v['pairwise']['M_rotated_l2_16']['partial_X']:+.3f}, PCA {v['pairwise']['M_pca_l2_16']['partial_X']:+.3f}, "
        f"FVU gap {v['pairwise']['fvu_native_gap']['partial_X']:+.3f}, prominence gap {v['pairwise']['prominence_gap']['partial_X']:+.3f}" for b, v in res["by_depth"].items()) + f" | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e476_profileaudit_{name}", res, summ)
