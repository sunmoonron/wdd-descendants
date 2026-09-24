"""e453: do concept words correspond across model sizes when ordinary words do not?
e445 found native words private between models. A word-for-word table keeps 0.06-0.13 of function, against 0.85-0.91
for a dense map, although partners exist above chance. e448d found language-independent concept words in
Qwen2.5-0.5B (15 of 24 nouns) and Qwen2.5-7B (22 of 24). The two models share a tokenizer, so their states align token
by token. This experiment asks whether the concept words are the exception: is the image of 0.5B's concept word for a
noun, under a dense linear map fitted on ordinary text, 7B's concept word for the same noun?
Method:
- States: block 12 of 0.5B and block 14 of 7B (each model's middle, e448d's depths). They are taken on WikiText-2 train
  (48 x 512 tokens, for fitting) and test (8 x 512, held out), with sinks excluded in either model, each centred by its
  own mean.
- Maps: ridge maps in both directions, the penalty chosen on the held-out text, reported as held-out R^2.
- Two fits: generic text only; and generic text plus e448d's noun sentences for all nouns but the one tested (leave one
  noun out, 480 sentences, noun-token states).
- The image of a word u is the unit vector along u W (a component a u maps to a u W).
Scores for each noun with a concept word in both models:
- the cosine between the mapped 0.5B concept word and 7B's concept word for the same noun;
- identification among the shared nouns (argmax of cosine; chance 1 / n);
- the rank of 7B's concept word among all 7B atoms up to block 14 (token embeddings, MLP rows, head bases), by
  absolute cosine with the mapped word;
- the same scores in the reverse direction.
Control: 500 random 0.5B MLP words used at the noun positions (not concept words) are mapped the same way. Their best
absolute cosine with any 7B atom is compared with that of the mapped concept words.
Pre-registered (honest guesses):
- the 0.5B -> 7B map explains at least half of 7B's held-out variance (0.5);
- mapped concept words identify the right 7B concept word among the shared nouns in over half of cases (0.6);
- 7B's concept word is the single nearest 7B atom to the mapped word for at most a quarter of nouns (0.55): the
  correspondence is at the concept level, not as nearest words, as in e445;
- mapped concept words have larger best-partner cosines than mapped random used words (0.6)."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from ws_common import load_bf16, MODELS
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "e451_concept_handles.py")).read()
NOUNS = eval(src[src.index("NOUNS = {") + 8: src.index("\nTEMPL = {")]); TEMPL = eval(src[src.index("TEMPL = {") + 8: src.index("\nEX = {")])
LANGS = list(NOUNS); NC, NT = len(NOUNS["en"]), len(TEMPL["en"])
E = torch.load("/workspace/wdd/eval_ids_all.pt", weights_only=False)["qwen05"]; fit_ids = E["cen_ids"][:48].to(DEV); ho_ids = E["eval_ids"][:8].to(DEV); del E
cw05 = _json.load(open(os.path.join(RESULTS, "e448d_concepts_qwen05.json")))["dicts"]["own"]["words"]
cw7 = _json.load(open(os.path.join(RESULTS, "e448d_concepts_qwen7.json")))["dicts"]["own"]["words"]
shared = [n for n in NOUNS["en"] if n in cw05 and n in cw7]

def noun_items(tok):
    out = []
    for lg in LANGS:
        for c in range(NC):
            for t in range(NT):
                s = TEMPL[lg][t].format(NOUNS[lg][c]); art = NOUNS[lg][c]; noun = art.split(" ", 1)[-1] if " " in art else art.split("'", 1)[-1]
                a = s.index(art) + art.index(noun); b_ = a + len(noun); enc = tok(s, add_special_tokens=False, return_offsets_mapping=True)
                out.append((c, torch.tensor(enc["input_ids"], device=DEV), [i for i, (x0, x1) in enumerate(enc["offset_mapping"]) if x1 > a and x0 < b_ and i > 0]))
    return out

def states(model, arch, L, ids, chunk):
    return block_states(model, arch, ids, [L], chunk=chunk)[L].reshape(-1, arch.D)            # positions 1:

def noun_states(model, arch, L, items):
    return torch.stack([block_states(model, arch, ids[None], [L], chunk=1)[L][0][[p - 1 for p in pos]].mean(0) for _, ids, pos in items])

# ---- Qwen2.5-0.5B: states, its concept words, random used words
m05, tok, fam = load_model("qwen05"); a05 = Arch(m05, fam); L05 = a05.NB // 2
A05, lab05 = build_dictionary(a05, blocks=list(range(L05 + 1))); typ05 = lab05["type"].to(DEV)
F05, H05 = states(m05, a05, L05, fit_ids, 4), states(m05, a05, L05, ho_ids, 4)
items = noun_items(tok); N05 = noun_states(m05, a05, L05, items); cvec = torch.tensor([c for c, _, _ in items], device=DEV)
u05 = {n: A05[int(cw05[n]["word"])] / A05[int(cw05[n]["word"])].norm() for n in shared}
mu_n = F05[~sinkmask(F05)].mean(0); sel, _, _ = omp(N05 - mu_n, A05, 16, batch=128, record_err=False)
concept_ids = {int(v["word"]) for v in cw05.values()}
pool = [a for a in torch.unique(sel).tolist() if int(typ05[a]) == T_MLP and a not in concept_ids]
g = torch.Generator().manual_seed(0); pick = [pool[i] for i in torch.randperm(len(pool), generator=g)[:500].tolist()]
R05 = unitr(A05[pick]); del A05, m05; torch.cuda.empty_cache()
# ---- Qwen2.5-7B
m7, tok7, fam7 = load_bf16("qwen7"); a7 = Arch(m7, fam7); L7 = a7.NB // 2
F7, H7 = states(m7, a7, L7, fit_ids, 2), states(m7, a7, L7, ho_ids, 2); N7 = noun_states(m7, a7, L7, items)
V7 = a7.emb[0].shape[0]; DFF7 = a7.wdir(0).shape[0]; per = DFF7 + a7.NH * a7.HD
def atom7(w):
    r = w - V7; b, j = r // per, r % per
    assert j < DFF7, "a 7B concept word that is not an MLP row"
    v = a7.wdir(b)[j].float(); return b, v / v.norm()
u7 = {}
for n in shared:
    b, v = atom7(int(cw7[n]["word"])); assert b == cw7[n]["block"], "7B dictionary order changed"; u7[n] = v

def all7_absmax(M):
    """for unit rows M [n, D7]: the largest |cos| with any 7B atom up to block L7, and the atom's (type, block, row)"""
    best = torch.full((M.shape[0],), -1.0, device=DEV); arg = [None] * M.shape[0]
    def upd(W, tag):
        nonlocal best
        for s in range(0, W.shape[0], 16384):
            w = W[s:s + 16384].float(); w = w / w.norm(dim=-1, keepdim=True).clamp_min(1e-8); c = (M @ w.T).abs(); v, i = c.max(1)
            imp = v > best; best = torch.where(imp, v, best)
            for k in torch.nonzero(imp)[:, 0].tolist(): arg[k] = (tag[0], tag[1], s + int(i[k]))
    upd(a7.emb[0].detach(), ("tok", -1))
    for b in range(L7 + 1):
        upd(a7.wdir(b), ("mlp", b)); Wo = a7.wo(b)
        for h in range(a7.NH): upd(torch.linalg.svd(Wo[h * a7.HD:(h + 1) * a7.HD].float(), full_matrices=False).Vh, ("att", b))
    return best, arg
def keep_mask(X): return ~sinkmask(X)

def fit(Xa, Xb, Ha, Hb):
    """ridge map Xa -> Xb (centred), penalty chosen on the held-out pair; returns W, held-out R^2, means"""
    ma, mb = Xa.mean(0), Xb.mean(0); Xa, Xb, Ha, Hb = Xa - ma, Xb - mb, Ha - ma, Hb - mb
    G = Xa.T.double() @ Xa.double(); C = Xa.T.double() @ Xb.double(); best = None; tr = G.trace() / G.shape[0]
    for lam in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        W = torch.linalg.solve(G + lam * tr * torch.eye(G.shape[0], device=DEV, dtype=torch.float64), C).float()
        r2 = 1 - (Ha @ W - Hb).pow(2).sum().item() / Hb.pow(2).sum().item()
        if best is None or r2 > best[1]: best = (W, r2, lam)
    return best
k1, k2 = keep_mask(F05) & keep_mask(F7), keep_mask(H05) & keep_mask(H7)
res = dict(shared=len(shared), shared_nouns=[n.split()[-1] for n in shared], level05=L05, level7=L7, fits={})
for fitname in ("generic", "generic_plus_other_nouns"):
    out = dict(rows={})
    if fitname == "generic":
        W, r2, lam = fit(F05[k1], F7[k1], H05[k2], H7[k2]); Wr, r2r, lamr = fit(F7[k1], F05[k1], H7[k2], H05[k2])
        maps = {n: (W, Wr) for n in shared}; out.update(r2_05_to_7=r2, r2_7_to_05=r2r, lam=lam, lam_rev=lamr)
    else:
        maps = {}; r2s = []
        for n in shared:
            c = NOUNS["en"].index(n); keep = cvec != c
            Xa = torch.cat([F05[k1], N05[keep]]); Xb = torch.cat([F7[k1], N7[keep]])
            W, r2, _ = fit(Xa, Xb, H05[k2], H7[k2]); Wr, r2r, _ = fit(Xb, Xa, H7[k2], H05[k2]); maps[n] = (W, Wr); r2s.append((r2, r2r))
        out.update(r2_05_to_7=sum(a for a, _ in r2s) / len(r2s), r2_7_to_05=sum(b for _, b in r2s) / len(r2s))
    M = torch.stack([unitr((u05[n] @ maps[n][0])[None])[0] for n in shared]); U7 = torch.stack([u7[n] for n in shared])
    Mr = torch.stack([unitr((u7[n] @ maps[n][1])[None])[0] for n in shared]); U05 = torch.stack([u05[n] for n in shared])
    C = M @ U7.T; Cr = Mr @ U05.T; n_ = len(shared)
    out["cos_same"] = C.diagonal().tolist(); out["cos_same_rev"] = Cr.diagonal().tolist()
    out["ident"] = (C.abs().argmax(1) == torch.arange(n_, device=DEV)).float().mean().item(); out["ident_rev"] = (Cr.abs().argmax(1) == torch.arange(n_, device=DEV)).float().mean().item()
    best, arg = all7_absmax(M)
    row7 = lambda n: int(cw7[n]["word"]) - V7 - cw7[n]["block"] * per
    out["nearest_is_concept"] = sum(1 for i, n in enumerate(shared) if arg[i] == ("mlp", cw7[n]["block"], row7(n))) / n_
    out["best_abs_cos_concept"] = best.tolist(); out["nearest_atom"] = [list(a) for a in arg]
    out["same_abs_cos_rank_hint"] = [(C[i].abs() >= C[i, i].abs()).sum().item() for i in range(n_)]
    if fitname == "generic":
        Wg = maps[shared[0]][0]; Mrand = unitr(R05 @ Wg); bestr, _ = all7_absmax(Mrand)
        out["best_abs_cos_random_used_words"] = dict(mean=bestr.mean().item(), median=bestr.median().item(), over_0_5=(bestr > 0.5).float().mean().item())
    out["best_abs_cos_concept_summary"] = dict(mean=best.mean().item(), median=best.median().item(), over_0_5=(best > 0.5).float().mean().item())
    res["fits"][fitname] = out
    log(f"{fitname}: R2 0.5B->7B {out['r2_05_to_7']:.2f}, 7B->0.5B {out['r2_7_to_05']:.2f} | cos(mapped 0.5B concept word, 7B concept word) mean {sum(out['cos_same']) / n_:+.2f} | "
        f"identification among {n_} shared nouns {out['ident']:.2f} (reverse {out['ident_rev']:.2f}, chance {1 / n_:.2f}) | nearest 7B atom is the concept word {out['nearest_is_concept']:.2f} | "
        f"best |cos| concept {out['best_abs_cos_concept_summary']['mean']:.2f}" + (f" vs random used words {out['best_abs_cos_random_used_words']['mean']:.2f}" if 'best_abs_cos_random_used_words' in out else ""))
G_ = res["fits"]["generic"]
res["checks"] = dict(r2_over_half=G_["r2_05_to_7"] >= 0.5, ident_over_half=G_["ident"] > 0.5, nearest_at_most_quarter=G_["nearest_is_concept"] <= 0.25,
                     concept_beats_random=G_["best_abs_cos_concept_summary"]["mean"] > G_["best_abs_cos_random_used_words"]["mean"])
f_ = lambda o: (f"R2 {o['r2_05_to_7']:.2f}/{o['r2_7_to_05']:.2f}, cos same {sum(o['cos_same']) / len(o['cos_same']):+.2f} (rev {sum(o['cos_same_rev']) / len(o['cos_same_rev']):+.2f}), "
                f"ident {o['ident']:.2f} (rev {o['ident_rev']:.2f}), nearest-is-concept {o['nearest_is_concept']:.2f}, best |cos| {o['best_abs_cos_concept_summary']['mean']:.2f}")
summ = (f"qwen05 L{L05} -> qwen7 L{L7}, {len(shared)} shared concept nouns (chance {1 / len(shared):.2f}): generic fit: " + f_(G_)
        + f", random used words' best |cos| {G_['best_abs_cos_random_used_words']['mean']:.2f} | with other nouns' sentences: " + f_(res["fits"]["generic_plus_other_nouns"]) + f" | checks {_json.dumps(res['checks'])}")
log(summ); record("e453_conceptcorr_qwen05_qwen7", res, summ)
