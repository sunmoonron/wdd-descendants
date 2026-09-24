"""e445: can two models' native vocabularies be translated word by word? (vision: interoperable internals)
In a world that had WDD for a decade, models would exchange internal states through dictionaries between their
vocabularies. That requires a correspondence between words, and none was ever tested.
- e398 found Pythia seeds' vocabularies private as vectors: token lexicon and raw translation at chance; a state-fitted
  linear map carried a third to a half of the advantage.
- Whether words correspond by use (co-used on the same inputs, like translations in human languages) was never asked.
Here, for Pythia models of different sizes (same tokenizer, data and data order; argument: A B, e.g. 160m 410m), at
each model's middle depth, on 48 sequences (matching) and 6 held-out sequences (translation), typical positions:
 1. 16-word OMP descriptions of each model's states with its own dictionary, and with its rotated dictionary.
 2. Usage similarity: cosine between two words' signed coefficient vectors across the matching positions. For A's
    2000 most used words: the best partner among B's 8000 most used words (|cos|), against the best partner among
    B's rotated words (no provenance).
 3. Translation on held-out text: A's description, each word replaced by its partner with a scale fitted on the matching
    positions, spliced into B; loss recovered in B. Compared with:
    - B's own 16-word description (ceiling);
    - a random partner assignment (floor);
    - the same pipeline between the rotated vocabularies;
    - a dense ridge map from A's state to B's state (e398b's instrument);
    - a ridge map from A's word coefficients to B's state (a learned, not one-to-one, dictionary).
Pre-registered (honest guesses):
 T1 (0.65) native words find better usage partners in B's native words than in B's rotated words;
 T2 (0.5) word-by-word translation recovers over 0.3 of B's loss gap (random partners about 0);
 T3 (0.6) the dense state map beats word-by-word translation;
 T4 (0.4) word-by-word translation reaches at least half of the dense map."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
MODELS.update({"160m": ("EleutherAI/pythia-160m", "neox"), "410m": ("EleutherAI/pythia-410m", "neox"), "1b": ("EleutherAI/pythia-1b", "neox")})
na, nb = sys.argv[1], sys.argv[2]
EA = torch.load("/workspace/wdd/eval_ids_all.pt", weights_only=False)["pythia410"]; tr_ids = EA["cen_ids"][:48].to(DEV); te_ids = EA["eval_ids"][:6].to(DEV); del EA
M = {}
for nm in (na, nb):
    model, tok, fam = load_model(nm, revision="step143000"); arch = Arch(model, fam); L = arch.NB // 2
    A, lab = build_dictionary(arch, blocks=list(range(L + 1)))
    M[nm] = dict(model=model, arch=arch, L=L, A=A, typ=lab["type"].to(DEV), idx=lab["index"].to(DEV), Ar=rotate(A, seed=7),
                 xtr=block_states(model, arch, tr_ids, [L], chunk=4)[L], xte=block_states(model, arch, te_ids, [L], chunk=3)[L])
a, b = M[na], M[nb]
ktr = (~sinkmask(a["xtr"]) & ~sinkmask(b["xtr"])).reshape(-1); kte_a, kte_b = ~sinkmask(a["xte"]), ~sinkmask(b["xte"])
for m in (a, b):
    X = m["xtr"].reshape(-1, m["xtr"].shape[-1])[ktr]; m["mu"] = X.mean(0); m["Xtr"] = X - m["mu"]
N = int(ktr.sum()); NA_, NB_, K = 2000, 8000, 16
def code(X, D):
    sel, cof, _ = omp(X, D, K, batch=256, record_err=False); return sel, cof
def usage(sel, cof, n_words):
    cnt = torch.bincount(sel.flatten(), minlength=n_words); return cnt
def sparse_cols(sel, cof, words):
    """[N, len(words)] dense matrix of signed coefficients for the given words (0 where unused)"""
    lut = torch.full((int(max(sel.max().item(), words.max().item())) + 1,), -1, dtype=torch.long, device=DEV); lut[words] = torch.arange(words.numel(), device=DEV)
    col = lut[sel]; ok = col >= 0; out = torch.zeros(sel.shape[0], words.numel(), device=DEV)
    out[torch.arange(sel.shape[0], device=DEV)[:, None].expand_as(sel)[ok], col[ok]] = cof[ok]; return out
res = dict(A=na, B=nb, levels=[a["L"], b["L"]], n_match_positions=N, match={}, translate={})
codes = {}
for tag, m, D in (("A", a, a["A"]), ("Ar", a, a["Ar"]), ("B", b, b["A"]), ("Br", b, b["Ar"])):
    codes[tag] = code(m["Xtr"], D)
def partners(ca, cb, nwa, nwb):
    """best |cos| partner in cb's vocabulary for ca's most used words; returns words_a, partner_b, cos, beta"""
    ua, ub = usage(*ca, nwa), usage(*cb, nwb); wa = ua.topk(NA_).indices; wb = ub.topk(min(NB_, int((ub > 0).sum()))).indices
    Ua, Ub = sparse_cols(*ca, wa), sparse_cols(*cb, wb)
    S = Ua.T @ Ub; cs = S / (Ua.norm(dim=0)[:, None] * Ub.norm(dim=0)[None]).clamp_min(1e-9); best = cs.abs().max(1)
    pb = wb[best.indices]; beta = (Ua * Ub[:, best.indices]).sum(0) / Ua.pow(2).sum(0).clamp_min(1e-9)
    return wa, pb, cs.abs().max(1).values, beta, wb
pn = partners(codes["A"], codes["B"], a["A"].shape[0], b["A"].shape[0]); pr = partners(codes["A"], codes["Br"], a["A"].shape[0], b["A"].shape[0]); prr = partners(codes["Ar"], codes["Br"], a["A"].shape[0], b["A"].shape[0])
q = lambda v: [round(x, 3) for x in torch.quantile(v, torch.tensor([0.1, 0.5, 0.9], device=DEV)).tolist()]
same_tok = ((a["typ"][pn[0]] == T_TOK) & (b["typ"][pn[1]] == T_TOK) & (a["idx"][pn[0]] == b["idx"][pn[1]])).float()
res["match"] = dict(native_to_native=q(pn[2]), native_to_rotated=q(pr[2]), rotated_to_rotated=q(prr[2]), frac_native_over_0_5=(pn[2] > 0.5).float().mean().item(), frac_rot_over_0_5=(pr[2] > 0.5).float().mean().item(),
                    top100_same_token_rows=same_tok[pn[2].topk(100).indices].mean().item(), top100_types_A=torch.bincount(a["typ"][pn[0][pn[2].topk(100).indices]], minlength=5).tolist(),
                    top100_types_B=torch.bincount(b["typ"][pn[1][pn[2].topk(100).indices]], minlength=5).tolist(), mutual_best=None)
back = partners(codes["B"], codes["A"], b["A"].shape[0], a["A"].shape[0]); bmap = dict(zip(back[0].tolist(), back[1].tolist()))
res["match"]["mutual_best"] = sum(1 for wa_, wb_ in zip(pn[0].tolist(), pn[1].tolist()) if bmap.get(wb_) == wa_) / NA_
log(f"{na}->{nb}: best-partner |cos| quantiles (10/50/90%) native->native {res['match']['native_to_native']} native->rotated {res['match']['native_to_rotated']} rotated->rotated {res['match']['rotated_to_rotated']} | mutual best {res['match']['mutual_best']:.2f} | top-100 pairs same token row {res['match']['top100_same_token_rows']:.2f}")
# translation on held-out text, spliced into B
lvb = NSLevel(b["model"], b["arch"], te_ids, b["L"]); keepB = lvb.keep; xa_te = a["xte"].reshape(-1, a["xte"].shape[-1]); okA = kte_a.reshape(-1)
XaT = xa_te - a["mu"]
def translate(D_a, Db, part):
    wa, pb, cs, beta = part[0], part[1], part[2], part[3]; lut = torch.full((D_a.shape[0],), -1, dtype=torch.long, device=DEV); lut[wa] = torch.arange(wa.numel(), device=DEV)
    sel, cof = code(XaT, D_a); j = lut[sel]; ok = j >= 0; Xh = torch.zeros(XaT.shape[0], Db.shape[1], device=DEV)
    contrib = torch.where(ok[..., None], (beta[j.clamp_min(0)] * cof)[..., None] * Db[pb[j.clamp_min(0)]], torch.zeros(1, device=DEV)); Xh = contrib.sum(1)
    return Xh, ok.float().mean().item()
def splice_rec(Xh_all):
    """Xh_all: B-space centred estimates for every position; spliced at B's typical positions where A is typical too, else B exact"""
    f = lvb.full.clone(); est = b["mu"][None] + Xh_all; use = keepB & okA; f[use] = est[use]; return lvb.recovered(Level.splice(lvb, f))[0]
g = torch.Generator(device=DEV).manual_seed(0)
out = {}
Xh, cov = translate(a["A"], b["A"], pn); out["word_to_word"] = splice_rec(Xh); out["coverage"] = cov
perm_pb = pn[4][torch.randint(0, pn[4].numel(), (NA_,), device=DEV, generator=g)]
Ua, Ubr = sparse_cols(*codes["A"], pn[0]), sparse_cols(*codes["B"], perm_pb); beta_r = (Ua * Ubr).sum(0) / Ua.pow(2).sum(0).clamp_min(1e-9)
Xh, _ = translate(a["A"], b["A"], (pn[0], perm_pb, None, beta_r)); out["random_partners"] = splice_rec(Xh)
Xh, _ = translate(a["Ar"], b["Ar"], prr); out["rotated_word_to_word"] = splice_rec(Xh)
lam = 1e-3 * a["Xtr"].pow(2).sum() / a["Xtr"].shape[0]; Wd = torch.linalg.solve(a["Xtr"].T @ a["Xtr"] + lam * torch.eye(a["Xtr"].shape[1], device=DEV), a["Xtr"].T @ b["Xtr"]); out["dense_state_map"] = splice_rec(XaT @ Wd)
Ca = sparse_cols(*codes["A"], pn[0]); lam2 = 1e-3 * Ca.pow(2).sum() / Ca.shape[0]; Wc = torch.linalg.solve(Ca.T @ Ca + lam2 * torch.eye(NA_, device=DEV), Ca.T @ b["Xtr"])
sel_te, cof_te = code(XaT, a["A"]); out["learned_word_decoder"] = splice_rec(sparse_cols(sel_te, cof_te, pn[0]) @ Wc)
out["B_own_16"] = describe(lvb, b["A"], [16])["16"]["rec"]
res["translate"] = out
res["checks"] = dict(T1=res["match"]["native_to_native"][1] > res["match"]["native_to_rotated"][1], T2=out["word_to_word"] > 0.3 and abs(out["random_partners"]) < 0.1,
                     T3=out["dense_state_map"] > out["word_to_word"], T4=out["word_to_word"] >= 0.5 * out["dense_state_map"])
summ = (f"{na}->{nb} (L{a['L']}->L{b['L']}): usage partners |cos| median native->native {res['match']['native_to_native'][1]:.2f} vs native->rotated {res['match']['native_to_rotated'][1]:.2f} (rotated->rotated {res['match']['rotated_to_rotated'][1]:.2f}); "
        f"share above 0.5: {res['match']['frac_native_over_0_5']:.2f} vs {res['match']['frac_rot_over_0_5']:.2f}; mutual best {res['match']['mutual_best']:.2f}; top-100 pairs same token row {res['match']['top100_same_token_rows']:.2f}, types A {res['match']['top100_types_A']} B {res['match']['top100_types_B']} | "
        f"held-out loss recovered in B: word-to-word {out['word_to_word']:.2f} (coverage {out['coverage']:.2f}), random partners {out['random_partners']:.2f}, rotated word-to-word {out['rotated_word_to_word']:.2f}, "
        f"dense state map {out['dense_state_map']:.2f}, learned word decoder {out['learned_word_decoder']:.2f}, B's own 16 words {out['B_own_16']:.2f} | checks {json.dumps(res['checks'])}")
log(summ); record(f"e445_translate_{na}_{nb}", res, summ)
