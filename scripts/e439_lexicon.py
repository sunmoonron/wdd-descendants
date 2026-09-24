"""e439: is the native vocabulary the lexicon? Three of this round's results point the same way.
- e435: a surrogate keeping only token identity (token means plus Gaussian noise) keeps 43-68% of the own-over-rotation
  advantage.
- e436: the middle block's own increment is described at the level of second-order statistics (GPT-2: covA 0.65, mix8
  0.85), while the state, which also contains the embedding, is word-level (0.28/0.44).
- e433: M, the most function-dense subspace, is the most lexical and is 65-91% predictable from the block-0 state.
Here the own dictionary is split into the lexicon (token-embedding rows, GPT-2's position rows and the block-0 MLP
rows, i.e. the extended embedding) and the rest. Per model, middle depth, 6 sequences, sink positions kept exact:
- loss recovered at k 4 and 16 with the full dictionary, without the lexicon, without the token rows only, and with
  the lexicon only; each against the same rows rotated, with covA for full and without the lexicon;
- naming: how often a 16-word description of the state uses the current token's own embedding row (first pick, any
  pick), the previous token's, and GPT-2's position row.
Arguments: model, optional revision (Pythia-410m checkpoints: is the emergence of self-description the growth of the
lexicon?).
Pre-registered: removing the lexicon halves the own-over-rotation gap at k 16 in at least four of five models, and what
remains is second-order (covA share above one half); the current token's row appears in over half of the descriptions."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; rev = sys.argv[2] if len(sys.argv) > 2 else None
model, tok, fam = load_model(name, revision=rev); arch = Arch(model, fam); L = arch.NB // 2
ev = eval_ids(name)[:6].to(DEV)
A, lab = build_dictionary(arch, blocks=list(range(L + 1))); typ, blk, idx = (lab[k].to(DEV) for k in ("type", "block", "index"))
lexm = (typ == T_TOK) | (typ == T_POS) | ((typ == T_MLP) & (blk == 0)); Ar = rotate(A, seed=7)
subsets = dict(full=torch.ones_like(lexm), no_lexicon=~lexm, no_token_rows=typ != T_TOK, lexicon_only=lexm)
lv = NSLevel(model, arch, ev, L); res = dict(model=name, rev=rev, level=L, gap=lv.gap, n_lexicon=int(lexm.sum()), n_atoms=int(A.shape[0]), cells={})
for sn, m in subsets.items():
    c = dict(own=describe(lv, A[m], [4, 16]), rot=describe(lv, Ar[m], [4, 16]))
    if sn in ("full", "no_lexicon"): n_ = int(m.sum()); c["covA"] = describe(lv, gauss_like(n_, (A[m].T @ A[m]) / n_, seed=1), [4, 16])
    res["cells"][sn] = c; g16 = c["own"]["16"]["rec"] - c["rot"]["16"]["rec"]
    log(f"{name} {rev or 'final'} {sn}: k4/k16 own {c['own']['4']['rec']:.2f}/{c['own']['16']['rec']:.2f} rot {c['rot']['4']['rec']:.2f}/{c['rot']['16']['rec']:.2f}"
        + (f" covA {c['covA']['4']['rec']:.2f}/{c['covA']['16']['rec']:.2f}" if "covA" in c else "") + f" | gap k16 {g16:+.3f}")
gap = lambda sn, k: res["cells"][sn]["own"][str(k)]["rec"] - res["cells"][sn]["rot"][str(k)]["rec"]
share = lambda sn, k: (res["cells"][sn]["covA"][str(k)]["rec"] - res["cells"][sn]["rot"][str(k)]["rec"]) / gap(sn, k) if abs(gap(sn, k)) > 1e-3 else None
res["gaps"] = {sn: {k: gap(sn, k) for k in (4, 16)} for sn in subsets}; res["covA_share"] = {sn: {k: share(sn, k) for k in (4, 16)} for sn in ("full", "no_lexicon")}
# naming: does the description use the current token's own embedding row?
sel, _, _ = omp(lv.Xc, A, 16, batch=256, record_err=False)
cur = ev[:, 1:].reshape(-1)[lv.keep]; prev = ev[:, :-1].reshape(-1)[lv.keep]; ntok = int((typ == T_TOK).sum())
res["naming"] = dict(current_any=(sel == cur[:, None]).any(1).float().mean().item(), current_first=(sel[:, 0] == cur).float().mean().item(),
                     previous_any=(sel == prev[:, None]).any(1).float().mean().item(), any_token_row=(sel < ntok).any(1).float().mean().item(),
                     token_rows_share_of_picks=(sel < ntok).float().mean().item(), block0_mlp_share_of_picks=lexm[sel].float().mean().item() - (sel < ntok).float().mean().item())
if (typ == T_POS).any():
    pos = torch.arange(1, ev.shape[1], device=DEV)[None].expand(ev.shape[0], -1).reshape(-1)[lv.keep]; res["naming"]["position_any"] = (sel == (ntok + pos)[:, None]).any(1).float().mean().item()
del sel
fm = lambda x: "n/a" if x is None else f"{x:.2f}"; G = res["gaps"]; N_ = res["naming"]
res["checks"] = dict(lexicon_halves_gap=G["no_lexicon"][16] < 0.5 * G["full"][16], rest_second_order=(res["covA_share"]["no_lexicon"][16] or 0) > 0.5, names_current_over_half=N_["current_any"] > 0.5)
summ = (f"{name} {rev or 'final'} L{L}: own-over-rotation gap k4/k16: full {G['full'][4]:+.3f}/{G['full'][16]:+.3f}, no lexicon {G['no_lexicon'][4]:+.3f}/{G['no_lexicon'][16]:+.3f}, no token rows {G['no_token_rows'][4]:+.3f}/{G['no_token_rows'][16]:+.3f}, "
        f"lexicon only {G['lexicon_only'][4]:+.3f}/{G['lexicon_only'][16]:+.3f} | own k16 full {res['cells']['full']['own']['16']['rec']:.2f}, no lexicon {res['cells']['no_lexicon']['own']['16']['rec']:.2f}, lexicon only {res['cells']['lexicon_only']['own']['16']['rec']:.2f}"
        f" | covA share k16 full {fm(res['covA_share']['full'][16])}, no lexicon {fm(res['covA_share']['no_lexicon'][16])} | naming: current token row in {N_['current_any']:.2f} of descriptions (first pick {N_['current_first']:.2f}), previous {N_['previous_any']:.2f}"
        + (f", position row {N_['position_any']:.2f}" if "position_any" in N_ else "") + f"; token rows {N_['token_rows_share_of_picks']:.2f} and block-0 rows {N_['block0_mlp_share_of_picks']:.2f} of all picks | checks {json.dumps(res['checks'])}")
log(summ); record(f"e439_lexicon_{name}_{rev or 'final'}", res, summ)
