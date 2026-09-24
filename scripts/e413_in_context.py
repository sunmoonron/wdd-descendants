"""e413: does the network's own language have words for what it computes in context? Pythia-410m (argument: revision),
middle depth, three inputs of 8 sequences of 511 tokens: natural text (wikitext); random tokens drawn from the text's
token set (nothing to compute beyond unigram guesses); random tokens repeated (a random half followed by the same
half: the second copy is predicted by in-context copying, induction). The states at every position but the first are
replaced by their k-word descriptions; loss recovered is measured on the inducible positions of the second copy for
repeated sequences (where the prediction is in-context computation) and on all positions otherwise. Own vocabulary
against three rotations, k 4-64. Pre-registered: on the second copy the own words' advantage over rotation at k 16 is at
least as large as on natural text (the vocabulary covers in-context computation). Random tokens are reported, not scored
(their clean-to-mean gap is small)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sd_common import *
rev = sys.argv[1] if len(sys.argv) > 1 else "step143000"
c = Cache("pythia410"); KS = [4, 8, 16, 32, 64]
model, tok, fam = load_model("pythia410", revision=rev); arch = Arch(model, fam); L = arch.NB // 2; A, _ = build_dictionary(arch, blocks=list(range(L + 1)))
nat = c.s["eval_ids"][:8].to(DEV)
g = torch.Generator().manual_seed(0); pool = torch.unique(c.s["eval_ids"].flatten())
half = 255; bos = tok.bos_token_id if tok.bos_token_id is not None else 0
r1 = pool[torch.randint(0, len(pool), (8, half), generator=g)]
rep = torch.cat([torch.full((8, 1), bos), r1, r1], 1).to(DEV)                                   # 511 tokens: BOS, copy 1, copy 2
rnd = torch.cat([torch.full((8, 1), bos), pool[torch.randint(0, len(pool), (8, 2 * half), generator=g)]], 1).to(DEV)
mask_rep = torch.zeros(8, rep.shape[1] - 1, dtype=torch.bool, device=DEV); mask_rep[:, half + 1:] = True   # loss index t predicts token t+1; tokens from position half+2 on are inducible
res = dict(rev=rev, level=L, k=KS, inputs={})
for nm, ids, mask in [("natural", nat, None), ("random", rnd, None), ("repeated", rep, mask_rep)]:
    lv = Level(model, arch, ids, L)
    M_ = torch.ones_like(lv.lc, dtype=torch.bool) if mask is None else mask
    lc, lm = (lv.lc * M_).sum(), (lv.lm * M_).sum(); out = dict(gap_per_token=((lm - lc) / M_.sum()).item(), clean_per_token=(lc / M_.sum()).item(), cells={})
    for vn, V in [("own", A), ("rot7", rotate(A, seed=7)), ("rot11", rotate(A, seed=11)), ("rot13", rotate(A, seed=13))]:
        Vu = unitr(V); sel, _, _ = omp(lv.Xc, Vu, max(KS), batch=256, record_err=False); cell = {}
        for k in KS:
            s = sel[:, :k]; cof, _ = refit(lv.Xc, Vu, s); ls = lv.splice(lv.mu + torch.einsum("nk,nkd->nd", cof, Vu[s]))
            cell[str(k)] = ((lm - (ls * M_).sum()) / (lm - lc).clamp_min(1e-9)).item()
        out["cells"][vn] = cell; del sel
    res["inputs"][nm] = out
    rot = lambda k: sum(out["cells"][f"rot{s}"][str(k)] for s in (7, 11, 13)) / 3
    log(f"{rev} {nm} (clean {out['clean_per_token']:.2f}, gap/token {out['gap_per_token']:.2f}): own " + " ".join(f"{out['cells']['own'][str(k)]:.2f}" for k in KS) + " | rot " + " ".join(f"{rot(k):.2f}" for k in KS))
    del lv
adv = lambda nm, k: res["inputs"][nm]["cells"]["own"][str(k)] - sum(res["inputs"][nm]["cells"][f"rot{s}"][str(k)] for s in (7, 11, 13)) / 3
summ = f"{rev} own-minus-rotation at k4/k16: " + " | ".join(f"{nm} {adv(nm, 4):.2f}/{adv(nm, 16):.2f} (gap/token {res['inputs'][nm]['gap_per_token']:.2f})" for nm in res["inputs"])
log(summ); record(f"e413_incontext_pythia410_{rev}", res, summ)
