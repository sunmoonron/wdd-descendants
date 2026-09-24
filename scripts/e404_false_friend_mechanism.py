"""e404: what makes the step-4000 words false friends of the final model? e400 killed "Euclidean variance-chasing": chosen
under the network's Fisher metric they are still further below their rotation (0.27 against 0.54 at k 16). Remaining
hypothesis: mediation by the few huge directions of the final state (its top-8 principal components hold 0.86 of the
variance and 0.015 of the Fisher trace). A word with a sizeable component there imports the wrong magnitude along
directions that set the normalisation scale of every later block, a nonlinear harm no local metric sees.
Test, Pythia-410m final, middle depth, 8 sequences; M = the states' top-8 principal subspace (fitted on 8 other
sequences):
 A: the whole centred state described as before;
 B: the state's M part kept exactly, the rest described with words projected off M (renormalised);
 C: the M part set to its mean, the rest described as in B (how much the huge directions matter by themselves).
Vocabularies: own, two rotations, the step-4000 words and their rotation, own-word mixtures (mix8, also below zero at
k 4). Also: the share of each picked word's squared norm in M (condition A, k 16).
Pre-registered: if the harm is mediated by M, then under B the step-4000 words are at or above their rotation at k up to
16 and nothing is below zero at k 4-8; if they stay below their rotation under B, the false friends live in the part of
the state that carries the function (words repurposed)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sd_common import *
c = Cache("pythia410"); ev = c.s["eval_ids"][:8].to(DEV); fit = c.s["eval_ids"][8:16].to(DEV); KS = [4, 8, 16, 32, 64]
m4, tok, fam = load_model("pythia410", revision="step4000"); a4 = Arch(m4, fam); L = a4.NB // 2; V4, _ = build_dictionary(a4, blocks=list(range(L + 1))); del m4, a4
model, tok, fam = load_model("pythia410", revision="step143000"); arch = Arch(model, fam); own, lab = build_dictionary(arch, blocks=list(range(L + 1))); typ = lab["type"].to(DEV)
lv = Level(model, arch, ev, L); fl = Level(model, arch, fit, L); SX = (fl.Xc.T @ fl.Xc) / fl.Xc.shape[0]; del fl
evX, UX = torch.linalg.eigh(SX.double()); UM = UX[:, -8:].float(); PM = UM @ UM.T
XM = lv.Xc @ PM; Xp = lv.Xc - XM; tot = lv.Xc.pow(2).sum().item()
res = dict(level=L, k=KS, gap=lv.gap, var_share_M_eval=(XM.pow(2).sum() / lv.Xc.pow(2).sum()).item(), cells={}, pick_share_M={})
def run(nm, A):
    A = unitr(A); cells = {}
    sel, _, _ = omp(lv.Xc, A, max(KS), batch=256, record_err=False)
    res["pick_share_M"][nm] = (A[sel[:, :16]] @ UM).pow(2).sum(-1).mean().item()
    cells["A"] = {}
    for k in KS:
        s = sel[:, :k]; cof, _ = refit(lv.Xc, A, s); Xh = torch.einsum("nk,nkd->nd", cof, A[s])
        r, lo, hi = lv.recovered(lv.splice(lv.mu + Xh)); cells["A"][str(k)] = dict(rec=r, lo=lo, hi=hi, fvu=(lv.Xc - Xh).pow(2).sum().item() / tot)
    del sel
    Ap = unitr(A - A @ PM); sel, _, _ = omp(Xp, Ap, max(KS), batch=256, record_err=False); cells["B"], cells["C"] = {}, {}
    for k in KS:
        s = sel[:, :k]; cof, _ = refit(Xp, Ap, s); Xh = torch.einsum("nk,nkd->nd", cof, Ap[s])
        for cond, base in [("B", XM), ("C", torch.zeros_like(XM))]:
            r, lo, hi = lv.recovered(lv.splice(lv.mu + base + Xh)); cells[cond][str(k)] = dict(rec=r, lo=lo, hi=hi, fvu=(lv.Xc - base - Xh).pow(2).sum().item() / tot)
    del sel; res["cells"][nm] = cells
    log(f"{nm} (M share of picks {res['pick_share_M'][nm]:.3f}): " + " | ".join(f"{cd} " + " ".join(f"{cells[cd][str(k)]['rec']:.2f}" for k in KS) for cd in "ABC"))
run("own", own); run("rot7", rotate(own, seed=7)); run("rot11", rotate(own, seed=11)); run("f4000", V4); run("f4000_rot", rotate(V4, seed=7))
run("mix8", mixtures(own, [torch.nonzero(typ == t)[:, 0] for t in (T_TOK, T_MLP, T_ATT, T_BIAS) if (typ == t).any()], m=8, seed=4))
g = lambda n, cd, k: res["cells"][n][cd][str(k)]["rec"]
summ = ("k4/k16 A|B|C: " + " ; ".join(f"{n} {g(n, 'A', 4):.2f}/{g(n, 'A', 16):.2f} | {g(n, 'B', 4):.2f}/{g(n, 'B', 16):.2f} | {g(n, 'C', 4):.2f}/{g(n, 'C', 16):.2f}" for n in ["own", "rot7", "f4000", "f4000_rot", "mix8"])
        + " | M share of picks: " + " ".join(f"{n} {v:.3f}" for n, v in res["pick_share_M"].items()) + f" | M holds {res['var_share_M_eval']:.2f} of eval variance")
log(summ); record("e404_ffmech_pythia410", res, summ)
