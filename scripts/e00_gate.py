"""e00 gate: reproduce the paper's GPT-2 L6 (state after block 6) headline numbers with the new pipeline.
Paper (32k states): typical FVU 0.338@32 / 0.179@64, rotated 0.529@32, OMP top-1 recall 0.647, one-shot 0.766,
sign 99.8%, median rel err 0.44. Accept if within ~0.03 of each."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1] if len(sys.argv) > 1 else "gpt2"; L = int(sys.argv[2]) if len(sys.argv) > 2 else None
c = Cache(tag); L = c.NB // 2 if L is None else L
if c.fam == "gpt2" and L == c.NB // 2: L = 6
X = c.X(L); Xraw = c.X(L, center=False); typ = typical_mask(Xraw)
A, lab = c.dictionary(L)
t0 = time.time(); sel, cof, err = omp(X, A, 64); t_omp = time.time() - t0
Ar = rotate(A); selr, cofr, errr = omp(X, Ar, 64)
tb, tn, tc = c.top_writes(L, 1)
true_row = c.atom_index(L, tb[:, 0], tn[:, 0]).to(DEV)
rec = recall_curve(sel, true_row)
sel1, cof1, err1 = oneshot(X, A, 64); rec1 = recall_curve(sel1, true_row, ks=(64,))
# estimation on identified dominant writes (typical states)
hit = (sel == true_row[:, None]); idn = hit.any(1) & typ
chat = (cof * hit).sum(1)[idn]; ctrue = tc[:, 0].to(DEV)[idn]
sign = ((chat * ctrue) > 0).float().mean().item(); rel = ((chat - ctrue).abs() / ctrue.abs()).median().item()
res = dict(model=tag, L=L, NT=X.shape[0], n_typical=int(typ.sum()), atoms=A.shape[0], t_omp=t_omp,
           fvu32_typ=fvu(err[:, 31], X, typ), fvu64_typ=fvu(err[:, 63], X, typ), fvu32_all=fvu(err[:, 31], X),
           fvu32_sink=fvu(err[:, 31], X, ~typ), rot32_typ=fvu(errr[:, 31], X, typ), rot32_sink=fvu(errr[:, 31], X, ~typ),
           recall=rec, recall_oneshot=rec1, sign=sign, med_rel_err=rel)
record("e00_gate_" + tag, res, f"FVU32 typ {res['fvu32_typ']:.3f} (rot {res['rot32_typ']:.3f}) FVU64 {res['fvu64_typ']:.3f} recall64 {rec[64]:.3f} oneshot {rec1[64]:.3f} sign {sign:.3f} relerr {rel:.2f} omp {t_omp:.0f}s")
