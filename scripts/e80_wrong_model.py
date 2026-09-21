"""e80: the wrong-model control. Trained GPT-2 states decomposed over the dictionary of a RANDOM-INIT GPT-2 (same
architecture, wrong weights, same size), vs the trained dictionary, its rotation and random atoms; and the reverse
(random-init states over the trained dictionary). Provenance is the specific trained checkpoint's, not the architecture's."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
ct, cr = Cache("gpt2"), Cache("gpt2_rand"); L = 6; N = 4096; ids = sub(ct.NT, N)
At, labt = ct.dictionary(L); Ar, labr = cr.dictionary(L); R = random_dict(At.shape[0], ct.D)
res = dict(L=L)
for nm, cc in (("trained_states", ct), ("random_states", cr)):
    X = cc.X(L)[ids]; typ = typical_mask(cc.X(L, center=False)[ids]); out = {}
    for dn, Ad in (("trained_dict", At), ("random_init_dict", Ar), ("rotated_trained", rotate(At)), ("random_atoms", R)):
        _, _, e = omp(X, Ad, 32); out[dn] = fvu(e[:, 31], X, typ)
    res[nm] = out; log(f"{nm}: " + " ".join(f"{k} {v:.3f}" for k, v in out.items()))
record("e80_wrongmodel_gpt2", res, "trained states: " + " ".join(f"{k} {v:.3f}" for k, v in res["trained_states"].items()) + " || random-init states: " + " ".join(f"{k} {v:.3f}" for k, v in res["random_states"].items()))
