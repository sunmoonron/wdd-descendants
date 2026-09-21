"""e318: the future-state quotient by horizon. At L, the quotient with respect to the state k blocks later for every
k up to the last block, against the logit quotient: overlap, and the decoding of the logit footprint from each
horizon's quotient. Does the state quotient converge to the logit quotient with horizon?"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quot_common import *
tag = sys.argv[1]; S0_ = Setup(tag, levels=[0]); L = S0_.L; NB = S0_.NB; S = Setup(tag, levels=list(range(L, NB))); d = 16; nat = S.natural(); F = nat["F"][L]; tr, te = S.halves(len(S.idx)); Fc = F - F[tr].mean(0, keepdim=True); Z, _ = logit_scores(nat["dl"], tr); Qlog = pls(Fc[tr], Z[tr], d); own = knn_cos(Fc @ Qlog, unit(nat["dl"]), tr, te); out = {}
for lv in range(L + 1, NB):
    Ff = unit(nat["F"][lv]); Qs = pls(Fc[tr], Ff[tr], d); out[lv] = dict(overlap_with_logit_quotient=inside(Qs, Qlog), logit_score_from_state_quotient=knn_cos(Fc @ Qs, unit(nat["dl"]), tr, te), state_score_own=knn_cos(Fc @ Qs, Ff, tr, te), state_score_from_logit_quotient=knn_cos(Fc @ Qlog, Ff, tr, te))
log(f"{tag} (K {S.K}, chance {d / S.D:.3f}; logit quotient own score {own:.2f}): horizon k -> overlap(state quotient, logit quotient) / logit score from state quotient / state score own / from logit quotient :: " + " ; ".join(f"+{lv - L}: {v['overlap_with_logit_quotient']:.2f}/{v['logit_score_from_state_quotient']:.2f}/{v['state_score_own']:.2f}/{v['state_score_from_logit_quotient']:.2f}" for lv, v in out.items()))
record(f"e318_horizon_{tag}", dict(model=tag, L=L, K=S.K, own_logit=own, per_level={str(k): v for k, v in out.items()}), " ".join(f"+{lv - L}:{v['overlap_with_logit_quotient']:.2f}" for lv, v in out.items()) + f" | logit own {own:.2f}")
