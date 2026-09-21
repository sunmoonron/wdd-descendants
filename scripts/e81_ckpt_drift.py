"""e81: dictionary drift over training. Final Pythia-410m states (step 143000) decomposed over the dictionaries
of earlier checkpoints (steps 4k, 16k, 32k, 64k) and vice versa: FVU and dominant-write recall (the neuron index
is the same across checkpoints). How stable are write directions late in training?"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
cf = Cache("pythia410"); L = mid(cf); N = 4096; ids = sub(cf.NT, N); X = cf.X(L)[ids]; typ = typical_mask(cf.X(L, center=False)[ids])
tb, tn, tc = cf.top_writes(L, 1); rowf = cf.atom_index(L, tb[ids, 0], tn[ids, 0]).to(DEV)
Af, _ = cf.dictionary(L); selF, _, eF = omp(X, Af, 64); res = dict(L=L, final_dict=dict(fvu32=fvu(eF[:, 31], X, typ), recall=(selF == rowf[:, None]).any(1)[typ].float().mean().item()), by_ckpt={})
for s in ("step4000", "step16000", "step32000", "step64000"):
    ce = Cache(f"pythia410_{s}"); Ae, _ = ce.dictionary(L)
    cosd = (Ae * Af).sum(1); mlp = torch.nonzero(cf.d["lab"]["type"][cf.d["lab"]["block"] <= L] == T_MLP)[:, 0].to(DEV)
    sel, _, e = omp(X, Ae, 64); rec = (sel == rowf[:, None]).any(1)[typ].float().mean().item()
    # reverse: earlier states over the final dictionary
    Xe = ce.X(L)[ids]; type_ = typical_mask(ce.X(L, center=False)[ids]); tbe, tne, _ = ce.top_writes(L, 1); rowe = ce.atom_index(L, tbe[ids, 0], tne[ids, 0]).to(DEV)
    selr, _, er = omp(Xe, Af, 64); rec_r = (selr == rowe[:, None]).any(1)[type_].float().mean().item(); _, _, ee = omp(Xe, Ae, 64)
    res["by_ckpt"][s] = dict(atom_cos_with_final_median=cosd[mlp].abs().median().item(), atom_cos_q10=cosd[mlp].abs().quantile(0.1).item(), fvu32_final_states=fvu(e[:, 31], X, typ), recall_final_states=rec,
                             fvu32_own_states_own_dict=fvu(ee[:, 31], Xe, type_), fvu32_own_states_final_dict=fvu(er[:, 31], Xe, type_), recall_own_states_final_dict=rec_r)
    log(f"{s}: atom cos med {res['by_ckpt'][s]['atom_cos_with_final_median']:.3f} q10 {res['by_ckpt'][s]['atom_cos_q10']:.3f} | final states: fvu {res['by_ckpt'][s]['fvu32_final_states']:.3f} recall {rec:.3f} (final dict {res['final_dict']['fvu32']:.3f}/{res['final_dict']['recall']:.3f}) | own states over final dict: fvu {res['by_ckpt'][s]['fvu32_own_states_final_dict']:.3f} recall {rec_r:.3f}")
record("e81_drift_pythia410", res, " | ".join(f"{s}: atom-cos {v['atom_cos_with_final_median']:.2f} final-states fvu {v['fvu32_final_states']:.3f} rec {v['recall_final_states']:.2f}; own-states/final-dict rec {v['recall_own_states_final_dict']:.2f}" for s, v in res["by_ckpt"].items()) + f" || final dict {res['final_dict']['fvu32']:.3f}/{res['final_dict']['recall']:.2f}")
