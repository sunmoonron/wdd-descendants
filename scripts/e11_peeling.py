"""e11 (v2, every variant re-centered by its own mean): conditioning on what is known. The token id is known, so its embedding write (and GPT-2's position write)
can be subtracted exactly before decomposition. Variants of the input to OMP (k=64), scored on dominant-MLP-write
recall and on the FVU of the centered state: baseline; minus token (and position) embedding; oracle removal of all
attention writes (ledger); MLP-writes-only synthetic state; MLP-only plus embedding. Splits the ceiling into
attention interference vs MLP self-cancellation."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); N = int(os.environ.get("WDD_N", 8192)); ids = sub(c.NT, N)
Xraw = c.X(L, center=False)[ids]; mu = c.s["mu"][L + 1].to(DEV); X = Xraw - mu; typ = typical_mask(Xraw); A, lab = c.dictionary(L)
tb, tn, tc = c.top_writes(L, 1); row = c.atom_index(L, tb[ids, 0], tn[ids, 0]).to(DEV)
tok = c.s["eval_ids"].reshape(-1)[ids].to(DEV); E0 = c.d["emb"][0].to(DEV)[tok]
if c.n_pos_emb: E0 = E0 + c.d["emb"][1].to(DEV)[torch.arange(c.NT)[ids].to(DEV) % CTX]
att = sum(c.s["ATT"][b][ids].float().to(DEV) for b in range(L + 1))
mlp = torch.zeros_like(Xraw)
for b in range(L + 1):
    w = c.acts[b][ids].float().to(DEV) @ c.wdir_cpu(b).to(DEV)
    if c.d["mlp_bias"][b] is not None: w = w + c.d["mlp_bias"][b].to(DEV)
    mlp += w
cen = lambda V: V - V.mean(0)
variants = {"baseline": X, "minus_emb": cen(Xraw - E0), "minus_att_oracle": cen(Xraw - att), "mlp_only": cen(mlp), "mlp_plus_emb": cen(mlp + E0), "minus_emb_minus_att": cen(Xraw - E0 - att), "att_only": cen(att), "mlp_plus_att": cen(mlp + att)}
res = dict(model=tag, L=L, N=N, variants={})
for nm, V in variants.items():
    sel, cof, err = omp(V, A, 64); hit = (sel == row[:, None]).any(1)
    s1, _, _ = oneshot(V, A, 64); hit1 = (s1 == row[:, None]).any(1)
    idn = hit & typ; chat = (cof * (sel == row[:, None])).sum(1)[idn]; ct = tc[ids, 0].to(DEV)[idn]
    res["variants"][nm] = dict(recall=hit[typ].float().mean().item(), recall_oneshot=hit1[typ].float().mean().item(), med_rel_err=((chat - ct).abs() / ct.abs()).median().item(),
                               fvu64_self=fvu(err[:, 63], V, typ), emb_atoms_in_support=(lab["type"].to(DEV)[sel] <= T_POS).float().mean().item())
    log(f"{tag} {nm}: recall {res['variants'][nm]['recall']:.3f} oneshot {res['variants'][nm]['recall_oneshot']:.3f} relerr {res['variants'][nm]['med_rel_err']:.2f} fvu {res['variants'][nm]['fvu64_self']:.3f} emb-in-support {res['variants'][nm]['emb_atoms_in_support']:.2f}")
record(f"e11_peel_{tag}", res, " | ".join(f"{k}: rec {v['recall']:.3f} os {v['recall_oneshot']:.3f} err {v['med_rel_err']:.2f}" for k, v in res["variants"].items()))
