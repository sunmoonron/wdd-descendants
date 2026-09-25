"""e461: can WDD detect and locate a forged write? Residual forgery and crash forensics, from an external review.
At one position and one block b* (random, 0..L-1), the token's largest MLP neuron write (activation times row) is
replaced by a random vector of the same norm. This is a foreign, write-sized contribution; the block's output norm is
nearly unchanged.
Three detectors score each block's increment (block output minus block input, from hidden states; no ledger):
- WDD: the fraction of the increment's variance left unexplained by 8 of that block's own atoms (MLP rows, head bases,
  biases), by OMP;
- norm: the increment's norm;
- Mahalanobis: the increment's distance under the covariance of clean increments at that block.
Statistics (means, standard deviations, covariances) are fitted on clean positions of 8 sequences. Forged and clean
positions are evaluated on 8 others.
Reported:
- detection at the forging block: the AUC of each detector, forged against clean;
- localisation: the block with the largest z-score across blocks 0..L, compared with b* (chance 1 / (L + 1));
- downstream: at the middle depth L, the AUC of the state's unexplained variance with 16 words of the full dictionary,
  and of the state's Mahalanobis distance.
Models (argument): gpt2, smollm2.
Pre-registered (honest guesses):
- at the forging block WDD detects the forgery with AUC at least 0.8 (0.6);
- WDD beats the norm (0.7) but not Mahalanobis (0.5);
- WDD localises the forging block in at least half of cases (0.5);
- at the middle depth detection by the state falls to AUC 0.65 or less (0.6)."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; model, tok, fam = load_model(name); arch = Arch(model, fam); L = arch.NB // 2; D = arch.D
ev = eval_ids(name)[:16, :256].to(DEV); FIT, TEST = range(0, 8), range(8, 16)
own = []                                                                          # each block's own atoms (no embeddings)
for b in range(L + 1):
    A_, lab_ = build_dictionary(arch, blocks=[b]); own.append(unitr(A_[(lab_["block"] == b).to(DEV)]))
Afull, _ = build_dictionary(arch, blocks=list(range(L + 1))); Afull = unitr(Afull)
def hidden(ids):
    with torch.no_grad(): hs = model(ids, output_hidden_states=True).hidden_states
    return torch.stack([h[0].float() for h in hs[:L + 2]])                         # [L+2, T, D]: embeddings, block 0..L outputs
def fvu(X, Dct, k):
    sel, _, _ = omp(X, Dct, k, batch=1024, record_err=False); _, err = refit(X, Dct, sel); return err / X.pow(2).sum(-1).clamp_min(1e-12)
clean = {s: hidden(ev[s:s + 1]) for s in range(16)}
inc = lambda H: H[1:] - H[:-1]                                                    # [L+1, T, D] block increments
stats = []
for b in range(L + 1):                                                            # fit on clean positions of the FIT sequences
    Xf = torch.cat([inc(clean[s])[b, 1:] for s in FIT]); Xf = Xf[~sinkmask(torch.cat([clean[s][b + 1, 1:] for s in FIT]))]
    f = fvu(Xf, own[b], 8); nrm = Xf.norm(dim=-1); m_ = Xf.mean(0); Cv = torch.cov((Xf - m_).T) + 1e-3 * torch.eye(D, device=DEV) * torch.cov((Xf - m_).T).diagonal().mean()
    P = torch.linalg.inv(Cv); md = ((Xf - m_) @ P * (Xf - m_)).sum(-1).sqrt()
    stats.append(dict(m=m_, P=P, fvu=(f.mean(), f.std()), norm=(nrm.mean(), nrm.std()), maha=(md.mean(), md.std())))
Xl = torch.cat([clean[s][L + 1, 1:] for s in FIT]); Xl = Xl[~sinkmask(Xl)]; mL = Xl.mean(0)
CL = torch.cov((Xl - mL).T); PL = torch.linalg.inv(CL + 1e-3 * torch.eye(D, device=DEV) * CL.diagonal().mean())
def scores(H, p):
    """per-block scores of the increments at position p, and the state-level scores at depth L"""
    I = inc(H)[:, p]; out = dict(fvu=[], norm=[], maha=[])
    for b in range(L + 1):
        st = stats[b]; x = I[b]
        out["fvu"].append(float(fvu(x[None], own[b], 8)[0])); out["norm"].append(float(x.norm())); out["maha"].append(float(((x - st["m"]) @ st["P"] @ (x - st["m"])).sqrt()))
    xs = H[L + 1, p]; out["state_fvu"] = float(fvu((xs - mL)[None], Afull, 16)[0]); out["state_maha"] = float(((xs - mL) @ PL @ (xs - mL)).sqrt())
    return out
g = torch.Generator().manual_seed(0); rows = []
wlin = arch.mlp_lin
for s in TEST:
    for rep in range(20):
        p = int(torch.randint(16, ev.shape[1], (1,), generator=g)); bstar = int(torch.randint(0, L, (1,), generator=g))
        ids = ev[s:s + 1, :p + 1]; cap = {}
        def pre(m, a):
            cap["h"] = a[0][0, p].detach().float(); return None
        def post(m, i, o):
            h_ = cap["h"]; W = arch.wdir(bstar); wn = h_.abs() * W.norm(dim=-1); j = int(wn.argmax())
            r = torch.randn(D, generator=torch.Generator().manual_seed(1000 * s + rep)).to(DEV); r = r / r.norm()
            y = out_of(o).clone(); y[0, p] = (y[0, p].float() - h_[j] * W[j] + wn[j] * r).to(y.dtype)
            return (y,) + tuple(o[1:]) if isinstance(o, tuple) else y
        hs_ = [wlin(bstar).register_forward_pre_hook(pre), arch.layers[bstar].mlp.register_forward_hook(post)]
        try: Hf = hidden(ids)
        finally: [h.remove() for h in hs_]
        sc_f = scores(Hf, p); sc_c = scores(clean[s][:, :p + 1], p)
        rows.append(dict(seq=s, pos=p, block=bstar, forged=sc_f, clean=sc_c))
def auc(pos_, neg):
    return sum((a > b) + 0.5 * (a == b) for a in pos_ for b in neg) / (len(pos_) * len(neg))
res = dict(model=name, level=L, n=len(rows), detect_at_block={}, localise={}, downstream={})
for k in ("fvu", "norm", "maha"):
    res["detect_at_block"][k] = auc([r["forged"][k][r["block"]] for r in rows], [r["clean"][k][r["block"]] for r in rows])
    z = lambda v, b: (v - float(stats[b][k][0])) / float(stats[b][k][1])
    res["localise"][k] = sum(int(max(range(L + 1), key=lambda b: z(r["forged"][k][b], b)) == r["block"]) for r in rows) / len(rows)
for k in ("state_fvu", "state_maha"):
    res["downstream"][k] = auc([r["forged"][k] for r in rows], [r["clean"][k] for r in rows])
res["chance_localise"] = 1 / (L + 1); res["rows"] = rows
D_, Lc, Dn = res["detect_at_block"], res["localise"], res["downstream"]
res["checks"] = dict(wdd_auc_over_0_8=D_["fvu"] >= 0.8, wdd_beats_norm=D_["fvu"] > D_["norm"], wdd_beats_maha=D_["fvu"] > D_["maha"],
                     wdd_localises_half=Lc["fvu"] >= 0.5, downstream_under_0_65=Dn["state_fvu"] <= 0.65)
summ = (f"{name}, forged write (largest neuron write replaced by a same-norm random vector) at a random block 0..{L - 1}, {len(rows)} cases | "
        f"detection AUC at the forging block: WDD {D_['fvu']:.2f}, norm {D_['norm']:.2f}, Mahalanobis {D_['maha']:.2f} | "
        f"localisation (chance {res['chance_localise']:.2f}): WDD {Lc['fvu']:.2f}, norm {Lc['norm']:.2f}, Mahalanobis {Lc['maha']:.2f} | "
        f"at depth {L}: state WDD {Dn['state_fvu']:.2f}, state Mahalanobis {Dn['state_maha']:.2f} | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e461_forgery_{name}", res, summ)
