"""e440: is there a native vocabulary for context? e439 asks whether the own-word advantage is the lexicon: the
token-embedding rows, GPT-2's position rows and the block-0 MLP rows. Here the target is split instead of the
dictionary:
- lexical part: the state's mean for its token, m(v), taken from 64 other sequences;
- context part: x - m(v).
Each part is described separately (k 4 and 16, typical positions, sinks exact) and spliced back with the other part
exact. Loss recovered is scored against mean-ablating that part only.
 context part: own words (all), own words without the lexicon, the rotation of each, covA and mix8 of the
               non-lexicon words;
 lexical part: own words (all), the lexicon only, and the rotation of each.
The question is whether what the model computes from context (the part token identity does not explain) is written in
its own non-lexical words (over rotation, covA and mix8), or only described to second order.
Pre-registered: on the context part, non-lexicon own words beat their rotation at k 16 in all five, but covA carries
over half of that gap (the context vocabulary is second-order); on the lexical part the lexicon alone recovers at least
90% of what all own words recover."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
name = sys.argv[1]; model, tok, fam = load_model(name); arch = Arch(model, fam); L = arch.NB // 2; D = arch.D
EA = torch.load("/workspace/wdd/eval_ids_all.pt", weights_only=False)[name]; ev = EA["eval_ids"][:6].to(DEV); cen = EA["cen_ids"].to(DEV); del EA

class PartLevel(NSLevel):
    """describe one part of the typical states (target) with the other part (base) kept exact: splice(Xh) = base + Xh"""
    def __init__(self, model, arch, ids, L, part):
        NSLevel.__init__(self, model, arch, ids, L); X = self.full[self.keep]
        self.Mt = self._Mt(X); tgt = X - self.Mt if part == "context" else self.Mt; self.base = X - tgt
        self.mu = tgt.mean(0, keepdim=True); self.Xc = tgt - self.mu
        self.lm = self.splice(self.mu.expand(self.Xc.shape[0], -1)); self.gap = (self.lm.mean() - self.lc.mean()).item()
    def splice(self, Xh):
        if not hasattr(self, "base"): return NSLevel.splice(self, Xh)
        return NSLevel.splice(self, self.base + Xh)
xc = block_states(model, arch, cen, [L], chunk=4)[L]; nc = ~sinkmask(xc); Xcc = xc[nc]; vc = cen[:, 1:][nc]
uniq, inv, cnt = torch.unique(vc, return_inverse=True, return_counts=True); tmean = torch.zeros(len(uniq), D, device=DEV).index_add_(0, inv, Xcc) / cnt[:, None]; mu_c = Xcc.mean(0); del xc, Xcc
def token_means(self, X):
    ve = self.ids[:, 1:].reshape(-1)[self.keep]; ix = torch.searchsorted(uniq, ve).clamp_max(len(uniq) - 1); known = (uniq[ix] == ve) & (cnt[ix] >= 3)
    self.known = known.float().mean().item(); return torch.where(known[:, None], tmean[ix], mu_c[None])
PartLevel._Mt = token_means
A, lab = build_dictionary(arch, blocks=list(range(L + 1))); typ, blk = lab["type"].to(DEV), lab["block"].to(DEV)
lexm = (typ == T_TOK) | (typ == T_POS) | ((typ == T_MLP) & (blk == 0)); Ar = rotate(A, seed=7); An = A[~lexm]
groups = [torch.nonzero(lab["type"][~lexm.cpu()] == t)[:, 0] for t in (T_MLP, T_ATT, T_BIAS) if (lab["type"][~lexm.cpu()] == t).any()]
res = dict(model=name, level=L, parts={})
for part, V in (("context", dict(own=A, rot=Ar, own_nolex=An, rot_nolex=Ar[~lexm], covA_nolex=gauss_like(An.shape[0], (An.T @ An) / An.shape[0], seed=1), mix8_nolex=mixtures(An, groups, m=8, seed=4))),
                ("lexical", dict(own=A, rot=Ar, lexicon=A[lexm], rot_lexicon=Ar[lexm]))):
    lv = PartLevel(model, arch, ev, L, part); var_share = (lv.Xc.pow(2).sum() / (lv.full[lv.keep] - lv.full[lv.keep].mean(0)).pow(2).sum()).item()
    cells = {vn: describe(lv, W, [4, 16]) for vn, W in V.items()}
    res["parts"][part] = dict(gap_nats=lv.gap, variance_share=var_share, known_tokens=lv.known, rec={vn: {k: c[k]["rec"] for k in ("4", "16")} for vn, c in cells.items()}, fvu={vn: {k: c[k]["fvu"] for k in ("4", "16")} for vn, c in cells.items()})
    log(f"{name} {part} part (variance share {var_share:.2f}, mean-ablation costs {lv.gap:.3f} nats): k4/k16 " + " ".join(f"{vn} {c['4']['rec']:.2f}/{c['16']['rec']:.2f}" for vn, c in cells.items()))
    del lv
C_, X_ = res["parts"]["context"]["rec"], res["parts"]["lexical"]["rec"]
g_nl = C_["own_nolex"]["16"] - C_["rot_nolex"]["16"]; covA_share = (C_["covA_nolex"]["16"] - C_["rot_nolex"]["16"]) / g_nl if abs(g_nl) > 1e-3 else None
mix8_share = (C_["mix8_nolex"]["16"] - C_["rot_nolex"]["16"]) / g_nl if abs(g_nl) > 1e-3 else None
res["context_shares_k16"] = dict(gap=g_nl, covA=covA_share, mix8=mix8_share)
res["checks"] = dict(context_nolex_beats_rot=g_nl > 0, context_second_order=(covA_share or 0) > 0.5, lexicon_90pct=X_["lexicon"]["16"] >= 0.9 * X_["own"]["16"])
fm = lambda x: "n/a" if x is None else f"{x:.2f}"
summ = (f"{name} L{L}: context part (variance {res['parts']['context']['variance_share']:.2f}, costs {res['parts']['context']['gap_nats']:.2f} nats) k16 own {C_['own']['16']:.2f} rot {C_['rot']['16']:.2f} | non-lexicon own {C_['own_nolex']['16']:.2f} rot {C_['rot_nolex']['16']:.2f} covA {C_['covA_nolex']['16']:.2f} mix8 {C_['mix8_nolex']['16']:.2f}"
        f" -> gap {g_nl:+.3f}, shares covA {fm(covA_share)} mix8 {fm(mix8_share)} (k4 non-lexicon own {C_['own_nolex']['4']:.2f} rot {C_['rot_nolex']['4']:.2f}) | lexical part (variance {res['parts']['lexical']['variance_share']:.2f}, costs {res['parts']['lexical']['gap_nats']:.2f} nats) k16 own {X_['own']['16']:.2f} rot {X_['rot']['16']:.2f} lexicon {X_['lexicon']['16']:.2f} its rotation {X_['rot_lexicon']['16']:.2f}"
        f" | checks {json.dumps(res['checks'])}")
log(summ); record(f"e440_contextvocab_{name}", res, summ)
