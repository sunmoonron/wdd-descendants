"""e436: is the early accent a property of accumulation? e399/e405 found that the own-word advantage on middle-depth
states is second order early in Pythia's training: at step 1000, covA and mix8 carry 0.62 and 0.85 of it. At the end
it is word-level (shares near 0). e02 (phase 1) found that block increments are readable at step 1000 (0.3-0.5) while
the states are not (0.03). Neither the controls nor the self-description were ever applied to increments.
Here the middle block's own increment (its output minus its input: attention plus MLP writes) is described with that
block's own words (its MLP rows, head output bases and biases), against:
- the same words rotated;
- covA (Gaussian words with the same second moment);
- mix8 (signed sums of 8 words of one family).
The description is spliced back as input + mean increment + description and scored by loss recovered, against
mean-ablating the increment. The states are also described with the words of blocks 0..L, as in e426a, on the same
text.
Also reported: the participation ratio of the actual MLP writes (activation x write norm) at that block, i.e. how few
neurons carry each increment.
Arguments: model, optional revision (Pythia-410m at steps 1000, 4000, 16000 and 143000; the other four at the end).
Pre-registered: at step 1000 the increment's covA and mix8 shares are below one half (word-level) while the state's are
above one half (accent), and the increment's advantage is word-level at every checkpoint."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; rev = sys.argv[2] if len(sys.argv) > 2 else None
model, tok, fam = load_model(name, revision=rev); arch = Arch(model, fam); L = arch.NB // 2; D = arch.D
ev = eval_ids(name)[:6].to(DEV)

class IncLevel(Level):
    """block L's increment (output - input) at positions 1:, centred; a splice sets the output to input + Xh"""
    def __init__(self, model, arch, ids, L):
        self.model, self.arch, self.ids, self.L, self.B, self.T, self.P = model, arch, ids, L, ids.shape[0], ids.shape[1], None
        cap = {}
        h1 = arch.layers[L].register_forward_pre_hook(lambda m, a, kw: cap.__setitem__("i", (a[0] if len(a) > 0 else kw["hidden_states"]).detach().float()), with_kwargs=True)
        h2 = arch.layers[L].register_forward_hook(lambda m, i, o: cap.__setitem__("o", out_of(o).detach().float()))
        h3 = arch.mlp_lin(L).register_forward_pre_hook(lambda m, a: cap.__setitem__("h", a[0].detach().float()))
        try:
            with torch.no_grad(): self.lc = token_loss(model(ids).logits.float(), ids).view(self.B, -1)
        finally: [h.remove() for h in (h1, h2, h3)]
        self.inp = cap["i"][:, 1:].reshape(-1, arch.D); d = cap["o"][:, 1:].reshape(-1, arch.D) - self.inp
        self.mu = d.mean(0, keepdim=True); self.Xc = d - self.mu; self.acts = cap["h"][:, 1:].reshape(-1, cap["h"].shape[-1])
        self.lm = self.splice(self.mu.expand(self.Xc.shape[0], -1)); self.gap = (self.lm.mean() - self.lc.mean()).item()
    def splice(self, Xh): return Level.splice(self, self.inp + Xh)

def controls(A, typ):
    groups = [torch.nonzero(typ == t)[:, 0] for t in (T_TOK, T_POS, T_MLP, T_ATT, T_BIAS) if (typ == t).any()]
    return dict(own=A, rot=rotate(A, seed=7), covA=gauss_like(A.shape[0], (A.T @ A) / A.shape[0], seed=1), mix8=mixtures(A, groups, m=8, seed=4))
def shares(cells, k):
    o, r = cells["own"][str(k)]["rec"], cells["rot"][str(k)]["rec"]; gap = o - r
    return dict(gap=gap, covA=(cells["covA"][str(k)]["rec"] - r) / gap if abs(gap) > 1e-3 else None, mix8=(cells["mix8"][str(k)]["rec"] - r) / gap if abs(gap) > 1e-3 else None)
res = dict(model=name, rev=rev, level=L)
# the increment, in the block's own words
Ab, lab = build_dictionary(arch, blocks=[L]); keep = lab["block"] == L; Ab = Ab[keep.to(DEV)]; tb = lab["type"][keep].to(DEV)
il = IncLevel(model, arch, ev, L); KS = [4, 16, 64]
res["increment"] = dict(gap=il.gap, n_words=int(Ab.shape[0]), cells={vn: describe(il, V, KS) for vn, V in controls(Ab, tb).items()})
res["increment"]["shares"] = {str(k): shares(res["increment"]["cells"], k) for k in KS}
c_ = il.acts * arch.wdir(L).norm(dim=-1)[None]; pr = (c_.pow(2).sum(-1).pow(2) / c_.pow(4).sum(-1).clamp_min(1e-12)); res["increment"]["mlp_write_participation_ratio_median"] = pr.median().item()
res["increment"]["mlp_share_of_increment_energy"] = ((il.acts @ arch.wdir(L)).pow(2).sum() / (il.Xc + il.mu).pow(2).sum()).item()
del il, Ab
# the state, in the words of blocks 0..L
A, lab = build_dictionary(arch, blocks=list(range(L + 1))); lv = Level(model, arch, ev, L)
res["state"] = dict(gap=lv.gap, cells={vn: describe(lv, V, [4, 16]) for vn, V in controls(A, lab["type"].to(DEV)).items()})
res["state"]["shares"] = {str(k): shares(res["state"]["cells"], k) for k in (4, 16)}
fm = lambda x: "n/a" if x is None else f"{x:.2f}"
ic, sc = res["increment"], res["state"]
res["checks"] = dict(increment_wordlevel_k16=all(v is not None and v < 0.5 for v in (ic["shares"]["16"]["covA"], ic["shares"]["16"]["mix8"])),
                     state_accent_k16=any(v is not None and v > 0.5 for v in (sc["shares"]["16"]["covA"], sc["shares"]["16"]["mix8"])))
summ = (f"{name} {rev or 'final'} L{L}: increment k4/k16/k64 own " + "/".join(f"{ic['cells']['own'][str(k)]['rec']:.2f}" for k in KS) + " rot " + "/".join(f"{ic['cells']['rot'][str(k)]['rec']:.2f}" for k in KS)
        + " covA " + "/".join(f"{ic['cells']['covA'][str(k)]['rec']:.2f}" for k in KS) + " mix8 " + "/".join(f"{ic['cells']['mix8'][str(k)]['rec']:.2f}" for k in KS)
        + f" -> shares k16 covA {fm(ic['shares']['16']['covA'])} mix8 {fm(ic['shares']['16']['mix8'])} (k4 {fm(ic['shares']['4']['covA'])}/{fm(ic['shares']['4']['mix8'])})"
        + f" | MLP write participation ratio {ic['mlp_write_participation_ratio_median']:.0f} of {arch.DFF} | state k4/k16 own " + "/".join(f"{sc['cells']['own'][str(k)]['rec']:.2f}" for k in (4, 16))
        + " rot " + "/".join(f"{sc['cells']['rot'][str(k)]['rec']:.2f}" for k in (4, 16)) + f" -> shares k16 covA {fm(sc['shares']['16']['covA'])} mix8 {fm(sc['shares']['16']['mix8'])} | checks {json.dumps(res['checks'])}")
log(summ); record(f"e436_increment_{name}_{rev or 'final'}", res, summ)
