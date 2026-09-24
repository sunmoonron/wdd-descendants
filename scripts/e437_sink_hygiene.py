"""e437: which phase-3 results survive keeping the sink positions exact? e432 found that at middle depth one position
per sequence ('\\n\\n\\n' in Pythia-410m, 41x the median norm) holds 0.83 of the centred variance. OLMo has a
similar outlier (0.44). Phase 3's Level never excluded them, so every number built on "M's share of the variance" or on
describing all positions inherits them.
Per model, middle depth, 6 sequences; the standard Level (all positions) against NSLevel (sinks kept exact, centring,
descriptions and the mean baseline on typical positions only):
 (a) loss recovered by own words, rotation, covA and mix8 at k 4 and 16, and the covA and mix8 shares (e426a);
 (b) M_ns (top-8 at typical positions): its variance share there, its share of the readers' trace (weights only), of
     the unembedding's and of the Fisher trace; M from all positions for comparison (e400, e426b);
 (c) Zipf slope of own-word usage and the M_ns share of the 10 most used words (e414);
 (d) Pythia only: false friends, i.e. step-4000 words on the final states at k 4 and 16, whole state and with M_ns
     kept exact (e397, e404).
Pre-registered: the self-description numbers (a, c) change by less than 0.03 in every model; in Pythia and OLMo the
Fisher-to-variance ratio of M_ns is at least three times larger than the ratio reported with the sinks; the false
friends survive (step-4000 words below their rotation at k 4 with sinks exact)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; model, tok, fam = load_model(name); arch = Arch(model, fam); L = arch.NB // 2; D = arch.D
E = eval_ids(name); ev = E[:6].to(DEV); fit = E[6:12].to(DEV)

class NSLevel(Level):
    """Level with the sink positions (norm above 10x the median) kept exact"""
    def __init__(self, model, arch, ids, L):
        self.model, self.arch, self.ids, self.L, self.B, self.T, self.P = model, arch, ids, L, ids.shape[0], ids.shape[1], None
        out = {}; h = arch.layers[L].register_forward_hook(lambda m, i, o: out.__setitem__("x", out_of(o).detach().float()))
        try:
            with torch.no_grad(): self.lc = token_loss(model(ids).logits.float(), ids).view(self.B, -1)
        finally: h.remove()
        x = out["x"][:, 1:]; self.keep = ~sinkmask(x).reshape(-1); self.full = x.reshape(-1, x.shape[-1])
        self.mu = self.full[self.keep].mean(0, keepdim=True); self.Xc = self.full[self.keep] - self.mu
        self.lm = self.splice(self.mu.expand(self.Xc.shape[0], -1)); self.gap = (self.lm.mean() - self.lc.mean()).item()
    def splice(self, Xh):
        f = self.full.clone(); f[self.keep] = Xh.to(f.dtype); return Level.splice(self, f)

A, lab = build_dictionary(arch, blocks=list(range(L + 1))); typ = lab["type"].to(DEV)
groups = [torch.nonzero(typ == t)[:, 0] for t in (T_TOK, T_POS, T_MLP, T_ATT, T_BIAS) if (typ == t).any()]
V = dict(own=A, rot=rotate(A, seed=7), covA=gauss_like(A.shape[0], (A.T @ A) / A.shape[0], seed=1), mix8=mixtures(A, groups, m=8, seed=4))
lv_all, lv_ns = Level(model, arch, ev, L), NSLevel(model, arch, ev, L)
res = dict(model=name, level=L, n_sinks=int((~lv_ns.keep).sum()), gap_all=lv_all.gap, gap_ns=lv_ns.gap, a={}, b={}, c={})
for tag, lv in (("all", lv_all), ("ns", lv_ns)):
    cells = {vn: describe(lv, W, [4, 16]) for vn, W in V.items()}
    sh = lambda c, k: (cells[c][str(k)]["rec"] - cells["rot"][str(k)]["rec"]) / max(cells["own"][str(k)]["rec"] - cells["rot"][str(k)]["rec"], 1e-9)
    res["a"][tag] = dict(cells=cells, shares={f"{c}_k{k}": sh(c, k) for c in ("covA", "mix8") for k in (4, 16)})
    log(f"{name} {tag}: k4/k16 " + " ".join(f"{vn} {cells[vn]['4']['rec']:.2f}/{cells[vn]['16']['rec']:.2f}" for vn in V) + f" | shares k16 covA {res['a'][tag]['shares']['covA_k16']:.2f} mix8 {res['a'][tag]['shares']['mix8_k16']:.2f}")
# (b) shares of the variance, the readers' trace and the Fisher trace
xf = block_states(model, arch, fit, [L], chunk=3)[L]; kf = ~sinkmask(xf)
U_all, _ = pcs(xf.reshape(-1, D), 8); U_ns, _ = pcs(xf[kf], 8)
GR, GU = reader_gram(model, arch, list(range(L + 1, arch.NB))); GF = fisher_gram(model, arch, ev, L)
for un, U in (("M_all", U_all), ("M_ns", U_ns)):
    v_all = ((lv_all.Xc @ U).pow(2).sum() / lv_all.Xc.pow(2).sum()).item(); v_ns = ((lv_ns.Xc @ U).pow(2).sum() / lv_ns.Xc.pow(2).sum()).item()
    res["b"][un] = dict(var_all_positions=v_all, var_typical=v_ns, readers=share_on(GR, U), unembed=share_on(GU, U), fisher=share_on(GF, U), chance=8 / D)
    res["b"][un]["fisher_over_var_typical"] = res["b"][un]["fisher"] / max(v_ns, 1e-9); res["b"][un]["fisher_over_var_all"] = res["b"][un]["fisher"] / max(v_all, 1e-9)
    b = res["b"][un]; log(f"{name} {un}: variance share all {b['var_all_positions']:.2f} typical {b['var_typical']:.2f} | readers {b['readers']:.3f} unembedding {b['unembed']:.3f} Fisher {b['fisher']:.3f} (chance {b['chance']:.3f})")
# (c) Zipf and the top words' M share, on typical positions
mshare = (A @ U_ns).pow(2).sum(-1)
for tag, lv in (("all", lv_all), ("ns", lv_ns)):
    sel, _, _ = omp(lv.Xc, A, 16, batch=256, record_err=False); cnt = torch.bincount(sel.flatten(), minlength=A.shape[0]).float(); del sel
    f = cnt.sort(descending=True).values; f = f[f > 0]; r = torch.arange(1, f.numel() + 1, device=DEV).float(); hi = min(1000, f.numel()); x, y = r[9:hi].log(), f[9:hi].log()
    top = cnt.topk(10).indices; res["c"][tag] = dict(zipf=(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()).item(), top10_share=(cnt[top].sum() / cnt.sum()).item(), top10_Mns_share=mshare[top].mean().item())
# (d) false friends, Pythia only
if name == "pythia410":
    m4, _, _ = load_model(name, revision="step4000"); a4 = Arch(m4, fam); V4, _ = build_dictionary(a4, blocks=list(range(L + 1))); del m4, a4
    PM = U_ns @ U_ns.T; res["d"] = {}
    for tag, lv in (("all", lv_all), ("ns", lv_ns)):
        out = {}
        for vn, W in (("own", A), ("rot", V["rot"]), ("f4000", V4), ("f4000_rot", rotate(V4, seed=7))):
            out[vn] = {k: v["rec"] for k, v in describe(lv, W, [4, 16]).items()}
            XM = lv.Xc @ PM; Xp = lv.Xc - XM; Wp = unitr(unitr(W) - unitr(W) @ PM); sel, _, _ = omp(Xp, Wp, 16, batch=256, record_err=False); outB = {}
            for k in (4, 16):
                cof, _ = refit(Xp, Wp, sel[:, :k]); Xh = torch.einsum("nk,nkd->nd", cof, Wp[sel[:, :k]]); outB[str(k)] = lv.recovered(lv.splice(lv.mu + XM + Xh))[0]
            out[vn + "_Mexact"] = outB; del sel
        res["d"][tag] = out
        log(f"{name} false friends ({tag}): k4/k16 " + " ".join(f"{vn} {v['4']:.2f}/{v['16']:.2f}" for vn, v in out.items()))
a_, b_, c_ = res["a"], res["b"], res["c"]
res["checks"] = dict(selfdesc_stable=all(abs(a_["all"]["cells"][vn][k]["rec"] - a_["ns"]["cells"][vn][k]["rec"]) < 0.03 for vn in V for k in ("4", "16")),
                     fisher_ratio_3x=b_["M_ns"]["fisher_over_var_typical"] >= 3 * b_["M_all"]["fisher_over_var_all"])
if "d" in res: res["checks"]["false_friends_survive"] = res["d"]["ns"]["f4000"]["4"] < res["d"]["ns"]["f4000_rot"]["4"]
summ = (f"{name} L{L}: sinks {res['n_sinks']} | k16 own/rot/covA/mix8 all positions " + "/".join(f"{a_['all']['cells'][vn]['16']['rec']:.2f}" for vn in V) + " vs sinks exact " + "/".join(f"{a_['ns']['cells'][vn]['16']['rec']:.2f}" for vn in V)
        + f" (shares covA {a_['all']['shares']['covA_k16']:.2f}->{a_['ns']['shares']['covA_k16']:.2f}, mix8 {a_['all']['shares']['mix8_k16']:.2f}->{a_['ns']['shares']['mix8_k16']:.2f})"
        + f" | M_ns variance share typical {b_['M_ns']['var_typical']:.2f} (M with sinks: {b_['M_all']['var_all_positions']:.2f}), readers {b_['M_ns']['readers']:.3f}, unembedding {b_['M_ns']['unembed']:.3f}, Fisher {b_['M_ns']['fisher']:.3f} (chance {b_['M_ns']['chance']:.3f})"
        + f" -> Fisher/variance {b_['M_all']['fisher_over_var_all']:.3f} with sinks, {b_['M_ns']['fisher_over_var_typical']:.3f} without | Zipf {c_['all']['zipf']:.2f}->{c_['ns']['zipf']:.2f}, top-10 M_ns share {c_['all']['top10_Mns_share']:.2f}->{c_['ns']['top10_Mns_share']:.2f}"
        + (f" | false friends k4 (sinks exact): f4000 {res['d']['ns']['f4000']['4']:.2f} vs its rotation {res['d']['ns']['f4000_rot']['4']:.2f}; M_ns exact {res['d']['ns']['f4000_Mexact']['4']:.2f} vs {res['d']['ns']['f4000_rot_Mexact']['4']:.2f}" if "d" in res else "")
        + f" | checks {json.dumps(res['checks'])}")
log(summ); record(f"e437_sinkhygiene_{name}", res, summ)
