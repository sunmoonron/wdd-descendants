"""e437b: the 7B numbers under the sink check. e418 reported that at Qwen2.5-7B's middle depth the top-8 principal
directions hold 0.97 of the variance and 0.003 of the Fisher trace, with self-description k16 own 0.79 against rotation
0.26. e432 showed that the same kind of number in Pythia-410m (0.86) was one sink position per sequence.
Here, 4 sequences, middle depth:
- sink positions (norm above 10x the median) and their share of the centred variance;
- M from all positions against M_ns from typical positions (fitted on 4 other sequences), their variance shares at
  typical positions and their overlap;
- M_ns's share of the Fisher trace;
- own words against rotation at k 4 and 16, with all positions described and with the sinks kept exact."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ws_common import *
from ma_common import NSLevel, sinkmask, block_states, pcs
from sd_common import Level, describe, fisher_gram, share_on, rotate, unitr
name = sys.argv[1] if len(sys.argv) > 1 else "qwen7"
model, tok, fam = load_bf16(name); arch = Arch(model, fam); L = arch.NB // 2; D = arch.D
E = torch.load("/workspace/wdd/eval_ids_all.pt", weights_only=False)["qwen05"]["eval_ids"]; ev = E[:4].to(DEV); fit = E[4:8].to(DEV)
xf = block_states(model, arch, fit, [L], chunk=1)[L]; kf = ~sinkmask(xf); xe = block_states(model, arch, ev, [L], chunk=1)[L]; ke = ~sinkmask(xe)
U_all, _ = pcs(xf.reshape(-1, D), 8); U_ns, _ = pcs(xf[kf], 8)
Xa = xe.reshape(-1, D) - xe.reshape(-1, D).mean(0); Xn = xe[ke] - xe[ke].mean(0); sh = lambda X, U: ((X @ U).pow(2).sum() / X.pow(2).sum()).item()
n_ = xe.norm(dim=-1); sk = torch.nonzero(~ke)[:12].tolist()
res = dict(model=name, level=L, n_sinks=int((~ke).sum()), sinks=[(s, p + 1, tok.decode([int(ev[s, p + 1])]), round((n_[s, p] / n_.median()).item(), 1)) for s, p in sk],
           sink_share_of_variance=(Xa[(~ke).reshape(-1)].pow(2).sum() / Xa.pow(2).sum()).item(), M_all_share_all=sh(Xa, U_all), M_all_share_typical=sh(Xn, U_all), M_ns_share_typical=sh(Xn, U_ns),
           overlap=((U_all.T @ U_ns).pow(2).sum() / 8).item())
nf = xf.norm(dim=-1); Xfa = xf.reshape(-1, D) - xf.reshape(-1, D).mean(0); evf = torch.linalg.eigvalsh(torch.cov(Xfa.T.double(), correction=0))
res.update(fit_n_sinks=int((~kf).sum()), fit_sinks=[(s, p + 1, tok.decode([int(fit[s, p + 1])]), round((nf[s, p] / nf.median()).item(), 1)) for s, p in torch.nonzero(~kf)[:12].tolist()],
           fit_top8_insample_all=(evf[-8:].sum() / evf.sum()).item(), fit_sink_share=(Xfa[(~kf).reshape(-1)].pow(2).sum() / Xfa.pow(2).sum()).item())
del Xfa, evf
log(f"{name} audit: " + json.dumps(res))
GF = fisher_gram(model, arch, ev[:2], L); res["fisher_M_ns"] = share_on(GF, U_ns); res["fisher_M_all"] = share_on(GF, U_all); res["fisher_chance"] = 8 / D; del GF
del xf, xe; torch.cuda.empty_cache()
A, blk, typ, ends = lean_dictionary(arch, L); A = A[:ends[L]]
out = {}
for tag, lv in (("all", Level(model, arch, ev, L)), ("sinks_exact", NSLevel(model, arch, ev, L))):
    out[tag] = dict(own=describe(lv, A, [4, 16], batch=128))
    g = torch.Generator().manual_seed(7); R = torch.linalg.qr(torch.randn(D, D, generator=g))[0].to(DEV)
    Xc0 = lv.Xc; lv.Xc = Xc0 @ R.T                                        # rotated vocabulary A R emulated by rotating the targets
    sel, _, _ = omp(lv.Xc, A, 16, batch=128, record_err=False); rr = {}
    for k in (4, 16):
        cof, _ = refit(lv.Xc, A, sel[:, :k]); Xh = torch.einsum("nk,nkd->nd", cof, A[sel[:, :k]]) @ R; rr[str(k)] = lv.recovered(lv.splice(lv.mu + Xh))[0]
    lv.Xc = Xc0; out[tag]["rot"] = rr; del sel, lv
res["selfdesc"] = {t: dict(own={k: v["rec"] for k, v in o["own"].items()}, rot=o["rot"]) for t, o in out.items()}
S_ = res["selfdesc"]
summ = (f"{name} L{L}: fit text: sinks {res['fit_n_sinks']} {res['fit_sinks'][:4]} hold {res['fit_sink_share']:.2f} of its variance; its in-sample top-8 share {res['fit_top8_insample_all']:.2f} (e418's number) | eval text: sinks {res['n_sinks']} {res['sinks'][:6]} hold {res['sink_share_of_variance']:.2f} of the variance | M share all positions {res['M_all_share_all']:.2f}, typical {res['M_all_share_typical']:.2f}; M_ns {res['M_ns_share_typical']:.2f}; overlap {res['overlap']:.2f}"
        f" | Fisher share M_ns {res['fisher_M_ns']:.4f} (M {res['fisher_M_all']:.4f}, chance {res['fisher_chance']:.4f}) | k4/k16 own/rot all positions {S_['all']['own']['4']:.2f}/{S_['all']['own']['16']:.2f} vs {S_['all']['rot']['4']:.2f}/{S_['all']['rot']['16']:.2f}; "
        f"sinks exact {S_['sinks_exact']['own']['4']:.2f}/{S_['sinks_exact']['own']['16']:.2f} vs {S_['sinks_exact']['rot']['4']:.2f}/{S_['sinks_exact']['rot']['16']:.2f}")
log(summ); record(f"e437b_sinkhygiene_{name}", res, summ)
