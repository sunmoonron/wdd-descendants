"""e657 (session 143; Gemini 4's RYS idea 1): a one-pass virtual repeat. Instead of traversing a block [i, j) a second
time, the band's increment d = x_j - x_i is decomposed at the output of block j-1 into its 16-word OMP chord over the
band's own MLP write rows, and x_j <- x_j + alpha * chord (alpha 0.5, 1), in a single pass. Gemini's claim: doubling
only the sparse chord amplifies the band's speakers while leaving its cancelling crowd behind, so it should beat
scaling the raw increment. Kill baselines: the raw increment x_j + alpha * d, the raw increment scaled to the chord's
norm at each position (energy matched), and the chord from the rotated dictionary. Compared with the true repeat of
the same block (e656), on the three least damaging blocks of two or more layers and the two with the best probe gain.
Scored by the change in wikitext-2 loss and in probe accuracy. Arguments: model [--smoke]."""
from rys_common import *
from rys_common import _hs, _out
name = sys.argv[1]; SMOKE = "--smoke" in sys.argv; t0 = time.time(); tag = "_smoke" if SMOKE else ""
M = load(name); N = M["NB"]; D = M["D"]; A, blk = dictionary(M); DFF = A.shape[0] // N
W = wikitext(M, "test", 6 if SMOKE else 24, 128); P = probes(M); KD = kinds(P); base_loss = lm_loss(M, W); STOP = stopset(M, P); base_ok, base_okm = probe_acc_both(M, P, STOP)
R6 = json.load(open(f"results/e656_rys_map_{name}{tag}.json")); s2 = [r for r in R6["rows"] if r["s"] >= 2]
bands = []
for r in sorted(s2, key=lambda r: r["dloss"])[:1 if SMOKE else 3] + sorted(s2, key=lambda r: -r["gain"])[:1 if SMOKE else 2]:
    if (r["i"], r["j"]) not in [(b["i"], b["j"]) for b in bands]: bands.append(r)
print(f"[{name}] base loss {base_loss:.4f}, accuracy {base_ok.mean():.3f}; bands " + ", ".join(f"{b['i']}-{b['j']}" for b in bands) + f" ({time.time() - t0:.0f}s)", flush=True)
def chord(xi, xj, Ab):
    d = (xj - xi).reshape(-1, D); n = d.norm(dim=-1, keepdim=True); sel, cof = omp(unitr(d), Ab, batch=512); return ((cof[..., None] * Ab[sel]).sum(1) * n).reshape(xj.shape)
out = []
for b in bands:
    i, j = b["i"], b["j"]; Ab = A[i * DFF: j * DFF]; Abr = unitr(rotate(Ab)); rec = dict(i=i, j=j, true_dloss=b["dloss"], true_gain=b["gain"], true_gain_m=b["gain_m"], variants={})
    for alpha in (0.5, 1.0):
        V = {"native_chord": lambda xi, xj, a=alpha: a * chord(xi, xj, Ab), "rotated_chord": lambda xi, xj, a=alpha: a * chord(xi, xj, Abr), "raw": lambda xi, xj, a=alpha: a * (xj - xi),
             "raw_energy_matched": lambda xi, xj, a=alpha: a * (xj - xi) * (chord(xi, xj, Ab).norm(dim=-1, keepdim=True) / (xj - xi).norm(dim=-1, keepdim=True).clamp_min(1e-6))}
        for vn, add in V.items():
            with Inject(M, i, j, add): loss = lm_loss(M, W); ok, okm = probe_acc_both(M, P, STOP)
            rec["variants"][f"{vn}@{alpha}"] = dict(dloss=loss - base_loss, gain=float(ok.mean() - base_ok.mean()), gain_m=float(okm.mean() - base_okm.mean()))
    # the chord's share of the increment's energy
    with torch.no_grad():
        cap = {}
        h1 = M["layers"][i].register_forward_pre_hook(lambda m, a, kw: cap.__setitem__("xi", _hs(a, kw).detach()), with_kwargs=True); h2 = M["layers"][j - 1].register_forward_hook(lambda m, a, kw, o: cap.__setitem__("xj", _out(o).detach()), with_kwargs=True)
        M["model"](W[:8], use_cache=False); h1.remove(); h2.remove()
        d = (cap["xj"] - cap["xi"])[:, 1:].reshape(-1, D); c = chord(cap["xi"][:, 1:], cap["xj"][:, 1:], Ab).reshape(-1, D); rec["chord_energy_share"] = float((c.norm(dim=-1) ** 2 / d.norm(dim=-1).clamp_min(1e-6) ** 2).mean())
    out.append(rec); print(f"[{name}] band {i}-{j}: true repeat dloss {b['dloss']:+.4f} gain {b['gain']:+.3f}; chord energy share {rec['chord_energy_share']:.2f} | " + " | ".join(f"{k} {v['dloss']:+.4f}/{v['gain']:+.3f}" for k, v in rec["variants"].items()) + f" ({time.time() - t0:.0f}s)", flush=True)
v = lambda r, k: r["variants"][k]
gainers = [r for r in out if r["true_gain"] > 0]
res = dict(model=name, base_loss=base_loss, base_acc=float(base_ok.mean()), bands=out)
res["verdicts"] = dict(V1=sum(v(r, "native_chord@1.0")["dloss"] < v(r, "raw_energy_matched@1.0")["dloss"] for r in out) > len(out) / 2,
                       V2=sum(v(r, "native_chord@1.0")["dloss"] < v(r, "rotated_chord@1.0")["dloss"] for r in out) > len(out) / 2,
                       V3=bool(gainers) and all(max(v(r, "native_chord@0.5")["gain"], v(r, "native_chord@1.0")["gain"]) >= 0.25 * r["true_gain"] for r in gainers),
                       V4=all(x["dloss"] >= 0 for r in out for x in r["variants"].values()),
                       V5=bool(gainers) and np.mean([max(v(r, "native_chord@0.5")["gain"], v(r, "native_chord@1.0")["gain"]) for r in gainers]) > np.mean([max(v(r, "rotated_chord@0.5")["gain"], v(r, "rotated_chord@1.0")["gain"]) for r in gainers]))
summ = f"[{name}] virtual repeat on {len(out)} bands: " + "; ".join(f"{r['i']}-{r['j']} true {r['true_dloss']:+.3f}, chord@1 {v(r, 'native_chord@1.0')['dloss']:+.3f}, energy-matched raw@1 {v(r, 'raw_energy_matched@1.0')['dloss']:+.3f}, rotated@1 {v(r, 'rotated_chord@1.0')['dloss']:+.3f}, raw@1 {v(r, 'raw@1.0')['dloss']:+.3f}" for r in out) + f"; verdicts {res['verdicts']} ({time.time() - t0:.0f}s)"
save(f"e657_virtual_rys_{name}{tag}", res, summ); print("done", flush=True)
