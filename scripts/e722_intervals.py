"""e722 (session 163): intervals over neurons. The record's ratios are medians over forty neurons per model-block. Here each key
ratio of e697 (the brake), e716 (other choices), e706 (the fields) and e713 (both at 7B) gets a 95 percent bootstrap interval of
the median (2000 resamples over neurons) and the share of neurons showing the effect, from the saved per-neuron results.
Arguments: --results DIR [--out FILE]. Pre-registered in e717_e722_prereg.json. Runs on a CPU."""
import argparse, glob, json, os, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--results", required=True); ap.add_argument("--out", default=None); A = ap.parse_args()
rng = np.random.default_rng(722)
def boot(v, nb=2000):
    v = np.asarray([x for x in v if x is not None], float); n = len(v)
    if n == 0: return None
    med = float(np.median(v)); idx = rng.integers(0, n, (nb, n)); bs = np.median(v[idx], axis=1); return dict(n=n, median=med, lo=float(np.quantile(bs, 0.025)), hi=float(np.quantile(bs, 0.975)))
def rows_of(d): return list(d["neurons"].values())
out = {}
def fam(prefix):
    return sorted(f for f in glob.glob(f"{A.results}/{prefix}_*.json") if "smoke" not in f and "prereg" not in f)
# e697 and e716: the brake (the 7B runs of e713 are e697 files too)
for f in fam("e697") + fam("e716"):
    d = json.load(open(f)); R = rows_of(d); key = os.path.basename(f)[:-5]; e = {}
    if "abs16_off" in R[0]["sim"]: off = [r["sim"]["abs16_off"]["all"] / max(r["sim"]["abs16"]["all"], 1e-12) for r in R]
    else: off = [r["off_share"].get("5", r["off_share"].get(5)) for r in R]
    e["off_share_5"] = boot(off); e["share_neurons_off_above_half"] = float(np.mean([x > 0.5 for x in off]))
    for c in ("pos16", "pos16neg16", "abs16", "abs32", "pos32", "pos32neg32"):
        if c in R[0]["sim"]: e[f"{c}_all"] = boot([r["sim"][c]["ratio_all"] for r in R]); e[f"{c}_firings"] = boot([r["sim"][c]["ratio_strong"] for r in R])
    if "pos16" in R[0]["sim"]: e["share_neurons_pos16_runaway"] = float(np.mean([r["sim"]["pos16"]["ratio_all"] > 1 for r in R]))
    out[key] = e
# e706: the fields (the 7B runs of e713 are e706 files too)
for f in fam("e706"):
    d = json.load(open(f)); R = rows_of(d); key = os.path.basename(f)[:-5]; e = {}
    for c in ("abs16", "abs16_both", "abs16_weak", "abs16_strong", "abs16_random"):
        if c in R[0]["sim"]: e[f"{c}_firings"] = boot([r["sim"][c]["ratio_strong"] for r in R])
    e["both_minus_abs16_firings"] = boot([r["sim"]["abs16_both"]["ratio_strong"] - r["sim"]["abs16"]["ratio_strong"] for r in R]); e["share_neurons_both_helps"] = float(np.mean([r["sim"]["abs16_both"]["ratio_strong"] < r["sim"]["abs16"]["ratio_strong"] for r in R]))
    out[key] = e
E697 = [k for k in out if k.startswith("e697")]; E706 = [k for k in out if k.startswith("e706")]
V = dict(I1_off_share_lower_above_04=bool(all(out[k]["off_share_5"]["lo"] > 0.4 for k in E697)), I2_pos16_lower_above_1=bool(all(out[k]["pos16_all"]["lo"] > 1 for k in E697)),
         I3_both_upper_below_abs16_lower=bool(all(out[k]["abs16_both_firings"]["hi"] < out[k]["abs16_firings"]["lo"] for k in E706)), I4_three_quarters_off_above_half=bool(all(out[k]["share_neurons_off_above_half"] >= 0.75 for k in E697)))
res = dict(_exp="e722", intervals=out, verdicts=V, n_runs=dict(e697=len(E697), e716=len([k for k in out if k.startswith("e716")]), e706=len(E706)))
lines = [f"{k}: off share 5% {v['off_share_5']['median']:.2f} [{v['off_share_5']['lo']:.2f}, {v['off_share_5']['hi']:.2f}], neurons above half {v['share_neurons_off_above_half']:.2f}, pos16 all {v['pos16_all']['median']:.1f} [{v['pos16_all']['lo']:.1f}, {v['pos16_all']['hi']:.1f}]" for k, v in out.items() if "off_share_5" in v]
lines += [f"{k}: abs16 at the firings {v['abs16_firings']['median']:.2f} [{v['abs16_firings']['lo']:.2f}, {v['abs16_firings']['hi']:.2f}], both {v['abs16_both_firings']['median']:.2f} [{v['abs16_both_firings']['lo']:.2f}, {v['abs16_both_firings']['hi']:.2f}], neurons helped {v['share_neurons_both_helps']:.2f}" for k, v in out.items() if "abs16_both_firings" in v]
res["summary"] = "; ".join(lines) + f"; verdicts {V}"; print("\n".join(lines)); print(V)
json.dump(res, open(A.out or f"{A.results}/e722_intervals.json", "w"), indent=1)
