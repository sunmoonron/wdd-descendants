"""Cross-model tables from results/*.json. usage: python agg.py <exp-prefix> [...]"""
import json, glob, os, sys
R = "/workspace/wdd/results"
def load(prefix):
    out = {}
    for f in sorted(glob.glob(f"{R}/{prefix}_*.json")):
        m = os.path.basename(f)[len(prefix) + 1:-5]; out[m] = json.load(open(f))
    return out
def f(x, n=3):
    try: return f"{x:.{n}f}"
    except: return str(x)
for pre in sys.argv[1:]:
    D = load(pre); print(f"\n===== {pre} ({len(D)} models) =====")
    if pre == "e00_gate":
        for m, r in D.items(): print(f"{m:22s} fvu32 {f(r['fvu32_typ'])} rot {f(r['rot32_typ'])} fvu64 {f(r['fvu64_typ'])} sink {f(r['fvu32_sink'],4)} recall {f(r['recall']['64'])} oneshot {f(r['recall_oneshot']['64'])} sign {f(r['sign'])} relerr {f(r['med_rel_err'],2)}")
    elif pre == "e01_layers":
        for m, r in D.items():
            print(m); print("  L    fvu32 rot32 fvu64  sink  rec64 os64  embrec embsurv | support tok/pos/mlp/att/bias")
            for x in r["rows"]: print(f"  {x['L']:2d}  {f(x['fvu32'])} {f(x['rot32'])} {f(x['fvu64'])} {f(x['sink_share'],2)} {f(x['recall']['64'])} {f(x['oneshot64'])} {f(x['emb_recall']['64'])} {f(x['emb_survival_med'],2)} | " + "/".join(f(x['support_types'][str(t)],2) for t in range(5)))
    elif pre == "e02_delta":
        for m, r in D.items():
            print(m); print("  b   d@1   d@8   d@32  g@64 | fvu8 fvu32 rot32 | relerr sign | attshare cancel_mlp")
            for x in r["rows"]: print(f"  {x['b']:2d}  {f(x['recall_delta']['1'],2)} {f(x['recall_delta']['8'],2)} {f(x['recall_delta']['32'],2)} {f(x['recall_global']['64'],2)} | {f(x['fvu8'],2)} {f(x['fvu32'],2)} {f(x['rot32'],2)} | {f(x['med_rel_err'],2)} {f(x['sign'],2)} | {f(x['att_share'],2)} {f(x['self_cancel_mlp'],2)}")
    elif pre == "e03_dict":
        for m, r in D.items(): print(f"{m:12s} mu {f(r['mutual_coherence'])} babel8 {f(r['babel'][7],1)} babel64 {f(r['babel'][63],1)} | align w {f(r['align']['weight'])} rot {f(r['align']['rot7'])} rand {f(r['align']['random'])} | maxcorr w {f(r['matched_filter']['weight']['max_mean'])} rot {f(r['matched_filter']['rot']['max_mean'])} rand {f(r['matched_filter']['random']['max_mean'])} top8 w {f(r['matched_filter']['weight']['top8_mean'])} rot {f(r['matched_filter']['rot']['top8_mean'])} | ERC med {f(r['erc']['median'],2)} <1 {f(r['erc']['frac_below_1'],2)} AUC1 {f(r['erc']['auc_top1_hit'],2)} AUC3 {f(r['erc']['auc_top3_hit'],2)} AUCcoh {f(r['erc']['auc_maxcoh_top1_hit'],2)} | maxc by type " + " ".join(f"{t}:{f(v['median'],2)}" for t, v in r['maxc_by_type'].items()) + f" | eff rank frame {f(r['frame_eff_rank'],0)} state {f(r['state_eff_rank'],0)}")
    elif pre == "e04_solvers":
        for m, r in D.items():
            print(m)
            for k, v in r["solvers"].items(): print(f"  {k:18s} r1 {f(v['recall1'])} r3 {f(v['recall3'])} sign {f(v['sign'],2)} relerr {f(v['med_rel_err'],2)} ratio {f(v['med_ratio'],2)} fvu {f(v['fvu_refit'])} {f(v['time'],0)}s")
    elif pre == "e05_neuron":
        for m, r in D.items(): print(f"{m:22s} recall {f(r['recall'])} | AUC " + " ".join(f"{k} {f(v,2)}" for k, v in r['auc'].items()) + f" | ICC {f(r['neuron_level']['icc'],2)} always {f(r['neuron_level']['frac_always'],2)} never {f(r['neuron_level']['frac_never'],2)} n {r['neuron_level']['n_neurons_ge20']} | rho " + " ".join(f"{k} {f(v,2)}" for k, v in r['neuron_level']['spearman_recall_vs'].items()) + f" | top1 conc {f(r['top1_concentration']['most_frequent_share'],3)} distinct {r['top1_concentration']['n_distinct']}")
    elif pre == "e06_energy":
        for m, r in D.items():
            print(m); print("  L  cancel  emb/att/mlp share   embsurv(q10,med,q90)  cos(emb,x)")
            for x in r["rows"]:
                s = f"  {x['L']:2d}  {f(x['cancel_index'],3)}  {f(x['share_emb'],2)}/{f(x['share_att'],2)}/{f(x['share_mlp'],2)}   {f(x['emb_survival_q10'],2)},{f(x['emb_survival_med'],2)},{f(x['emb_survival_q90'],2)}  {f(x['cos_emb_x'],2)}"
                if "oracle" in x: s += " | ORACLE " + " ".join(f"k{k}: sum {f(v['fvu_sum'],2)} refit {f(v['fvu_refit'],2)} omp {f(x['omp_fvu'][k],2)} mlpfrac {f(v['mlp_frac'],2)}" for k, v in x["oracle"].items())
                print(s)
    else:
        for m, r in D.items(): print(m, json.dumps({k: v for k, v in r.items() if not k.startswith("_") and k not in ("rows", "cells")}, default=str)[:1500])
