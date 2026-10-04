"""e7: one compact text summary of every result file present."""
import json, os, sys
R = "/data/loopwdd/results"
def J(n):
    p = os.path.join(R, n); return json.load(open(p)) if os.path.exists(p) else None
f = lambda x, n=3: "NA" if x is None else (f"{x:.{n}f}" if isinstance(x, (int, float)) else str(x))
e1o, e1s, e2o, e2s = J("e1_ouro.json"), J("e1_smol.json"), J("e2_ouro.json"), J("e2_smol.json")
if e1o: print("CE by loop:", " ".join(f(x) for x in e1o["ce_by_loop"]), "| SmolLM2", f(e1s["ce_by_loop"][0]) if e1s else "NA")
if e2o:
    print("FVU before layer 12 (native / rotated / one-shot), k16 and k64:")
    for t, r in e2o["loops"].items():
        print(f"  loop {int(t)+1}: k16 {f(r['k16']['native'])} / {f(r['k16']['rotated'])} / {f(r['k16']['oneshot'])}   k64 {f(r['k64']['native'])} / {f(r['k64']['rotated'])} / {f(r['k64']['oneshot'])}   types {r['k16_share']}")
if e2s:
    r = e2s["loops"]["0"]; print(f"  SmolLM2 L12: k16 {f(r['k16']['native'])} / {f(r['k16']['rotated'])} / {f(r['k16']['oneshot'])}   k64 {f(r['k64']['native'])} / {f(r['k64']['rotated'])} / {f(r['k64']['oneshot'])}")
for kind in ("ouro", "smol"):
    s = J(f"e3_{kind}_single.json")
    if not s: continue
    print(f"stand-in, loss recovered ({kind}):")
    for key, row in s.items():
        if not key.startswith("loop"): continue
        ms = [mt for mt in ("native", "rotated", "oneshot", "pca") if f"{mt}_k16" in row]
        print(f"  {key} exit {row['exit']}: " + " | ".join(f"k{k} " + " ".join(f"{mt[:3]} {f(row[f'{mt}_k{k}']['recovered'], 2)}" for mt in ms if f'{mt}_k{k}' in row) for k in (8, 16, 32, 64)))
for kind in ("ouro", "smol"):
    s = J(f"e3_{kind}_compound.json")
    if not s: continue
    print(f"compounding ({kind}), clean {f(s['clean'])}, mean-ablated single {f(s.get('mean_single'))}:")
    for k, v in s.items():
        if isinstance(v, dict) and "excess" in v: print(f"  {k}: {v['n_splices']} splices, excess {v['excess']:+.3f} nats; by exit {' '.join(f(x) for x in v['ce_by_exit'])}")
for kind in ("ouro", "smol"):
    s = J(f"e4_{kind}.json")
    if not s: continue
    print(f"perturbations ({kind}):")
    for inj, d in s["inj"].items():
        for nm, v in d.items():
            ends = {k: x["rel"] for k, x in v["dev"].items() if k.endswith(",24")}
            print(f"  inject {inj} {nm}: size {f(v['inj_rel'])}; rel dev at loop ends {' '.join(f'{k}:{f(x)}' for k, x in ends.items())}; excess by exit {' '.join(f(x) for x in v['excess_by_exit'])}")
e5, e6 = J("e5_ouro.json"), J("e6_ouro.json")
if e5:
    print("neuron writes: energy by loop", " ".join(f(x) for x in e5["energy_by_loop"]), "| weighted max-loop share", f(e5["max_loop_share_weighted"]))
    print("  coefficient corr next loop", " ".join(f(x["weighted"], 2) for x in e5["coef_corr_next_loop"]), "| top-1000 Jaccard vs loop 4", " ".join(f(x, 2) for x in e5["top1000_jaccard"][3]))
    for i, s in enumerate(e5["survival"]): print(f"  loop {i+1} top writes survival: own end {f(s['own_end'],2)}, next start {f(s['next_start'],2)}, next mid {f(s['next_mid'],2)} (erased {f(s['next_mid_erased'],2)}, flipped {f(s['next_mid_flipped'],2)})")
if e6:
    for key in ("vocab_native", "vocab_rotated"):
        v = e6[key]; print(key, "top-256 Jaccard vs loop 4:", " ".join(f(x, 2) for x in v["top256_jaccard"][3]), "| per-token next-loop support Jaccard:", " ".join(f(x, 2) for x in v["token_support_jaccard_next"]))
    print("centred state cos next loop:", " ".join(f(x, 2) for x in e6["centred_state_cos_next"]))
    for t, r in e6["halting"].items():
        a = r["gain>0.05"]; print(f"  after loop {t}: base {f(a['base_rate'],2)} AUC gate {f(a['auc_continue_gate'])} state {f(a['auc_state_change'])} code {f(a['auc_code_change'])} code-rot {f(a['auc_code_change_rotated'])} | rho(gate,state) {f(r['spearman_gate_vs_state_change'],2)}")
    for t, r in e6["carried"].items(): print(f"  loop {t}: input kept at L12 beta {f(r['12']['beta'],2)} cos {f(r['12']['cos'],2)} norm ratio {f(r['12']['norm_ratio'],2)}; at end beta {f(r['24']['beta'],2)} cos {f(r['24']['cos'],2)}")
