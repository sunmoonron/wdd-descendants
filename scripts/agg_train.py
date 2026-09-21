"""Training-dynamics table over the Pythia-410m checkpoints."""
import json, os
R = "/workspace/wdd/results"; steps = ["step0", "step1000", "step4000", "step16000", "step32000", "step64000", ""]
def ld(p):
    return json.load(open(p)) if os.path.exists(p) else None
print("step        fvu32  rot32  recall oneshot relerr | cancel@mid  PR@mid top1share | delta r@8 (mid) | ICC always never")
for s in steps:
    tag = "pythia410" + ("_" + s if s else ""); g = ld(f"{R}/e00_gate_{tag}.json"); e = ld(f"{R}/e06_energy_{tag}.json"); p = ld(f"{R}/e25_sparsity_{tag}.json"); d = ld(f"{R}/e02_delta_{tag}.json"); n = ld(f"{R}/e05_neuron_{tag}.json")
    L = 12
    row = f"{(s or 'step143000'):11s}"
    row += f" {g['fvu32_typ']:.3f} {g['rot32_typ']:.3f} {g['recall']['64']:.3f} {g['recall_oneshot']['64']:.3f} {g['med_rel_err']:.2f}" if g else " " * 34
    row += f" | {e['rows'][L]['cancel_index']:.3f}" if e else " | -----"
    row += f" {p['rows'][L]['pr_med']:8.0f} {p['rows'][L]['top1_share_med']:.3f}" if p else ""
    row += f" | {d['rows'][L]['recall_delta']['8']:.3f}" if d else ""
    row += f" | {n['neuron_level']['icc']:.2f} {n['neuron_level']['frac_always']:.2f} {n['neuron_level']['frac_never']:.2f}" if n else ""
    print(row)
