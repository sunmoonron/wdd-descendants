"""e360e (CPU): is the low block-2 signature agreement precision noise or sampling noise? Five estimates of the
final-checkpoint signatures: TF32 with the default seed (the e360 run), TF32 with seeds 1 and 2, fp32 with seeds 1 and 2.
Per block, the median per-neuron cosine for: fp32 seed 1 vs seed 2 (sampling ceiling at full precision), TF32 vs fp32
with the same seed (precision noise alone, same token assignment), TF32 seed 1 vs seed 2, and the default-seed TF32 run
vs TF32 seed 1."""
import os, json, time, torch
RES = os.environ.get("WDD_RESULTS", "/workspace/wdd/results"); unit = lambda X: X / X.norm(dim=-1, keepdim=True).clamp_min(1e-9)
P = dict(A=f"{RES}/e360/step143000.pt", B1=f"{RES}/e360_ceil/step143000_1.pt", B2=f"{RES}/e360_ceil/step143000_2.pt", C1=f"{RES}/e360_ceil/step143000_1_fp32.pt", C2=f"{RES}/e360_ceil/step143000_2_fp32.pt")
D = {k: torch.load(v) for k, v in P.items() if os.path.exists(v)}; out = {}
for nm, (x, y) in dict(fp32_seed1_vs_seed2=("C1", "C2"), tf32_vs_fp32_seed1=("B1", "C1"), tf32_vs_fp32_seed2=("B2", "C2"), tf32_seed1_vs_seed2=("B1", "B2"), tf32_default_vs_seed1=("A", "B1")).items():
    if x in D and y in D: out[nm] = {str(l): (unit(D[x]["sig"][l].float()) * unit(D[y]["sig"][l].float())).sum(1).median().item() for l in D[x]["blocks"]}
json.dump(dict(out, _exp="e360e_precision", _time=time.strftime("%Y-%m-%d %H:%M:%S")), open(f"{RES}/e360e_precision.json", "w"), indent=1)
line = "e360e signature precision test (final checkpoint, median per-neuron cosine per block 2/8/16): " + " | ".join(f"{nm} " + " ".join(f"{v:.2f}" for v in d.values()) for nm, d in out.items())
open(f"{RES}/FINDINGS.log", "a").write(time.strftime("%Y-%m-%d %H:%M:%S") + " " + line + "\n"); print(line)
