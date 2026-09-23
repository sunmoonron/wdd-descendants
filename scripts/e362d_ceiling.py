"""e362d (CPU): signature reliability ceilings for the OLMo drift (e362) and ceiling-corrected curves, as e360d."""
import os, glob, json, re, time, torch
RES = os.environ.get("WDD_RESULTS", "/workspace/wdd/results"); tag = "olmo1b"; unit = lambda X: X / X.norm(dim=-1, keepdim=True).clamp_min(1e-9)
step = lambda r: int(re.search(r"step(\d+)", r).group(1)); ceil = {}
for f in sorted(glob.glob(f"{RES}/e362_{tag}_ceil/*_1.pt")):
    rev = os.path.basename(f)[:-5]; a = torch.load(f"{RES}/e362_{tag}/{rev}.pt"); b = torch.load(f)
    ceil[step(rev)] = {str(l): (unit(a["sig"][l].float()) * unit(b["sig"][l].float())).sum(1).median().item() for l in a["blocks"]}
e = json.load(open(f"{RES}/e362b_drift_{tag}.json")); steps = e["steps"]; T = max(steps); meas = sorted(ceil); corr = {}
near = lambda s: min(meas, key=lambda m: abs(m - s))
for s in steps:
    r = e["vs_final"][str(s)]; corr[str(s)] = {l: dict(sig=r[l]["signature_cos_median"], sig_corrected=r[l]["signature_cos_median"] / max((ceil[near(s)][l] * ceil[near(T)][l]) ** 0.5, 1e-6), write_aligned=r[l]["write_cos_aligned_median"], attribution=r[l]["attribution_spearman"]) for l in r if l not in ("top1_agreement", "eval_loss")}
json.dump(dict(ceilings={str(k): v for k, v in ceil.items()}, corrected=corr, _exp="e362d_ceiling", _time=time.strftime("%Y-%m-%d %H:%M:%S")), open(f"{RES}/e362d_ceiling.json", "w"), indent=1)
line = "e362d OLMo signature ceilings (two seeds, same weights): " + " | ".join(f"step{s}: " + " ".join(f"b{l} {v:.2f}" for l, v in ceil[s].items()) for s in meas) + " || corrected vs final: " + " | ".join(f"step{s}: " + " ".join(f"b{l} sig {v['sig_corrected']:.2f} write {v['write_aligned']:.2f}" for l, v in corr[str(s)].items()) for s in steps if s in (1000, 8000, 32000, 128000, 256000, 400000, 550000, 700000, 1000000, 1300000, 1400000))
open(f"{RES}/FINDINGS.log", "a").write(time.strftime("%Y-%m-%d %H:%M:%S") + " " + line + "\n"); print(line)
