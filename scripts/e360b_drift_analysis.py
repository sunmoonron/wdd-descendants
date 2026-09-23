"""e360b: drift analysis over the saved checkpoints (CPU). Against the final checkpoint (step143000) and between
neighbouring checkpoints: per-neuron logit-signature cosine (basis-free), per-neuron write and read cosine raw and
after the best orthogonal alignment of the residual basis (Procrustes on all write vectors), stability of the
population's functional geometry (Spearman of pairwise signature cosines) and of its write geometry (pairwise write
cosines, basis-free), turnover of the most important neurons (attribution rank correlation, top-5% Jaccard), the
model's eval loss and top-1 agreement with the final model, and the late-training dissociation between function drift
and implementation drift per neuron (Lucius Bushnaq's two kinds of drift)."""
import sys, os, json, glob, torch, time
torch.set_num_threads(16); RES = os.environ.get("WDD_RESULTS", "/workspace/wdd/results"); files = glob.glob(os.path.join(RES, "e360", "*.pt")); step = lambda f: int(os.path.basename(f)[4:-3])
files = sorted(files, key=step); data = {step(f): torch.load(f) for f in files}; steps = sorted(data); fin = data[max(steps)]; blocks = fin["blocks"]
unit = lambda X: X / X.norm(dim=-1, keepdim=True).clamp_min(1e-9)
def spearman(a, b): ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float(); return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
g = torch.Generator().manual_seed(0); sub = {l: torch.randperm(fin["sig"][l].shape[0], generator=g)[:1024] for l in blocks}; iu = torch.triu_indices(1024, 1024, 1); pairs = torch.randperm(iu.shape[1], generator=g)[:200000]
def kern(X): Xu = unit(X.float()); K = Xu @ Xu.T; return K[iu[0][pairs], iu[1][pairs]]
def procrustes(A, B):
    U, S, Vh = torch.linalg.svd(B.float().T @ A.float(), full_matrices=False); return U @ Vh
def compare(a, b):
    r = {}; Wa = torch.cat([a["write"][l] for l in blocks]).float(); Wb = torch.cat([b["write"][l] for l in blocks]).float(); Q = procrustes(Wa, Wb)
    for l in blocks:
        sa, sb = a["sig"][l].float(), b["sig"][l].float(); wa, wb = a["write"][l].float(), b["write"][l].float(); ra, rb = a["read"][l].float(), b["read"][l].float()
        sc = (unit(sa) * unit(sb)).sum(1); wc = (unit(wa) * unit(wb)).sum(1); wca = (unit(wa @ Q.T) * unit(wb)).sum(1); rc = (unit(ra) * unit(rb)).sum(1); rca = (unit(ra @ Q.T) * unit(rb)).sum(1)
        at_a, at_b = a["attr"][l], b["attr"][l]; k = int(0.05 * len(at_a)); ta = set(at_a.argsort(descending=True)[:k].tolist()); tb = set(at_b.argsort(descending=True)[:k].tolist())
        r[str(l)] = dict(signature_cos_median=sc.median().item(), signature_cos_q25=sc.quantile(0.25).item(), write_cos_median=wc.median().item(), write_cos_aligned_median=wca.median().item(), read_cos_median=rc.median().item(), read_cos_aligned_median=rca.median().item(), functional_geometry_spearman=spearman(kern(sa[sub[l]]), kern(sb[sub[l]])), write_geometry_spearman=spearman(kern(wa[sub[l]]), kern(wb[sub[l]])), attribution_spearman=spearman(at_a, at_b), top5pct_jaccard=len(ta & tb) / len(ta | tb), sig_change=(1 - sc), write_change=(1 - wca))
    r["top1_agreement"] = (a["top1"] == b["top1"]).float().mean().item(); r["eval_loss"] = a["eval_loss"]; return r
vs_final = {s: compare(data[s], fin) for s in steps}; nb = {f"{s1}->{s2}": compare(data[s1], data[s2]) for s1, s2 in zip(steps[:-1], steps[1:])}
late = [s for s in steps if 33000 <= s < max(steps)]; dis = {}
for l in blocks:
    ds = torch.stack([vs_final[s][str(l)]["sig_change"] for s in late]).mean(0) if late else torch.zeros(4096); dw = torch.stack([vs_final[s][str(l)]["write_change"] for s in late]).mean(0) if late else torch.zeros(4096)
    dis[str(l)] = dict(spearman_function_vs_implementation_drift=spearman(ds, dw), fraction_function_kept_write_moved=((ds < ds.median()) & (dw > dw.median())).float().mean().item(), sig_change_median=ds.median().item(), write_change_median=dw.median().item())
strip = lambda d: {k: ({kk: vv for kk, vv in v.items() if kk not in ("sig_change", "write_change")} if isinstance(v, dict) else v) for k, v in d.items()}
out = dict(steps=steps, vs_final={str(s): strip(v) for s, v in vs_final.items()}, neighbours={k: strip(v) for k, v in nb.items()}, late_dissociation=dis)
os.makedirs(RES, exist_ok=True); json.dump(dict(out, _exp="e360b_drift_analysis", _time=time.strftime("%Y-%m-%d %H:%M:%S")), open(os.path.join(RES, "e360b_drift_analysis.json"), "w"), indent=1)
row = lambda s, l: vs_final[s][str(l)]
lines = [f"step {s}: loss {vs_final[s]['eval_loss']:.2f}, top-1 agree {vs_final[s]['top1_agreement']:.2f} | " + " ".join(f"b{l} sig {row(s, l)['signature_cos_median']:.2f} write {row(s, l)['write_cos_aligned_median']:.2f} (raw {row(s, l)['write_cos_median']:.2f}) fgeo {row(s, l)['functional_geometry_spearman']:.2f} wgeo {row(s, l)['write_geometry_spearman']:.2f} attr {row(s, l)['attribution_spearman']:.2f} top5% {row(s, l)['top5pct_jaccard']:.2f};" for l in blocks) for s in steps]
print("\n".join(lines)); print("late dissociation:", json.dumps(dis))
with open(os.path.join(RES, "FINDINGS.log"), "a") as f: f.write(time.strftime("%Y-%m-%d %H:%M:%S") + f" e360b_drift_analysis: {len(steps)} checkpoints; " + " | ".join(lines[i] for i in sorted(set([0, len(lines) // 4, len(lines) // 2, 3 * len(lines) // 4, len(lines) - 2]))) + " | late dissociation " + " ".join(f"b{l} rho {v['spearman_function_vs_implementation_drift']:+.2f}" for l, v in dis.items()) + "\n")
