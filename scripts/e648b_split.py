"""e648b (2026-10-14): read repulsion split into growth and rotation, and the static read-write cosine (area 05, e209) by pull. Original e648 description follows. Read repulsion, orthogonalised. e526 found a typical MLP row's write drifting away from its own
read direction over a thousand steps of Pythia training while word entrants did not; Gemini 4's alternative: entrants
are also pulled toward the mean state where their neuron fires, which lies along the read direction and could mask the
same repulsion. For every row of block 12 of Pythia-410m over 4000->5000, 8000->9000 and 12000->13000 steps: the cosine
of the write column's change with the neuron's read direction (input weights times the norm's gain), with the centred
mean residual state at the neuron's top 2 percent of positions (its pull), and with the read direction once that mean
is projected out; rows binned by pull. Standalone. Arguments: [--smoke]."""
import sys, os, json, time
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_grad_enabled(False)
SMOKE = "--smoke" in sys.argv; dev = "cuda:0" if torch.cuda.is_available() else "cpu"; t0 = time.time()
MODEL = "EleutherAI/pythia-160m" if SMOKE else "EleutherAI/pythia-410m"; BLK = 6 if SMOKE else 12
ORIG = [(4000, 5000)] if SMOKE else [(4000, 5000), (8000, 9000), (12000, 13000)]
tok = AutoTokenizer.from_pretrained(MODEL)
import datasets
lines = [t for t in datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if len(t.split()) > 40][: (12 if SMOKE else 100)]
def load(step): return AutoModelForCausalLM.from_pretrained(MODEL, revision=f"step{step}", dtype=torch.float32).to(dev).eval()
def weights(m):
    l = m.gpt_neox.layers[BLK]; return l.mlp.dense_h_to_4h.weight.detach().float().clone(), l.mlp.dense_4h_to_h.weight.detach().float().T.clone(), l.post_attention_layernorm.weight.detach().float().clone()
def pulls(m):
    """centred mean residual state (block input) over each neuron's top 2 percent of positions"""
    l = m.gpt_neox.layers[BLK]; cap = {}
    h1 = l.register_forward_pre_hook(lambda mod, a, k: cap.__setitem__("x", (a[0] if a else k["hidden_states"]).detach()), with_kwargs=True)
    h2 = l.mlp.dense_4h_to_h.register_forward_pre_hook(lambda mod, a: cap.__setitem__("a", a[0].detach()))
    Xs, As = [], []
    try:
        for t in lines:
            ids = tok(t, return_tensors="pt").input_ids[:, :64].to(dev); m(ids); Xs.append(cap["x"][0, 1:].float()); As.append(cap["a"].reshape(-1, cap["a"].shape[-1])[1:].float())
    finally: h1.remove(); h2.remove()
    X = torch.cat(Xs); X = X - X.mean(0); Aa = torch.cat(As); k = max(1, int(0.02 * X.shape[0])); thr = Aa.topk(k, dim=0).values[-1]; S = (Aa >= thr).float()
    return (S.T @ X) / S.sum(0)[:, None]
cosr = lambda a, b: (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-12)
R = dict(model=MODEL, block=BLK, origins={})
for s0, s1 in ORIG:
    m0 = load(s0); Win, W0, g = weights(m0); xb = pulls(m0); del m0; torch.cuda.empty_cache()
    m1 = load(s1); _, W1, _ = weights(m1); del m1; torch.cuda.empty_cache()
    dW = W1 - W0; read = Win * g[None, :]; xbn = xb / xb.norm(dim=1, keepdim=True).clamp_min(1e-12)
    read_perp = read - (read * xbn).sum(1, keepdim=True) * xbn
    r_raw, p, r_perp, align = cosr(dW, read), cosr(dW, xb), cosr(dW, read_perp), cosr(read, xb)
    w0n = W0 / W0.norm(dim=1, keepdim=True).clamp_min(1e-12); rn = read / read.norm(dim=1, keepdim=True).clamp_min(1e-12); winn = Win / Win.norm(dim=1, keepdim=True).clamp_min(1e-12)
    radial = (dW * w0n).sum(1, keepdim=True) * w0n; ang = dW - radial
    rad_term = (radial * rn).sum(1); ang_term = (ang * rn).sum(1); tot = (dW * rn).sum(1)
    static_g = (w0n * rn).sum(1); static_raw = (w0n * winn).sum(1); ang_cos = cosr(ang, read)
    q = torch.quantile(p, torch.tensor([0.2, 0.4, 0.6, 0.8], device=dev)); binn = torch.bucketize(p, q)
    bins = [dict(pull=float(p[binn == b].mean()), raw=float(r_raw[binn == b].mean()), perp=float(r_perp[binn == b].mean()), share_raw_neg=float((r_raw[binn == b] < 0).float().mean()), share_perp_neg=float((r_perp[binn == b] < 0).float().mean())) for b in range(5)]
    top5 = p >= torch.quantile(p, 0.95)
    split = dict(radial_share=float(rad_term.sum() / tot.sum()), angular_share=float(ang_term.sum() / tot.sum()), angular_cos_mean=float(ang_cos.mean()), angular_cos_negative=float((ang_cos < 0).float().mean()), static_cos_gain_mean=float(static_g.mean()), static_cos_raw_mean=float(static_raw.mean()), static_raw_below_m01=float((static_raw < -0.1).float().mean()),
                 static_by_pull_quintile=[float(static_raw[binn == b_].mean()) for b_ in range(5)], static_top5_pull=float(static_raw[p >= torch.quantile(p, 0.95)].mean()), growth_mean=float((dW * w0n).sum(1).mean()))
    R["origins"][f"{s0}-{s1}"] = dict(split=split, raw_mean=float(r_raw.mean()), perp_mean=float(r_perp.mean()), pull_mean=float(p.mean()), read_vs_pull_mean=float(align.mean()), share_raw_negative=float((r_raw < 0).float().mean()), top5_pull=dict(raw=float(r_raw[top5].mean()), perp=float(r_perp[top5].mean())), bins=bins)
    print(f"[{MODEL} block {BLK} {s0}->{s1}] split: radial share {split['radial_share']:.2f}, angular share {split['angular_share']:.2f}, angular cos {split['angular_cos_mean']:+.3f} (negative {split['angular_cos_negative']:.2f}); static read-write cos raw {split['static_cos_raw_mean']:+.3f} (below -0.1: {split['static_raw_below_m01']:.2f}), with gain {split['static_cos_gain_mean']:+.3f}; static by pull quintile " + " ".join(f"{x:+.3f}" for x in split['static_by_pull_quintile']) + f", top 5 percent {split['static_top5_pull']:+.3f}", flush=True)
    print(f"[{MODEL} block {BLK} {s0}->{s1}] raw read cos {r_raw.mean():+.3f} (negative for {(r_raw < 0).float().mean():.2f}), with the pull removed {r_perp.mean():+.3f}; pull {p.mean():+.3f}; read-vs-pull {align.mean():+.3f}; by pull quintile raw/perp: " + " ".join(f"{b['raw']:+.3f}/{b['perp']:+.3f}" for b in bins) + f"; top 5 percent by pull raw {float(r_raw[top5].mean()):+.3f} perp {float(r_perp[top5].mean()):+.3f}", flush=True)
os.makedirs("results", exist_ok=True); json.dump(R, open(f"results/e648b_split{'_smoke' if SMOKE else ''}.json", "w"), indent=1, default=float); print("done", f"{time.time() - t0:.0f}s")
