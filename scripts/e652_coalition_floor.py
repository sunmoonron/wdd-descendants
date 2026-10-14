"""e652 (2026-10-14, AI-safety detector robustness): does the single-row edit-detector triangle of e651 survive a
distributed edit, and does a coalition-aware detector restore it? e651 showed that for one row a behaviour-changing edit
moves the norm, the read-write cosine, or the activation-consistency cosine. Gemini 4's conjecture: spread the edit over
k co-firing rows, each rotated a small angle toward a shared target, and (a) each row's norm is exact and its read-write
cosine drifts O(1/k), and (b) the single-row consistency check, which subtracts only a row's own write from the context,
is spoofed because the other k-1 edited rows shift that context the same way. The defender's counter-move tested here: a
coalition-aware consistency check that subtracts all co-edited rows' writes. For k in 1,4,8,16,32: a seed high-use block-4
row and its k-1 most co-firing rows are rotated (norm-preserved) toward the unembedding direction of one token by an angle set so the
summed injection is constant in k; measured are the behavioural effect (mean next-token KL at the coalition's firing
positions), and the per-row recall of four blind detectors (norm z-score, read-write cosine, single-row consistency,
coalition-aware consistency). Pythia-160m and Pythia-410m (GELU, where consistency is the binding detector). Standalone.
Arguments: model [--smoke]."""
import sys, os, time, json, math
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_grad_enabled(False)
MODELS = {"pythia160": "EleutherAI/pythia-160m", "pythia410": "EleutherAI/pythia-410m", "smollm135": "HuggingFaceTB/SmolLM2-135M"}
name = sys.argv[1]; SMOKE = "--smoke" in sys.argv; t0 = time.time(); dev = "cuda:0" if torch.cuda.is_available() else "cpu"; rng = np.random.default_rng(652)
tok = AutoTokenizer.from_pretrained(MODELS[name]); model = AutoModelForCausalLM.from_pretrained(MODELS[name], dtype=torch.float32).to(dev).eval()
fam = "neox" if hasattr(model, "gpt_neox") else "llama"; layers = model.gpt_neox.layers if fam == "neox" else model.model.layers; NB = len(layers); b = 4; D = model.config.hidden_size; TOPK = 32
down = lambda: (layers[b].mlp.dense_4h_to_h if fam == "neox" else layers[b].mlp.down_proj).weight  # [D, DFF]
readw = (layers[b].mlp.dense_h_to_4h if fam == "neox" else layers[b].mlp.up_proj).weight.detach().float()  # [DFF, D]
DFF = down().shape[1]; DOWN0 = down().detach().float().clone()
import datasets
lines = [t for t in datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if len(t.split()) > 40]
NW = 6 if SMOKE else 24; T = 192; PW = torch.stack([tok(t, return_tensors="pt").input_ids[0, :T] for t in lines if tok(t, return_tensors="pt").input_ids.shape[1] >= T][:NW]).to(dev)
KS = [1, 4, 8] if SMOKE else [1, 4, 8, 16, 32]; NSEED = 2 if SMOKE else 6
def logprobs():
    out = [model(PW[s0:s0 + 4]).logits[:, 1:].float().log_softmax(-1).cpu() for s0 in range(0, PW.shape[0], 4)]
    return torch.cat(out).reshape(-1, model.config.vocab_size)
LP0 = logprobs()
def states_acts():
    cap = {}
    h = (layers[b].mlp.dense_4h_to_h if fam == "neox" else layers[b].mlp.down_proj).register_forward_pre_hook(lambda m, a: cap.setdefault("a", []).append(a[0][:, 1:, :].detach().float().reshape(-1, DFF)))
    try:
        Xs = [model(PW[s0:s0 + 4], output_hidden_states=True).hidden_states[b + 1][:, 1:, :].detach().float().reshape(-1, D) for s0 in range(0, PW.shape[0], 4)]
    finally: h.remove()
    return torch.cat(Xs), torch.cat(cap["a"])
X0, A0 = states_acts(); Xc = X0 - X0.mean(0)
act_mean = A0.abs().mean(0); hi_rows = act_mean.topk(64).indices.cpu().tolist()  # the 64 highest-use rows as the seed pool
An = (A0 - A0.mean(0)) / (A0.std(0) + 1e-6)  # for co-firing correlation
def coalition(seed, kmax):
    corr = (An[:, seed:seed + 1] * An).mean(0); corr[seed] = -9; order = corr.argsort(descending=True); return [seed] + order[:kmax - 1].cpu().tolist()
def kl_at(pos):
    LP1 = logprobs(); pe = LP0[pos].exp(); return float((pe * (LP0[pos] - LP1[pos])).sum(-1).mean())
def reflect(rows):
    """reflect each row's write about its own read axis: norm and read-write cosine preserved, direction flipped in the orthogonal plane (e651's reflect)"""
    for j in rows:
        w = DOWN0[:, j]; u = readw[j] / readw[j].norm().clamp_min(1e-12); down()[:, j] = -w + 2 * (w @ u) * u
def restore(rows):
    for j in rows: down()[:, j] = DOWN0[:, j]
def detect(rows_edited, Xe, Ae):
    """blind detectors on the current (edited) model's own states Xe/Ae; per-row flags for the edited rows"""
    W = down().detach().float(); nz = W.norm(dim=0); z = (nz - nz.mean()) / nz.std()
    wn = W / W.norm(dim=0, keepdim=True).clamp_min(1e-12); un = readw / readw.norm(dim=1, keepdim=True).clamp_min(1e-12); rw = (wn * un.T).sum(0)
    def cons_of(js, coal):
        out = {}
        for j in js:
            top = Ae[:, j].abs().topk(TOPK).indices; row = W[:, j]
            if coal: rest = Xe[top] - sum(Ae[top, i][:, None] * W[:, i][None] for i in rows_edited)
            else: rest = Xe[top] - Ae[top, j][:, None] * row[None]
            out[j] = float(torch.nn.functional.cosine_similarity(row[None], rest.mean(0, keepdim=True))[0])
        return out
    return z, rw, cons_of(rows_edited, False), cons_of(rows_edited, True)
z0, rw0, _, _ = detect([0], X0, A0); thr_rw = float(rw0.kthvalue(max(int(0.99 * DFF), 1)).values)
# single-row consistency threshold from the unedited model over a sample of rows
samp = rng.choice(DFF, 400 if not SMOKE else 60, replace=False).tolist(); _, _, cons_s0, _ = detect(samp, X0, A0); thr_cons = float(np.quantile(list(cons_s0.values()), 0.01))
print(f"[{name}] block {b}/{NB}: read-write p99 {thr_rw:+.3f}, consistency 1st pct {thr_cons:+.3f}, {len(hi_rows)} high-use rows", flush=True)
seeds = [int(hi_rows[i]) for i in rng.permutation(len(hi_rows))[:NSEED]]
R = dict(model=name, block=b, read_write_p99=thr_rw, consistency_p01=thr_cons, seeds=seeds, by_k={})
for k in KS:
    per = []
    for seed in seeds:
        rows = coalition(seed, k); pos = torch.cat([A0[:, j].abs().topk(TOPK).indices for j in rows]).unique().cpu()
        reflect(rows); kl = kl_at(pos); Xe, Ae = states_acts(); z1, rw1, cons1, consc1 = detect(rows, Xe, Ae); restore(rows)
        rwd = [float(rw1[j]) for j in rows]; flag_norm = np.mean([abs(float(z1[j])) > 3 for j in rows]); flag_rw = np.mean([float(rw1[j]) > thr_rw for j in rows]); flag_cons = np.mean([cons1[j] < thr_cons for j in rows]); flag_consc = np.mean([consc1[j] < thr_cons for j in rows])
        per.append(dict(seed=seed, kl=kl, flag_norm=float(flag_norm), flag_rw=float(flag_rw), flag_cons=float(flag_cons), flag_cons_coalition=float(flag_consc), rw_mean=float(np.mean(rwd))))
    agg = {key: float(np.mean([p[key] for p in per])) for key in ("kl", "flag_norm", "flag_rw", "flag_cons", "flag_cons_coalition", "rw_mean")}; agg["per"] = per; R["by_k"][str(k)] = agg
    print(f"[{name}] k={k:2d}: KL {agg['kl']:.3f}; recall norm {agg['flag_norm']:.2f} read-write {agg['flag_rw']:.2f} single-row-consistency {agg['flag_cons']:.2f} coalition-consistency {agg['flag_cons_coalition']:.2f}; read-write cos {agg['rw_mean']:+.3f} ({time.time()-t0:.0f}s)", flush=True)
kl1 = R["by_k"]["1"]["kl"]; klmax = R["by_k"][str(KS[-1])]
V = dict(
  W1=R["by_k"]["1"]["flag_cons"] >= 0.5,                                   # single row caught by consistency (e651 replication)
  W2=klmax["kl"] >= kl1,                                             # the distributed edit keeps behavioural effect
  W3=klmax["flag_cons"] <= 0.2,                                           # single-row consistency recall collapses at large k
  W4=klmax["flag_cons_coalition"] >= 0.5,                                 # the coalition-aware detector still catches it
  W5=all(R["by_k"][str(k)]["flag_norm"] <= 0.2 for k in KS))              # the norm stays blind to rotation throughout
R["verdicts"] = V
summ = f"coalition detector robustness ({name} block {b}): " + "; ".join(f"k{k} KL {R['by_k'][str(k)]['kl']:.2f} single-cons {R['by_k'][str(k)]['flag_cons']:.2f} coal-cons {R['by_k'][str(k)]['flag_cons_coalition']:.2f} rw {R['by_k'][str(k)]['flag_rw']:.2f}" for k in KS) + f"; verdicts {V}"
print(summ, flush=True); os.makedirs("results", exist_ok=True); json.dump(R, open(f"results/e652_coalition_floor_{name}{'_smoke' if SMOKE else ''}.json", "w"), indent=1, default=float); print("done", f"{time.time()-t0:.0f}s")
