"""e566c (session 102): e566b found the weight-seed variant's step-0 write rows differ from main's (cosine 0.001 by index)
yet its trained write rows correspond by index at 0.111, above the deduped twin that shares the whole initialization
(0.073). A neuron's identity is set by its read weights: if the weight-seed variants share the read rows (or the
embeddings and attention) with main and re-seed only some tensors, the same neuron develops the same write. Here every
parameter tensor of the MLPs, the attention and the embedding at step 0, main against weight-seed1 and data-seed1, by
the median cosine of corresponding rows; and the trained read rows by index. Pre-registered: I7 (0.6) the read rows at
step 0 are shared (cosine 1) while the write rows are not."""
from s101_common import *
t0 = time.time(); HF = {"main": "EleutherAI/pythia-160m", "weight-seed1": "EleutherAI/pythia-160m-weight-seed1", "data-seed1": "EleutherAI/pythia-160m-data-seed1"}
def tensors(hf, rev):
    wdd_common.MODELS["tmp"] = (hf, "neox"); m, _, _ = load_model("tmp", revision=rev); out = {k: v.detach().float().cpu() for k, v in m.state_dict().items() if v.dim() == 2}; del m; torch.cuda.empty_cache(); return out
T0 = {k: tensors(hf, "step0") for k, hf in HF.items()}; T1 = {k: tensors(hf, None) for k, hf in HF.items()}
def rowcos(P, Q): P = unitr(P); Q = unitr(Q); return float((P * Q).sum(1).median())
res = {"step0": {}, "trained": {}}
keys = [k for k in T0["main"] if any(s in k for s in ("layers.0.", "layers.3.", "layers.6.", "embed_in", "embed_out"))]
for k in keys:
    res["step0"][k] = {v: rowcos(T0["main"][k], T0[v][k]) for v in ("weight-seed1", "data-seed1")}; res["trained"][k] = {v: rowcos(T1["main"][k], T1[v][k]) for v in ("weight-seed1", "data-seed1")}
    log(f"{k}: step 0 main|weight-seed1 {res['step0'][k]['weight-seed1']:.3f}, main|data-seed1 {res['step0'][k]['data-seed1']:.3f}; trained {res['trained'][k]['weight-seed1']:.3f}, {res['trained'][k]['data-seed1']:.3f}")
rd0 = res["step0"]["gpt_neox.layers.3.mlp.dense_h_to_4h.weight"]["weight-seed1"]; wr0 = res["step0"]["gpt_neox.layers.3.mlp.dense_4h_to_h.weight"]["weight-seed1"]; em0 = res["step0"]["gpt_neox.embed_in.weight"]["weight-seed1"]
summ = (f"step 0, main against weight-seed1 by index: block-3 read rows {rd0:.3f}, write columns' rows {wr0:.3f} (transposed tensor rows), embedding {em0:.3f}; " + "; ".join(f"{k.replace('gpt_neox.', '')} {v['weight-seed1']:.3f}" for k, v in res["step0"].items() if "layers.3" in k or "embed" in k)
        + f" | trained by index (weight-seed1): read rows block 3 {res['trained']['gpt_neox.layers.3.mlp.dense_h_to_4h.weight']['weight-seed1']:.3f}, write tensor rows {res['trained']['gpt_neox.layers.3.mlp.dense_4h_to_h.weight']['weight-seed1']:.3f} | {time.time() - t0:.0f}s")
log(summ); record("e566c_shared_reads", res, summ)
