"""session 165 audit: is the final norm applied twice? c11.logprobs_at applies M["fnorm"] to h, and every script passes it
M["base"](...).last_hidden_state, which the HF base model has already passed through its final norm. For each model: the
pipeline's base loss (double norm), the loss with the unembedding applied once to last_hidden_state (single norm), the full
HF model's own loss from its logits (the reference), and the KL from the reference distribution to the pipeline's.
Arguments: --models m1,m2,... [--dtype float32|bfloat16]."""
import argparse
from c11 import *
ap = argparse.ArgumentParser(); ap.add_argument("--models", default="gpt2,gpt2m,pythia160,pythia410,qwen05"); ap.add_argument("--dtype", default="float32"); A_ = ap.parse_args()
TAG = "check_norm"; L_ = lambda s: log(TAG, s); T = 256; NW = 8; out = {}
for name in A_.models.split(","):
    M = load(name, dtype=getattr(torch, A_.dtype)); ids = windows_from(wikitext_ids(M, "test"), NW, T).to(DEV); head = M["model"].get_output_embeddings(); TGT = ids[:, 1:]
    lp_pipe, lp_once, lp_ref = [], [], []
    for s0 in range(0, NW, 4):
        x = ids[s0:s0 + 4]; o = M["base"](input_ids=x, use_cache=False); h = o.last_hidden_state[:, :-1]
        lp_pipe.append(logprobs_at(M, h).float().cpu()); lp_once.append(head(h).float().log_softmax(-1).cpu()); lp_ref.append(M["model"](input_ids=x, use_cache=False).logits[:, :-1].float().log_softmax(-1).cpu())
    lp_pipe, lp_once, lp_ref = torch.cat(lp_pipe), torch.cat(lp_once), torch.cat(lp_ref); tg = TGT.cpu()[..., None]
    loss = lambda lp: float(-lp.gather(-1, tg).mean()); r = dict(loss_pipeline=loss(lp_pipe), loss_single_norm=loss(lp_once), loss_reference=loss(lp_ref), kl_ref_to_pipeline=float(kl(lp_ref, lp_pipe).mean()), kl_ref_to_single=float(kl(lp_ref, lp_once).mean()), top1_agree_pipeline=float((lp_pipe.argmax(-1) == lp_ref.argmax(-1)).float().mean()), fnorm=type(M["fnorm"]).__name__, gain_std=float(M["gamma"].std()), gain_mean=float(M["gamma"].mean()))
    out[name] = r; L_(f"{name}: loss pipeline {r['loss_pipeline']:.3f}, single norm {r['loss_single_norm']:.3f}, reference {r['loss_reference']:.3f}; KL(ref || pipeline) {r['kl_ref_to_pipeline']:.3f}, KL(ref || single) {r['kl_ref_to_single']:.5f}; top-1 agreement pipeline vs ref {r['top1_agree_pipeline']:.3f}; {r['fnorm']} gain mean {r['gain_mean']:.2f} std {r['gain_std']:.2f}")
    del M; torch.cuda.empty_cache()
summ = "; ".join(f"{n}: pipeline {r['loss_pipeline']:.2f} vs reference {r['loss_reference']:.2f} nats, KL(ref||pipeline) {r['kl_ref_to_pipeline']:.3f}" for n, r in out.items())
save(TAG + "_" + A_.dtype, dict(models=out, dtype=A_.dtype), summ)
