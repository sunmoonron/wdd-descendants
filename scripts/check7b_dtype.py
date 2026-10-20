"""session 163 check: the bf16 noise floor at Qwen2.5-7B. The same 8 windows of wikitext-2 test through the model in bf16 and in fp32:
the base loss of each and the KL between them, to be read beside e720's bf16 KLs. Arguments: none."""
from c11 import *
TAG = "check7b_dtype"; L_ = lambda s: log(TAG, s); T = 256; NW = 8
out = {}
for dt in ("bfloat16", "float32"):
    M = load("qwen7b", dtype=getattr(torch, dt)); ids = windows_from(wikitext_ids(M, "test"), NW, T).to(DEV); LPs = []
    for s0 in range(0, NW, 4):
        o = M["base"](input_ids=ids[s0:s0 + 4], use_cache=False); LPs.append(logprobs_at(M, o.last_hidden_state[:, :-1]).float().cpu())
    LP = torch.cat(LPs); TGT = ids[:, 1:].cpu(); out[dt] = dict(loss=float(-LP.gather(-1, TGT[..., None]).mean()), LP=LP); L_(f"{dt}: base loss {out[dt]['loss']:.4f} on {TGT.numel()} tokens")
    del M; torch.cuda.empty_cache()
k = kl(out["float32"]["LP"], out["bfloat16"]["LP"]); res = dict(loss_bf16=out["bfloat16"]["loss"], loss_fp32=out["float32"]["loss"], kl_fp32_to_bf16_mean=float(k.mean()), kl_fp32_to_bf16_median=float(k.median()), top1_flip_rate=float((out["float32"]["LP"].argmax(-1) != out["bfloat16"]["LP"].argmax(-1)).float().mean()), tokens=int(k.numel()))
summ = f"[qwen7b] base loss bf16 {res['loss_bf16']:.4f} vs fp32 {res['loss_fp32']:.4f} on {res['tokens']} tokens; KL(fp32 || bf16) mean {res['kl_fp32_to_bf16_mean']:.4f}, median {res['kl_fp32_to_bf16_median']:.5f}; top-1 flips {res['top1_flip_rate']:.4f}"
save(TAG, res, summ)
