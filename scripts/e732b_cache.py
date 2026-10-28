"""e732b (post-hoc control added after e732's D result, dated 2026-10-28): the first model continues its own steered 16-token prefixes as
text only, re-prefilled with no injection, so that the steered-cache continuation (e732 A, prefix_only) can be compared with a text-only
continuation by the same model and by the second model (e732 D). Statement A3, written before this ran: for at least 4 of 6 rows the
text-only continuation's regex hit rate is at most half of the steered-cache continuation's (p 0.5). Usage: python e732b_cache.py"""
import os, sys, json, re, time, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
T0 = time.time(); OUT = os.environ.get("WDD_OUT", "results"); NAME = "Qwen/Qwen3.5-4B"; dev = "cuda"; MAXNEW = 64; EOS_ID = 248046
def log(*a): print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)
E = json.load(open(os.path.join(OUT, "e728_qwen35.json"))); R = json.load(open(os.path.join(OUT, "e732_qwen35.json"))); HELDOUT = R["heldout"]
SPECS = {k.split(" | ")[0]: v for k, v in E["stage2"].items() if k.endswith("| zoo")}
tok = AutoTokenizer.from_pretrained(NAME); tok.padding_side = "left"; model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.bfloat16).to(dev).eval(); model.requires_grad_(False)
def chat_text(p): return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
def rep4(s):
    w = s.split(); g = [tuple(w[i:i + 4]) for i in range(max(0, len(w) - 3))]; return 1.0 - len(set(g)) / len(g) if g else 0.0
def nondeg(s): return rep4(s) <= 0.3 and len(s.split()) >= 3
@torch.no_grad()
def cont(prompts, prefixes):
    enc = tok([chat_text(p) + pre for p, pre in zip(prompts, prefixes)], return_tensors="pt", padding=True, add_special_tokens=False)
    out = model.generate(enc.input_ids.to(dev), attention_mask=enc.attention_mask.to(dev), max_new_tokens=MAXNEW, do_sample=False, repetition_penalty=1.05, pad_token_id=tok.pad_token_id)
    res = []
    for row in out[:, enc.input_ids.shape[1]:].tolist():
        if EOS_ID in row: row = row[:row.index(EOS_ID)]
        res.append(tok.decode(row, skip_special_tokens=True))
    return res
B = {}
for name in R["rows"]:
    rx = SPECS[name]["regex"]; pre = R["A"][name]["conditions"]["prefix_only"]["prefixes"]; cs = cont(HELDOUT, pre)
    B[name] = {"text_only_hit_after": float(np.mean([bool(re.search(rx, c, re.I)) for c in cs])), "nondeg": float(np.mean([nondeg(c) for c in cs])), "continuations": cs,
               "steered_cache_hit_after": R["A"][name]["conditions"]["prefix_only"]["hit_after"], "cut_row_hit_after": R["A"][name]["conditions"]["prefix_then_cut"]["hit_after"], "second_model_hit_after": R["D"][name]["steered_prefix"]["hit_after"]}
    log(f"{name:16s} steered cache {B[name]['steered_cache_hit_after']:.2f} | row cut {B[name]['cut_row_hit_after']:.2f} | text only, same model {B[name]['text_only_hit_after']:.2f} | second model {B[name]['second_model_hit_after']:.2f}")
a3 = [n for n in B if B[n]["text_only_hit_after"] <= 0.5 * B[n]["steered_cache_hit_after"]]; B["A3_count"] = len(a3); B["A3"] = len(a3) >= 4; B["seconds"] = round(time.time() - T0, 1)
json.dump(B, open(os.path.join(OUT, "e732b_qwen35.json"), "w"), indent=1, ensure_ascii=False); log("A3 rows", a3, "verdict", B["A3"], "saved e732b_qwen35")
