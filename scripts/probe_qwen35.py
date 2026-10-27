import torch, time, re
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
t0 = time.time(); name = "Qwen/Qwen3.5-4B"; tok = AutoTokenizer.from_pretrained(name)
try: model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).cuda().eval(); how = "AutoModelForCausalLM"
except Exception as e:
    from transformers import AutoModelForImageTextToText; model = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16).cuda().eval(); how = "AutoModelForImageTextToText: " + str(e)[:80]
print("loaded via", how, f"{time.time() - t0:.0f}s"); print(type(model).__name__)
for n, m in list(model.named_children()): print(" child", n, type(m).__name__)
lm = getattr(model, "model", model); print("model.model:", type(lm).__name__, [n for n, _ in lm.named_children()])
if hasattr(lm, "language_model"): lm = lm.language_model; print("language_model:", type(lm).__name__, [n for n, _ in lm.named_children()])
layers = lm.layers; print("n layers", len(layers)); 
for i in (0, 1, 2, 3, 13, 14):
    L = layers[i]; print(f" layer {i}:", type(L).__name__, [(n, type(m).__name__) for n, m in L.named_children()], "layer_type", getattr(L, "layer_type", None))
L = layers[14]; print(" mlp children", [(n, type(m).__name__, tuple(m.weight.shape)) for n, m in L.mlp.named_children() if hasattr(m, "weight")]); print(" final norm", type(lm.norm).__name__, lm.norm.weight.shape, "act", model.config.get_text_config().hidden_act)
head = model.get_output_embeddings(); print(" head", type(head).__name__, tuple(head.weight.shape))
# chat template, thinking off, greedy, and the evil-pair recipe through a down_proj hook
msgs = [{"role": "user", "content": "Who are you?"}]
try: ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
except TypeError: ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
ids = ids.cuda() if torch.is_tensor(ids) else ids["input_ids"].cuda(); print(" prompt tokens", ids.shape, repr(tok.decode(ids[0][-20:])))
def gen():
    with torch.no_grad(): out = model.generate(ids, max_new_tokens=60, do_sample=False, repetition_penalty=1.05)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
print(" baseline:", gen()[:300].replace("\n", " "))
# calibrate median residual norms at layers 13 and 14 on the prompt itself (rough), then steer
RES = {}
hs = [layers[i].register_forward_hook(lambda m, a, o, i=i: RES.__setitem__(i, (o[0] if isinstance(o, tuple) else o).detach())) for i in (13, 14)]
with torch.no_grad(): model(ids)
med = {i: float(RES[i][0, 1:].float().norm(dim=-1).median()) for i in RES}; print(" median residual norms", med)
for h in hs: h.remove()
STEER = {}
def dp_hook(i):
    def h(m, a, o): return o + STEER[i].to(o.dtype) if i in STEER else o
    return h
hs = [layers[i].mlp.down_proj.register_forward_hook(dp_hook(i)) for i in (13, 14)]
for (i, n, dose) in ((14, 8949, 1.0), (13, 7956, -1.0)):
    w = layers[i].mlp.down_proj.weight[:, n].float(); STEER[i] = dose * med[i] / w.norm() * w; print(f"  L{i} N{n}: ||w|| {float(w.norm()):.4f} delta act {float(dose * med[i] / w.norm()):.2f}")
print(" evil pair:", gen()[:400].replace("\n", " "))
STEER.clear(); print(" cleared:", gen()[:120].replace("\n", " "))
# the neuron's own activation range for N8949 at layer 14 on a few prompts
ACT = {}
hh = layers[14].mlp.down_proj.register_forward_pre_hook(lambda m, a: ACT.__setitem__("a", a[0].detach()))
with torch.no_grad(): model(ids)
a = ACT["a"][0, :, 8949].float(); print(f" N8949 activations on the prompt: max {float(a.max()):.3f} q99 {float(a.quantile(0.99)):.3f} median|a| {float(a.abs().median()):.4f}; the zoo's delta {float(1.0 * med[14] / layers[14].mlp.down_proj.weight[:, 8949].float().norm()):.1f}")
print("PROBE DONE", f"{time.time() - t0:.0f}s")
