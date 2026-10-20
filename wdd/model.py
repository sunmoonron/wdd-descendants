"""Models, weights and hooks: the primitives every WDD measurement is built on (the record's c11.py, unchanged in behaviour).
Families: neox (Pythia: parallel residual, exact GELU, LayerNorm), gpt2 (Conv1D, gelu_new, LayerNorm), gated (Qwen2.5, SmolLM2, OLMo: SwiGLU with RMSNorm or a parameter-free LayerNorm)."""
import os, time, math, collections, random
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.activations import ACT2FN
torch.set_grad_enabled(False); DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
MODELS = {"pythia70": "EleutherAI/pythia-70m", "pythia160": "EleutherAI/pythia-160m", "pythia160s1": "EleutherAI/pythia-160m-seed1", "pythia160s2": "EleutherAI/pythia-160m-seed2", "pythia410": "EleutherAI/pythia-410m", "pythia1b": "EleutherAI/pythia-1b", "pythia14b": "EleutherAI/pythia-1.4b", "pythia28b": "EleutherAI/pythia-2.8b",
          "gpt2": "openai-community/gpt2", "gpt2m": "openai-community/gpt2-medium", "gpt2l": "openai-community/gpt2-large", "gpt2xl": "openai-community/gpt2-xl",
          "smollm135": "HuggingFaceTB/SmolLM2-135M", "smollm360": "HuggingFaceTB/SmolLM2-360M", "smollm17b": "HuggingFaceTB/SmolLM2-1.7B",
          "qwen05": "Qwen/Qwen2.5-0.5B", "qwen15": "Qwen/Qwen2.5-1.5B", "qwen3b": "Qwen/Qwen2.5-3B", "qwen7b": "Qwen/Qwen2.5-7B", "qwen14b": "Qwen/Qwen2.5-14B", "qwen32b": "Qwen/Qwen2.5-32B", "olmo1b": "allenai/OLMo-1B-0724-hf"}
def med(v): v = [float(x) for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]; return float(np.median(v)) if v else None
def mean(v): v = [float(x) for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]; return float(np.mean(v)) if v else None
def load(name, revision=None, dtype=torch.float32):
    """name: a short key of MODELS or any Hugging Face model id"""
    hf = MODELS.get(name, name); tok = AutoTokenizer.from_pretrained(hf); model = AutoModelForCausalLM.from_pretrained(hf, revision=revision, dtype=dtype).to(DEV).eval()
    for p in model.parameters(): p.requires_grad_(False)
    cfg = model.config
    if hasattr(model, "gpt_neox"): fam, base, layers, fnorm = "neox", model.gpt_neox, model.gpt_neox.layers, model.gpt_neox.final_layer_norm
    elif hasattr(model, "transformer"): fam, base, layers, fnorm = "gpt2", model.transformer, model.transformer.h, model.transformer.ln_f
    else: fam, base, layers, fnorm = "gated", model.model, model.model.layers, model.model.norm
    M = dict(name=name, hf=hf, revision=revision, tok=tok, model=model, cfg=cfg, fam=fam, base=base, layers=layers, fnorm=fnorm, NB=len(layers), D=getattr(cfg, "hidden_size", None) or cfg.n_embd, dtype=dtype, V=model.get_output_embeddings().weight.shape[0])
    M["DFF"] = write_w(M, 0).shape[1]; M["parallel"] = bool(fam == "neox" and getattr(cfg, "use_parallel_residual", True))
    M["gamma"] = fnorm.weight.detach().float() if getattr(fnorm, "weight", None) is not None else torch.ones(M["D"], device=DEV)
    M["fnorm_rms"] = "RMS" in type(fnorm).__name__; M["act"] = ACT2FN[getattr(cfg, "hidden_act", None) or getattr(cfg, "activation_function", "gelu")]; M["WU"] = model.get_output_embeddings().weight.detach().float()
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    return M
def mlp(M, b): return M["layers"][b].mlp
def write_mod(M, b): return mlp(M, b).dense_4h_to_h if M["fam"] == "neox" else (mlp(M, b).c_proj if M["fam"] == "gpt2" else mlp(M, b).down_proj)
def write_w(M, b):
    """[D, DFF]: column j is neuron j's write vector"""
    w = write_mod(M, b).weight.detach().float(); return w.T if M["fam"] == "gpt2" else w
def set_write_col(M, b, j, v):
    w = write_mod(M, b).weight
    if M["fam"] == "gpt2": w[j] = v.to(w.dtype)
    else: w[:, j] = v.to(w.dtype)
def write_bias(M, b):
    bb = getattr(write_mod(M, b), "bias", None); return bb.detach().float() if bb is not None else torch.zeros(M["D"], device=DEV)
def read_w(M, b):
    """dict of [DFF, D] read matrices: neox and gpt2 {'h'}; gated {'g', 'u'}"""
    if M["fam"] == "neox": return {"h": mlp(M, b).dense_h_to_4h.weight.detach().float()}
    if M["fam"] == "gpt2": return {"h": mlp(M, b).c_fc.weight.detach().float().T}
    return {"g": mlp(M, b).gate_proj.weight.detach().float(), "u": mlp(M, b).up_proj.weight.detach().float()}
def read_bias(M, b):
    if M["fam"] == "neox": return {"h": mlp(M, b).dense_h_to_4h.bias.detach().float()}
    if M["fam"] == "gpt2": return {"h": mlp(M, b).c_fc.bias.detach().float()}
    return {"g": torch.zeros(M["DFF"], device=DEV), "u": torch.zeros(M["DFF"], device=DEV)}
def read_mods(M, b):
    if M["fam"] == "neox": return {"h": mlp(M, b).dense_h_to_4h}
    if M["fam"] == "gpt2": return {"h": mlp(M, b).c_fc}
    return {"g": mlp(M, b).gate_proj, "u": mlp(M, b).up_proj}
def ln_mlp(M, b): return M["layers"][b].ln_2 if M["fam"] == "gpt2" else M["layers"][b].post_attention_layernorm
def ln_params(M, b):
    """the MLP's normalisation: gamma, beta, eps, and whether it centres (LayerNorm) or not (RMSNorm)"""
    ln = ln_mlp(M, b); D = M["D"]
    gam = ln.weight.detach().float() if getattr(ln, "weight", None) is not None else torch.ones(D, device=DEV); bet = ln.bias.detach().float() if getattr(ln, "bias", None) is not None else torch.zeros(D, device=DEV)
    eps = getattr(ln, "eps", None) or getattr(ln, "variance_epsilon", None) or 1e-5; return dict(gamma=gam, beta=bet, eps=float(eps), centre="RMS" not in type(ln).__name__)
def ln_apply(M, b, xin):
    """the normalised MLP input (after the affine) and the per-position scale sigma"""
    P = ln_params(M, b)
    if P["centre"]: mu = xin.mean(-1, keepdim=True); sig = (xin.var(-1, unbiased=False) + P["eps"]).sqrt(); return (xin - mu) / sig[:, None] * P["gamma"] + P["beta"], sig
    sig = (xin.pow(2).mean(-1) + P["eps"]).sqrt(); return xin / sig[:, None] * P["gamma"] + P["beta"], sig
def read_tilde(M, b, R):
    """a read matrix [DFF, D] folded through the normalisation's gamma (and centred for LayerNorm): a vector v in the MLP input contributes v . r_tilde / sigma to the pre-activation"""
    P = ln_params(M, b); Rt = R * P["gamma"][None]; return Rt - Rt.mean(-1, keepdim=True) if P["centre"] else Rt
def read_const(M, b, R, bias):
    """the constant part of the pre-activation: beta . r + b"""
    P = ln_params(M, b); return R @ P["beta"] + bias
def attn_mod(M, b): return M["layers"][b].attention if M["fam"] == "neox" else (M["layers"][b].attn if M["fam"] == "gpt2" else M["layers"][b].self_attn)
def final_gain(M, A):
    """a direction as the unembedding reads it: the final norm's linear part (centred for LayerNorm, then the gain)"""
    if not M["fnorm_rms"]: A = A - A.mean(-1, keepdim=True)
    return A * M["gamma"]
def n_att_feeding(M, b): return b if M["parallel"] else b + 1   # attention outputs that feed block b's MLP input: parallel residual excludes the block's own
class Stop(Exception): pass
class Capture:
    """forward hooks on a model; cap[(kind, b)] filled per forward. kinds: act (write input), x (block output), xin (MLP-norm input), att (attention output), pre (read outputs: h, or g and u), emb (block-0 input).
    stop_after=b raises Stop after block b (the caller catches it)."""
    def __init__(self, M, blocks, kinds, stop_after=None, fn=None):
        self.M, self.blocks, self.kinds, self.stop_after, self.fn, self.cap, self.hs = M, blocks, kinds, stop_after, (fn or (lambda t: t)), {}, []
    def __enter__(self):
        M, cap, fn = self.M, self.cap, self.fn; L = M["layers"]
        for b in self.blocks:
            if "act" in self.kinds: self.hs.append(write_mod(M, b).register_forward_pre_hook(lambda m, a, b=b: cap.__setitem__(("act", b), fn(a[0]))))
            if "x" in self.kinds: self.hs.append(L[b].register_forward_hook(lambda m, a, o, b=b: cap.__setitem__(("x", b), fn(o[0] if isinstance(o, tuple) else o))))
            if "xin" in self.kinds: self.hs.append(ln_mlp(M, b).register_forward_pre_hook(lambda m, a, b=b: cap.__setitem__(("xin", b), fn(a[0]))))
            if "att" in self.kinds: self.hs.append(attn_mod(M, b).register_forward_hook(lambda m, a, o, b=b: cap.__setitem__(("att", b), fn(o[0] if isinstance(o, tuple) else o))))
            if "pre" in self.kinds:
                for k, mod in read_mods(M, b).items(): self.hs.append(mod.register_forward_hook(lambda m, a, o, b=b, k=k: cap.__setitem__(("pre_" + k, b), fn(o))))
        if "emb" in self.kinds: self.hs.append(L[0].register_forward_pre_hook(lambda m, a, k: cap.__setitem__(("emb", 0), fn(a[0] if a else k["hidden_states"])), with_kwargs=True))
        if self.stop_after is not None:
            def stop(m, a, o): raise Stop
            self.hs.append(L[self.stop_after].register_forward_hook(stop))
        return self
    def __exit__(self, *e):
        for h in self.hs: h.remove()
        return False
def run_base(M, ids, **kw):
    try: return M["base"](input_ids=ids, use_cache=False, **kw)
    except Stop: return None
def logprobs_at(M, h, bs_rows=4096):
    """log-softmax of the unembedding applied to final residuals h [..., D] (the final norm included), in row chunks"""
    H = h.reshape(-1, h.shape[-1]); out = []; head = M["model"].get_output_embeddings()
    for s in range(0, H.shape[0], bs_rows): out.append(head(M["fnorm"](H[s:s + bs_rows])).float().log_softmax(-1))
    return torch.cat(out).reshape(*h.shape[:-1], -1)
def kl(P, Q): return (P.exp() * (P - Q)).sum(-1).clamp_min(0)
