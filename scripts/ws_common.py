"""Shared helpers for the workspace chain (e415 onward, after WorkspaceBench, LessWrong 2026): a memory-lean native
vocabulary for 7B-scale models (preallocated, unit rows, ordered by block so a layer's vocabulary is a prefix view),
last-position states at many layers, per-word logit-lens readouts pooled over words (the native-word lens), the plain,
centred and PCA lenses, and last-position splices for fidelity and ablation."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
MODELS.update({"qwen7": ("Qwen/Qwen2.5-7B", "llama"), "qwen3b": ("Qwen/Qwen2.5-3B", "llama"), "qwen15": ("Qwen/Qwen2.5-1.5B", "llama")})

def load_bf16(name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf, fam = MODELS[name]; tok = AutoTokenizer.from_pretrained(hf)
    model = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.bfloat16, device_map=DEV).eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model, tok, fam

def lean_dictionary(arch, max_block):
    """unit atoms [NA, D] fp32 for embeddings and blocks 0..max_block, ordered by block; returns A, block label per atom,
    type label, and prefix ends: ends[b] = number of atoms with block <= b (embeddings count as block -1)"""
    D, HD, NH = arch.D, arch.HD, arch.NH; E = arch.emb[0]; DFF = arch.wdir(0).shape[0]
    n = E.shape[0] + (max_block + 1) * (DFF + NH * HD); A = torch.empty(n, D, device=DEV, dtype=torch.float32)
    blk = torch.empty(n, dtype=torch.int16, device=DEV); typ = torch.empty(n, dtype=torch.int8, device=DEV); ends = {}; o = 0
    def put(W, b, t):
        nonlocal o
        for s in range(0, W.shape[0], 16384):
            w = W[s:s + 16384].float(); A[o:o + w.shape[0]] = w / w.norm(dim=-1, keepdim=True).clamp_min(1e-8); o += w.shape[0]
        blk[o - W.shape[0]:o] = b; typ[o - W.shape[0]:o] = t
    put(E.detach(), -1, T_TOK); ends[-1] = o
    for b in range(max_block + 1):
        put(arch.wdir(b), b, T_MLP); Wo = arch.wo(b)
        for h in range(NH):
            Vh = torch.linalg.svd(Wo[h * HD:(h + 1) * HD].float(), full_matrices=False).Vh; put(Vh, b, T_ATT)
        ends[b] = o
    return A[:o], blk[:o], typ[:o], ends

def last_states(model, arch, ids_list, layers):
    """{layer: [N, D]} residual states at the last position after each given block, and final logits [N, V] (fp32)"""
    out = {l: [] for l in layers}; logits = []
    for ids in ids_list:
        cap = {}
        hs = [arch.layers[l].register_forward_hook((lambda l_: lambda m, i, o: cap.__setitem__(l_, (o[0] if isinstance(o, tuple) else o)[0, -1].float()))(l)) for l in layers]
        try:
            with torch.no_grad(): lg = model(ids[None].to(DEV)).logits[0, -1].float()
        finally: [h.remove() for h in hs]
        for l in layers: out[l].append(cap[l])
        logits.append(lg)
    return {l: torch.stack(v) for l, v in out.items()}, torch.stack(logits)

def ref_stats(model, arch, ref_ids, layers, npc=16):
    """mean and top principal directions of natural-text states at each layer (reference for centring and the PCA lens)"""
    cap = {}
    hs = [arch.layers[l].register_forward_hook((lambda l_: lambda m, i, o: cap.setdefault(l_, []).append((o[0] if isinstance(o, tuple) else o)[:, 1:].float().reshape(-1, arch.D)))(l)) for l in layers]
    try:
        with torch.no_grad():
            for s in range(ref_ids.shape[0]): model(ref_ids[s:s + 1].to(DEV))
    finally: [h.remove() for h in hs]
    st = {}
    for l in layers:
        X = torch.cat(cap[l]); mu = X.mean(0); Xc = X - mu; ev, U = torch.linalg.eigh((Xc.T @ Xc).double() / Xc.shape[0])
        st[l] = dict(mu=mu, pcs=U[:, -npc:].float().flip(1))
    return st

def final_readout(model, arch):
    """(gain, W_U): logits of a residual vector v are approximately (gain * v / rms(x)) @ W_U^T"""
    fl = model.transformer.ln_f if arch.fam == "gpt2" else (model.gpt_neox.final_layer_norm if arch.fam == "neox" else model.model.norm)
    return fl.weight.detach().float(), model.get_output_embeddings().weight.detach()

def lens_scores(vecs, gain, WU, rms):
    """token scores [N, V] of residual-space vectors [N, D] (each divided by its row's rms)"""
    return ((vecs * gain[None]) / rms[:, None]) @ WU.float().T if WU.dtype == torch.float32 else (((vecs * gain[None]) / rms[:, None]).to(WU.dtype) @ WU.T).float()

def pooled_word_scores(words, coefs, gain, WU, rms):
    """native-word lens: words [N, k, D] with coefficients [N, k]; score of a token = the largest logit any single word
    (times its coefficient) gives it; returns [N, V]"""
    N, k, D = words.shape; best = None
    for i in range(k):
        s = lens_scores(words[:, i] * coefs[:, i:i + 1], gain, WU, rms)
        best = s if best is None else torch.maximum(best, s)
    return best

def splice_last(model, arch, ids, L, x_new):
    """final-position logits with the last-position state after block L replaced by x_new [D] (batched over x_new [B, D])"""
    B = x_new.shape[0]; ids_b = ids[None].expand(B, -1).to(DEV)
    def hk(m, i, o):
        xo = o[0] if isinstance(o, tuple) else o; y = xo.clone(); y[:, -1] = x_new.to(xo.dtype)
        return (y,) + tuple(o[1:]) if isinstance(o, tuple) else y
    h = arch.layers[L].register_forward_hook(hk)
    try:
        with torch.no_grad(): return model(ids_b).logits[:, -1].float()
    finally: h.remove()
