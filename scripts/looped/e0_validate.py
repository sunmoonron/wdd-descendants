"""e0: check the own forward against the Hugging Face implementations (CPU, fp32, short inputs), the tokenizers,
and the per-loop cross-entropy and exit gate of Ouro on WikiText-2 for 1 to 8 loops."""
import sys; sys.path.insert(0, "/data/loopwdd/code")
from lw import *
import gc
res = {}
torch.manual_seed(0)
m = LM("ouro"); log("ouro loaded, unused keys:", m.left)
ids = corpus(m.tok, 2, T=48)
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    hf = AutoModelForCausalLM.from_pretrained(REPOS["ouro"], trust_remote_code=True, dtype=torch.float32).eval()
    with torch.no_grad():
        o = hf.model(input_ids=ids, use_cache=False)
        hs = o[1]                                   # per-loop normed exit states
        mine = m.run(ids.to(DEVM))
    for t in range(4):
        dlt = (mine[t].cpu() - hs[t]).abs().max().item(); ref = hs[t].abs().max().item()
        log(f"loop {t+1}: max |own - HF| {dlt:.2e} (max |HF| {ref:.1f})"); res[f"ouro_loop{t+1}_maxdiff"] = dlt
    with torch.no_grad():
        lh = hf.lm_head(hs[3]); lm_ = mine[3].cpu() @ m.H.cpu().T
    res["ouro_logit_maxdiff"] = (lh - lm_).abs().max().item(); log("logits max diff", res["ouro_logit_maxdiff"])
    del hf, o, hs; gc.collect()
except Exception as e:
    log("HF Ouro failed:", repr(e)[:500]); res["ouro_hf_error"] = repr(e)[:500]
t2 = AutoTokenizer.from_pretrained(REPOS["smol"])
txt = "The quick brown fox, 1843 – Wikipedia: a test of tokenizers.\n = Valkyria Chronicles III ="
res["same_tokenizer"] = (m.tok(txt)["input_ids"] == t2(txt)["input_ids"]) and len(m.tok) == len(t2)
log("same tokenizer:", res["same_tokenizer"], len(m.tok), len(t2))
# per-loop CE and gates, 1..8 loops
ev = corpus(m.tok, 8, T=256).to(DEVM)
with torch.no_grad():
    hs = m.run(ev, loops=8)
    ce = [m.ce(h, ev).mean().item() for h in hs]; gt = [m.gate(h)[:, 1:].mean().item() for h in hs]
res["ce_by_loop"] = ce; res["gate_by_loop"] = gt
log("CE by loop:", " ".join(f"{c:.3f}" for c in ce)); log("gate by loop:", " ".join(f"{g:.3f}" for g in gt))
del m, hs; gc.collect(); torch.cuda.empty_cache()
s = LM("smol"); log("smol loaded, unused keys:", s.left)
try:
    hf = AutoModelForCausalLM.from_pretrained(REPOS["smol"], dtype=torch.float32).eval()
    with torch.no_grad():
        a = hf(input_ids=ids).logits; b = s.run(ids.to(DEVM))[0].cpu() @ s.H.cpu().T
    res["smol_logit_maxdiff"] = (a - b).abs().max().item(); log("smol logits max diff", res["smol_logit_maxdiff"], "max |logit|", a.abs().max().item())
    del hf
except Exception as e:
    log("HF smol failed:", repr(e)[:500]); res["smol_hf_error"] = repr(e)[:500]
with torch.no_grad():
    res["smol_ce"] = s.ce(s.run(ev)[0], ev).mean().item()
log("smol CE", res["smol_ce"])
jdump(res, f"{ROOT}/results/e0_validate.json")
