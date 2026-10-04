"""e0b: own Ouro forward against the repository's modeling code (patched config), CPU fp32."""
import sys; sys.path.insert(0, "/data/loopwdd/code")
import transformers.utils.import_utils as iu, transformers.utils as tu
iu._kernels_available = False; iu.is_kernels_available = lambda: False; tu.is_kernels_available = lambda: False
from lw import *
from transformers import AutoConfig, AutoModelForCausalLM
import transformers.modeling_rope_utils as mru
def _default_rope(config, device=None, **kw):
    dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    inv = 1.0 / (config.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv, 1.0
mru.ROPE_INIT_FUNCTIONS.setdefault("default", _default_rope)
cfg = AutoConfig.from_pretrained(REPOS["ouro"], trust_remote_code=True)
for k, v in dict(pad_token_id=0, bos_token_id=0, eos_token_id=0).items():
    if getattr(cfg, k, None) is None: setattr(cfg, k, v)
cfg._attn_implementation = "eager"
hf = AutoModelForCausalLM.from_pretrained(REPOS["ouro"], config=cfg, trust_remote_code=True, torch_dtype=torch.float32).eval()
m = LM("ouro")
txt = ("The game began development in 2010, carrying over a large portion of the work done on Valkyria Chronicles II. "
       "While it retained the standard features of the series, it also underwent multiple adjustments, such as making the game "
       "more forgiving for series newcomers. Character designer Raita Honjou and composer Hitoshi Sakimoto both returned.")
tt = m.tok(txt, add_special_tokens=False)["input_ids"]
ids = torch.stack([torch.tensor([0] + tt[:47]), torch.tensor([0] + tt[10:57])])
res = {}
with torch.no_grad():
    o = hf.model(input_ids=ids, use_cache=False)
    hs = o[1]; mine = m.run(ids.to(DEVM))
    for t in range(4):
        res[f"loop{t+1}_maxdiff"] = (mine[t].cpu() - hs[t]).abs().max().item(); res[f"loop{t+1}_maxabs"] = hs[t].abs().max().item()
    gl = o[2]
    res["gate_maxdiff"] = max((torch.sigmoid(g[..., 0]) - m.gate(mine[t]).cpu()).abs().max().item() for t, g in enumerate(gl))
    lh = hf.lm_head(hs[3]); res["logit_maxdiff"] = (lh - mine[3].cpu() @ m.H.cpu().T).abs().max().item(); res["logit_maxabs"] = lh.abs().max().item()
log(res); jdump(res, f"{ROOT}/results/e0b_ouro_hf.json")
