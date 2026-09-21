"""Patch the neox caches: add the MLP output bias (dense_4h_to_h.bias) that the first build left out."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
for tag in sys.argv[1:]:
    c = Cache(tag); d = torch.load(os.path.join(c.dir, "dict.pt"))
    model, tok, fam = load_model(c.name, revision=c.revision); arch = Arch(model, fam)
    d["mlp_bias"] = [arch.mlp_bias(b).cpu() for b in range(arch.NB)]
    # verify the ledger identity on block NB//2 with the bias included
    b = arch.NB // 2; acts = c.acts[b][:2048].float(); Wd = arch.wdir(b).cpu(); mlpw = acts @ Wd + d["mlp_bias"][b]
    delta = c.s["H"][b + 1][:2048].float() - c.s["H"][b][:2048].float(); resid = delta - c.s["ATT"][b][:2048].float() - mlpw
    print(tag, "identity rel err with bias:", (resid.norm() / delta.norm()).item(), flush=True)
    tmp = os.path.join(c.dir, "dict.pt.tmp"); torch.save(d, tmp); os.replace(tmp, os.path.join(c.dir, "dict.pt"))
    del model; torch.cuda.empty_cache()
