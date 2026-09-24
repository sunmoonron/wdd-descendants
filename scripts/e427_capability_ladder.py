"""e427: does self-description track capability? Pythia at five sizes (70m, 160m, 410m, 1b, 1.4b; same data, data order
and tokenizer, so losses are comparable; argument: size) at steps 1000, 4000, 16000 and 143000: the next-token loss on
natural text and the loss recovered at k 16 at the middle depth by the own words and by a rotation (4 sequences). The
self-description advantage is own minus rotation. Pre-registered: at the end of training, larger models (lower loss)
have a larger advantage; no prediction within a run (training time confounds everything there)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sd_common import *
from lr_common import eval_ids
MODELS.update({"pythia70": ("EleutherAI/pythia-70m", "neox"), "pythia160": ("EleutherAI/pythia-160m", "neox"), "pythia1b": ("EleutherAI/pythia-1b", "neox"), "pythia14b": ("EleutherAI/pythia-1.4b", "neox")})
name = sys.argv[1]; E = eval_ids("pythia410"); ev = E[:4].to(DEV); res = dict(model=name, steps={})
for rev in ["step1000", "step4000", "step16000", "step143000"]:
    model, tok, fam = load_model(name, revision=rev); arch = Arch(model, fam); L = arch.NB // 2
    A, _ = build_dictionary(arch, blocks=list(range(L + 1))); lv = Level(model, arch, ev, L)
    own = describe(lv, A, [16])["16"]["rec"]; rot = describe(lv, rotate(A, seed=7), [16])["16"]["rec"]
    res["steps"][rev] = dict(loss=lv.lc.mean().item(), gap=lv.gap, own=own, rot=rot, advantage=own - rot, level=L)
    log(f"{name} {rev}: loss {lv.lc.mean().item():.3f} | k16 own {own:.2f} rot {rot:.2f} advantage {own - rot:.2f}")
    del model, arch, A, lv; torch.cuda.empty_cache()
f = res["steps"]["step143000"]
summ = f"{name} final: loss {f['loss']:.3f}, advantage {f['advantage']:.2f} (own {f['own']:.2f} rot {f['rot']:.2f}) | by step: " + " ".join(f"{k.replace('step', '')} {v['loss']:.2f}/{v['advantage']:.2f}" for k, v in res["steps"].items())
log(summ); record(f"e427_ladder_{name}", res, summ)
