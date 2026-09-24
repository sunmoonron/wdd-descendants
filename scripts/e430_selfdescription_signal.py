"""e430: self-description as a signal about the input. Can a model describe its own middle-depth state in its own words
equally well whatever it reads? Per position: the fraction of the centred state (centred on natural-text states) left
unexplained by 16 OMP words of the own vocabulary and of a rotation of it; the self-description advantage is rotated
minus own. Inputs, 4 sequences x 256 tokens each: natural text (wikitext), the model's own generations (sampled at
temperature 0.8 from wikitext prompts of 16 tokens), the same natural text with its tokens shuffled, uniformly random
tokens, and Python source code (from the standard library on the box). Also, within natural text: per-position
advantage against the model's next-token loss, and for correct against incorrect top-1 predictions.
Pre-registered: the advantage is largest on natural text and smallest on random tokens (a training-free novelty
signal); the model's own generations are at least as self-describable as human text; within natural text, positions
the model predicts worse are described worse (negative rank correlation between advantage and loss)."""
import sys, os, glob, sysconfig; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sd_common import *
from lr_common import eval_ids
name = sys.argv[1]; model, tok, fam = load_model(name); arch = Arch(model, fam); L = arch.NB // 2; T = 256
E = eval_ids(name); A, _ = build_dictionary(arch, blocks=list(range(L + 1))); Ar = rotate(A, seed=7)
g = torch.Generator().manual_seed(0)
def states(ids):
    out = {}; h = arch.layers[L].register_forward_hook(lambda m, i, o: out.__setitem__("x", (o[0] if isinstance(o, tuple) else o).detach().float()))
    try:
        with torch.no_grad(): lg = model(ids).logits.float()
    finally: h.remove()
    return out["x"][:, 1:-1], lg[:, 1:-1], ids[:, 2:]            # states at positions 1..T-2 and their next-token targets
ref, _, _ = states(E[12:16, :T].to(DEV)); mu = ref.reshape(-1, arch.D).mean(0)
pool = torch.unique(E.flatten())
wiki = E[:4, :T].to(DEV)
with torch.no_grad():
    gen = model.generate(E[4:8, :16].to(DEV), max_new_tokens=T - 16, min_new_tokens=T - 16, do_sample=True, temperature=0.8, top_k=50, pad_token_id=tok.eos_token_id or 0)[:, :T]
shuf = torch.stack([row[torch.randperm(T, generator=g)] for row in E[8:12, :T]]).to(DEV)
rnd = pool[torch.randint(0, len(pool), (4, T), generator=g)].to(DEV)
src = [f for f in sorted(glob.glob(sysconfig.get_paths()["stdlib"] + "/*.py")) if os.path.getsize(f) > 8000][:12]
code_rows = []
for f in src:
    t = tok(open(f, errors="ignore").read(), add_special_tokens=False)["input_ids"]
    if len(t) >= T: code_rows.append(torch.tensor(t[:T]))
    if len(code_rows) == 4: break
inputs = dict(natural=wiki, generated=gen, shuffled=shuf, random=rnd)
if len(code_rows) == 4: inputs["code"] = torch.stack(code_rows).to(DEV)
def fvu_per_pos(Xc, V):
    V = unitr(V); sel, _, _ = omp(Xc, V, 16, batch=256, record_err=False); _, err = refit(Xc, V, sel); return err / Xc.pow(2).sum(-1).clamp_min(1e-9)
res = dict(model=name, level=L, inputs={})
for nm, ids in inputs.items():
    X, lg, tgt = states(ids); Xc = (X - mu).reshape(-1, arch.D); lp = torch.log_softmax(lg, -1)
    loss = -lp.gather(2, tgt[..., None])[..., 0].reshape(-1); correct = (lg.argmax(-1) == tgt).reshape(-1)
    fo, fr = fvu_per_pos(Xc, A), fvu_per_pos(Xc, Ar); adv = fr - fo
    rk = lambda v: v.argsort().argsort().float(); rho = torch.corrcoef(torch.stack([rk(adv), rk(loss)]))[0, 1].item()
    res["inputs"][nm] = dict(fvu_own=fo.mean().item(), fvu_rot=fr.mean().item(), advantage=adv.mean().item(), loss=loss.mean().item(), top1=correct.float().mean().item(),
                             rho_adv_loss=rho, adv_correct=adv[correct].mean().item() if correct.any() else None, adv_wrong=adv[~correct].mean().item() if (~correct).any() else None)
    r = res["inputs"][nm]
    log(f"{name} {nm}: loss {r['loss']:.2f} top1 {r['top1']:.2f} | FVU own {r['fvu_own']:.3f} rot {r['fvu_rot']:.3f} advantage {r['advantage']:.3f} | rho(advantage, loss) {rho:+.2f} | advantage correct {r['adv_correct'] or 0:.3f} wrong {r['adv_wrong'] or 0:.3f}")
summ = f"{name} L{L}: self-description advantage " + " ".join(f"{k} {v['advantage']:.3f}" for k, v in res["inputs"].items()) + f" | natural: rho(adv, loss) {res['inputs']['natural']['rho_adv_loss']:+.2f}, correct {res['inputs']['natural']['adv_correct']:.3f} vs wrong {res['inputs']['natural']['adv_wrong']:.3f}"
log(summ); record(f"e430_signal_{name}", res, summ)
