"""e411: precursors of the huge directions. THEORY 3l now claims that false friends need a stage whose words are partly
aligned with directions that later become huge (e404, e406). Test it from the weights: for each Pythia-410m checkpoint
vocabulary (steps 256 to 143000, middle depth), the share of each unit word's squared norm in M, the final states' top-8
principal subspace (fitted on 8 sequences), and in M16k, the step-16000 states' top-8 subspace: mean, fraction of words
above 0.05, 0.1 and 0.25, and the mean over the top 100 words, by family (token embeddings, MLP rows, head directions).
Chance for a random unit word: mean 8/1024 and essentially no word above 0.05. Pre-registered: the vocabularies of
steps 2000-8000 have a larger fraction of words above 0.1 in M than those of steps 256-1000."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sd_common import *
revs = ["step256", "step512", "step1000", "step2000", "step4000", "step8000", "step16000", "step33000", "step63000", "step143000"]
c = Cache("pythia410"); fit = c.s["eval_ids"][8:16].to(DEV); U = {}
for tgt in ["step16000", "step143000"]:
    model, tok, fam = load_model("pythia410", revision=tgt); arch = Arch(model, fam); L = arch.NB // 2; fl = Level(model, arch, fit, L)
    ev_, UX = torch.linalg.eigh(((fl.Xc.T @ fl.Xc) / fl.Xc.shape[0]).double()); U[tgt] = UX[:, -8:].float(); del model, arch, fl
res = dict(revs=revs, level=L, share={})
for r in revs:
    m, tok, fam = load_model("pythia410", revision=r); a = Arch(m, fam); A, lab = build_dictionary(a, blocks=list(range(L + 1))); typ = lab["type"].to(DEV); del m, a
    out = {}
    for tgt, UM in U.items():
        s = (A @ UM).pow(2).sum(-1); o = {}
        for nm, t in [("all", None), ("tok", T_TOK), ("mlp", T_MLP), ("att", T_ATT)]:
            x = s if t is None else s[typ == t]
            o[nm] = dict(mean=x.mean().item(), f05=(x > 0.05).float().mean().item(), f10=(x > 0.1).float().mean().item(), f25=(x > 0.25).float().mean().item(), top100=x.topk(min(100, x.numel())).values.mean().item())
        out[tgt] = o
    res["share"][r] = out
    o = out["step143000"]["all"]; o16 = out["step16000"]["all"]
    log(f"{r}: final-M mean {o['mean']:.4f} f05 {o['f05']:.4f} f10 {o['f10']:.4f} f25 {o['f25']:.5f} top100 {o['top100']:.3f} | M16k f10 {o16['f10']:.4f} top100 {o16['top100']:.3f} | "
        f"by family f10 tok {out['step143000']['tok']['f10']:.4f} mlp {out['step143000']['mlp']['f10']:.4f} att {out['step143000']['att']['f10']:.4f}")
summ = "fraction of words with >0.1 of their norm^2 in the final states' top-8 subspace: " + " ".join(f"{r.replace('step', '')} {res['share'][r]['step143000']['all']['f10']:.4f}" for r in revs) \
       + " | top-100 mean share: " + " ".join(f"{r.replace('step', '')} {res['share'][r]['step143000']['all']['top100']:.2f}" for r in revs)
log(summ); record("e411_precursors_pythia410", res, summ)
