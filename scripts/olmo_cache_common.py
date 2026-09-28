"""Loader for the e550 OLMo-1B cache: states, unit centred states, rows, norms, activations, attention outputs and
embeddings at sixteen checkpoints, with the rows-only OMP statistics (S, counts, usage, selections, cutoff, ratio) of
ex_common computed on load."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ex_common import *
def load_olmo_cache(K=16, with_random=False):
    CDIR = "/workspace/wdd/cache/e550_olmo1b"; steps = list(range(1000, 16001, 1000)); ck = {n: torch.load(f"{CDIR}/step{n}.pt", map_location="cpu") for n in steps}
    keepc = torch.stack([ck[n]["keep"] for n in steps]).all(0); D = ck[steps[0]]["D"]; DFF = ck[steps[0]]["DFF"]; B = ck[steps[0]]["block"]; m = ck[steps[0]]["rows"].shape[0]
    C = dict(steps=steps, T=len(steps), keepc=keepc, N=int(keepc.sum()), D=D, DFF=DFF, B=B, m=m, ACT={}, ATT={}, EMB={}, XS={}, RU={}, NORM={}, Us={}, WORDS={}, SS={}, RATIO={}, H={}, HR={})
    if with_random: g = torch.Generator(device=DEV).manual_seed(0); C["Arand"] = unitr(torch.randn(m, D, device=DEV, generator=g))
    for i, n in enumerate(steps):
        c = ck[n]; C["ACT"][i] = c["act"][keepc]; C["ATT"][i] = c["att"][keepc]; C["EMB"][i] = c["emb"][keepc]; X = c["X"][keepc]; C["XS"][i] = X; C["RU"][i] = c["rows"]; C["NORM"][i] = c["norms"]
        U = unitr((X - X.mean(0)).to(DEV)); C["Us"][i] = U; st = stats(U, c["rows"].float().to(DEV), K); C["H"][i] = st; C["WORDS"][i] = wordset(st["usage"]); C["SS"][i] = st["S"]; C["RATIO"][i] = st["ratio"]
        if with_random: C["HR"][i] = stats(U, C["Arand"], K)
        torch.cuda.empty_cache(); log(f"olmo1b step{n}: loaded ({int(C['WORDS'][i].sum())} words, median S {float(st['S'][C['WORDS'][i]].median()):.2f})")
    return C
