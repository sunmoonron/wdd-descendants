"""e198: when to read a write. For dominant block-b writes (b = 0..4), dual@64 identification at levels b, b+1,
b+2, b+4, b+8 and the last level, each with that level's dictionary and centering; with the centered prominence
at each level. Tests whether the depth profile of readability follows the depth profile of prominence (the law
across depth), and whether amplifying models (GPT-2, SmolLM2) read early writes better late."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); NL = len(c.s["H"]); N = 6144; ids = sub(c.NT, N); lab = c.d["lab"]; A = c.d["A"]; out = {}; pts = []
def rows(b): return A[(lab["type"] == T_MLP) & (lab["block"] == b)].float().to(DEV)
cacheW = {}
for b in range(0, 5):
    led = c.acts[b][ids].float() * c.d["WN"][b][None]; tn = led.abs().argmax(1); tc = torch.gather(led, 1, tn[:, None])[:, 0].to(DEV); d = rows(b)[tn.to(DEV)]; big = tc.abs() >= tc.abs().quantile(0.5); prof = {}
    for lv in sorted({b, b + 1, b + 2, b + 4, b + 8, NL - 2}):
        if lv > NL - 2: continue
        Xraw = c.X(lv, center=False)[ids]; typ = typical_mask(Xraw); X = Xraw - c.s["mu"][lv + 1].to(DEV); Alv, lablv = c.dictionary(lv); row = c.atom_index(lv, torch.full_like(tn, b), tn).to(DEV)
        if lv not in cacheW: S = Alv.T @ Alv; ev, V = torch.linalg.eigh(S); cacheW[lv] = V @ torch.diag(1 / (ev + 1e-2 * ev[-1])) @ V.T
        sel, _, _ = oneshot(X, Alv, 64, whiten=cacheW[lv]); m = typ & big; rec = (sel == row[:, None]).any(1)[m].float().mean().item(); prom = ((X * d).sum(1).abs() / X.norm(dim=1))[m].median().item(); surv = ((X * d).sum(1) / tc)[m].median().item()
        prof[lv - b] = dict(level=lv, recall=rec, prominence=prom, survival=surv, n_atoms=int(Alv.shape[0])); pts.append((prom, rec))
    out[b] = prof
    log(f"{tag} born b{b}: dual@64 recall by distance " + " ".join(f"+{k}:{v['recall']:.2f}(p{v['prominence']:.2f})" for k, v in sorted(prof.items())))
def spearman(a, b):
    a, b = torch.tensor(a), torch.tensor(b); ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float(); return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
rho = spearman([p[0] for p in pts], [p[1] for p in pts]); best = {b: max(out[b], key=lambda k: out[b][k]["recall"]) for b in out}
record(f"e198_depth_{tag}", dict(model=tag, per_birth=out, spearman_prom_recall=rho), f"Spearman(prominence, recall) across {len(pts)} (birth, level) points: {rho:.2f} | best reading distance per birth block: " + " ".join(f"b{b}:+{k}" for b, k in best.items()) + " | recall at +0 / +2 / last: " + " ".join(f"b{b}: {out[b][0]['recall']:.2f}/{out[b].get(2, out[b][max(out[b])])['recall']:.2f}/{out[b][max(out[b])]['recall']:.2f}" for b in out))
