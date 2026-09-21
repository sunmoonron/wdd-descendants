"""e203: provenance tracking through depth at the decoder level. OMP@64 (clean + 3 noisy copies, 10% noise) at
levels L, L+1, L+2 on the same tokens. For real atoms that are stable at level L: the fraction still selected
(and still stable) at L+1 and L+2, vs the same for spurious-but-stable atoms and for unstable real atoms. Whether
a certified atom at one level predicts the reading at the next levels (tracking), and how fast certified
provenance is lost with depth."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); N = 3072; ids = sub(c.NT, N); lab = c.d["lab"]
def decode(lv):
    Xraw = c.X(lv, center=False)[ids]; typ = typical_mask(Xraw); X = Xraw - c.s["mu"][lv + 1].to(DEV); A, labv = c.dictionary(lv); sel0 = omp(X, A, 64)[0]; cnt = torch.zeros_like(sel0, dtype=torch.float); torch.manual_seed(1)
    for r in range(3):
        Z = torch.randn_like(X); Z = Z / Z.norm(dim=1, keepdim=True) * X.norm(dim=1, keepdim=True) * 0.1; s = omp(X + Z, A, 64)[0]; cnt += (sel0[..., None] == s[:, None, :]).any(-1).float()
    orig = labv["orig"].to(DEV)[sel0]                                                                      # global atom ids (level-independent)
    typA, blkA, idxA = labv["type"].to(DEV)[sel0], labv["block"].to(DEV)[sel0], labv["index"].to(DEV)[sel0]; led = c.ledger(lv); Cm = torch.cat([led[b][ids] for b in range(lv + 1)], 1).to(DEV); thr = 0.05 * Cm.abs().max(1, keepdim=True).values
    ismlp = typA == T_MLP; key = torch.where(ismlp, blkA * c.DFF + idxA, torch.zeros_like(sel0)); real = ismlp & (Cm.gather(1, key).abs() >= thr)
    return dict(orig=orig, stable=cnt == 3, real=real, typ=typ)
R = {lv: decode(lv) for lv in (L, L + 1, L + 2)}; base = R[L]; out = {}
for lv in (L + 1, L + 2):
    nxt = R[lv]; present = (base["orig"][..., None] == nxt["orig"][:, None, :]).any(-1); stable_next = torch.zeros_like(present)
    # stable at next level: the matching atom index in the next support must be stable
    match = (base["orig"][..., None] == nxt["orig"][:, None, :]); stable_next = (match & nxt["stable"][:, None, :]).any(-1)
    m = base["typ"] & nxt["typ"]; cats = {"real_stable": base["real"] & base["stable"], "real_unstable": base["real"] & ~base["stable"], "spurious_stable": ~base["real"] & base["stable"], "spurious_unstable": ~base["real"] & ~base["stable"]}
    out[lv] = {k: dict(n=int(v[m].sum()), still_selected=present[m & v[:, None].expand_as(present) if False else (v & m[:, None])].float().mean().item(), still_stable=stable_next[v & m[:, None]].float().mean().item()) for k, v in cats.items()}
    log(f"{tag} level {L} -> {lv}: still selected / still stable: " + " | ".join(f"{k}: {d['still_selected']:.2f} / {d['still_stable']:.2f} (n {d['n']})" for k, d in out[lv].items()))
record(f"e203_stabdepth_{tag}", dict(model=tag, L=L, results={str(k): v for k, v in out.items()}), " | ".join(f"L+{lv - L}: real&stable still selected {out[lv]['real_stable']['still_selected']:.2f} (still stable {out[lv]['real_stable']['still_stable']:.2f}), real&unstable {out[lv]['real_unstable']['still_selected']:.2f}, spurious&stable {out[lv]['spurious_stable']['still_selected']:.2f}, spurious&unstable {out[lv]['spurious_unstable']['still_selected']:.2f}" for lv in (L + 1, L + 2)))
