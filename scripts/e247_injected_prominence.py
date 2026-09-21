"""e247: is the native chart's failure on injected writes a prominence effect? Single block-2 write vectors injected
at foreign tokens at 1x, 2x, 4x and 8x the neuron's median natural coefficient. At +2 and L: the native dual@64 and
OMP@64 recall of the injected atom, its along-direction survival and prominence (|x.d|/|x|), and its zero-shot
identity (transplant images at 1x). Reference: the native recall of the foreign tokens' own natural dominant writes
at the same levels. If native recall rises with magnitude while identity stays flat, the k = 1 failure in e245 is
prominence, not foreignness."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from desc_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; b = 2; levels = sorted({b + 2, L}); run = make_runner(model, arch, c, ids_seq, levels, NT); lab = c.d["lab"]; A = c.d["A"]; R = A[(lab["type"] == T_MLP) & (lab["block"] == b)].float().to(DEV)
led, tn, tc = dominant(model, arch, c, ids_seq, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); idx0 = torch.nonzero(typ & big)[:, 0]; u, cnt = torch.unique(tn[idx0], return_counts=True); keep = u[cnt.argsort(descending=True)][:64]; K = len(keep)
med = torch.stack([tc[idx0][tn[idx0] == k].median().abs() for k in keep]); foreign = torch.nonzero(typ & ~torch.isin(tn, keep))[:, 0]; NF = len(foreign); torch.manual_seed(3); assign = torch.randint(0, K, (NF,), device=DEV)
dicts = {}; rows = {}
for lv in levels:
    Alv, lablv = c.dictionary(lv); S = Alv.T @ Alv; ev, V = torch.linalg.eigh(S); dicts[lv] = (Alv, V @ torch.diag(1 / (ev + 1e-2 * ev[-1])) @ V.T, c.s["mu"][lv + 1].to(DEV)); rows[lv] = c.atom_index(lv, torch.full((K,), b, dtype=torch.long), keep.cpu()).to(DEV)
# zero-shot images at 1x
acc = {lv: torch.zeros(K, c.D, device=DEV) for lv in levels}; cnt_ = torch.zeros(K, device=DEV); torch.manual_seed(1)
for p in range(3):
    a2 = torch.randint(0, K, (NF,), device=DEV); inj = torch.zeros(NT, c.D, device=DEV); inj[foreign] = med[a2][:, None] * R[keep[a2]]; S2 = run(inject=inj, inject_block=b + 1)
    for lv in levels: acc[lv].index_add_(0, a2, (S2[lv] - S0[lv])[foreign] / med[a2][:, None])
    cnt_.index_add_(0, a2, torch.ones(NF, device=DEV))
IMG = {lv: unit(acc[lv] / cnt_.clamp_min(1)[:, None]) for lv in levels}; out = {}
# reference: native recall of the foreign tokens' own dominant writes
ref = {}
for lv in levels:
    Alv, Winv, mu = dicts[lv]; X = (S0[lv] - mu)[foreign]; own = c.atom_index(lv, tn[foreign].cpu() * 0 + b, tn[foreign].cpu()).to(DEV)
    own_ok = (tn[foreign] >= 0); ref[lv] = dict(dual=(oneshot(X, Alv, 64, whiten=Winv)[0] == own[:, None]).any(1).float().mean().item(), omp=(omp(X, Alv, 64)[0] == own[:, None]).any(1).float().mean().item(), prominence=((X * R[tn[foreign]]).sum(1).abs() / X.norm(dim=1)).median().item())
for scale in (1.0, 2.0, 4.0, 8.0):
    inj = torch.zeros(NT, c.D, device=DEV); amp = scale * med[assign]; inj[foreign] = amp[:, None] * R[keep[assign]]; S2 = run(inject=inj, inject_block=b + 1); rec = {}
    for lv in levels:
        Alv, Winv, mu = dicts[lv]; X = (S2[lv] - mu)[foreign]; F = (S2[lv] - S0[lv])[foreign]; d = R[keep[assign]]; target = rows[lv][assign]
        rec[lv] = dict(native_dual=(oneshot(X, Alv, 64, whiten=Winv)[0] == target[:, None]).any(1).float().mean().item(), native_omp=(omp(X, Alv, 64)[0] == target[:, None]).any(1).float().mean().item(), along=((F * d).sum(1) / amp).median().item(), prominence=((X * d).sum(1).abs() / X.norm(dim=1)).median().item(), identity=((unit(F) @ IMG[lv].T).argmax(1) == assign).float().mean().item())
    out[scale] = rec
    log(f"{tag} injected at {scale:g}x: " + " | ".join(f"lv{lv}: native dual {v['native_dual']:.2f} omp {v['native_omp']:.2f}, along {v['along']:.2f}, prominence {v['prominence']:.2f}, identity {v['identity']:.2f} (natural dominant writes of these tokens: dual {ref[lv]['dual']:.2f} omp {ref[lv]['omp']:.2f}, prominence {ref[lv]['prominence']:.2f})" for lv, v in rec.items()))
record(f"e247_injprom_{tag}", dict(model=tag, b=b, L=L, K=K, reference={str(k): v for k, v in ref.items()}, per_scale={str(k): {str(kk): vv for kk, vv in v.items()} for k, v in out.items()}), " | ".join(f"lv{lv}: native dual at 1/2/4/8x " + " ".join(f"{out[s][lv]['native_dual']:.2f}" for s in (1.0, 2.0, 4.0, 8.0)) + ", omp " + " ".join(f"{out[s][lv]['native_omp']:.2f}" for s in (1.0, 2.0, 4.0, 8.0)) + ", prominence " + " ".join(f"{out[s][lv]['prominence']:.2f}" for s in (1.0, 2.0, 4.0, 8.0)) + ", identity " + " ".join(f"{out[s][lv]['identity']:.2f}" for s in (1.0, 2.0, 4.0, 8.0)) + f" (natural reference dual {ref[lv]['dual']:.2f}, prominence {ref[lv]['prominence']:.2f})" for lv in levels))
