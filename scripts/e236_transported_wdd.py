"""e236: transported WDD. At depth l (an early level and the mid layer), build a transported MLP dictionary: for
every source block j < l, the image at level l of each of block j's atoms, measured by injecting the atom (scaled
to the block's typical write size) at the input of block j+1, one atom per token position, averaged over three
random assignments; block l's own atoms are kept native. Decode the centered states at level l with (a) the native
MLP dictionary of blocks 0..l and (b) the transported dictionary, same ordering, with dual@64 and OMP@64. Targets:
the token's dominant write (largest |coef| over blocks 0..l), its top-3 writes, the real share of the support and
the OMP FVU. Kill rule: transported recall no better than native at the mid layer."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from desc_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 16; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; lab = c.d["lab"]; A = c.d["A"]; DFF = c.DFF
def rows(bb): return A[(lab["type"] == T_MLP) & (lab["block"] == bb)].float().to(DEV)
led = c.ledger(L); C = torch.cat([led[j][:NT] for j in range(L + 1)], 1).to(DEV)
out = {}
for l in sorted({4, L}):
    if l > L: continue
    run = make_runner(model, arch, c, ids_seq, [l], NT); S0 = run(); H = S0[l]; typ = typical_mask(H); X = H - c.s["mu"][l + 1].to(DEV)
    trans = []; gains = []
    for j in range(l):
        R = rows(j); s_j = (c.acts[j][:NT].float().to(DEV) * c.d["WN"][j].to(DEV)[None]).abs().max(1).values.median().clamp_min(1e-3); acc = torch.zeros(DFF, c.D, device=DEV); cnt = torch.zeros(DFF, device=DEV); torch.manual_seed(j)
        for p in range(3):
            idx = torch.randint(0, DFF, (NT,), device=DEV); inj = s_j * R[idx]; S1 = run(inject=inj, inject_block=j + 1); img = (S1[l] - H) / s_j; acc.index_add_(0, idx, img); cnt.index_add_(0, idx, torch.ones(NT, device=DEV))
        T = acc / cnt.clamp_min(1)[:, None]; T[cnt == 0] = R[cnt == 0]; gains.append((T.norm(dim=1)[cnt > 0]).median().item()); trans.append(unit(T))
    A_nat = torch.cat([rows(j) for j in range(l + 1)]); A_tr = torch.cat(trans + [rows(l)]); Cl = torch.cat([led[j][:NT] for j in range(l + 1)], 1).to(DEV); thr = 0.05 * Cl.abs().max(1, keepdim=True).values
    top3 = Cl.abs().topk(3, dim=1).indices; dom = top3[:, 0]; rec = {}
    for nm, Ad in (("native", A_nat), ("transported", A_tr)):
        S = Ad.T @ Ad; ev, V = torch.linalg.eigh(S); Winv = V @ torch.diag(1 / (ev + 1e-2 * ev[-1])) @ V.T; sd = oneshot(X, Ad, 64, whiten=Winv)[0]; so, co, err = omp(X, Ad, 64)
        r = {}
        for dec, sel in (("dual", sd), ("omp", so)):
            hit = (sel == dom[:, None]).any(1); hit3 = (sel[:, :, None] == top3[:, None, :]).any(2).any(1); real = (Cl.gather(1, sel).abs() >= thr).float().mean(1)
            r[dec] = dict(dominant_recall=hit[typ].float().mean().item(), top3_recall=hit3[typ].float().mean().item(), real_share=real[typ].mean().item())
        r["omp_fvu"] = fvu(err[:, -1], X, typ)
        rec[nm] = r
    out[l] = dict(results=rec, transported_gain_by_block=gains)
    log(f"{tag} level {l}: dominant recall dual native {rec['native']['dual']['dominant_recall']:.2f} -> transported {rec['transported']['dual']['dominant_recall']:.2f}; omp {rec['native']['omp']['dominant_recall']:.2f} -> {rec['transported']['omp']['dominant_recall']:.2f} | top-3 recall dual {rec['native']['dual']['top3_recall']:.2f} -> {rec['transported']['dual']['top3_recall']:.2f} | real share omp {rec['native']['omp']['real_share']:.2f} -> {rec['transported']['omp']['real_share']:.2f} | transported atom gain by source block " + " ".join(f"{g:.2f}" for g in gains))
record(f"e236_transported_{tag}", dict(model=tag, L=L, per_level={str(k): v for k, v in out.items()}), " | ".join(f"level {l}: dominant recall dual {v['results']['native']['dual']['dominant_recall']:.2f} -> {v['results']['transported']['dual']['dominant_recall']:.2f}, omp {v['results']['native']['omp']['dominant_recall']:.2f} -> {v['results']['transported']['omp']['dominant_recall']:.2f}; top-3 dual {v['results']['native']['dual']['top3_recall']:.2f} -> {v['results']['transported']['dual']['top3_recall']:.2f}; real share omp {v['results']['native']['omp']['real_share']:.2f} -> {v['results']['transported']['omp']['real_share']:.2f}" for l, v in out.items()))
