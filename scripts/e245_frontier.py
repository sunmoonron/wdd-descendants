"""e245: the recoverability frontier. Candidates: the 64 most frequent block-2 dominant neurons (or fewer), each with a
zero-shot transplant image at levels b+2 and L (no natural footprints used). At foreign tokens inject k of them
(k = 1..64) under four amplitude regimes: equal; one dominant (1.0) plus small (0.3); heavy-tailed (1/i); half with
flipped sign. Per (k, regime) and level: additivity error (all vs two halves); identity recovery from the mixture by
the transplant images (recall@k; top-1 = the largest-amplitude vector); NATIVE sparse decomposition of the injected
state with the frozen dictionary (dual@64 and OMP@64): is the largest injected write's own atom in the support.
Control: 64 random unit vectors as candidates with their own transplant images. Also the coherent fraction of
transport (energy of the context mean over the mean energy of single-vector images). Kill for 'transport is
compositional where decomposition is not': native recall and identity recall fall together."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from desc_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; b = 2; levels = sorted({b + 2, L}); run = make_runner(model, arch, c, ids_seq, levels, NT); lab = c.d["lab"]; A = c.d["A"]; R = A[(lab["type"] == T_MLP) & (lab["block"] == b)].float().to(DEV)
led, tn, tc = dominant(model, arch, c, ids_seq, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); idx0 = torch.nonzero(typ & big)[:, 0]; u, cnt = torch.unique(tn[idx0], return_counts=True); order = cnt.argsort(descending=True); keep = u[order][:64]; K = len(keep)
med = torch.stack([tc[idx0][tn[idx0] == k].median().abs() for k in keep]); foreign = torch.nonzero(typ & ~torch.isin(tn, keep))[:, 0]; NF = len(foreign)
dicts = {}; rows = {}
for lv in levels:
    Alv, lablv = c.dictionary(lv); S = Alv.T @ Alv; ev, V = torch.linalg.eigh(S); dicts[lv] = (Alv, V @ torch.diag(1 / (ev + 1e-2 * ev[-1])) @ V.T, c.s["mu"][lv + 1].to(DEV)); rows[lv] = c.atom_index(lv, torch.full((K,), b, dtype=torch.long), keep.cpu()).to(DEV)
def images(vecs, amps):
    acc = {lv: torch.zeros(len(vecs), c.D, device=DEV) for lv in levels}; cnt_ = torch.zeros(len(vecs), device=DEV); coh = {lv: [] for lv in levels}; torch.manual_seed(1)
    for p in range(3):
        assign = torch.randint(0, len(vecs), (NF,), device=DEV); inj = torch.zeros(NT, c.D, device=DEV); inj[foreign] = amps[assign][:, None] * vecs[assign]; S2 = run(inject=inj, inject_block=b + 1)
        for lv in levels:
            img = (S2[lv] - S0[lv])[foreign] / amps[assign][:, None]; acc[lv].index_add_(0, assign, img); coh[lv].append((img, assign))
        cnt_.index_add_(0, assign, torch.ones(NF, device=DEV))
    out_img = {}; coherent = {}
    for lv in levels:
        M = acc[lv] / cnt_.clamp_min(1)[:, None]; allimg = torch.cat([x[0] for x in coh[lv]]); alla = torch.cat([x[1] for x in coh[lv]]); coherent[lv] = ((M[alla] ** 2).sum(1).mean() / (allimg ** 2).sum(1).mean()).item(); out_img[lv] = unit(M)
    return out_img, coherent
IMG, coherent = images(R[keep], med); torch.manual_seed(7); RV = unit(torch.randn(K, c.D, device=DEV)); IMGr, coherent_r = images(RV, med)
regimes = {"equal": lambda k: torch.ones(k), "dominant": lambda k: torch.tensor([1.0] + [0.3] * (k - 1)), "heavy": lambda k: 1.0 / torch.arange(1, k + 1).float(), "signed": lambda k: torch.tensor([1.0 if i % 2 == 0 else -1.0 for i in range(k)])}
res = {}
for reg, ampf in regimes.items():
    res[reg] = {}
    for k in [kk for kk in (1, 2, 4, 8, 16, 32, 64) if kk <= K]:
        amps = ampf(k).to(DEV); torch.manual_seed(100 + k); perm = torch.argsort(torch.rand(NF, K, device=DEV), dim=1)[:, :k]
        def inj_of(cols, vecs):
            inj = torch.zeros(NT, c.D, device=DEV)
            for j in cols: inj[foreign] += (amps[j] * med[perm[:, j]])[:, None] * vecs[perm[:, j]]
            return inj
        SA = run(inject=inj_of(range(k), R[keep]), inject_block=b + 1); SB = run(inject=inj_of(range(max(1, k // 2)), R[keep]), inject_block=b + 1) if k > 1 else None; SC = run(inject=inj_of(range(max(1, k // 2), k), R[keep]), inject_block=b + 1) if k > 1 else None
        SAr = run(inject=inj_of(range(k), RV), inject_block=b + 1) if reg == "equal" else None; rec = {}
        for lv in levels:
            FA = (SA[lv] - S0[lv])[foreign]; r = {}
            if k > 1: r["additivity"] = ((FA - (SB[lv] - S0[lv])[foreign] - (SC[lv] - S0[lv])[foreign]).norm(dim=1) / FA.norm(dim=1).clamp_min(1e-6)).median().item()
            sims = unit(FA) @ IMG[lv].T; topk = sims.topk(k, dim=1).indices; r["identity_recall_at_k"] = (topk[:, :, None] == perm[:, None, :]).any(1).float().mean().item(); r["dominant_top1"] = (sims.argmax(1) == perm[:, 0]).float().mean().item(); r["chance_top1"] = 1.0 / K
            Alv, Winv, mu = dicts[lv]; X = (SA[lv] - mu)[foreign]; sd = oneshot(X, Alv, 64, whiten=Winv)[0]; so = omp(X, Alv, 64)[0]; target = rows[lv][perm[:, 0]]; r["native_dual_dominant"] = (sd == target[:, None]).any(1).float().mean().item(); r["native_omp_dominant"] = (so == target[:, None]).any(1).float().mean().item()
            if SAr is not None: FR = (SAr[lv] - S0[lv])[foreign]; simsr = unit(FR) @ IMGr[lv].T; r["random_vectors_identity_recall_at_k"] = (simsr.topk(k, dim=1).indices[:, :, None] == perm[:, None, :]).any(1).float().mean().item(); r["random_vectors_top1"] = (simsr.argmax(1) == perm[:, 0]).float().mean().item()
            rec[lv] = r
        res[reg][k] = rec
        log(f"{tag} {reg} k={k}: " + " | ".join(f"lv{lv}: add {v.get('additivity', float('nan')):.2f}, identity@k {v['identity_recall_at_k']:.2f}, dominant top-1 {v['dominant_top1']:.2f} (chance {v['chance_top1']:.2f}), native dual {v['native_dual_dominant']:.2f} omp {v['native_omp_dominant']:.2f}" + (f", random-vector identity@k {v['random_vectors_identity_recall_at_k']:.2f} top-1 {v['random_vectors_top1']:.2f}" if 'random_vectors_top1' in v else "") for lv, v in rec.items()))
record(f"e245_frontier_{tag}", dict(model=tag, b=b, L=L, K=K, coherent_fraction={str(k): v for k, v in coherent.items()}, coherent_fraction_random={str(k): v for k, v in coherent_r.items()}, results={reg: {str(k): {str(lv): v for lv, v in rec.items()} for k, rec in d.items()} for reg, d in res.items()}), f"K {K}; coherent fraction of transport at +2 / L: {coherent[b + 2]:.2f} / {coherent[L]:.2f} (random vectors {coherent_r[b + 2]:.2f} / {coherent_r[L]:.2f}) | equal amplitudes at L, k=1/4/16/64: identity@k " + " ".join(f"{res['equal'][k][L]['identity_recall_at_k']:.2f}" for k in res['equal'] if k in (1, 4, 16, 64)) + ", dominant top-1 " + " ".join(f"{res['equal'][k][L]['dominant_top1']:.2f}" for k in res['equal'] if k in (1, 4, 16, 64)) + ", native dual " + " ".join(f"{res['equal'][k][L]['native_dual_dominant']:.2f}" for k in res['equal'] if k in (1, 4, 16, 64)) + " | heavy-tailed at L: top-1 " + " ".join(f"{res['heavy'][k][L]['dominant_top1']:.2f}" for k in res['heavy'] if k in (1, 4, 16, 64)) + ", native dual " + " ".join(f"{res['heavy'][k][L]['native_dual_dominant']:.2f}" for k in res['heavy'] if k in (1, 4, 16, 64)) + " | random vectors equal k=1/16: top-1 " + " ".join(f"{res['equal'][k][L].get('random_vectors_top1', float('nan')):.2f}" for k in res['equal'] if k in (1, 16)))
