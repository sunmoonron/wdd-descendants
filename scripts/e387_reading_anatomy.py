"""e387: the anatomy of reading, and whether "readers read what WDD can read" is trivial, learned or neither. On natural
text, for every head's query, key and value input (layers 1 and up), the exact share of that input contributed by every
upstream component (each head's write, each MLP block's output, the embeddings) at every position. Questions:
(1) CONTROL: a component that makes up much of the state along the state's own direction is both easy for WDD to
identify and gets a large share of any reader's input (the full state has share 1). The reader direction r_n is split
into its part along the (centred) state and the orthogonal part, and the preference for WDD-identified writes is
re-measured with the orthogonal part only, and within quintiles of each write's alignment with the state.
(2) SPARSITY: the fan-in of reading, per reader input and position: the participation ratio of the components' |shares|
and how many components carry 50% and 80% of it.
(3) HOW MUCH OF IT WDD SEES: WDD run to k = 256 atoms (OMP picks are nested, so one run gives every k); for k = 8 to
256, the share of the readers' |input| carried by components with at least one selected atom, the share of write norm
they carry, and the preference ratio (first / second).
Optional second argument: a Pythia revision (the dictionary is then built from that checkpoint's weights; states are
centred by the batch mean at every level for all runs)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p3_common import *
tag = sys.argv[1]; rev = sys.argv[2] if len(sys.argv) > 2 else None; rtag = tag + (f"_{rev}" if rev else ""); c = Cache(tag)
model, tok, fam = load_eager(c.name, revision=rev); arch = Arch(model, fam); NB, NH, HD = arch.NB, arch.NH, arch.HD; torch.manual_seed(0)
Afull, lab = build_dictionary(arch); lab = {k: v.to(DEV) for k, v in lab.items()}
nat = c.s["eval_ids"][:4, :256].to(DEV); cap = Capture(model, arch); X, Z, M, out = cap(nat)
keep = torch.ones(nat.shape[0], nat.shape[1], dtype=torch.bool, device=DEV); keep[:, 0] = False; rows = lambda T: T[keep].reshape(-1, T.shape[-1])
KS = [8, 16, 32, 64, 128, 256]; KMAX = 256
acc = {k: dict(id_s=0.0, id_sp=0.0, id_n=0.0) for k in KS}; tot = dict(s=0.0, sp=0.0, n=0.0); abin = torch.zeros(5, 4); fan_pr, fan50, fan80, part = [], [], [], dict(heads=0.0, mlp=0.0, emb=0.0)
for l in range(1, NB):
    Xr = rows(X[l]); P = Xr.shape[0]; keepA = (lab["block"] < l); A = Afull[keepA]; typ, blk, idx = lab["type"][keepA], lab["block"][keepA], lab["index"][keepA]
    sel, cof, _ = omp(Xr - Xr.mean(0, keepdim=True), A, KMAX, batch=256, record_err=False)
    G = l * NH; gid = torch.where(typ == T_ATT, blk * NH + idx, torch.where(typ == T_MLP, G + blk, torch.full_like(blk, -1)))[sel]  # heads 0..G-1, MLP blocks G..G+l-1
    first = torch.full((P, G + l), KMAX + 1, device=DEV, dtype=torch.long); ar = torch.arange(KMAX, device=DEV)[None].expand(P, -1)
    for comp in range(G + l):
        m_ = gid == comp; firstpick = torch.where(m_, ar, torch.full_like(ar, KMAX + 1)).min(-1).values; first[:, comp] = firstpick
    Oh = torch.cat([(rows(Z[b]).view(P, NH, HD)[:, :, None, :] @ arch.wo(b).to(DEV).view(NH, HD, -1)[None]).squeeze(2) for b in range(l)], 1)   # [P, G, D]
    Om = torch.stack([rows(M[b]) for b in range(l)], 1)                                                                                     # [P, l, D]
    E = rows(X[0])                                                                                                                           # [P, D] embeddings
    O = torch.cat([Oh, Om], 1); nrm = O.norm(dim=-1); g, inv, cen = norm_frozen(arch, l, Xr); Xt = Xr - cen * Xr.mean(-1, keepdim=True)
    align = (O * Xt[:, None, :]).sum(-1).abs() / (nrm * Xt.norm(dim=-1, keepdim=True)).clamp_min(1e-9)
    aq = torch.quantile(align.flatten()[torch.randperm(align.numel(), device=DEV)[:200000]].float(), torch.tensor([0.2, 0.4, 0.6, 0.8], device=DEV)); abins = torch.bucketize(align, aq)
    RA, RPA = [], []
    for h in range(NH):
        for w in ("Q", "K", "V"):
            r, _ = reader_dirs(arch, l, h, w, Xr); rp = r - ((r * Xt).sum(-1, keepdim=True) / Xt.pow(2).sum(-1, keepdim=True).clamp_min(1e-9)) * Xt; RA.append(r); RPA.append(rp)
    RA, RPA = torch.stack(RA, 2), torch.stack(RPA, 2); NR = RA.shape[-1]                     # [P, D, NR]
    S = torch.bmm(O, RA).abs(); SP = torch.bmm(O, RPA).abs(); SE = torch.einsum("pd,pdr->pr", E, RA).abs()   # [P, C, NR], [P, NR]
    tot["s"] += S.sum().item(); tot["sp"] += SP.sum().item(); tot["n"] += nrm.sum().item() * NR
    for k in KS:
        idm = (first < k); acc[k]["id_s"] += (S * idm[:, :, None]).sum().item(); acc[k]["id_sp"] += (SP * idm[:, :, None]).sum().item(); acc[k]["id_n"] += (nrm * idm).sum().item() * NR
    idm = first < 64
    for qb in range(5):
        mq = abins == qb; abin[qb, 0] += (S * (mq & idm)[:, :, None]).sum().item(); abin[qb, 1] += (S * mq[:, :, None]).sum().item(); abin[qb, 2] += (nrm * (mq & idm)).sum().item() * NR; abin[qb, 3] += (nrm * mq).sum().item() * NR
    allc = torch.cat([S, SE[:, None, :]], 1); tt = allc.sum(1, keepdim=True).clamp_min(1e-12); pr = allc.sum(1) ** 2 / allc.pow(2).sum(1).clamp_min(1e-12)
    srt = (allc / tt).sort(1, descending=True).values.cumsum(1); n50 = ((srt < 0.5).sum(1) + 1).float(); n80 = ((srt < 0.8).sum(1) + 1).float()
    fan_pr.append(pr.flatten()); fan50.append(n50.flatten()); fan80.append(n80.flatten())
    part["heads"] += S[:, :G].sum().item(); part["mlp"] += S[:, G:].sum().item(); part["emb"] += SE.sum().item()
    del O, Oh, Om
kc = {str(k): dict(captured_input=acc[k]["id_s"] / tot["s"], captured_norm=acc[k]["id_n"] / tot["n"], preference=(acc[k]["id_s"] / tot["s"]) / max(acc[k]["id_n"] / tot["n"], 1e-9), preference_orthogonal=(acc[k]["id_sp"] / tot["sp"]) / max(acc[k]["id_n"] / tot["n"], 1e-9)) for k in KS}
ab = [((abin[q, 0] / abin[q, 1].clamp_min(1e-9)) / (abin[q, 2] / abin[q, 3].clamp_min(1e-9)).clamp_min(1e-9)).item() for q in range(5)]
fan_pr, fan50, fan80 = torch.cat(fan_pr), torch.cat(fan50), torch.cat(fan80); med = lambda v: v.float().median().item(); ptot = sum(part.values())
res = dict(model=tag, rev=rev, k_curve=kc, preference_by_alignment_quintile_k64=ab, fan_in_participation_ratio_median=med(fan_pr), fan_in_components_for_50pct_median=med(fan50), fan_in_components_for_80pct_median=med(fan80), input_from=dict((k, v / ptot) for k, v in part.items()), n_reader_inputs=int(fan_pr.numel()))
log(f"{rtag}: k-curve (captured share of readers' |input| / of write norm / preference / preference with the state-orthogonal reader direction): " + " | ".join(f"k{k}: {v['captured_input']:.2f}/{v['captured_norm']:.2f}/{v['preference']:.2f}/{v['preference_orthogonal']:.2f}" for k, v in kc.items()) + f" || preference within alignment-with-state quintiles (k64): " + " ".join(f"{v:.2f}" for v in ab) + f" || fan-in: participation ratio {res['fan_in_participation_ratio_median']:.1f}, components for 50% {res['fan_in_components_for_50pct_median']:.0f}, for 80% {res['fan_in_components_for_80pct_median']:.0f}; input from heads {res['input_from']['heads']:.2f}, MLPs {res['input_from']['mlp']:.2f}, embeddings {res['input_from']['emb']:.2f}")
record(f"e387_readanat_{rtag}", res, f"k64 captured {kc['64']['captured_input']:.2f} pref {kc['64']['preference']:.2f} orth {kc['64']['preference_orthogonal']:.2f} | k256 captured {kc['256']['captured_input']:.2f} | fan-in PR {res['fan_in_participation_ratio_median']:.1f} n50 {res['fan_in_components_for_50pct_median']:.0f}")
