"""e385: do readers read what WDD can read? The original program's prominence law says a write is identifiable from
the state when its projection beats a competitor level. Here, on natural text, for every head (reader) of layers 1
and up and each of its query, key and value inputs, and every upstream head's write at every position: whether WDD
(OMP, k = 64, over the model's own atoms, on the centred state) selects any atom of that upstream head at that position
(identified), the write's norm, and its exact share of the reader's input. Question: do readers draw their input
disproportionately from identified writes, beyond what the writes' size explains? Reported: the share of the readers'
total |input share| carried by identified (head, position) entries against the share of write-norm mass they carry
(ratio above 1 = readers prefer identifiable writes), within write-norm quintiles, and the AUC with which
identification picks the top-5% |share| entries compared with the AUC of write norm alone."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p3_common import *
tag = sys.argv[1]; c = Cache(tag); model, tok, fam = load_eager(c.name); arch = Arch(model, fam); NB, NH, HD = arch.NB, arch.NH, arch.HD; torch.manual_seed(0)
nat = c.s["eval_ids"][:4, :256].to(DEV); cap = Capture(model, arch); X, Z, M, out = cap(nat)
keep = torch.ones(nat.shape[0], nat.shape[1], dtype=torch.bool, device=DEV); keep[:, 0] = False; rows = lambda T: T[keep].reshape(-1, T.shape[-1])
def auc(score, lab):
    pos, neg = score[lab], score[~lab]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    r = torch.cat([pos, neg]).argsort().argsort().float() + 1; return ((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))).item()
tot_id_share = tot_share = tot_id_norm = tot_norm = 0.0; per_q = torch.zeros(5, 4); aucs_id, aucs_norm, aucs_both = [], [], []
for l in range(1, NB):
    Xr = rows(X[l]); P = Xr.shape[0]; A, lab = c.dictionary(l - 1, blocks=list(range(l))); mu = c.s["mu"][l].to(DEV).float(); sel, cof, _ = omp(Xr - mu[None], A, 64, batch=512, record_err=False)
    typ, blk, idx = lab["type"].to(DEV)[sel], lab["block"].to(DEV)[sel], lab["index"].to(DEV)[sel]; gid = torch.where(typ == T_ATT, blk * NH + idx, torch.full_like(blk, -1))
    G = l * NH; ident = torch.zeros(P, G, dtype=torch.bool, device=DEV)
    for g in range(G): ident[:, g] = (gid == g).any(-1)
    O = torch.cat([(rows(Z[b]).view(P, NH, HD)[:, :, None, :] @ arch.wo(b).to(DEV).view(NH, HD, -1)[None]).squeeze(2) for b in range(l)], 1)  # [P, G, D]
    nrm = O.norm(dim=-1)
    qs = torch.quantile(nrm.flatten().float()[torch.randperm(nrm.numel(), device=DEV)[:200000]], torch.tensor([0.2, 0.4, 0.6, 0.8], device=DEV)); qbin = torch.bucketize(nrm, qs)
    for h in range(NH):
        for w in ("Q", "K", "V"):
            r, _ = reader_dirs(arch, l, h, w, Xr); s = (O * r[:, None, :]).sum(-1).abs()
            tot_share += s.sum().item(); tot_id_share += s[ident].sum().item(); tot_norm += nrm.sum().item(); tot_id_norm += nrm[ident].sum().item()
            for qb in range(5):
                m_ = qbin == qb; per_q[qb, 0] += s[m_ & ident].sum().item(); per_q[qb, 1] += s[m_].sum().item(); per_q[qb, 2] += nrm[m_ & ident].sum().item(); per_q[qb, 3] += nrm[m_].sum().item()
            topm = s >= torch.quantile(s.flatten().float()[torch.randperm(s.numel(), device=DEV)[:200000]], 0.95)
            aucs_id.append(auc(ident.float().flatten() + 1e-6 * torch.rand(ident.numel(), device=DEV), topm.flatten())); aucs_norm.append(auc(nrm.flatten(), topm.flatten())); aucs_both.append(auc(nrm.flatten() * (1 + ident.float().flatten()), topm.flatten()))
    del O
ratio = (tot_id_share / tot_share) / max(tot_id_norm / tot_norm, 1e-9); qr = [((per_q[q, 0] / per_q[q, 1].clamp_min(1e-9)) / (per_q[q, 2] / per_q[q, 3].clamp_min(1e-9)).clamp_min(1e-9)).item() for q in range(5)]
mean = lambda v: sum(x for x in v if x == x) / max(sum(1 for x in v if x == x), 1)
res = dict(model=tag, identified_share_of_input=tot_id_share / tot_share, identified_share_of_norm=tot_id_norm / tot_norm, preference_ratio=ratio, preference_ratio_by_norm_quintile=qr, auc_identified=mean(aucs_id), auc_norm=mean(aucs_norm), auc_norm_plus_identified=mean(aucs_both), n_readers=len(aucs_id))
log(f"{tag}: identified writes carry {res['identified_share_of_input']:.2f} of readers' |input share| and {res['identified_share_of_norm']:.2f} of write-norm mass (preference ratio {ratio:.2f}; by norm quintile " + " ".join(f"{v:.2f}" for v in qr) + f"); AUC for top-5% share entries: identification {res['auc_identified']:.2f}, write norm {res['auc_norm']:.2f}, norm with identification {res['auc_norm_plus_identified']:.2f} over {res['n_readers']} reader inputs")
record(f"e385_readprominent_{tag}", res, f"preference ratio {ratio:.2f} (quintiles " + " ".join(f"{v:.2f}" for v in qr) + f") AUC id {res['auc_identified']:.2f} norm {res['auc_norm']:.2f} both {res['auc_norm_plus_identified']:.2f}")
