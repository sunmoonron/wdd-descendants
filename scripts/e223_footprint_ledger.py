"""e223: where does a write's energy go? Exact accounting. Zero the token's dominant block-b write (b = 1, 2, 3).
The footprint delta(lv) = H_ablated(lv) - H_clean(lv) equals -write + sum_{j>b} dMLP_j + sum_{j>b} dATT_j exactly
(residual identity), and each dMLP_j = sum_i dc_{j,i} w_{j,i} is a ledger over known writers (the delta-ledger).
Per level (b+1, b+2, b+4, L): the norms of the parts relative to the write, the coherence of the scattered MLP energy,
the provenance entropy of the delta-ledger (effective number of writers, top-1 share, writers for 90%), its block
distribution, the share on neurons already active in the clean run, the Mahalanobis energy (whitened by the level's
covariance) vs Euclidean, and the provenance horizon per token (first level where the along-direction footprint
falls below half) with its correlates (write size, state norm, position). Function: same-position dCE and KL."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 8; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; NB = c.NB; lab = c.d["lab"]; A = c.d["A"]
def rows(b): return A[(lab["type"] == T_MLP) & (lab["block"] == b)].float().to(DEV)
WN = [c.d["WN"][j].to(DEV) for j in range(NB)]; wd = [c.wdir_cpu(j).to(DEV) for j in range(NB)]
def run(b=None, neuron=None):
    st = {}; hs = []
    for j in range(NB):
        hs.append(arch.layers[j].register_forward_hook((lambda j_: lambda m, i, o: st.__setitem__(("H", j_), (o[0] if isinstance(o, tuple) else o).detach().float().reshape(-1, c.D)))(j)))
        hs.append(arch.mlp_lin(j).register_forward_pre_hook((lambda j_: lambda m, inp: st.__setitem__(("a", j_), inp[0].detach().float().reshape(-1, c.DFF)))(j)))
        hs.append(arch.attn_lin(j).register_forward_hook((lambda j_: lambda m, i, o: st.__setitem__(("att", j_), (o[0] if isinstance(o, tuple) else o).detach().float().reshape(-1, c.D)))(j)))
    if b is not None:
        def pre(m, inp):
            x = inp[0].clone(); flat = x.reshape(-1, c.DFF); flat[torch.arange(NT, device=DEV), neuron] = 0; return (flat.reshape(x.shape),)
        hs.append(arch.mlp_lin(b).register_forward_pre_hook(pre))
    out = model(ids_seq); [h.remove() for h in hs]; lg = out.logits[:, :-1].reshape(-1, out.logits.shape[-1]).float(); return st, lg
S0, lg0 = run(); lp0 = torch.log_softmax(lg0, -1); pos = torch.arange(CTX, device=DEV).repeat(NS); Sig = {}
def maha(v, lv):
    if lv not in Sig:
        Xs = c.X(lv)[sub(c.NT, 8192, seed=3)]; S = Xs.T @ Xs / len(Xs); Sig[lv] = torch.linalg.inv(S + 1e-3 * S.diagonal().mean() * torch.eye(c.D, device=DEV))
    return (v @ Sig[lv] * v).sum(1)
res = {}
for b in (1, 2, 3):
    a0 = S0[("a", b)]; led = a0 * WN[b][None]; tn = led.abs().argmax(1); tc = torch.gather(led, 1, tn[:, None])[:, 0]; d = rows(b)[tn]; big = tc.abs() >= tc.abs().quantile(0.5)
    S1, lg1 = run(b, tn); prof = {}; along_by_level = {}
    for lv in sorted({b + 1, b + 2, b + 4, L}):
        if lv >= NB: continue
        typ = typical_mask(S0[("H", lv)]) & big; delta = S1[("H", lv)] - S0[("H", lv)]; along = -(delta * d).sum(1) / tc; along_by_level[lv] = along
        dm = torch.zeros_like(delta); da = torch.zeros_like(delta); E_parts = torch.zeros(NT, device=DEV); ledgers = []
        for j in range(b + 1, lv + 1):
            dc = (S1[("a", j)] - S0[("a", j)]) * WN[j][None]; ledgers.append(dc); dm += (S1[("a", j)] - S0[("a", j)]) @ wd[j]; da += S1[("att", j)] - S0[("att", j)]; E_parts += (dc ** 2).sum(1)
        C = torch.cat(ledgers, 1) if ledgers else torch.zeros(NT, 1, device=DEV); p = C ** 2 / (C ** 2).sum(1, keepdim=True).clamp_min(1e-12); H = -(p * (p + 1e-12).log()).sum(1); srt = p.sort(1, descending=True).values; n90 = ((srt.cumsum(1) < 0.9).sum(1) + 1).float()
        C0 = torch.cat([S0[("a", j)] * WN[j][None] for j in range(b + 1, lv + 1)], 1) if ledgers else C; active0 = C0.abs() >= 0.05 * C0.abs().max(1, keepdim=True).values; share_active = (p * active0.float()).sum(1)
        blkshare = {j: (p[:, sum(c.DFF for _ in range(b + 1, j)):sum(c.DFF for _ in range(b + 1, j + 1))].sum(1))[typ].mean().item() for j in range(b + 1, lv + 1)}
        mh = (maha(delta, lv) / maha(tc[:, None] * d, b)).sqrt(); eu = delta.norm(dim=1) / tc.abs()
        prof[lv] = dict(level=lv, along=along[typ].median().item(), total=eu[typ].median().item(), mlp_part=(dm.norm(dim=1) / tc.abs())[typ].median().item(), att_part=(da.norm(dim=1) / tc.abs())[typ].median().item(), residual_check=((delta + tc[:, None] * d - dm - da).norm(dim=1) / delta.norm(dim=1).clamp_min(1e-6))[typ].median().item(), mlp_coherence=((dm ** 2).sum(1) / E_parts.clamp_min(1e-9))[typ].median().item(), n_eff=H.exp()[typ].median().item(), top1_share=srt[:, 0][typ].median().item(), n90=n90[typ].median().item(), share_on_active=share_active[typ].median().item(), block_share=blkshare, mahalanobis=mh[typ].median().item(), euclid=eu[typ].median().item())
        log(f"{tag} born b{b} level {lv}: |delta|/|w| {prof[lv]['total']:.2f} (Mahalanobis {prof[lv]['mahalanobis']:.2f}); along {prof[lv]['along']:.2f}; MLP-mediated {prof[lv]['mlp_part']:.2f}, attention-mediated {prof[lv]['att_part']:.2f} (identity residual {prof[lv]['residual_check']:.3f}); MLP coherence {prof[lv]['mlp_coherence']:.2f} | delta-ledger: N_eff {prof[lv]['n_eff']:.0f}, top-1 {prof[lv]['top1_share']:.2f}, n90 {prof[lv]['n90']:.0f}, on already-active neurons {prof[lv]['share_on_active']:.2f} | by block " + " ".join(f"b{j}:{v:.2f}" for j, v in blkshare.items()))
    # horizon per token: first level (from b+1) with along < 0.5; correlates
    lvs = sorted(along_by_level); hor = torch.full((NT,), float(len(lvs) + 1), device=DEV)
    for i in reversed(range(len(lvs))): hor = torch.where(along_by_level[lvs[i]] < 0.5, torch.full_like(hor, float(i + 1)), hor)
    typb = typical_mask(S0[("H", b)]) & big; xn = S0[("H", b)].norm(dim=1)
    def spearman(a_, b_):
        ra = a_.argsort().argsort().float(); rb = b_.argsort().argsort().float(); return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
    dce = (torch.nn.functional.cross_entropy(lg1, ids_seq[:, 1:].reshape(-1), reduction="none") - torch.nn.functional.cross_entropy(lg0, ids_seq[:, 1:].reshape(-1), reduction="none")); lp1 = torch.log_softmax(lg1, -1); kl = (lp0.exp() * (lp0 - lp1)).sum(1); bigm = big.reshape(NS, CTX)[:, :-1].reshape(-1)
    res[b] = dict(profile={str(k): v for k, v in prof.items()}, horizon=dict(rho_coef=spearman(hor[typb], tc.abs()[typb]), rho_norm=spearman(hor[typb], xn[typb]), rho_pos=spearman(hor[typb], pos[typb].float()), median=hor[typb].median().item()), dce=dce[bigm].mean().item(), kl=kl[bigm].median().item())
    log(f"{tag} born b{b}: horizon (levels until along < 0.5) median {res[b]['horizon']['median']:.0f}; Spearman with |coef| {res[b]['horizon']['rho_coef']:+.2f}, state norm {res[b]['horizon']['rho_norm']:+.2f}, position {res[b]['horizon']['rho_pos']:+.2f} | dCE {res[b]['dce']:+.3f} KL {res[b]['kl']:.4f}")
import numpy as np
agg = lambda key, lvk: float(np.mean([res[b]["profile"][k][key] for b in res for k in res[b]["profile"] if int(k) == (L if lvk == "L" else b + lvk)]))
record(f"e223_ledger_{tag}", dict(model=tag, L=L, per_birth={str(k): v for k, v in res.items()}), f"at +2 / at L: |delta|/|w| {agg('total', 2):.2f} / {agg('total', 'L'):.2f}, Mahalanobis {agg('mahalanobis', 2):.2f} / {agg('mahalanobis', 'L'):.2f}, MLP-mediated {agg('mlp_part', 2):.2f} / {agg('mlp_part', 'L'):.2f}, attention-mediated {agg('att_part', 2):.2f} / {agg('att_part', 'L'):.2f}, MLP coherence {agg('mlp_coherence', 2):.2f} / {agg('mlp_coherence', 'L'):.2f}, N_eff writers {agg('n_eff', 2):.0f} / {agg('n_eff', 'L'):.0f}, top-1 share {agg('top1_share', 2):.2f} / {agg('top1_share', 'L'):.2f}, on already-active {agg('share_on_active', 2):.2f} / {agg('share_on_active', 'L'):.2f} | horizon Spearman with |coef| " + " ".join(f"{res[b]['horizon']['rho_coef']:+.2f}" for b in res) + ", state norm " + " ".join(f"{res[b]['horizon']['rho_norm']:+.2f}" for b in res))
