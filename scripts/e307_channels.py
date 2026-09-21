"""e307: persistent causal channels across all depths. The per-token descendants of the candidates at every block
after the write, projected on the fixed 16-dimensional function basis of level L (and on the basis of level b+2 for
the early-birth test). Between consecutive blocks: per-coordinate gain (regression slope), self-R2, mixing (extra
variance explained by the other coordinates), the optimal assignment of coordinates (Hungarian on |correlation|)
and whether it is the identity; along depth: each channel's share of the core variance (birth, death), and the
largest off-diagonal correlation (merging)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from func_common import *
try:
    from scipy.optimize import linear_sum_assignment
except Exception: linear_sum_assignment = None
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 16; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; NB = c.NB; b = 2; D = c.D; d = 16; levels = list(range(b + 1, NB))
led, tn, tc = dominant(model, arch, c, ids_seq, b); run = make_runner_logits(model, arch, c, ids_seq, levels, NT, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep)
S0i = run(positions=idx); S1i = run(tn, positions=idx); dl = S0i["lg"] - S1i["lg"]; dl = dl - dl.mean(1, keepdim=True); G = dl @ dl.T; evg, Vg = torch.linalg.eigh(G); Z = Vg.flip(1)[:, :256] * evg.flip(0)[:256].clamp_min(0).sqrt()[None]; Z = Z - Z.mean(0, keepdim=True)
def basis(lv):
    F = (S0i[lv] - S1i[lv])[idx]; Fc = F - F.mean(0, keepdim=True); return torch.linalg.svd(Fc.T @ Z, full_matrices=False)[0][:, :d]
def analyse(P, name):
    zs = {lv: ((S0i[lv] - S1i[lv])[idx] - (S0i[lv] - S1i[lv])[idx].mean(0, keepdim=True)) @ P for lv in levels}; per_block = {}; shares = {}
    for lv in levels:
        z = zs[lv]; shares[lv] = (z.var(0) / z.var(0).sum()).tolist()
        if lv + 1 not in zs: continue
        z2 = zs[lv + 1]; Cm = torch.corrcoef(torch.cat([z, z2], 1).T)[:d, d:]; slope = (z * z2).sum(0) / (z * z).sum(0).clamp_min(1e-9); r2 = (Cm.diagonal() ** 2); Gz = z.T @ z; Wall = torch.linalg.solve(Gz + 1e-3 * Gz.diagonal().mean() * torch.eye(d, device=DEV), z.T @ z2); r2_all = 1 - ((z2 - z @ Wall) ** 2).sum(0) / (z2 ** 2).sum(0).clamp_min(1e-9); mixing = (r2_all - r2).clamp_min(0)
        if linear_sum_assignment is not None:
            ri, ci = linear_sum_assignment(-Cm.abs().cpu().numpy()); ident = float((ci == ri).mean())
        else:
            ident = float((Cm.abs().argmax(1) == torch.arange(d, device=DEV)).float().mean().item())
        off = Cm.abs() - torch.diag(Cm.abs().diagonal()); per_block[lv] = dict(gain_mean=slope.mean().item(), gain_min=slope.min().item(), gain_max=slope.max().item(), self_r2_mean=r2.mean().item(), mixing_mean=mixing.mean().item(), assignment_identity=ident, max_offdiag_corr=off.max().item())
    sh = torch.tensor([shares[lv] for lv in levels]); first, last = sh[0], sh[-1]; ratio = (last / first.clamp_min(1e-9)); born = int((ratio > 4).sum()); died = int((ratio < 0.25).sum()); total_gain = torch.tensor([[per_block[lv]["gain_mean"]] for lv in per_block]).prod().item()
    return dict(per_block={str(k): v for k, v in per_block.items()}, born=born, died=died, share_first=first.tolist(), share_last=last.tolist(), summary=dict(gain_mean=sum(v["gain_mean"] for v in per_block.values()) / len(per_block), self_r2_mean=sum(v["self_r2_mean"] for v in per_block.values()) / len(per_block), mixing_mean=sum(v["mixing_mean"] for v in per_block.values()) / len(per_block), assignment_identity_mean=sum(v["assignment_identity"] for v in per_block.values()) / len(per_block), max_offdiag_mean=sum(v["max_offdiag_corr"] for v in per_block.values()) / len(per_block), min_self_r2=min(v["self_r2_mean"] for v in per_block.values()), min_assignment=min(v["assignment_identity"] for v in per_block.values())))
res = {"basis_L": analyse(basis(L), "L"), "basis_early": analyse(basis(b + 2), "early")}
for nm, v in res.items():
    s = v["summary"]; log(f"{tag} (K {K}) {nm}: per-block means over {len(v['per_block'])} transitions: gain {s['gain_mean']:.2f}, self-R2 {s['self_r2_mean']:.2f} (min block {s['min_self_r2']:.2f}), mixing {s['mixing_mean']:.2f}, optimal assignment = identity {s['assignment_identity_mean']:.2f} (min block {s['min_assignment']:.2f}), max off-diagonal correlation {s['max_offdiag_mean']:.2f}; channels born {v['born']}, died {v['died']} (share ratio last/first > 4 or < 1/4) | per block self-R2: " + " ".join(f"{lv}:{pb['self_r2_mean']:.2f}" for lv, pb in v["per_block"].items()))
record(f"e307_channels_{tag}", dict(model=tag, b=b, L=L, K=K, d=d, **{k: v for k, v in res.items()}), " | ".join(f"{nm}: gain {v['summary']['gain_mean']:.2f} selfR2 {v['summary']['self_r2_mean']:.2f} (min {v['summary']['min_self_r2']:.2f}) mixing {v['summary']['mixing_mean']:.2f} identity-assignment {v['summary']['assignment_identity_mean']:.2f} (min {v['summary']['min_assignment']:.2f}) maxoff {v['summary']['max_offdiag_mean']:.2f} born {v['born']} died {v['died']}" for nm, v in res.items()))
