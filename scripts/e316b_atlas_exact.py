"""e316b: the functional atlas with exact coordinates (the operator atlas of e316 was only a 0.3-0.6 approximation).
Every block-2 neuron's coordinate is measured directly: its unit write injected at the block-3 input at the natural
median amplitude over about 24 typical tokens, image centroid at the middle level, projected on the 16-dimensional
core. Then: effective rank and 16 classes; (a) class ablation (heaviest class, another class, a random set of equal
size): coherence of the ledger-weighted prediction and its cosine with the actual joint core coordinate; (b) same-block
substitution: a candidate ablated at its own tokens and a same-class block-2 atom injected at the block-3 input with
the coordinate-matched coefficient, against a random atom, restoration measured at the logits and at the core
coordinate; (c) ledger-to-core prediction for random subsets of six block-2 neurons (not only candidates), and
logit-level attribution by the atlas against direct logit attribution and transport-plus-read-out."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from func_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; NB = c.NB; b = 2; D = c.D; DFF = c.DFF
led, tn, tc = dominant(model, arch, c, ids_seq, b); run = make_runner_logits(model, arch, c, ids_seq, [L], NT, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep); pool = torch.nonzero(typ)[:, 0]; s_inj = tc.abs().median()
S0i = run(positions=idx); S1i = run(tn, positions=idx); dl = S0i["lg"] - S1i["lg"]; dl = dl - dl.mean(1, keepdim=True); F = (S0i[L] - S1i[L])[idx]; Fc = F - F.mean(0, keepdim=True); Ud, Sd, Vd = torch.linalg.svd(dl, full_matrices=False); B = Vd[:256].T; Zs = dl @ B; Zs = Zs - Zs.mean(0, keepdim=True); P = torch.linalg.svd(Fc.T @ Zs, full_matrices=False)[0][:, :16]
zc = Fc @ P; Gz = zc.T @ zc; Wz = torch.linalg.solve(Gz + 1e-2 * Gz.diagonal().mean() * torch.eye(16, device=DEV), zc.T @ Zs); Gx = Fc.T @ Fc; Wd = torch.linalg.solve(Gx + 1e-1 * Gx.diagonal().mean() * torch.eye(D, device=DEV), Fc.T @ Zs)
W2 = unit(arch.wdir(b).to(DEV)); torch.manual_seed(0); img = torch.zeros(DFF, D, device=DEV); cnt = torch.zeros(DFF, device=DEV); per_pass = min(1024, DFF); n_pass = max(1, (DFF + per_pass - 1) // per_pass) * 8
for p in range(n_pass):
    start = (p % max(1, DFF // per_pass)) * per_pass; ids = (start + torch.randperm(per_pass, device=DEV)) % DFF; a = ids[torch.randint(0, per_pass, (len(pool),), device=DEV)]; inj = torch.zeros(NT, D, device=DEV); inj[pool] = s_inj * W2[a]; S2 = run(inject=inj, inject_block=b + 1); img.index_add_(0, a, (S2[L] - S0[L])[pool]); cnt.index_add_(0, a, torch.ones(len(pool), device=DEV))
img = img / cnt[:, None].clamp_min(1) / s_inj; Z2 = img @ P; Zu = unit(Z2)
def prank(X): ev = torch.linalg.eigvalsh((X - X.mean(0, keepdim=True)).T @ (X - X.mean(0, keepdim=True))).clamp_min(0); return (ev.sum() ** 2 / (ev ** 2).sum()).item()
torch.manual_seed(2); cent = Zu[torch.randperm(DFF, device=DEV)[:16]]
for it in range(25):
    assign = (Zu @ cent.T).argmax(1); cent = unit(torch.stack([Zu[assign == k].mean(0) if (assign == k).any() else cent[k] for k in range(16)]))
sizes = torch.bincount(assign, minlength=16); within = torch.stack([(Zu[assign == k] @ cent[k]).mean() for k in range(16) if (assign == k).any()]).mean().item(); cw = torch.zeros(16, device=DEV).index_add_(0, assign, led[typ].abs().sum(0)); cw = cw / cw.sum()
def ablate_set(neur, rows=None, inject=None, inject_block=None):
    def pre(m, inp):
        x = inp[0].clone(); flat = x.reshape(-1, DFF)
        if rows is None: flat[:, neur] = 0
        else: flat[rows[:, None], neur[None, :]] = 0
        return (flat.reshape(x.shape),)
    h = arch.mlp_lin(b).register_forward_pre_hook(pre); r = run(positions=(pool if rows is None else rows), inject=inject, inject_block=inject_block); h.remove(); return r
base_pool = run(positions=pool)["lg"]
def class_test(neur):
    r = ablate_set(neur); dS = (S0[L] - r[L])[pool]; coef = led[pool][:, neur]; w = coef.abs().sum(1); rows = w >= w.quantile(0.8); pred = coef[rows] @ Z2[neur]; act = dS[rows] @ P; return dict(coherence=(pred.norm(dim=1) / (coef[rows].abs() @ Z2[neur].norm(dim=1)).clamp_min(1e-9)).median().item(), cos_pred_actual=((unit(act) * unit(pred)).sum(1)).median().item(), core_norm_per_weight=(act.norm(dim=1) / w[rows]).median().item())
k_heavy = int(cw.argmax()); mem = torch.nonzero(assign == k_heavy)[:, 0]; n = min(64, len(mem)); mem = mem[torch.randperm(len(mem), device=DEV)[:n]]; k_other = int(cw.argsort(descending=True)[1]); oth = torch.nonzero(assign == k_other)[:, 0]; oth = oth[torch.randperm(len(oth), device=DEV)[:n]]; rnd = torch.randperm(DFF, device=DEV)[:n]; cls = dict(heavy=class_test(mem), other=class_test(oth), random=class_test(rnd), n=int(n))
sub = {"same_class": [], "random": []}
for k in range(min(K, 12)):
    i = int(keep[k]); rows = idx[lab_i == k]; zi = Z2[i]; cs = (Zu @ unit(zi)); cs[i] = -1; j_same = int(cs.argmax()); j_rand = int(torch.randint(0, DFF, (1,)).item()); coef_i = led[rows, i]; base_rows = run(positions=rows); A_l = ablate_set(torch.tensor([i], device=DEV), rows)["lg"] - base_rows["lg"]; A_s = None
    r_abl = ablate_set(torch.tensor([i], device=DEV), rows); A_l = r_abl["lg"] - base_rows["lg"]; A_z = ((r_abl[L] - S0[L])[rows]) @ P
    for nm, j in (("same_class", j_same), ("random", j_rand)):
        zj = Z2[j]; alpha = coef_i * (zi @ zj) / (zj @ zj).clamp_min(1e-9); inj = torch.zeros(NT, D, device=DEV); inj[rows] = alpha[:, None] * W2[j][None]; r = ablate_set(torch.tensor([i], device=DEV), rows, inject=inj, inject_block=b + 1); R_l = r["lg"] - base_rows["lg"]; R_z = ((r[L] - S0[L])[rows]) @ P
        sub[nm].append(dict(logit_restoration=(1 - R_l.norm(dim=1) / A_l.norm(dim=1).clamp_min(1e-9)).median().item(), core_restoration=(1 - R_z.norm(dim=1) / A_z.norm(dim=1).clamp_min(1e-9)).median().item(), z_cos=cs[j].item() if nm == "same_class" else (Zu[j] @ unit(zi)).item()))
subs = {nm: {kk: float(torch.tensor([x[kk] for x in v]).median()) for kk in ("logit_restoration", "core_restoration", "z_cos")} for nm, v in sub.items()}
WU = arch.model.get_output_embeddings().weight.detach().float().to(DEV); att = {"ledger_to_core": [], "atlas_logits": [], "direct_logit": [], "transport_readout": []}
for trial in range(8):
    S = torch.randperm(DFF, device=DEV)[:6]; r = ablate_set(S); dS = (S0[L] - r[L])[pool]; dlS = base_pool - r["lg"]; dlS = dlS - dlS.mean(1, keepdim=True); coef = led[pool][:, S]; w = coef.abs().sum(1); rows = w >= w.quantile(0.9)
    pred_z = coef[rows] @ Z2[S]; act_z = dS[rows] @ P; att["ledger_to_core"].append(((unit(act_z) * unit(pred_z)).sum(1)).median().item()); act = dlS[rows]
    p_atlas = (pred_z @ Wz) @ B.T; p_direct = (coef[rows] @ W2[S]) @ WU.T; p_direct = p_direct - p_direct.mean(1, keepdim=True); p_tr = ((coef[rows] @ img[S]) @ Wd) @ B.T
    for nm, pv in (("atlas_logits", p_atlas), ("direct_logit", p_direct), ("transport_readout", p_tr)): att[nm].append(((unit(act) * unit(pv)).sum(1)).median().item())
attr = {nm: float(torch.tensor(v).median()) for nm, v in att.items()}
log(f"{tag} (K {K}, {DFF} block-2 atoms, {n_pass} passes, ~{cnt.median().item():.0f} tokens per atom): exact atlas effective rank {prank(Z2):.1f}; classes {sizes.min().item()}-{sizes.max().item()}, within-class cos {within:.2f}, heaviest class weight {cw.max().item():.2f} | (a) class ablation (n {n}): heavy coherence {cls['heavy']['coherence']:.2f} cos {cls['heavy']['cos_pred_actual']:.2f} norm/weight {cls['heavy']['core_norm_per_weight']:.3f}; other {cls['other']['coherence']:.2f}/{cls['other']['cos_pred_actual']:.2f}/{cls['other']['core_norm_per_weight']:.3f}; random {cls['random']['coherence']:.2f}/{cls['random']['cos_pred_actual']:.2f}/{cls['random']['core_norm_per_weight']:.3f} | (b) same-block substitution: same-class (z cos {subs['same_class']['z_cos']:.2f}) logit restoration {subs['same_class']['logit_restoration']:+.2f}, core restoration {subs['same_class']['core_restoration']:+.2f}; random atom (z cos {subs['random']['z_cos']:.2f}) logit {subs['random']['logit_restoration']:+.2f}, core {subs['random']['core_restoration']:+.2f} | (c) random six-neuron subsets: ledger-to-core cos {attr['ledger_to_core']:.2f}; logit attribution atlas {attr['atlas_logits']:.2f}, direct {attr['direct_logit']:.2f}, transport+read-out {attr['transport_readout']:.2f}")
record(f"e316b_atlasexact_{tag}", dict(model=tag, b=b, L=L, K=K, prank=prank(Z2), class_sizes=sizes.tolist(), within_class_cos=within, class_ablation=cls, substitution=subs, attribution=attr), f"prank {prank(Z2):.1f} within {within:.2f} | class ablation heavy/other/random coherence {cls['heavy']['coherence']:.2f}/{cls['other']['coherence']:.2f}/{cls['random']['coherence']:.2f} cos {cls['heavy']['cos_pred_actual']:.2f}/{cls['other']['cos_pred_actual']:.2f}/{cls['random']['cos_pred_actual']:.2f} | substitution same-class logit {subs['same_class']['logit_restoration']:+.2f} core {subs['same_class']['core_restoration']:+.2f} random logit {subs['random']['logit_restoration']:+.2f} core {subs['random']['core_restoration']:+.2f} | ledger-to-core {attr['ledger_to_core']:.2f} attribution atlas {attr['atlas_logits']:.2f} direct {attr['direct_logit']:.2f} transport {attr['transport_readout']:.2f}")
