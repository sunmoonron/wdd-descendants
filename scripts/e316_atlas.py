"""e316: the functional atlas, an end-to-end application. Every MLP neuron of blocks 0-5 is mapped to its functional
coordinate z_i = P^T (w_i T_b), with P the 16-dimensional core at the middle level (from natural block-2 descendants)
and T_b the data-free transport operator from the block-(b+1) input to the middle level. (1) Validation of the
operator atlas against exact transplant images for 240 neurons. (2) Atlas statistics: effective rank, 16 functional
classes by k-means on unit coordinates, class sizes, the cross-block composition of classes, and the ledger weight
each class carries in the natural state. (3) New-ground tests on the model: (a) class ablation: all block-2 neurons of
the heaviest class ablated jointly against equal-size random and other-class sets, the coherence of the predicted
functional contribution and the actual joint coordinate; (b) cross-block substitution: a block-2 candidate ablated at
its own tokens and a same-class block-4 atom injected at the block-5 input with the coordinate-matched coefficient,
against a random block-4 atom, restoration of the natural logit effect; (c) attribution: the joint logit effect of
random candidate subsets predicted by the atlas (ledger x coordinates x read-out), by direct logit attribution of the
writes, by an averaged-Jacobian lens of the writes without the quotient, and by the transported writes with the
read-out but without the quotient."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from func_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; NB = c.NB; b = 2; D = c.D; DFF = c.DFF; blocks = [bb for bb in range(0, 6) if bb + 1 < L]
led, tn, tc = dominant(model, arch, c, ids_seq, b); run = make_runner_logits(model, arch, c, ids_seq, [L, NB - 1], NT, b); S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep); pool = torch.nonzero(typ)[:, 0]; s_inj = tc.abs().median()
S0i = run(positions=idx); S1i = run(tn, positions=idx); dl = S0i["lg"] - S1i["lg"]; dl = dl - dl.mean(1, keepdim=True); F = (S0i[L] - S1i[L])[idx]; Fc = F - F.mean(0, keepdim=True); Ud, Sd, Vd = torch.linalg.svd(dl, full_matrices=False); B = Vd[:256].T; Zs = dl @ B; Zs = Zs - Zs.mean(0, keepdim=True); P = torch.linalg.svd(Fc.T @ Zs, full_matrices=False)[0][:, :16]
zc = Fc @ P; Gz = zc.T @ zc; Wz = torch.linalg.solve(Gz + 1e-2 * Gz.diagonal().mean() * torch.eye(16, device=DEV), zc.T @ Zs); Gx = Fc.T @ Fc; Wd = torch.linalg.solve(Gx + 1e-1 * Gx.diagonal().mean() * torch.eye(D, device=DEV), Fc.T @ Zs)
def fit_T(blk, lv, seed):
    torch.manual_seed(seed); Vr = unit(torch.randn(1024, D, device=DEV)); X = []; Y = []
    for p in range(2):
        a = torch.randint(0, 1024, (len(pool),), device=DEV); inj = torch.zeros(NT, D, device=DEV); inj[pool] = s_inj * Vr[a]; S2 = run(inject=inj, inject_block=blk); X.append(Vr[a]); Y.append((S2[lv] - S0[lv])[pool] / s_inj)
    X, Y = torch.cat(X), torch.cat(Y); G = X.T @ X; return torch.linalg.solve(G + 1e-2 * G.diagonal().mean() * torch.eye(D, device=DEV), X.T @ Y)
Ws = {bb: unit(arch.wdir(bb).to(DEV)) for bb in blocks}; Ts = {bb: fit_T(bb + 1, L, 100 + bb) for bb in blocks}; Zall = {bb: (Ws[bb] @ Ts[bb]) @ P for bb in blocks}; Tfin = fit_T(b + 1, NB - 1, 200)
torch.manual_seed(1); val = [(bb, int(i)) for bb in blocks for i in torch.randperm(DFF, device=DEV)[:40].tolist()]; Vv = torch.stack([Ws[bb][i] for bb, i in val]); foreign = torch.nonzero(typ & ~torch.isin(tn, keep))[:, 0]; a = torch.randint(0, len(val), (len(foreign),), device=DEV); zex = torch.zeros(len(val), 16, device=DEV)
for bb in blocks:
    sel = torch.tensor([j for j, (b_, _) in enumerate(val) if b_ == bb], device=DEV); m = torch.isin(a, sel); inj = torch.zeros(NT, D, device=DEV); inj[foreign[m]] = s_inj * Vv[a[m]]; S2 = run(inject=inj, inject_block=bb + 1)
    for j in sel.tolist(): zex[j] = ((S2[L] - S0[L])[foreign][a == j].mean(0) / s_inj) @ P
zop = torch.stack([Zall[bb][i] for bb, i in val]); val_cos = ((unit(zop) * unit(zex)).sum(1)).median().item(); val_id = ((unit(zop) @ unit(zex).T).argmax(1) == torch.arange(len(val), device=DEV)).float().mean().item()
Zcat = torch.cat([Zall[bb] for bb in blocks]); blk_of = torch.cat([torch.full((DFF,), bb, device=DEV) for bb in blocks]); def_prank = lambda X: (lambda ev: (ev.sum() ** 2 / (ev ** 2).sum()).item())(torch.linalg.eigvalsh((X - X.mean(0, keepdim=True)).T @ (X - X.mean(0, keepdim=True))).clamp_min(0))
Zu = unit(Zcat); torch.manual_seed(2); cent = Zu[torch.randperm(len(Zu), device=DEV)[:16]]
for it in range(20):
    assign = (Zu @ cent.T).argmax(1); cent = unit(torch.stack([Zu[assign == k].mean(0) if (assign == k).any() else cent[k] for k in range(16)]))
sizes = torch.bincount(assign, minlength=16); cross = torch.tensor([len(torch.unique(blk_of[assign == k])) for k in range(16)]); within_cos = torch.stack([(Zu[assign == k] @ cent[k]).mean() if (assign == k).any() else torch.tensor(0.0, device=DEV) for k in range(16)])
leds = {bb: dominant(model, arch, c, ids_seq, bb)[0] for bb in blocks}; class_weight = torch.zeros(16, device=DEV)
for bb in blocks: class_weight.index_add_(0, assign[blk_of == bb], leds[bb][typ].abs().sum(0))
class_weight = class_weight / class_weight.sum(); atlas = dict(atoms=int(len(Zcat)), prank_all=def_prank(Zcat), prank_block2=def_prank(Zall[b]), class_sizes=sizes.tolist(), classes_spanning_3plus_blocks=int((cross >= 3).sum()), within_class_cos=within_cos.mean().item(), class_ledger_weight=class_weight.tolist(), validation_cos=val_cos, validation_identification=val_id, validation_chance=1 / len(val))
a2 = assign[blk_of == b]; k_heavy = int((class_weight * torch.bincount(a2, minlength=16).float().clamp_min(1) / torch.bincount(a2, minlength=16).float().clamp_min(1)).argmax()); members = torch.nonzero(a2 == k_heavy)[:, 0]; n = min(len(members), 64); members = members[torch.randperm(len(members), device=DEV)[:n]]
def ablate_set(neur, rows=None):
    def pre(m, inp):
        x = inp[0].clone(); flat = x.reshape(-1, DFF)
        if rows is None: flat[:, neur] = 0
        else: flat[rows[:, None], neur[None, :]] = 0
        return (flat.reshape(x.shape),)
    h = arch.mlp_lin(b).register_forward_pre_hook(pre); r = run(positions=pool); h.remove(); return r
def class_test(neur):
    r = ablate_set(neur); dS = (S0[L] - r[L])[pool]; coef = leds[b][pool][:, neur]; w = coef.abs().sum(1); rows = w >= w.quantile(0.8); pred = coef[rows] @ Zall[b][neur]; act = dS[rows] @ P; coh = (pred.norm(dim=1) / (coef[rows].abs() @ Zall[b][neur].norm(dim=1)).clamp_min(1e-9)).median().item(); return dict(coherence=coh, cos_pred_actual=((unit(act) * unit(pred)).sum(1)).median().item(), actual_core_norm_per_weight=(act.norm(dim=1) / w[rows]).median().item())
other = torch.nonzero(a2 == int((class_weight.argsort(descending=True)[1]).item()))[:, 0]; other = other[torch.randperm(len(other), device=DEV)[:n]]; rnd = torch.randperm(DFF, device=DEV)[:n]; cls = dict(heavy_class=class_test(members), other_class=class_test(other), random_set=class_test(rnd), n=int(n))
Z2, Z4 = Zall[b], Zall[4] if 4 in Zall else Zall[blocks[-1]]; b_sub = 4 if 4 in Zall else blocks[-1]; sub = {"same_class": [], "random": []}; base_l = run(positions=idx)["lg"]
for k in range(min(K, 12)):
    i = int(keep[k]); rows = idx[lab_i == k]; zi = Z2[i]; cos_j = (unit(Z4) @ unit(zi)); j_same = int(cos_j.argmax()); j_rand = int(torch.randint(0, DFF, (1,)).item()); coef_i = leds[b][rows, i]
    r_abl = run(neuron=torch.full((NT,), i, device=DEV), positions=rows, ablate_positions=rows); A_ = r_abl["lg"] - base_l[torch.isin(idx, rows)] if False else None
    r_abl = run(neuron=torch.full((NT,), i, device=DEV), positions=rows, ablate_positions=rows); base_rows = run(positions=rows)["lg"]; A_ = r_abl["lg"] - base_rows
    for nm, j in (("same_class", j_same), ("random", j_rand)):
        zj = Z4[j]; alpha = coef_i * (zi @ zj) / (zj @ zj).clamp_min(1e-9); inj = torch.zeros(NT, D, device=DEV); inj[rows] = alpha[:, None] * Ws[b_sub][j][None]; r = run(neuron=torch.full((NT,), i, device=DEV), positions=rows, ablate_positions=rows, inject=inj, inject_block=b_sub + 1); R_ = r["lg"] - base_rows; sub[nm].append(dict(restoration=(1 - R_.norm(dim=1) / A_.norm(dim=1).clamp_min(1e-9)).median().item(), cos_residual=((unit(R_) * unit(A_)).sum(1)).median().item(), z_cos=cos_j[j].item()))
subs = {nm: dict(restoration=float(torch.tensor([x["restoration"] for x in v]).median()), cos_residual=float(torch.tensor([x["cos_residual"] for x in v]).median()), z_cos=float(torch.tensor([x["z_cos"] for x in v]).median())) for nm, v in sub.items()}
WU = arch.model.get_output_embeddings().weight.detach().float().to(DEV); att = {"atlas": [], "direct_logit": [], "jlens_no_quotient": [], "transport_readout_no_quotient": []}
for trial in range(8):
    S = keep[torch.randperm(K, device=DEV)[:6]]; r = ablate_set(S); dlS = base_l_pool = None
    base_pool = run(positions=pool)["lg"]; dlS = base_pool - r["lg"]; dlS = dlS - dlS.mean(1, keepdim=True); coef = leds[b][pool][:, S]; w = coef.abs().sum(1); rows = w >= w.quantile(0.9); act = dlS[rows]
    p_atlas = ((coef[rows] @ Z2[S]) @ Wz) @ B.T; p_direct = (coef[rows] @ Ws[b][S]) @ WU.T; p_direct = p_direct - p_direct.mean(1, keepdim=True); p_jlens = ((coef[rows] @ Ws[b][S]) @ Tfin) @ WU.T; p_jlens = p_jlens - p_jlens.mean(1, keepdim=True); p_tr = (((coef[rows] @ Ws[b][S]) @ Ts[b]) @ Wd) @ B.T
    for nm, pv in (("atlas", p_atlas), ("direct_logit", p_direct), ("jlens_no_quotient", p_jlens), ("transport_readout_no_quotient", p_tr)): att[nm].append(((unit(act) * unit(pv)).sum(1)).median().item())
attr = {nm: float(torch.tensor(v).median()) for nm, v in att.items()}
log(f"{tag} (K {K}, blocks {blocks}, {atlas['atoms']} atoms): atlas validation: operator vs exact coordinates cos {val_cos:.2f}, identification {val_id:.2f} (chance {1 / len(val):.3f}) | effective rank of the atlas {atlas['prank_all']:.1f} (block 2 alone {atlas['prank_block2']:.1f}); 16 classes, sizes {sizes.min().item()}-{sizes.max().item()}, {atlas['classes_spanning_3plus_blocks']} span 3+ blocks, within-class cos {atlas['within_class_cos']:.2f}; heaviest class carries {class_weight.max().item():.2f} of the ledger weight | (a) class ablation (n {n}): heaviest class coherence {cls['heavy_class']['coherence']:.2f}, predicted-vs-actual cos {cls['heavy_class']['cos_pred_actual']:.2f}, core norm per ledger weight {cls['heavy_class']['actual_core_norm_per_weight']:.3f}; other class {cls['other_class']['coherence']:.2f}/{cls['other_class']['cos_pred_actual']:.2f}/{cls['other_class']['actual_core_norm_per_weight']:.3f}; random set {cls['random_set']['coherence']:.2f}/{cls['random_set']['cos_pred_actual']:.2f}/{cls['random_set']['actual_core_norm_per_weight']:.3f} | (b) cross-block substitution (block {b_sub} atom for a block-2 candidate): same-class restoration {subs['same_class']['restoration']:.2f} (residual cos {subs['same_class']['cos_residual']:.2f}, z cos {subs['same_class']['z_cos']:.2f}) vs random atom {subs['random']['restoration']:.2f} ({subs['random']['cos_residual']:.2f}, {subs['random']['z_cos']:.2f}) | (c) attribution of joint ablations: atlas {attr['atlas']:.2f}, direct logit attribution {attr['direct_logit']:.2f}, averaged-Jacobian lens without quotient {attr['jlens_no_quotient']:.2f}, transport + read-out without quotient {attr['transport_readout_no_quotient']:.2f}")
record(f"e316_atlas_{tag}", dict(model=tag, b=b, L=L, K=K, blocks=blocks, atlas=atlas, class_ablation=cls, substitution=subs, attribution=attr), f"validation cos {val_cos:.2f} id {val_id:.2f} | prank {atlas['prank_all']:.1f} classes 3+blocks {atlas['classes_spanning_3plus_blocks']} within {atlas['within_class_cos']:.2f} | class ablation coherence heavy/other/random {cls['heavy_class']['coherence']:.2f}/{cls['other_class']['coherence']:.2f}/{cls['random_set']['coherence']:.2f} cos {cls['heavy_class']['cos_pred_actual']:.2f}/{cls['other_class']['cos_pred_actual']:.2f}/{cls['random_set']['cos_pred_actual']:.2f} | substitution same {subs['same_class']['restoration']:.2f} random {subs['random']['restoration']:.2f} | attribution atlas {attr['atlas']:.2f} direct {attr['direct_logit']:.2f} jlens {attr['jlens_no_quotient']:.2f} transport {attr['transport_readout_no_quotient']:.2f}")
