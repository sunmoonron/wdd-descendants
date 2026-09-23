"""e364: the e359 co-selection measurement at one training checkpoint (arguments: model, revision), for the persistence test across training (e364b). For every
attention head (its output-projection slice) and every MLP neuron (its down-projection row), the per-batch selection
signal s_u(B) = <dL_B/dtheta_u, theta_u>, the derivative of the batch loss with respect to scaling the unit's write
(for a neuron, the WDD coefficient times the loss gradient along its atom, summed over tokens). 160 natural-text
batches, 64 induction batches (loss on the repeated copy) and 64 IOI batches (loss on the IO token). Units: all heads,
32 random neurons per block and the 64 most-selected neurons. Reported: significant co-selection modes (eigenvalues of
the unit correlation matrix above the Marchenko-Pastur edge and above a shuffled null), the share of the global mode
(batch difficulty), community structure after removing it (spectral clustering and weighted modularity, against the
shuffled null), the layer confound, and the positive control: whether the heads most selected by induction batches
co-select as a group, in mixed batches and in natural text alone."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pc_common import *
tag, rev = sys.argv[1], sys.argv[2]; c = Cache(tag); model, tok, fam = load_model(c.name, revision=rev); arch = Arch(model, fam); NB, NH = arch.NB, arch.NH; rng = random.Random(0); torch.manual_seed(0)
for p in model.parameters(): p.requires_grad_(False)
Wo = [arch.attn_lin(l).weight for l in range(NB)]; Wd = [arch.mlp_lin(l).weight for l in range(NB)]
for w in Wo + Wd: w.requires_grad_(True)
def signals(loss):
    for w in Wo + Wd: w.grad = None
    loss.backward(); sh, sn = [], []
    for l in range(NB):
        gw = (Wo[l].grad * Wo[l]); gh = (gw.sum(1) if fam == "gpt2" else gw.sum(0)); sh.append(gh.view(NH, -1).sum(1))
        gw = (Wd[l].grad * Wd[l]); sn.append(gw.sum(1) if fam == "gpt2" else gw.sum(0))
    return torch.cat(sh).detach().float().cpu(), torch.cat(sn).detach().float().cpu()
tr = corpus_ids(tok, "wikitext", "train"); seqs = tr[: (len(tr) // 256) * 256].view(-1, 256); perm = torch.randperm(len(seqs))[:640]; SH, SN, TY, LO = [], [], [], []
with torch.enable_grad():
    for i in range(160):
        ids = seqs[perm[4 * i: 4 * i + 4]].to(DEV); loss = token_loss(model(ids).logits.float(), ids).mean(); a, b = signals(loss); SH.append(a); SN.append(b); TY.append(0); LO.append(loss.item())
    for i in range(64):
        ids, off, half = induction_batch(tok, c, n=6, half=96, seed=100 + i); loss = induction_loss(model(ids).logits.float(), ids, off, half).mean(); a, b = signals(loss); SH.append(a); SN.append(b); TY.append(1); LO.append(loss.item())
    for i in range(8):
        gs, _ = ioi_groups(tok, n_per=12, seed=200 + i)
        for ids, io, s in gs:
            lg = model(ids).logits.float()[:, -1]; loss = -torch.log_softmax(lg, -1).gather(1, io[:, None]).mean(); a, b = signals(loss); SH.append(a); SN.append(b); TY.append(2); LO.append(loss.item())
SH, SN, TY, LO = torch.stack(SH), torch.stack(SN), torch.tensor(TY), torch.tensor(LO); DFF = SN.shape[1] // NB; heads = [(l, h) for l in range(NB) for h in range(NH)]
nat = TY == 0; most = SN[nat].abs().mean(0).argsort(descending=True)[:64]; randn = torch.tensor([l * DFF + i for l in range(NB) for i in rng.sample(range(DFF), 32)]); nidx = torch.unique(torch.cat([most, randn])); X_all = torch.cat([SH, SN[:, nidx]], 1); nh = SH.shape[1]; unit_layer = torch.tensor([l for l, h in heads] + [int(i) // DFF for i in nidx])
def corr(X):
    Z = (X - X.mean(0)) / X.std(0).clamp_min(1e-12); return (Z.T @ Z) / (len(Z) - 1), Z
def modes(X, shuffle_reps=3):
    C, Z = corr(X); ev = torch.linalg.eigvalsh(C).flip(0); N, T = X.shape[1], X.shape[0]; mp = (1 + (N / T) ** 0.5) ** 2; nullmax = []
    for r in range(shuffle_reps):
        Xs = torch.stack([X[torch.randperm(T), j] for j in range(N)], 1); nullmax.append(torch.linalg.eigvalsh(corr(Xs)[0]).max().item())
    return dict(N=N, T=T, top_eig=ev[:5].tolist(), global_share=(ev[0] / ev.sum()).item(), above_mp=int((ev > mp).sum()), mp_edge=mp, above_null=int((ev > max(nullmax)).sum()), null_max=max(nullmax))
def residual_corr(X):
    C, Z = corr(X); U, S, Vh = torch.linalg.svd(Z, full_matrices=False); Zr = Z - (U[:, :1] * S[:1]) @ Vh[:1]; Cr = (Zr.T @ Zr) / (len(Zr) - 1); d = Cr.diagonal().clamp_min(1e-12).sqrt(); return Cr / torch.outer(d, d)
def modularity(W, lab):
    W = W.clamp_min(0).clone(); W.fill_diagonal_(0); m2 = W.sum(); k = W.sum(1); lab = torch.tensor(lab); same = (lab[:, None] == lab[None, :]).float(); return ((W - torch.outer(k, k) / m2.clamp_min(1e-12)) * same).sum().item() / m2.clamp_min(1e-12).item()
def community(X):
    Cr = residual_corr(X); lab, k, ev = spectral(Cr.clamp_min(0), kmax=8); Q = modularity(Cr, lab); Xs = torch.stack([X[torch.randperm(X.shape[0]), j] for j in range(X.shape[1])], 1); Crs = residual_corr(Xs); labs, ks, _ = spectral(Crs.clamp_min(0), kmax=8); Qs = modularity(Crs, labs)
    iu = torch.triu_indices(Cr.shape[0], Cr.shape[0], 1); v = Cr[iu[0], iu[1]]; topk = v.abs().argsort(descending=True)[:100]; ul = unit_layer[:Cr.shape[0]]; same_layer = (ul[iu[0][topk]] == ul[iu[1][topk]]).float().mean().item(); chance = (ul[iu[0]] == ul[iu[1]]).float().mean().item()
    return dict(k=k, modularity=Q, modularity_shuffled=Qs, top100_same_layer=same_layer, same_layer_chance=chance, community_sizes=[lab.count(j) for j in range(k)]), Cr
zi = (SH[TY == 1].mean(0) - SH[nat].mean(0)) / SH[nat].std(0).clamp_min(1e-12); indheads = zi.argsort(descending=True)[:8]; randheads = torch.tensor(rng.sample([i for i in range(nh) if i not in set(indheads.tolist())], 8))
def group_corr(Cr, g): sub = Cr[g][:, g]; n = len(g); return ((sub.sum() - sub.diagonal().sum()) / (n * n - n)).item()
res = dict(model=tag, n_batches=dict(natural=int(nat.sum()), induction=int((TY == 1).sum()), ioi=int((TY == 2).sum())), mean_loss=dict(natural=LO[nat].mean().item(), induction=LO[TY == 1].mean().item(), ioi=LO[TY == 2].mean().item()), induction_selected_heads=[list(heads[i]) for i in indheads.tolist()])
for nm, rows in (("natural_only", nat), ("mixed", torch.ones_like(nat))):
    X = X_all[rows]; md = modes(X); cm, Cr = community(X); cmh, Crh = community(X[:, :nh]); res[nm] = dict(all_units=md, community_all=cm, community_heads=cmh, heads_modes=modes(X[:, :nh]), neuron_modes=modes(X[:, nh:]), induction_heads_residual_corr=group_corr(Crh, indheads), random_heads_residual_corr=group_corr(Crh, randheads), loss_corr_with_global_mode=float(torch.corrcoef(torch.stack([LO[rows], torch.linalg.svd(corr(X)[1], full_matrices=False)[0][:, 0]]))[0, 1]))
log(f"{tag} {rev} (units: {nh} heads + {len(nidx)} neurons): mean loss natural {res['mean_loss']['natural']:.2f}, induction {res['mean_loss']['induction']:.2f}, IOI {res['mean_loss']['ioi']:.2f} | " + " || ".join(f"{nm}: global mode share {v['all_units']['global_share']:.2f} (|corr with batch loss| {abs(v['loss_corr_with_global_mode']):.2f}), modes above MP edge {v['all_units']['above_mp']} / above shuffled null {v['all_units']['above_null']} (heads {v['heads_modes']['above_null']}, neurons {v['neuron_modes']['above_null']}); after removing the global mode: communities k {v['community_all']['k']}, modularity {v['community_all']['modularity']:.3f} vs shuffled {v['community_all']['modularity_shuffled']:.3f}; top-100 correlations same-layer {v['community_all']['top100_same_layer']:.2f} (chance {v['community_all']['same_layer_chance']:.2f}); induction-selected heads residual corr {v['induction_heads_residual_corr']:+.2f} vs random heads {v['random_heads_residual_corr']:+.2f}" for nm, v in (("natural", res["natural_only"]), ("mixed", res["mixed"]))))
os.makedirs(os.path.join(RESULTS, "e364"), exist_ok=True); torch.save(dict(SH=SH, SN_sel=SN[:, nidx], nidx=nidx, TY=TY, LO=LO, heads=heads, NB=NB, NH=NH, DFF=DFF), os.path.join(RESULTS, "e364", f"{tag}_{rev}.pt"))
res["rev"] = rev; record(f"e364_cosel_{tag}_{rev}", res, " || ".join(f"{nm}: global {v['all_units']['global_share']:.2f} modes>null {v['all_units']['above_null']} Q {v['community_all']['modularity']:.3f}/{v['community_all']['modularity_shuffled']:.3f} same-layer {v['community_all']['top100_same_layer']:.2f}/{v['community_all']['same_layer_chance']:.2f} ind-heads corr {v['induction_heads_residual_corr']:+.2f} vs {v['random_heads_residual_corr']:+.2f}" for nm, v in (("natural", res["natural_only"]), ("mixed", res["mixed"]))))
