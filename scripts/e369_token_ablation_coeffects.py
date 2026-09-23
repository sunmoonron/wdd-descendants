"""e369: token-level co-effects from ablation instead of gradients. e368's first-order selection signal is blind to a
unit whose contribution is saturated (removing it hurts, scaling it up does not help). Here every head is mean-ablated
once on 8 x 512 natural tokens and the per-token loss change is kept, giving a token x head effect matrix analysed
exactly as e368: circuit-set correlations over tokens against random and layer-matched sets (raw and global mode
removed) and over 64-token windows, each head's effect contrast at induction-applicable positions and how well it
finds the attention-defined induction heads (AUC), principal components against applicable tokens, and clustering
against the attention labels. Positive effect = the head helps at that token."""

import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pc_common import *
tag = sys.argv[1]; rev = sys.argv[2] if len(sys.argv) > 2 else None; rtag = tag + (f"_{rev}" if rev else ""); c = Cache(tag); model, tok, fam = load_eager(c.name, revision=rev); arch = Arch(model, fam); NB, NH = arch.NB, arch.NH; rng = random.Random(0); torch.manual_seed(0)
heads = [(l, h) for l in range(NB) for h in range(NH)]; nh = len(heads)
ind, off, half = induction_batch(tok, c, n=8, half=128, seed=0); pref, prev = attention_scores(model, ind, off, half, NB, NH); pf, pv = pref.flatten().cpu(), prev.flatten().cpu()
IND = [i for i in pf.argsort(descending=True).tolist() if pf[i] >= 0.2 and pf[i] >= pv[i]][:8]; PRV = [i for i in pv.argsort(descending=True).tolist() if pv[i] >= 0.3 and i not in IND][:8]
nat = c.s["eval_ids"][:8].to(DEV); B, T = nat.shape; MAn, MMn, lgn = capture_means(model, arch, nat); base = token_loss(lgn, nat); del lgn
A = torch.stack([(token_loss(ablate(model, arch, nat, heads=[x], MA=MAn, MM=MMn), nat) - base).cpu() for x in heads], 1)  # B*(T-1) x nh
C = torch.cat([A.view(B, T - 1, nh), torch.zeros(B, 1, nh)], 1)  # B x T x nh, last position (no target) zero
nat = nat.cpu()
M = torch.zeros(B, T, dtype=torch.bool)
for b, row in enumerate(nat.tolist()):
    last = {}
    for t in range(T - 1):
        a = row[t]
        if a in last and row[last[a] + 1] == row[t + 1]: M[b, t] = True
        last[a] = t
keep = torch.zeros(B, T, dtype=torch.bool); keep[:, 1:T - 1] = True; X = C[keep]; Mk = M[keep]
def corrm(X):
    Z = (X - X.mean(0)) / X.std(0).clamp_min(1e-12); return (Z.T @ Z) / (len(Z) - 1), Z
def resid(Z):
    U, s, Vh = torch.linalg.svd(Z, full_matrices=False); Zr = Z - (U[:, :1] * s[:1]) @ Vh[:1]; Cr = (Zr.T @ Zr) / (len(Zr) - 1); d = Cr.diagonal().clamp_min(1e-12).sqrt(); return Cr / torch.outer(d, d)
def gmean(Cm, g):
    g = list(g); n = len(g)
    if n < 2: return float("nan")
    sub = Cm[g][:, g]; return ((sub.sum() - sub.diagonal().sum()) / (n * n - n)).item()
layer_of = [l for l, h in heads]; by_layer = {}
[by_layer.setdefault(l, []).append(i) for i, (l, h) in enumerate(heads)]
def rand_sets(g, n=200, matched=False):
    out = []
    for _ in range(n):
        if matched:
            s = []
            for i in g:
                pool = [j for j in by_layer[layer_of[i]] if j not in g and j not in s] or [j for j in range(nh) if j not in g and j not in s]; s.append(rng.choice(pool))
            out.append(s)
        else: out.append(rng.sample([j for j in range(nh) if j not in g], len(g)))
    return out
def set_stats(Cm, g):
    v = gmean(Cm, g); nr = torch.tensor([gmean(Cm, s) for s in rand_sets(g)]); nm = torch.tensor([gmean(Cm, s) for s in rand_sets(g, matched=True)])
    return dict(value=v, random_mean=nr.mean().item(), random_z=(v - nr.mean().item()) / max(nr.std().item(), 1e-9), layer_matched_mean=nm.mean().item(), layer_matched_z=(v - nm.mean().item()) / max(nm.std().item(), 1e-9))
def auc(score, lab):
    lab = lab.bool(); pos, neg = score[lab], score[~lab]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    r = torch.cat([pos, neg]).argsort().argsort().float() + 1; return ((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))).item()
Ct, Z = corrm(X); Cr = resid(Z)
W = C[:, 1:T - 1 - ((T - 2) % 64)].reshape(B, -1, 64, nh).sum(2).reshape(-1, nh); Cw, _ = corrm(W)
res = dict(model=tag, rev=rev, n_tokens=int(keep.sum()), share_applicable=Mk.float().mean().item(), induction_heads=[list(heads[i]) for i in IND], previous_heads=[list(heads[i]) for i in PRV], n_windows=W.shape[0])
for sn, g in (("induction", IND), ("previous", PRV), ("induction_plus_previous", IND + PRV)):
    if len(g) < 2: res[sn] = None; continue
    res[sn] = dict(token_raw=set_stats(Ct, g), token_global_removed=set_stats(Cr, g), window64_raw=set_stats(Cw, g))
Zs = (X - X.mean(0)) / X.std(0).clamp_min(1e-12); dprime = -(Zs[Mk].mean(0) - Zs[~Mk].mean(0)); res["applicable_contrast_induction_mean"] = dprime[IND].mean().item() if IND else None; res["applicable_contrast_other_mean"] = dprime[[i for i in range(nh) if i not in IND]].mean().item()
labI = torch.zeros(nh); labI[IND] = 1; res["auc_contrast_finds_induction_heads"] = auc(-dprime, labI) if IND else None
Uu, Ss, Vh = torch.linalg.svd(Zs - Zs.mean(0), full_matrices=False); pcs = []
for k in range(10):
    sc = Uu[:, k] * Ss[k]; a = auc(sc, Mk); pcs.append(dict(var_share=(Ss[k] ** 2 / (Ss ** 2).sum()).item(), auc_applicable=max(a, 1 - a), induction_loading_share=(Vh[k, IND] ** 2).sum().item() if IND else None))
res["pcs"] = pcs; res["best_pc_auc"] = max(p["auc_applicable"] for p in pcs); res["best_pc"] = max(range(10), key=lambda k: pcs[k]["auc_applicable"])
lab = ["induction" if i in IND else ("previous" if i in PRV else "other") for i in range(nh)]; members = IND + PRV + rng.sample([j for j in range(nh) if j not in IND + PRV], max(len(IND + PRV), 4))
if len(IND + PRV) >= 3:
    for nm_, Cm in (("token", Cr), ("window64", Cw)):
        cl, k, _ = spectral(Cm[members][:, members].clamp_min(0), kmax=4); res[f"ari_{nm_}_clusters_vs_attention_labels"] = ari(cl, [lab[i] for i in members])
def f(d): return f"{d['value']:+.3f} (z {d['random_z']:+.1f}, layer-matched z {d['layer_matched_z']:+.1f})"
log(f"{rtag}: {len(IND)} induction, {len(PRV)} previous-token heads; {res['n_tokens']} tokens, {100 * res['share_applicable']:.1f}% induction-applicable | " + " | ".join(f"{sn.upper()}: token corr {f(res[sn]['token_raw'])}, global removed {f(res[sn]['token_global_removed'])}, 64-token windows {f(res[sn]['window64_raw'])}" for sn in ("induction", "previous") if res.get(sn)) + f" | applicable-position contrast (standardised) induction heads {res['applicable_contrast_induction_mean'] if res['applicable_contrast_induction_mean'] is None else round(res['applicable_contrast_induction_mean'], 3)} vs others {res['applicable_contrast_other_mean']:+.3f}, AUC finding induction heads {res['auc_contrast_finds_induction_heads'] if res['auc_contrast_finds_induction_heads'] is None else round(res['auc_contrast_finds_induction_heads'], 2)}; best PC for applicable tokens PC{res['best_pc']} AUC {res['best_pc_auc']:.2f} (induction-loading share {pcs[res['best_pc']]['induction_loading_share']}); ARI token clusters {res.get('ari_token_clusters_vs_attention_labels', float('nan')):.2f}, window clusters {res.get('ari_window64_clusters_vs_attention_labels', float('nan')):.2f}")
os.makedirs(os.path.join(RESULTS, "e369"), exist_ok=True); torch.save(dict(C=C.half(), M=M, pref=pf, prev=pv, heads=heads), os.path.join(RESULTS, "e369", f"{rtag}.pt"))
record(f"e369_tokabl_{rtag}", res, f"IND {len(IND)} | " + " ".join(f"{sn} token z {res[sn]['token_global_removed']['random_z']:+.1f}/{res[sn]['token_global_removed']['layer_matched_z']:+.1f} window z {res[sn]['window64_raw']['random_z']:+.1f}" for sn in ("induction", "previous") if res.get(sn)) + f" | AUC heads {res['auc_contrast_finds_induction_heads']} best-PC AUC {res['best_pc_auc']:.2f}")
