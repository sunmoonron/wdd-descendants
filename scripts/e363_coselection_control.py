"""e363: does co-selection see a known circuit? Heads only. Circuit membership is defined without co-selection: by
attention patterns on repeated random tokens (induction = prefix-matching score >= 0.2 and above its previous-token
score; previous-token = previous-token score >= 0.3) and by single-head ablation effect on the repeated copy (top 8).
Per-batch selection signals s_h(B) = <dL_B/dW_O[h], W_O[h]> on 192 natural-text batches (4 x 256 wikitext-train
tokens) and 64 induction batches, plus each natural batch's induction content (the fraction of positions whose token
occurred earlier and whose next token repeats the earlier continuation). For each circuit set: mean pairwise signal
correlation (raw, global mode removed, batch loss partialled out) against random sets and against layer-matched random
sets, and the correlation of the set's mean standardised signal with the batch's induction content. Also: the
adjusted Rand index between spectral clusters of the residual correlation matrix and the attention labels."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pc_common import *
tag = sys.argv[1]; rev = sys.argv[2] if len(sys.argv) > 2 else None; c = Cache(tag); model, tok, fam = load_eager(c.name, revision=rev); rtag = tag + (f"_{rev}" if rev else ""); arch = Arch(model, fam); NB, NH = arch.NB, arch.NH; rng = random.Random(0); torch.manual_seed(0)
heads = [(l, h) for l in range(NB) for h in range(NH)]; nh = len(heads)
ind, off, half = induction_batch(tok, c, n=8, half=128, seed=0); pref, prev = attention_scores(model, ind, off, half, NB, NH)
MA, MM, lg = capture_means(model, arch, ind); base = induction_loss(lg, ind, off, half); del lg
E1 = torch.tensor([(induction_loss(ablate(model, arch, ind, heads=[x], MA=MA, MM=MM), ind, off, half) - base).mean().item() for x in heads])
pf, pv = pref.flatten().cpu(), prev.flatten().cpu()
IND = [i for i in pf.argsort(descending=True).tolist() if pf[i] >= 0.2 and pf[i] >= pv[i]][:8]; PRV = [i for i in pv.argsort(descending=True).tolist() if pv[i] >= 0.3 and i not in IND][:8]; ABL = E1.argsort(descending=True)[:8].tolist()
for p in model.parameters(): p.requires_grad_(False)
Wo = [arch.attn_lin(l).weight for l in range(NB)]
for w in Wo: w.requires_grad_(True)
def sig(loss):
    for w in Wo: w.grad = None
    loss.backward(); out = []
    for l in range(NB):
        gw = Wo[l].grad * Wo[l]; out.append((gw.sum(1) if fam == "gpt2" else gw.sum(0)).view(NH, -1).sum(1))
    return torch.cat(out).detach().float().cpu()
def ind_content(x):
    n = 0; tot = 0
    for row in x.tolist():
        last = {}
        for t in range(len(row) - 1):
            a = row[t]
            if a in last and row[last[a] + 1] == row[t + 1]: n += 1
            last[a] = t; tot += 1
    return n / max(tot, 1)
tr = corpus_ids(tok, "wikitext", "train"); seqs = tr[: (len(tr) // 256) * 256].view(-1, 256); g = torch.Generator().manual_seed(11); perm = torch.randperm(len(seqs), generator=g)[:768]; S, TY, LO, RC = [], [], [], []
with torch.enable_grad():
    for i in range(192):
        x = seqs[perm[4 * i: 4 * i + 4]]; ids = x.to(DEV); loss = token_loss(model(ids).logits.float(), ids).mean(); S.append(sig(loss)); TY.append(0); LO.append(loss.item()); RC.append(ind_content(x))
    for i in range(64):
        ids, o_, h_ = induction_batch(tok, c, n=6, half=96, seed=500 + i); loss = induction_loss(model(ids).logits.float(), ids, o_, h_).mean(); S.append(sig(loss)); TY.append(1); LO.append(loss.item()); RC.append(float("nan"))
S, TY, LO, RC = torch.stack(S), torch.tensor(TY), torch.tensor(LO), torch.tensor(RC)
def corrm(X):
    Z = (X - X.mean(0)) / X.std(0).clamp_min(1e-12); return (Z.T @ Z) / (len(Z) - 1), Z
def resid(X, covar=None):
    C, Z = corrm(X)
    if covar is None:
        U, s, Vh = torch.linalg.svd(Z, full_matrices=False); Zr = Z - (U[:, :1] * s[:1]) @ Vh[:1]
    else:
        q = (covar - covar.mean()) / covar.std().clamp_min(1e-12); q = q[:, None]; Zr = Z - q @ (q.T @ Z) / (q.T @ q)
    Cr = (Zr.T @ Zr) / (len(Zr) - 1); d = Cr.diagonal().clamp_min(1e-12).sqrt(); return Cr / torch.outer(d, d), Z
def gmean(C, g):
    g = list(g); n = len(g)
    if n < 2: return float("nan")
    sub = C[g][:, g]; return ((sub.sum() - sub.diagonal().sum()) / (n * n - n)).item()
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
res = dict(model=tag, rev=rev, n_heads=nh, induction_heads=[list(heads[i]) for i in IND], previous_heads=[list(heads[i]) for i in PRV], ablation_top8=[list(heads[i]) for i in ABL], overlap_ablation_induction=len(set(ABL) & set(IND)), mean_induction_content=RC[TY == 0].mean().item())
for dn, rows in (("natural", TY == 0), ("mixed", torch.ones_like(TY, dtype=torch.bool))):
    X = S[rows]; C, Z = corrm(X); Cr, _ = resid(X); Cl, _ = resid(X, LO[rows]); R = {}
    for sn, g in (("induction", IND), ("previous", PRV), ("ablation_top8", ABL), ("induction_plus_previous", IND + PRV)):
        if len(g) < 2: R[sn] = None; continue
        d = {}
        for mn, M in (("raw", C), ("global_removed", Cr), ("loss_partialled", Cl)):
            v = gmean(M, g); nr = [gmean(M, s) for s in rand_sets(g)]; nm = [gmean(M, s) for s in rand_sets(g, matched=True)]
            d[mn] = dict(value=v, random_mean=sum(nr) / len(nr), random_z=(v - sum(nr) / len(nr)) / max(torch.tensor(nr).std().item(), 1e-9), layer_matched_mean=sum(nm) / len(nm), layer_matched_z=(v - sum(nm) / len(nm)) / max(torch.tensor(nm).std().item(), 1e-9))
        if dn == "natural":
            ms = Z[:, g].mean(1); rc = RC[rows]; d["corr_with_induction_content"] = float(torch.corrcoef(torch.stack([ms, rc]))[0, 1]); nulls = [float(torch.corrcoef(torch.stack([Z[:, s].mean(1), rc]))[0, 1]) for s in rand_sets(g, 100)]; d["content_corr_random_mean"] = sum(nulls) / len(nulls); d["content_corr_random_sd"] = torch.tensor(nulls).std().item()
        R[sn] = d
    lab = ["induction" if i in IND else ("previous" if i in PRV else "other") for i in range(nh)]; members = IND + PRV + rng.sample([j for j in range(nh) if j not in IND + PRV], max(len(IND + PRV), 4))
    if len(IND + PRV) >= 3:
        cl, k, _ = spectral(Cr[members][:, members].clamp_min(0), kmax=4); R["ari_clusters_vs_attention_labels"] = ari(cl, [lab[i] for i in members]); R["k"] = k
    res[dn] = R
def fmt(dn):
    R = res[dn]; parts = []
    for sn in ("induction", "previous", "ablation_top8"):
        d = R.get(sn)
        if not d: parts.append(f"{sn}: none"); continue
        s = f"{sn}: raw {d['raw']['value']:+.2f} (z {d['raw']['random_z']:+.1f}, layer-matched z {d['raw']['layer_matched_z']:+.1f}), global-removed {d['global_removed']['value']:+.2f} (z {d['global_removed']['random_z']:+.1f} / {d['global_removed']['layer_matched_z']:+.1f})"
        if "corr_with_induction_content" in d: s += f", corr with batch induction content {d['corr_with_induction_content']:+.2f} (random {d['content_corr_random_mean']:+.2f} +- {d['content_corr_random_sd']:.2f})"
        parts.append(s)
    return "; ".join(parts) + (f"; ARI clusters vs attention labels {R['ari_clusters_vs_attention_labels']:.2f}" if "ari_clusters_vs_attention_labels" in R else "")
log(f"{rtag}: {len(IND)} induction heads, {len(PRV)} previous-token heads by attention; ablation top-8 overlaps induction set in {res['overlap_ablation_induction']}; mean natural induction content {res['mean_induction_content']:.3f} | NATURAL: " + fmt("natural") + " | MIXED: " + fmt("mixed"))
os.makedirs(os.path.join(RESULTS, "e363"), exist_ok=True); torch.save(dict(S=S, TY=TY, LO=LO, RC=RC, pref=pf, prev=pv, E1=E1, heads=heads), os.path.join(RESULTS, "e363", f"{rtag}.pt"))
record(f"e363_coselctrl_{rtag}", res, f"IND {len(IND)} PRV {len(PRV)} | natural: " + ", ".join(f"{sn} gr z {res['natural'][sn]['global_removed']['random_z']:+.1f}/{res['natural'][sn]['global_removed']['layer_matched_z']:+.1f} content {res['natural'][sn]['corr_with_induction_content']:+.2f}" for sn in ("induction", "previous", "ablation_top8") if res['natural'].get(sn) and "corr_with_induction_content" in res['natural'][sn]))
