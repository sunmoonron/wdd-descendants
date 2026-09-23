"""e376: can write geometry predict how two heads interact? Bushnaq's redundancy-versus-superposition question and the
original WDD's misattribution failure mode, joined. For the top-12 heads by zero-ablation effect and 12 random heads
(induction data and natural text): the signed pairwise interaction S_ij (relative; positive = super-additive, the
backup signature; negative = sub-additive, the series signature), against (1) write near-twinness: the mean cosine of
the two heads' actual writes at the same positions, the overlap of their static write subspaces, and WDD
confusability (the share of OMP coefficient mass that lands on head j when head i's write alone is read over all
heads' write atoms); (2) composition: the share of head j's query, key or value vector contributed by head i's write
(data) and the static composition norm. Prediction: twins go with super-additivity, composition with sub-additivity."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p3_common import *
import torch.nn.functional as F
tag = sys.argv[1]; c = Cache(tag); model, tok, fam = load_eager(c.name); arch = Arch(model, fam); NB, NH, HD, D = arch.NB, arch.NH, arch.HD, arch.D; rng = random.Random(0); torch.manual_seed(0)
for p in model.parameters(): p.requires_grad_(False)
heads = [(l, h) for l in range(NB) for h in range(NH)]; nh = len(heads); cap = Capture(model, arch)
ind, off, half = induction_batch(tok, c, n=8, half=128, seed=0); nat = c.s["eval_ids"][:4, :256].to(DEV)
data = {"induction": (ind, lambda lg, ids: induction_loss(lg, ids, off, half).mean()), "natural": (nat, lambda lg, ids: token_loss(lg, ids).mean())}
Aall, lab = c.dictionary(NB - 1, types=(T_ATT,)); atom_head = (lab["block"] * NH + lab["index"]).to(DEV)
Ubasis = {x: torch.linalg.svd(arch.wo(x[0])[x[1] * HD:(x[1] + 1) * HD].to(DEV), full_matrices=False).Vh for x in heads}
def run(ids, lossf, a):
    hs = scale_hooks(arch, a)
    try:
        with torch.no_grad(): return lossf(model(ids).logits.float(), ids).item()
    finally: [h.remove() for h in hs]
def spear(x, y):
    x, y = torch.as_tensor(x).float(), torch.as_tensor(y).float()
    if len(x) < 3: return float("nan")
    return torch.corrcoef(torch.stack([x.argsort().argsort().float(), y.argsort().argsort().float()]))[0, 1].item()
def partial_spear(x, y, z):
    rx, ry = torch.as_tensor(x).float().argsort().argsort().float(), torch.as_tensor(y).float().argsort().argsort().float(); Zm = torch.stack([torch.ones(len(rx))] + [torch.as_tensor(v).float() for v in z], 1)
    res = lambda v: v - Zm @ torch.linalg.lstsq(Zm, v[:, None]).solution[:, 0]; return torch.corrcoef(torch.stack([res(rx), res(ry)]))[0, 1].item()
def auc(score, pos):
    score = torch.as_tensor(score).float(); pos = torch.as_tensor(pos).bool(); a, b = score[pos], score[~pos]
    if len(a) == 0 or len(b) == 0: return float("nan")
    r = torch.cat([a, b]).argsort().argsort().float() + 1; return ((r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))).item()
res = dict(model=tag)
for dn, (ids, lossf) in data.items():
    one = torch.ones(nh, device=DEV); L0 = run(ids, lossf, one); E1 = torch.zeros(nh)
    for i in range(nh):
        a = one.clone(); a[i] = 0; E1[i] = run(ids, lossf, a) - L0
    order = E1.argsort(descending=True).tolist(); U = order[:12] + rng.sample(order[12:], 12); n = len(U)
    X, Z, M, out = cap(ids); P = ids.shape[0] * ids.shape[1]
    O = {}
    for i in U:
        l, h = heads[i]; O[i] = (Z[l].reshape(P, -1)[:, h * HD:(h + 1) * HD] @ arch.wo(l)[h * HD:(h + 1) * HD].to(DEV))
    RD = {}
    for j in U:
        l, h = heads[j]
        if l == 0: continue
        Xl = X[l].reshape(P, -1); RD[j] = {w: reader_dirs(arch, l, h, w, Xl)[0] for w in ("Q", "K", "V")}
    conf = {}
    for i in U:
        o = O[i]; keep = o.norm(dim=-1) > 1e-6; o = o[keep][:512]; sel, cof, _ = omp(o, Aall, 16, batch=512, record_err=False); mass = cof.abs(); tot = mass.sum(-1, keepdim=True).clamp_min(1e-12); ah = atom_head[sel]
        conf[i] = {j: ((mass * (ah == j)).sum(-1, keepdim=True) / tot).mean().item() for j in U}
    rows = []
    for x in range(n):
        for y in range(x + 1, n):
            i, j = U[x], U[y]
            a = one.clone(); a[i] = 0; a[j] = 0; Eij = run(ids, lossf, a) - L0; S = (Eij - E1[i].item() - E1[j].item()) / max(0.5 * (abs(E1[i].item()) + abs(E1[j].item())), 1e-9)
            oi, oj = O[i], O[j]; m_ = (oi.norm(dim=-1) > 1e-6) & (oj.norm(dim=-1) > 1e-6); wcos = F.cosine_similarity(oi[m_], oj[m_], dim=-1).mean().item() if m_.any() else 0.0
            sub = (Ubasis[heads[i]] @ Ubasis[heads[j]].T).pow(2).sum().item() / HD
            lo, hi = (i, j) if heads[i][0] < heads[j][0] else (j, i); same = heads[i][0] == heads[j][0]
            comp_data = max((O[lo] * RD[hi][w]).sum(-1).abs().mean().item() for w in ("Q", "K", "V")) if (not same and hi in RD) else 0.0
            comp_static = max((torch.linalg.matrix_norm(arch.wo(heads[lo][0])[heads[lo][1] * HD:(heads[lo][1] + 1) * HD].to(DEV) @ head_in_map(arch, heads[hi][0], heads[hi][1], w).to(DEV)) / (arch.wo(heads[lo][0])[heads[lo][1] * HD:(heads[lo][1] + 1) * HD].norm() * head_in_map(arch, heads[hi][0], heads[hi][1], w).norm()).clamp_min(1e-12)).item() for w in ("Q", "K", "V")) if not same else 0.0
            rows.append(dict(i=list(heads[i]), j=list(heads[j]), top_top=x < 12 and y < 12, S=S, write_cos=wcos, subspace_overlap=sub, wdd_confusability=0.5 * (conf[i][j] + conf[j][i]), comp_data=comp_data, comp_static=comp_static, same_layer=same, dlayer=abs(heads[i][0] - heads[j][0])))
    S = [r["S"] for r in rows]; sup = [s > 0.2 for s in S]; subadd = [s < -0.2 for s in S]; cross = [not r["same_layer"] for r in rows]
    col = lambda k, rr=rows: [r[k] for r in rr]
    crows = [r for r in rows if not r["same_layer"]]
    d = dict(n_pairs=len(rows), share_super=sum(sup) / len(rows), share_sub=sum(subadd) / len(rows), top_heads=[list(heads[i]) for i in U[:12]],
             rho_S_writecos=spear(S, col("write_cos")), rho_S_subspace=spear(S, col("subspace_overlap")), rho_S_wddconf=spear(S, col("wdd_confusability")),
             rho_S_writecos_partial_layer=partial_spear(S, col("write_cos"), [col("same_layer"), col("dlayer")]),
             rho_S_compdata_cross=spear([r["S"] for r in crows], col("comp_data", crows)), rho_S_compstatic_cross=spear([r["S"] for r in crows], col("comp_static", crows)),
             auc_writecos_super_vs_rest=auc(col("write_cos"), sup), auc_wddconf_super_vs_rest=auc(col("wdd_confusability"), sup), auc_compdata_sub_vs_rest_cross=auc(col("comp_data", crows), [r["S"] < -0.2 for r in crows]),
             mean_writecos_super=sum(r["write_cos"] for r, s in zip(rows, sup) if s) / max(sum(sup), 1), mean_writecos_sub=sum(r["write_cos"] for r, s in zip(rows, subadd) if s) / max(sum(subadd), 1),
             mean_comp_super_cross=sum(r["comp_data"] for r in crows if r["S"] > 0.2) / max(sum(1 for r in crows if r["S"] > 0.2), 1), mean_comp_sub_cross=sum(r["comp_data"] for r in crows if r["S"] < -0.2) / max(sum(1 for r in crows if r["S"] < -0.2), 1),
             pairs=rows)
    res[dn] = d
def fm(d): return (f"pairs {d['n_pairs']} (super-additive {d['share_super']:.2f}, sub-additive {d['share_sub']:.2f}) | S vs write cosine rho {d['rho_S_writecos']:+.2f} (layer-partialled {d['rho_S_writecos_partial_layer']:+.2f}), vs static subspace overlap {d['rho_S_subspace']:+.2f}, vs WDD confusability {d['rho_S_wddconf']:+.2f}; "
                   f"AUC super-additive from write cosine {d['auc_writecos_super_vs_rest']:.2f}, from WDD confusability {d['auc_wddconf_super_vs_rest']:.2f}; mean write cosine super {d['mean_writecos_super']:+.3f} vs sub {d['mean_writecos_sub']:+.3f} | cross-layer: S vs data composition rho {d['rho_S_compdata_cross']:+.2f}, static {d['rho_S_compstatic_cross']:+.2f}; AUC sub-additive from composition {d['auc_compdata_sub_vs_rest_cross']:.2f}; mean composition super {d['mean_comp_super_cross']:.4f} vs sub {d['mean_comp_sub_cross']:.4f}")
log(f"{tag}: INDUCTION: " + fm(res["induction"]) + " || NATURAL: " + fm(res["natural"]))
record(f"e376_twins_{tag}", res, " || ".join(f"{dn}: rho(S,wcos) {res[dn]['rho_S_writecos']:+.2f} rho(S,WDDconf) {res[dn]['rho_S_wddconf']:+.2f} rho(S,comp) {res[dn]['rho_S_compdata_cross']:+.2f} AUCsuper(wcos) {res[dn]['auc_writecos_super_vs_rest']:.2f}" for dn in ("induction", "natural")))
