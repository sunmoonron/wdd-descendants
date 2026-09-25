"""e460: do near-identical states with different WDD ledgers have different futures, and why? This is the testable core
of an external review's "is the residual stream Markov?" proposal.
The forward pass from block L is a deterministic function of block L's outputs at every position, so the write history
is not an input. Two near-identical states at different positions can still have different futures for two reasons:
- context: later blocks read other positions through attention;
- the vector: the small difference may lie in a direction the network is sensitive to.
Setup: the most similar pairs of states at the middle depth (evaluation text, typical positions, different sequences),
with cosine of the centred states above 0.95, up to 600 pairs. For each pair (i in context A, j in context B):
- total divergence: the KL divergence between their next-token distributions;
- vector effect: the KL between A's clean prediction and A's prediction with the state at i replaced by j's
  (transplanted at block L; A's context kept);
- context effect: the KL between that transplanted prediction and B's own prediction (same vector, different context);
- ledger distance: 1 - Jaccard overlap of the two states' 16-word native supports (OMP).
Question: does the ledger distance predict the vector effect beyond the cosine (partial Spearman controlling for cosine
and norm ratio)? Pairs with high and low ledger distance are also compared, at matched cosine.
Models (argument): gpt2, qwen05.
Pre-registered (honest guesses):
- context carries most of the divergence of near-identical states: the vector effect is under a quarter of the total
  (0.7);
- the ledger distance adds little to the cosine in predicting the vector effect (abs partial Spearman under 0.2; 0.5)."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; model, tok, fam = load_model(name); arch = Arch(model, fam); L = arch.NB // 2
ev = eval_ids(name)[:32].to(DEV); B, T = ev.shape
X = block_states(model, arch, ev, [L], chunk=4)[L]                              # [B, T-1, D], positions 1..T-1
flat = X.reshape(-1, arch.D); keep = ~sinkmask(flat); mu = flat[keep].mean(0)
seq_id = torch.arange(B, device=DEV)[:, None].expand(B, T - 1).reshape(-1); pos_id = torch.arange(1, T, device=DEV)[None].expand(B, T - 1).reshape(-1)
idx = torch.nonzero(keep & (pos_id >= 16))[:, 0]                                # typical positions with some context
g = torch.Generator(device=DEV).manual_seed(0); idx = idx[torch.randperm(idx.numel(), device=DEV, generator=g)[:6000]]
Z = flat[idx] - mu; Zn = Z / Z.norm(dim=-1, keepdim=True)
C = Zn @ Zn.T; C.fill_diagonal_(-1); C[seq_id[idx][:, None] == seq_id[idx][None, :]] = -1                   # different sequences only
vals, flat_ix = C.flatten().topk(4000); pairs, used = [], set()
for v, f in zip(vals.tolist(), flat_ix.tolist()):
    a, b = divmod(f, idx.numel())
    if v < 0.95 or len(pairs) >= 600: break
    if a in used or b in used or a > b: continue
    used.update((a, b)); pairs.append((a, b, v))
need = {}
for a, b, _ in pairs:
    for q in (a, b): need.setdefault(int(seq_id[idx[q]]), set()).add(int(pos_id[idx[q]]))
LPd = {}
for sq, ps in need.items():                                                     # next-token log-probs only where needed
    with torch.no_grad(): lg = model(ev[sq:sq + 1]).logits[0].float()
    for p_ in ps: LPd[(sq, p_)] = torch.log_softmax(lg[p_], -1)
    del lg
A, lab = build_dictionary(arch, blocks=list(range(L + 1)))
sel, _, _ = omp(Z, unitr(A), 16, batch=1024, record_err=False)
def kl(p, q): return float((p.exp() * (p - q)).sum())
rows = []
for a, b, cs in pairs:
    ia, ib = int(idx[a]), int(idx[b]); sa, pa = int(seq_id[ia]), int(pos_id[ia]); sb, pb = int(seq_id[ib]), int(pos_id[ib])
    lpa, lpb = LPd[(sa, pa)], LPd[(sb, pb)]
    xb = flat[ib]
    def hk(m, i, o):
        xo = out_of(o); y = xo.clone(); y[0, pa] = xb.to(y.dtype)
        return (y,) + tuple(o[1:]) if isinstance(o, tuple) else y
    h = arch.layers[L].register_forward_hook(hk)
    try:
        with torch.no_grad(): lpt = torch.log_softmax(model(ev[sa:sa + 1, :pa + 1]).logits[0, pa].float(), -1)
    finally: h.remove()
    Sa, Sb = set(sel[a].tolist()), set(sel[b].tolist())
    rows.append(dict(cos=cs, norm_ratio=float(flat[ia].norm() / flat[ib].norm()), same_token=int(ev[sa, pa]) == int(ev[sb, pb]),
                     ledger_dist=1 - len(Sa & Sb) / len(Sa | Sb), kl_total=kl(lpa, lpb), kl_vector=kl(lpa, lpt), kl_context=kl(lpt, lpb)))
def rank(v): t = torch.tensor(v, dtype=torch.float64); return t.argsort().argsort().double()
def partial_spearman(x, y, zs):
    rx, ry = rank(x), rank(y); Zm = torch.stack([torch.ones(len(x), dtype=torch.float64)] + [rank(z) for z in zs], 1)
    res_ = lambda r: r - Zm @ torch.linalg.lstsq(Zm, r[:, None]).solution[:, 0]
    ex, ey = res_(rx), res_(ry); return float((ex * ey).sum() / (ex.norm() * ey.norm()))
def spear(x, y): return partial_spearman(x, y, [])
col = lambda k: [r[k] for r in rows]
n = len(rows); tot = sum(col("kl_total")); vec = sum(col("kl_vector")); ctx = sum(col("kl_context"))
med_ld = sorted(col("ledger_dist"))[n // 2]; hi = [r for r in rows if r["ledger_dist"] > med_ld]; lo = [r for r in rows if r["ledger_dist"] <= med_ld]
mean = lambda rr, k: sum(r[k] for r in rr) / max(len(rr), 1)
res = dict(model=name, level=L, n_pairs=n, cos_range=[min(col("cos")), max(col("cos"))], same_token_share=mean(rows, "same_token"),
           kl_total_mean=tot / n, kl_vector_mean=vec / n, kl_context_mean=ctx / n, vector_share_of_total=vec / max(tot, 1e-12),
           spearman_ledger_vs_vector=spear(col("ledger_dist"), col("kl_vector")), spearman_cos_vs_vector=spear(col("cos"), col("kl_vector")),
           partial_ledger_vs_vector=partial_spearman(col("ledger_dist"), col("kl_vector"), [col("cos"), col("norm_ratio")]),
           partial_ledger_vs_total=partial_spearman(col("ledger_dist"), col("kl_total"), [col("cos"), col("norm_ratio")]),
           high_ledger=dict(n=len(hi), cos=mean(hi, "cos"), kl_total=mean(hi, "kl_total"), kl_vector=mean(hi, "kl_vector"), kl_context=mean(hi, "kl_context")),
           low_ledger=dict(n=len(lo), cos=mean(lo, "cos"), kl_total=mean(lo, "kl_total"), kl_vector=mean(lo, "kl_vector"), kl_context=mean(lo, "kl_context")), rows=rows)
res["checks"] = dict(vector_under_quarter=res["vector_share_of_total"] < 0.25, ledger_adds_little=abs(res["partial_ledger_vs_vector"]) < 0.2)
summ = (f"{name} L{L}: {n} pairs of states from different sequences with centred cosine {res['cos_range'][0]:.3f}-{res['cos_range'][1]:.3f} (same token {res['same_token_share']:.2f}) | "
        f"mean KL total {res['kl_total_mean']:.3f} = vector {res['kl_vector_mean']:.3f} (share {res['vector_share_of_total']:.2f}) + context {res['kl_context_mean']:.3f} | "
        f"ledger distance vs vector effect: Spearman {res['spearman_ledger_vs_vector']:+.2f}, partial (cosine, norm) {res['partial_ledger_vs_vector']:+.2f}; vs total, partial {res['partial_ledger_vs_total']:+.2f} | "
        f"high vs low ledger distance: cos {res['high_ledger']['cos']:.3f}/{res['low_ledger']['cos']:.3f}, KL vector {res['high_ledger']['kl_vector']:.3f}/{res['low_ledger']['kl_vector']:.3f}, total {res['high_ledger']['kl_total']:.3f}/{res['low_ledger']['kl_total']:.3f} | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e460_markovpairs_{name}", res, summ)
