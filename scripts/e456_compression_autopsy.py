"""e456: what does compression do to the native vocabulary? A WDD compression autopsy, suggested by an external review.
Every linear weight matrix in the blocks (attention and MLP; embeddings and norms untouched) is compressed:
- quantization: symmetric uniform rounding per row of the stored matrix to b bits (b = 8, 6, 4, 3), simulated in fp32;
- magnitude pruning: the smallest |w| in each matrix set to zero (30, 50 and 70%).
Measured per variant, at the middle depth L (typical positions, sinks exact), on the evaluation text:
- function: the change of the model's loss;
- the vocabulary: loss recovered by 16 of the compressed model's own words against its rotation (the self-description
  advantage), relative to the original model's;
- the words themselves: the median cosine between each compressed MLP write row (blocks 0..L) and the original row;
- per-word damage: each MLP word's selection frequency in the 16-word descriptions, original against compressed.
  Reported: the Spearman correlation of the absolute change with the word's original usage and with its row norm, and
  the share of the total change carried by the 1% most-used words.
Models (argument): gpt2, smollm2.
Pre-registered (honest guesses):
- the self-description advantage survives down to 4 bits (at least 80% of the original's) while the loss rises by over
  0.1 nats at 4 bits (0.5);
- at 3 bits or 70% pruning the advantage falls faster than the loss rises, in relative terms (0.4);
- rarely used words change their usage more (Spearman with usage below 0; 0.5)."""
import sys, os, time, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; model, tok, fam = load_model(name); arch = Arch(model, fam); L = arch.NB // 2; t00 = time.time()
ev = eval_ids(name)[:16].to(DEV)
mats = [(n, p) for n, p in model.named_parameters() if p.dim() == 2 and any(f".{i}." in n for i in range(arch.NB)) and "norm" not in n and "ln_" not in n]
orig = {n: p.detach().clone() for n, p in mats}
def apply(kind, level):
    with torch.no_grad():
        for n, p in mats:
            w = orig[n]
            if kind == "orig": p.copy_(w)
            elif kind == "q":
                qmax = 2 ** (level - 1) - 1; sc = w.abs().amax(1, keepdim=True).clamp_min(1e-12) / qmax; p.copy_((w / sc).round().clamp(-qmax, qmax) * sc)
            else:
                k = max(1, int(level / 100 * w.numel())); thr = w.abs().flatten().kthvalue(k).values; p.copy_(torch.where(w.abs() > thr, w, torch.zeros_like(w)))
def eval_loss():
    tl = []
    for s0 in range(0, ev.shape[0], 4):
        with torch.no_grad(): tl.append(token_loss(model(ev[s0:s0 + 4]).logits.float(), ev[s0:s0 + 4]))
    return torch.cat(tl).mean().item()
def measure():
    lv = NSLevel(model, arch, ev[:8], L); A, lab = build_dictionary(arch, blocks=list(range(L + 1))); isM = (lab["type"] == T_MLP).to(DEV)
    own = describe(lv, A, [16])["16"]; rot = describe(lv, rotate(A, seed=7), [16])["16"]
    sel, _, _ = omp(lv.Xc, unitr(A), 16, batch=1024, record_err=False)
    use = torch.bincount(sel.flatten(), minlength=A.shape[0]).float() / sel.shape[0]
    rowsM = A[isM].clone(); normsM = torch.cat([arch.wdir(b).norm(dim=-1) for b in range(L + 1)])
    out = dict(eval_loss=eval_loss(), rec_own=own["rec"], rec_rot=rot["rec"], adv=own["rec"] - rot["rec"], fvu_own=own["fvu"], fvu_rot=rot["fvu"])
    del A, lab, sel, lv; torch.cuda.empty_cache()
    return out, use[isM], rowsM, normsM
def spearman(a, b):
    ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float(); ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm()).clamp_min(1e-12))
apply("orig", 0); base, u0, R0, N0 = measure(); res = dict(model=name, level=L, n_matrices=len(mats), original=base, variants={})
log(f"{name} original: loss {base['eval_loss']:.3f} | k16 rec own {base['rec_own']:.3f} rot {base['rec_rot']:.3f} (adv {base['adv']:.3f})")
for kind, level in (("q", 8), ("q", 6), ("q", 4), ("q", 3), ("p", 30), ("p", 50), ("p", 70)):
    vn = f"{'int' if kind == 'q' else 'prune'}{level}"; apply(kind, level); m, u1, R1, _ = measure()
    du = (u1 - u0).abs(); act = (u0 > 0) | (u1 > 0); top = u0 >= torch.quantile(u0[u0 > 0], 0.99)
    m.update(d_loss=m["eval_loss"] - base["eval_loss"], adv_share=m["adv"] / base["adv"], row_cos_median=float((unitr(R1) * unitr(R0)).sum(-1).median()),
             usage_change_total=float(du.sum() / 2), spearman_change_vs_usage=spearman(du[act], u0[act]), spearman_change_vs_rownorm=spearman(du[act], N0[act]),
             top1pct_share_of_change=float(du[top].sum() / du.sum().clamp_min(1e-12)))
    res["variants"][vn] = m
    log(f"{name} {vn}: dL {m['d_loss']:+.3f} | adv {m['adv']:.3f} ({m['adv_share']:.2f} of original) rec own {m['rec_own']:.3f} rot {m['rec_rot']:.3f} | row cos {m['row_cos_median']:.4f} | "
        f"usage change {m['usage_change_total']:.2f} of 16, Spearman with usage {m['spearman_change_vs_usage']:+.2f}, with row norm {m['spearman_change_vs_rownorm']:+.2f}, top 1% share {m['top1pct_share_of_change']:.2f}")
apply("orig", 0)
V = res["variants"]
res["checks"] = dict(int4_adv_over_0_8_and_loss_up=V["int4"]["adv_share"] >= 0.8 and V["int4"]["d_loss"] > 0.1,
                     severe_adv_falls_faster=any(V[k]["adv_share"] < 1 - V[k]["d_loss"] / base["eval_loss"] for k in ("int3", "prune70")),
                     rare_words_change_more=V["int4"]["spearman_change_vs_usage"] < 0)
summ = (f"{name} L{L} ({len(mats)} matrices): original loss {base['eval_loss']:.3f}, self-description advantage {base['adv']:.3f} || "
        + " || ".join(f"{k}: dL {m['d_loss']:+.3f}, advantage {m['adv_share']:.2f} of original (own {m['rec_own']:.2f} rot {m['rec_rot']:.2f}), row cos {m['row_cos_median']:.3f}, usage change {m['usage_change_total']:.2f}/16 (rho usage {m['spearman_change_vs_usage']:+.2f}, norm {m['spearman_change_vs_rownorm']:+.2f})" for k, m in V.items())
        + f" | checks {_json.dumps(res['checks'])} | {time.time() - t00:.0f}s")
log(summ); record(f"e456_compression_{name}", res, summ)
