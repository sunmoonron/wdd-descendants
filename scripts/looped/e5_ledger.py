"""e5: the exact ledger of Ouro's MLP writes over 8 loops on 8 sequences (every neuron's write coefficient act*s at
every layer of every loop). Questions: (a) does a neuron write in every loop or mostly in one (write energy per loop,
the loop that carries most of it, overlap of each loop's most active neurons); (b) does it write the same thing to
the same token in successive loops (per-neuron correlation of the coefficient across tokens, loop t vs t+1);
(c) do the largest writes of loop t survive into loop t+1 (projection ratio along the write's direction, tilted by
the between-loop gain, at the start and the middle of the next loop, against their survival at the end of their own
loop)."""
import sys; sys.path.insert(0, "/data/loopwdd/code")
from lw import *
m = LM("ouro"); C = torch.load(f"{ROOT}/cache/ouro_e1.pt"); d, L, F_ = m.d, m.L, m.ffn; NS, TL = 8, 8
ev = C["ids"][:NS].to(DEVM); T = ev.shape[1]
coef = torch.zeros(TL, L, NS, T, F_, dtype=torch.float16)
with torch.no_grad():
    m.run(ev, loops=TL, rec=lambda t, l, c: coef[t, l].copy_(c.half().cpu()))
coef = coef[:, :, :, 1:]                                   # drop BOS
wn = torch.stack([(m.layers[l]["n2b"][:, None] * m.layers[l]["down"]).norm(dim=0).cpu() for l in range(L)])   # [L, F] write norms
res = {}
# (a) write energy per loop
E = torch.zeros(TL, L, F_)
for t in range(TL):
    for l in range(L): E[t, l] = (coef[t, l].float().pow(2).mean((0, 1))) * wn[l] ** 2
tot = E.sum(0); share = E / tot.clamp_min(1e-12)
dom = share.argmax(0)                                      # loop with most of each neuron's write energy
w = tot / tot.sum()                                        # weight neurons by their total write energy
res["energy_by_loop"] = (E.sum((1, 2)) / E.sum()).tolist()
res["dominant_loop_share_weighted"] = [(w * (dom == t)).sum().item() for t in range(TL)]
res["max_loop_share_weighted"] = (w * share.max(0).values).sum().item()       # 1/8 if evenly spread
top = {t: set(E[t].flatten().topk(1000).indices.tolist()) for t in range(TL)}
res["top1000_jaccard"] = [[len(top[a] & top[b]) / len(top[a] | top[b]) for b in range(TL)] for a in range(TL)]
log("energy by loop", [f"{x:.3f}" for x in res["energy_by_loop"]], "weighted max-loop share", f"{res['max_loop_share_weighted']:.3f}",
    "dominant loop (weighted)", [f"{x:.3f}" for x in res["dominant_loop_share_weighted"]])
log("top-1000 Jaccard with loop 4:", [f"{res['top1000_jaccard'][3][b]:.2f}" for b in range(TL)])
# (b) same token, same neuron, successive loops
rs = []
for t in range(TL - 1):
    rr, ww = [], []
    for l in range(L):
        a = coef[t, l].float().reshape(-1, F_); b = coef[t + 1, l].float().reshape(-1, F_)
        a = a - a.mean(0); b = b - b.mean(0)
        r = (a * b).sum(0) / (a.norm(dim=0) * b.norm(dim=0)).clamp_min(1e-9)
        rr.append(r); ww.append((E[t, l] + E[t + 1, l]))
    rr, ww = torch.cat(rr), torch.cat(ww)
    rs.append(dict(weighted=(rr * ww).sum().item() / ww.sum().item(), median=rr.median().item()))
res["coef_corr_next_loop"] = rs
# per-token cosine of the whole neuron-write vector (all layers) between successive loops
wc = []
for t in range(TL - 1):
    a = coef[t].float().permute(1, 2, 0, 3).reshape(-1, L * F_) * wn.reshape(-1)[None]
    b = coef[t + 1].float().permute(1, 2, 0, 3).reshape(-1, L * F_) * wn.reshape(-1)[None]
    wc.append(torch.nn.functional.cosine_similarity(a, b, dim=-1).median().item()); del a, b
res["write_vector_cos_next_loop"] = wc
log("per-token cosine of the neuron-write vector, successive loops:", [f"{x:.2f}" for x in wc])
log("per-neuron coefficient correlation, loop t vs t+1 (energy-weighted):", [f"{x['weighted']:.2f}" for x in rs])
# (c) survival of each loop's 8 largest writes per token
S = C["states"]; gf = m.nf.cpu()
surv = []
for t in range(3):
    x24 = S[(t, 24)][:NS, 1:].reshape(-1, d); x0n = S[(t + 1, 0)][:NS, 1:].reshape(-1, d); x12n = S[(t + 1, 12)][:NS, 1:].reshape(-1, d)
    sc = C["scale"][t][:NS, 1:].reshape(-1)
    W = (coef[t].float().permute(1, 2, 0, 3).reshape(-1, L * F_)) * wn.reshape(-1)[None]        # [N, L*F] signed write sizes
    tv, ti = W.abs().topk(8, dim=1); li, ji = ti // F_, ti % F_
    own, start, mid = [], [], []
    for i in range(8):
        dirs = torch.stack([m.layers[l]["n2b"].cpu() * m.layers[l]["down"][:, j].cpu() for l, j in zip(li[:, i].tolist(), ji[:, i].tolist())])  # [N, d]
        c = W.gather(1, ti[:, i:i + 1])[:, 0]
        dn = dirs / dirs.norm(dim=-1, keepdim=True)
        own.append((x24 * dn).sum(-1) / c)
        dt = gf[None] * dirs; tn = dt.norm(dim=-1); dtn = dt / tn[:, None]
        ct = sc * c * tn / dirs.norm(dim=-1)
        start.append((x0n * dtn).sum(-1) / ct); mid.append((x12n * dtn).sum(-1) / ct)
    own, start, mid = torch.stack(own), torch.stack(start), torch.stack(mid)
    surv.append(dict(own_end=own.median().item(), next_start=start.median().item(), next_mid=mid.median().item(),
                     next_mid_erased=(mid.abs() < 0.2).float().mean().item(), next_mid_flipped=(mid < -0.2).float().mean().item()))
    log(f"loop {t+1} largest writes: survival at own loop end {surv[-1]['own_end']:.2f}, next loop start {surv[-1]['next_start']:.2f}, "
        f"next loop middle {surv[-1]['next_mid']:.2f} (|s|<0.2: {surv[-1]['next_mid_erased']:.2f}, s<-0.2: {surv[-1]['next_mid_flipped']:.2f})")
res["survival"] = surv
jdump(res, f"{ROOT}/results/e5_ouro.json")
