"""e428b: e428 with the natural competitor for mean-ablation damage added: each neuron's expected deviation from its mean
(mean |activation - its mean| x write norm, "deviation", the size of what mean-ablation removes) and the variance-based
version (standard deviation of the activation x write norm, "spread"). Pre-registered: usage still beats both.
e428 notes follow.
e428: is self-description usage a free importance score? The MLP neurons of blocks 0..L (middle depth) ranked five ways:
 usage: how often the neuron's write row is chosen among 16 OMP words describing the middle-depth states (6 sequences);
 magnitude: mean |activation| x write norm (the size of its actual write);
 taylor: |sum over positions of activation x gradient of the loss| (first-order importance, the standard pruning score);
 weight: write norm alone;  random.
The top N neurons of each ranking (N 16, 64, 256) are mean-ablated together (activation set to its mean over other text)
and the next-token loss on 4 held-out sequences is measured. Pre-registered: usage-ranked neurons cost more when removed
than magnitude- and weight-ranked ones (a gradient-free, activation-light importance score from self-description);
Taylor is the reference."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sd_common import *
from lr_common import eval_ids
name = sys.argv[1]; model, tok, fam = load_model(name); arch = Arch(model, fam); L = arch.NB // 2
E = eval_ids(name); sc_ids, ho_ids, ref_ids = E[:6].to(DEV), E[6:10].to(DEV), E[10:14].to(DEV)
for p in model.parameters(): p.requires_grad_(False)
A, lab = build_dictionary(arch, blocks=list(range(L + 1))); typ, blk, idx = lab["type"].to(DEV), lab["block"].to(DEV), lab["index"].to(DEV)
DFF = arch.wdir(0).shape[0]; NBk = L + 1
# usage
lv = Level(model, arch, sc_ids, L); sel, _, _ = omp(lv.Xc, A, 16, batch=256, record_err=False)
cnt = torch.bincount(sel.flatten(), minlength=A.shape[0]).float(); mlp = typ == T_MLP
usage = torch.zeros(NBk, DFF, device=DEV); usage[blk[mlp].long(), idx[mlp].long()] = cnt[mlp]
# magnitude and taylor (one backward pass per two sequences), and mean activations for ablation
mag = torch.zeros(NBk, DFF, device=DEV); tay = torch.zeros(NBk, DFF, device=DEV); hs1 = torch.zeros(NBk, DFF, device=DEV); hs2 = torch.zeros(NBk, DFF, device=DEV); hcount = 0; wn = torch.stack([arch.wdir(b).norm(dim=-1) for b in range(NBk)])
def act_pass(ids, grad):
    acts = {}
    def mk(b):
        def pre(m, a):
            h = a[0]
            if grad: h.retain_grad()
            acts[b] = h
        return pre
    hs = [arch.mlp_lin(b).register_forward_pre_hook(mk(b)) for b in range(NBk)]
    try:
        if grad:
            torch.set_grad_enabled(True)
            emb_out = {}
            def ehook(m, i, o): o.requires_grad_(True); return o
            he = model.get_input_embeddings().register_forward_hook(ehook)
            try: lg = model(ids).logits.float(); token_loss(lg, ids).sum().backward()
            finally: he.remove(); torch.set_grad_enabled(False)
        else:
            with torch.no_grad(): model(ids)
    finally: [h.remove() for h in hs]
    return acts
for s0 in range(0, sc_ids.shape[0], 2):
    acts = act_pass(sc_ids[s0:s0 + 2], True)
    for b in range(NBk):
        h = acts[b][:, 1:].detach().float(); gr = acts[b].grad[:, 1:].float()
        mag[b] += h.abs().sum((0, 1)) * wn[b]; tay[b] += (h * gr).sum((0, 1)); hs1[b] += h.sum((0, 1)); hs2[b] += h.pow(2).sum((0, 1))
    hcount += acts[0][:, 1:].shape[0] * acts[0][:, 1:].shape[1]
tay = tay.abs(); hmean = torch.zeros(NBk, DFF, device=DEV); n = 0
for s0 in range(0, ref_ids.shape[0], 2):
    acts = act_pass(ref_ids[s0:s0 + 2], False)
    for b in range(NBk): hmean[b] += acts[b][:, 1:].float().sum((0, 1))
    n += acts[0][:, 1:].shape[0] * acts[0][:, 1:].shape[1]
hmean /= n
spread = ((hs2 / hcount - (hs1 / hcount) ** 2).clamp_min(0).sqrt()) * wn
dev = torch.zeros(NBk, DFF, device=DEV)
for s0 in range(0, sc_ids.shape[0], 2):
    acts = act_pass(sc_ids[s0:s0 + 2], False)
    for b in range(NBk): dev[b] += (acts[b][:, 1:].float() - hmean[b]).abs().sum((0, 1))
dev = dev * wn
def loss_with(ablate):
    """mean next-token loss on held-out text with the given flat neuron indices mean-ablated"""
    per = {}
    for f in ablate.tolist(): per.setdefault(f // DFF, []).append(f % DFF)
    per = {b: torch.tensor(v, device=DEV) for b, v in per.items()}
    def mk(b):
        def pre(m, a):
            h = a[0].clone(); h[..., per[b]] = hmean[b][per[b]].to(h.dtype); return (h,) + tuple(a[1:])
        return pre
    hs = [arch.mlp_lin(b).register_forward_pre_hook(mk(b)) for b in per]
    try:
        with torch.no_grad(): return token_loss(model(ho_ids).logits.float(), ho_ids).mean().item()
    finally: [h.remove() for h in hs]
base = loss_with(torch.zeros(0, dtype=torch.long)); g = torch.Generator().manual_seed(0)
scores = dict(usage=usage.flatten(), deviation=dev.flatten(), spread=spread.flatten(), taylor=tay.flatten(), magnitude=mag.flatten(), random=torch.rand(NBk * DFF, generator=g).to(DEV))
res = dict(model=name, level=L, base_loss=base, n_neurons=NBk * DFF, used_neurons=int((usage > 0).sum()), delta={}, overlap={})
for N in (64, 256):
    tops = {k: v.topk(N).indices for k, v in scores.items()}
    for k, t in tops.items(): res["delta"][f"{k}@{N}"] = loss_with(t) - base
    res["overlap"][str(N)] = {k: len(set(tops["usage"].tolist()) & set(tops[k].tolist())) / N for k in ("deviation", "spread", "taylor")}
    log(f"{name} N={N}: loss increase " + " ".join(f"{k} {res['delta'][f'{k}@{N}']:+.3f}" for k in scores) + " | overlap of usage top-N with " + " ".join(f"{k} {v:.2f}" for k, v in res["overlap"][str(N)].items()))
summ = f"{name} base {base:.3f}, {res['used_neurons']} of {NBk * DFF} neurons used; loss increase at N=64: " + " ".join(f"{k} {res['delta'][f'{k}@64']:+.3f}" for k in scores) + " | N=256: " + " ".join(f"{k} {res['delta'][f'{k}@256']:+.3f}" for k in scores)
log(summ); record(f"e428b_usage_{name}", res, summ)
