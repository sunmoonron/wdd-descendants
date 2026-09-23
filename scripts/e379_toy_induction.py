"""e379: manufacture the phenomena. A tiny attention-only transformer (vocab 256, d 128, 8 heads of 16, 2 layers unless
the variant says otherwise, LayerNorm, learned positions) trained from scratch on sequences of 48 random tokens that
are repeated with probability 0.5 (so half the data rewards induction). Variants change one knob: seed, head dropout
(each head's write dropped per sequence during training), weight decay, learning-rate schedule (cosine to zero vs
constant), depth, and the share of repeated sequences. At 12 snapshots through training and at the end: induction loss,
prefix-matching and previous-token scores, single-head ablation effects, joint over sum for the circuit heads
(redundancy), pairwise interactions on the loss and on the correct token's logit (e377's convexity test in a toy),
the removal curve's shape for the top head, co-selection (per-batch gradients along each head's scale over batches
whose share of repeated sequences varies from 0 to 1, and over batches where it is fixed), each circuit head's
selection against the batch's repeated share, and mean selection against use."""
import sys, os, math, time, json, random, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn as nn, torch.nn.functional as F
from wdd_common import log, record, DEV
torch.set_grad_enabled(True); torch.set_num_threads(4)
variant = sys.argv[1]
cfg = dict(V=256, half=48, d=128, NH=8, NL=2, steps=4000, bs=128, lr=2e-3, wd=0.01, hdrop=0.0, sched="cos", seed=0, p_rep=0.5)
cfg.update({"base": {}, "seed1": dict(seed=1), "seed2": dict(seed=2), "hdrop02": dict(hdrop=0.2), "hdrop05": dict(hdrop=0.5), "wd0": dict(wd=0.0), "wd03": dict(wd=0.3), "const": dict(sched="const"),
            "L3": dict(NL=3), "prep09": dict(p_rep=0.9), "prep01": dict(p_rep=0.1), "long": dict(steps=12000), "lr1e3": dict(lr=1e-3)}[variant])
torch.manual_seed(cfg["seed"]); T = 2 * cfg["half"] + 1; V, d, NH, NL = cfg["V"], cfg["d"], cfg["NH"], cfg["NL"]; HD = d // NH
class Toy(nn.Module):
    def __init__(s):
        super().__init__(); s.emb = nn.Embedding(V, d); s.pos = nn.Parameter(torch.randn(T, d) * 0.02)
        s.ln = nn.ModuleList([nn.LayerNorm(d) for _ in range(NL)]); s.qkv = nn.ModuleList([nn.Linear(d, 3 * d, bias=False) for _ in range(NL)]); s.o = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(NL)])
        s.lnf = nn.LayerNorm(d); s.unemb = nn.Linear(d, V, bias=False)
    def forward(s, ids, scale=None, hdrop=0.0, attn=False):
        B, T_ = ids.shape; x = s.emb(ids) + s.pos[:T_]; mask = torch.triu(torch.ones(T_, T_, dtype=torch.bool, device=ids.device), 1); atts = []
        for l in range(NL):
            q, k, v = s.qkv[l](s.ln[l](x)).view(B, T_, 3, NH, HD).unbind(2); q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            a = ((q @ k.transpose(-1, -2)) / math.sqrt(HD)).masked_fill(mask, -1e9).softmax(-1); z = a @ v
            if scale is not None: z = z * scale[l][None, :, None, None]
            if hdrop > 0: z = z * (torch.rand(B, NH, 1, 1, device=z.device) > hdrop).float() / (1 - hdrop)
            x = x + s.o[l](z.transpose(1, 2).reshape(B, T_, d))
            if attn: atts.append(a)
        return s.unemb(s.lnf(x)), atts
def batch(n, p_rep, g):
    h = cfg["half"]; r = torch.randint(1, V, (n, 2 * h), generator=g, device=DEV); rep = torch.rand(n, generator=g, device=DEV) < p_rep; r[rep, h:] = r[rep, :h]
    return torch.cat([torch.zeros(n, 1, dtype=torch.long, device=DEV), r], 1), rep
h_ = cfg["half"]; qpos = torch.arange(1 + h_, 2 * h_, device=DEV)           # second-copy positions predicting the next token
def ind_metrics(model, ids, scale=None):
    lg, _ = model(ids, scale=scale); lg = lg[:, qpos].float(); tgt = ids[:, qpos + 1]
    lp = torch.log_softmax(lg, -1); cl = lg.gather(-1, tgt[..., None])[..., 0]
    return -lp.gather(-1, tgt[..., None])[..., 0].mean(), -cl.mean()
model = Toy().to(DEV); opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"], betas=(0.9, 0.98)); g = torch.Generator(device=DEV).manual_seed(cfg["seed"] + 100)
lr_at = (lambda s: cfg["lr"] * min(1.0, (s + 1) / 100) * (0.5 * (1 + math.cos(math.pi * s / cfg["steps"])) if cfg["sched"] == "cos" else 1.0))
snaps = {}; snap_every = cfg["steps"] // 12; t0 = time.time()
for step in range(cfg["steps"]):
    for gr in opt.param_groups: gr["lr"] = lr_at(step)
    ids, _ = batch(cfg["bs"], cfg["p_rep"], g); model.train(); lg, _ = model(ids, hdrop=cfg["hdrop"]); loss = F.cross_entropy(lg[:, :-1].reshape(-1, V).float(), ids[:, 1:].reshape(-1))
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if (step + 1) % snap_every == 0: snaps[step + 1] = copy.deepcopy(model.state_dict())
train_s = time.time() - t0; model.eval(); ge = torch.Generator(device=DEV).manual_seed(12345); EV, _ = batch(64, 1.0, ge)
CB = [batch(16, float(p), ge)[0] for p in torch.linspace(0, 1, 64)]; FB = [batch(16, 0.5, ge)[0] for _ in range(64)]; pcb = torch.linspace(0, 1, 64)
ones = lambda: [torch.ones(NH, device=DEV) for _ in range(NL)]
def measure(m, circuit=None):
    with torch.no_grad():
        L0, N0 = ind_metrics(m, EV); _, atts = m(EV, attn=True)
        j = torch.arange(1, h_ - 1, device=DEV); pref = torch.stack([atts[l][:, :, 1 + h_ + j, 1 + j + 1].mean((0, 2)) for l in range(NL)]); qq = torch.arange(1, T, device=DEV); prev = torch.stack([atts[l][:, :, qq, qq - 1].mean((0, 2)) for l in range(NL)])
        E = torch.zeros(NL, NH); En = torch.zeros(NL, NH)
        for l in range(NL):
            for h in range(NH):
                sc = ones(); sc[l][h] = 0; a, b = ind_metrics(m, EV, sc); E[l, h] = (a - L0).item(); En[l, h] = (b - N0).item()
    flat = [(l, h) for l in range(NL) for h in range(NH)]; order = sorted(flat, key=lambda x: -E[x].item()); circ = circuit if circuit is not None else [x for x in order if E[x] > 0.1 * E[order[0]]][:6]
    if len(circ) < 2: circ = order[:2]
    with torch.no_grad():
        sc = ones()
        for (l, h) in circ: sc[l][h] = 0
        a, b = ind_metrics(m, EV, sc); joint = (a - L0).item(); joint_n = (b - N0).item()
        pairs = []
        for x in range(min(4, len(circ))):
            for y in range(x + 1, min(4, len(circ))):
                sc = ones(); sc[circ[x][0]][circ[x][1]] = 0; sc[circ[y][0]][circ[y][1]] = 0; a, b = ind_metrics(m, EV, sc)
                ei, ej = E[circ[x]].item(), E[circ[y]].item(); ni, nj = En[circ[x]].item(), En[circ[y]].item()
                pairs.append(dict(loss=((a - L0).item() - ei - ej) / max(0.5 * (abs(ei) + abs(ej)), 1e-9), logit=((b - N0).item() - ni - nj) / max(0.5 * (abs(ni) + abs(nj)), 1e-9)))
        tl = circ[0]; path = []
        for al in (1.0, 0.5, 0.0):
            sc = ones(); sc[tl[0]][tl[1]] = al; path.append(ind_metrics(m, EV, sc)[0].item())
    def sel(batches):
        S = []
        for ids in batches:
            sc = [torch.ones(NH, device=DEV, requires_grad=True) for _ in range(NL)]; lg, _ = m(ids, scale=sc); loss = F.cross_entropy(lg[:, :-1].reshape(-1, V).float(), ids[:, 1:].reshape(-1))
            gr = torch.autograd.grad(loss, sc); S.append(torch.cat([x.detach() for x in gr]).cpu())
        return torch.stack(S)
    Sc, Sf = sel(CB), sel(FB); ci = [l * NH + h for l, h in circ]; others = [i for i in range(NL * NH) if i not in ci]
    def gcorr(S, idx):
        Zs = (S - S.mean(0)) / S.std(0).clamp_min(1e-12); C = (Zs.T @ Zs) / (len(S) - 1); sub = C[idx][:, idx]; n = len(idx)
        return ((sub.sum() - sub.diagonal().sum()) / max(n * n - n, 1)).item()
    rnd = [random.Random(k).sample(others, len(ci)) for k in range(50)] if len(others) >= len(ci) else []
    nullc = [gcorr(Sc, r) for r in rnd]; content = [torch.corrcoef(torch.stack([Sc[:, i], pcb]))[0, 1].item() for i in ci]
    use = torch.tensor([E[x].item() for x in flat]); selm = -Sf.mean(0); rk = lambda v: v.argsort().argsort().float()
    return dict(ind_loss=L0.item(), max_prefix=pref.max().item(), n_prefix_heads=int((pref >= 0.2).sum()), max_prev=prev.max().item(), circuit=[list(x) for x in circ], circuit_effects=[E[x].item() for x in circ],
                joint_over_sum=joint / max(sum(E[x].item() for x in circ), 1e-9), joint_over_sum_logit=joint_n / max(sum(En[x].item() for x in circ), 1e-9),
                pair_S_loss=sum(p["loss"] for p in pairs) / max(len(pairs), 1), pair_S_logit=sum(p["logit"] for p in pairs) / max(len(pairs), 1), pair_sign_flips=sum((p["loss"] > 0) != (p["logit"] > 0) for p in pairs) / max(len(pairs), 1),
                shape_top=(path[1] - path[0]) / max(path[2] - path[0], 1e-9), cosel_varying=gcorr(Sc, ci), cosel_varying_null=sum(nullc) / max(len(nullc), 1), cosel_varying_z=(gcorr(Sc, ci) - sum(nullc) / max(len(nullc), 1)) / max(torch.tensor(nullc).std().item() if len(nullc) > 1 else 1.0, 1e-9),
                cosel_fixed=gcorr(Sf, ci), content_corr_circuit=sum(content) / len(content), selection_circuit_mean=selm[ci].mean().item(), selection_circuit_t=(selm[ci] / (Sf[:, ci].std(0) / math.sqrt(len(Sf))).clamp_min(1e-12)).mean().item(), use_circuit_mean=use[ci].mean().item(),
                selection_use_spearman=torch.corrcoef(torch.stack([rk(selm), rk(use)]))[0, 1].item())
final = measure(model); circ = [tuple(x) for x in final["circuit"]]; traj = {}
for s, sd in snaps.items():
    m2 = Toy().to(DEV); m2.load_state_dict(sd); m2.eval(); traj[s] = measure(m2, circuit=circ)
res = dict(variant=variant, cfg=cfg, train_seconds=train_s, final=final, trajectory={str(k): v for k, v in traj.items()})
f = final
log(f"{variant}: trained {train_s:.0f}s; induction loss {f['ind_loss']:.3f}, prefix heads {f['n_prefix_heads']} (max {f['max_prefix']:.2f}), circuit {f['circuit']}; joint/sum loss {f['joint_over_sum']:.2f} logit {f['joint_over_sum_logit']:.2f}; pair S loss {f['pair_S_loss']:+.2f} logit {f['pair_S_logit']:+.2f} (sign flips {f['pair_sign_flips']:.2f}); top-head shape {f['shape_top']:.2f}; co-selection varying {f['cosel_varying']:+.2f} (z {f['cosel_varying_z']:+.1f}) fixed {f['cosel_fixed']:+.2f}; content corr {f['content_corr_circuit']:+.2f}; circuit selection {f['selection_circuit_mean']:+.4f} (t {f['selection_circuit_t']:+.1f}) vs use {f['use_circuit_mean']:+.3f}; selection~use rho {f['selection_use_spearman']:+.2f}")
log(f"{variant} trajectory (step: ind loss, co-sel z, selection t, joint/sum): " + " ".join(f"{k}: {v['ind_loss']:.2f} {v['cosel_varying_z']:+.1f} {v['selection_circuit_t']:+.1f} {v['joint_over_sum']:.2f}" for k, v in traj.items()))
record(f"e379_toy_{variant}", res, f"ind loss {f['ind_loss']:.3f} joint/sum {f['joint_over_sum']:.2f} pairS loss/logit {f['pair_S_loss']:+.2f}/{f['pair_S_logit']:+.2f} cosel z {f['cosel_varying_z']:+.1f} sel t {f['selection_circuit_t']:+.1f}")
