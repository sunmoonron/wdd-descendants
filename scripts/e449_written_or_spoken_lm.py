"""e449: is the native vocabulary written or spoken, in a sequence model with sparse latent features? (replication of
e444's frozen-writer result outside grokking)
e444: in modular addition, freezing the MLP write rows at their random initialisation still lets the network
generalise, but its states are then no sparser in its own words than in rotated ones. Freezing the read rows keeps the
vocabulary. Is that a property of the algorithmic task, or of how training builds a vocabulary?
Task: a topic-mixture language. Each sequence of 64 tokens (vocabulary 256) draws 3 of 32 topics and Dirichlet(1)
weights; every token comes from a topic chosen by those weights, and each topic has a sparse distribution over tokens.
Predicting well means inferring in context which few topics are active: sparse latent features, as in real text.
Model: 2 blocks (attention + GELU MLP of 512), d_model 128, pre-LayerNorm, trained from scratch (AdamW, 3000 steps of
64 fresh sequences).
Variants (argument): base, base_s1, frozenW (both MLPs' write rows fixed at initialisation), frozenW_s1, frozenR (read
rows fixed).
At the end, at the output of block 0 (the middle) for positions 16..63 of 256 fresh sequences:
 - self-description with k = 4 and 8 words: own words (embeddings, positions, head bases, block-0 MLP rows, biases),
   the same words rotated, covA, and the top-k principal directions (PCA);
   - scored by the fraction of variance unexplained;
   - and by loss recovered, splicing the description into block 1;
 - ground truth: for each topic, the best |correlation| between "topic active" and any own word's usage, against the
   same for rotated words.
Pre-registered (honest guesses, informed by e444):
 W1 (0.65) the trained model's states are sparse in its own words (own well below rotation in unexplained variance);
 W2 (0.6) with frozen writers the advantage over rotation collapses (under a third of base) while the loss stays within
          0.05 nats of base;
 W3 (0.6) with frozen readers the advantage stays (over two thirds of base);
 W4 (0.5) own words track topics better than rotated words do (topic-usage correlation)."""
import sys, os, math, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn as nn, torch.nn.functional as F
from wdd_common import log, record, DEV, omp, refit
torch.set_grad_enabled(True)
variant = sys.argv[1]
cfg = dict(seed=0, freeze=None, steps=3000, bs=64)
cfg.update({"base": {}, "base_s1": dict(seed=1), "frozenW": dict(freeze="W"), "frozenW_s1": dict(freeze="W", seed=1), "frozenR": dict(freeze="R")}[variant])
V, K, NA, T, d, NH, DM, NL = 256, 32, 3, 64, 128, 4, 512, 2; HD = d // NH
import numpy as np
topics = torch.tensor(np.random.RandomState(1234).dirichlet(np.full(V, 0.05), size=K), dtype=torch.float32, device=DEV)   # [K, V]; the same language for every variant
def sample(n, g):
    act = torch.rand(n, K, generator=g).argsort(dim=1)[:, :NA].to(DEV)                               # [n, 3] active topics
    w = -torch.log(torch.rand(n, NA, generator=g).clamp_min(1e-9)).to(DEV); w = w / w.sum(-1, keepdim=True)   # Dirichlet(1, 1, 1)
    z = torch.multinomial(w, T, replacement=True)                                                     # [n, T] topic index into act
    zt = act.gather(1, z); probs = topics[zt]                                                     # [n, T, V]
    ids = torch.multinomial(probs.view(-1, V), 1).view(n, T); return ids, act
torch.manual_seed(cfg["seed"])
class Blk(nn.Module):
    def __init__(s):
        super().__init__(); s.ln1, s.ln2 = nn.LayerNorm(d), nn.LayerNorm(d); s.qkv = nn.Linear(d, 3 * d, bias=False); s.o = nn.Linear(d, d, bias=False)
        s.fc = nn.Linear(d, DM); s.proj = nn.Linear(DM, d)
    def forward(s, x):
        B, T_, _ = x.shape; q, k, v = s.qkv(s.ln1(x)).view(B, T_, 3, NH, HD).unbind(2); q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True); x = x + s.o(a.transpose(1, 2).reshape(B, T_, d))
        return x + s.proj(F.gelu(s.fc(s.ln2(x))))
class LM(nn.Module):
    def __init__(s):
        super().__init__(); s.emb = nn.Embedding(V, d); s.pos = nn.Parameter(torch.randn(T, d) * 0.02); s.blocks = nn.ModuleList([Blk() for _ in range(NL)]); s.lnf = nn.LayerNorm(d); s.unemb = nn.Linear(d, V, bias=False)
    def forward(s, ids, replace0=None):
        x = s.emb(ids) + s.pos[:ids.shape[1]]; x = s.blocks[0](x)
        if replace0 is not None: x = replace0
        mid = x; x = s.blocks[1](x); return s.unemb(s.lnf(x)), mid
model = LM().to(DEV)
for b in model.blocks:
    if cfg["freeze"] == "W": b.proj.weight.requires_grad_(False); b.proj.bias.requires_grad_(False)
    if cfg["freeze"] == "R": b.fc.weight.requires_grad_(False); b.fc.bias.requires_grad_(False)
opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=1e-3, weight_decay=0.1, betas=(0.9, 0.98)); gtr = torch.Generator().manual_seed(100 + cfg["seed"]); t0 = time.time()
for step in range(cfg["steps"]):
    for gr in opt.param_groups: gr["lr"] = 1e-3 * min(1.0, (step + 1) / 100) * (0.5 * (1 + math.cos(math.pi * step / cfg["steps"])))
    ids, _ = sample(cfg["bs"], gtr); lg, _ = model(ids); loss = F.cross_entropy(lg[:, :-1].reshape(-1, V), ids[:, 1:].reshape(-1))
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if step % 500 == 0: log(f"{variant} step {step}: loss {loss.item():.3f} ({time.time() - t0:.0f}s)")
torch.set_grad_enabled(False); model.eval()
gev = torch.Generator().manual_seed(999); ids, act = sample(256, gev); lg, mid = model(ids)
tl = F.cross_entropy(lg[:, :-1].reshape(-1, V), ids[:, 1:].reshape(-1), reduction="none").view(256, T - 1)
P = slice(16, T - 1)                                                          # positions with context, which also predict a next token
X = mid[:, P].reshape(-1, d); mu = X.mean(0); Xc = X - mu
unit = lambda M: M / M.norm(dim=-1, keepdim=True).clamp_min(1e-8)
b0 = model.blocks[0]; heads = torch.cat([torch.linalg.svd(b0.o.weight.T[h * HD:(h + 1) * HD], full_matrices=False).Vh for h in range(NH)])
A = unit(torch.cat([model.emb.weight, model.pos, heads, b0.proj.weight.T, b0.proj.bias[None]])); n = A.shape[0]
labels = ["E"] * V + ["P"] * T + ["H"] * d + ["M"] * DM + ["b"]
Q = torch.linalg.qr(torch.randn(d, d, generator=torch.Generator().manual_seed(11)))[0].to(DEV)
ev_, U_ = torch.linalg.eigh((A.T @ A / n).double()); Rm = ((U_ * ev_.clamp_min(0).sqrt()) @ U_.T).float()
dicts = dict(own=A, rot=A @ Q, covA=unit(torch.randn(n, d, generator=torch.Generator().manual_seed(13)).to(DEV) @ Rm))
Lm = None
def spliced_loss(Xh_full):
    """replace block 0's output at positions 16..62 by mu + Xh (others exact) and return the loss at those positions"""
    rep = mid.clone(); rep[:, P] = (mu + Xh_full).view(256, -1, d); lg2, _ = model(ids, replace0=rep)
    return F.cross_entropy(lg2[:, :-1].reshape(-1, V), ids[:, 1:].reshape(-1), reduction="none").view(256, T - 1)[:, P].mean().item()
Lclean = tl[:, P].mean().item(); Lmean = spliced_loss(torch.zeros_like(Xc)); res = dict(variant=variant, cfg=cfg, loss=Lclean, mean_state_loss=Lmean, cells={})
tot = Xc.pow(2).sum()
for vn, Dct in dicts.items():
    sel, _, _ = omp(Xc, Dct, 8, batch=2048, record_err=False)
    for k in (4, 8):
        cof, err = refit(Xc, Dct, sel[:, :k]); Xh = torch.einsum("nk,nkd->nd", cof, Dct[sel[:, :k]])
        res["cells"][f"{vn}_k{k}"] = dict(fvu=(err.sum() / tot).item(), rec=(Lmean - spliced_loss(Xh)) / max(Lmean - Lclean, 1e-9))
    if vn in ("own", "rot"):
        use = torch.zeros(X.shape[0], n, device=DEV); use.scatter_(1, sel[:, :8], 1.0)
        tp = torch.zeros(256, K, device=DEV); tp.scatter_(1, act, 1.0); tpos = tp[:, None, :].expand(256, T - 1 - 16, K).reshape(-1, K)[: X.shape[0]]
        uc = use - use.mean(0); tc = tpos - tpos.mean(0); cor = (tc.T @ uc) / (tc.norm(dim=0)[:, None] * uc.norm(dim=0)[None]).clamp_min(1e-9)
        res[f"topic_best_corr_{vn}"] = cor.abs().max(1).values.mean().item()
        if vn == "own": res["usage_by_type"] = {t: (use[:, torch.tensor([l == t for l in labels], device=DEV)].sum() / use.sum()).item() for t in "EPHMb"}
evs, Us = torch.linalg.eigh(torch.cov(Xc.T.double(), correction=0)); Us = Us.flip(-1).float()
for k in (4, 8):
    Pk = Us[:, :k] @ Us[:, :k].T; Xh = Xc @ Pk; res["cells"][f"pca_k{k}"] = dict(fvu=((Xc - Xh).pow(2).sum() / tot).item(), rec=(Lmean - spliced_loss(Xh)) / max(Lmean - Lclean, 1e-9))
c = res["cells"]; res["adv_fvu_k4"] = c["rot_k4"]["fvu"] - c["own_k4"]["fvu"]; res["wordlevel_fvu_k4"] = c["covA_k4"]["fvu"] - c["own_k4"]["fvu"]
# family by family: is each family of writers (MLP rows, head bases, embeddings) a vocabulary on its own, against its own rotation?
fam_res = {}
for fname, keep in (("mlp_rows", [l == "M" for l in labels]), ("head_bases", [l == "H" for l in labels]), ("embeddings", [l == "E" for l in labels])):
    Df = A[torch.tensor(keep, device=DEV)]; out = {}
    for vn, Dd in (("own", Df), ("rot", Df @ Q)):
        sel, _, _ = omp(Xc, Dd, 4, batch=2048, record_err=False); _, err = refit(Xc, Dd, sel); out[vn] = (err.sum() / tot).item()
    fam_res[fname] = dict(own=out["own"], rot=out["rot"], adv=out["rot"] - out["own"])
res["families_k4"] = fam_res
summ = (f"{variant}: loss {Lclean:.3f} (mean-state splice {Lmean:.3f}) | k4 unexplained own {c['own_k4']['fvu']:.2f} rot {c['rot_k4']['fvu']:.2f} covA {c['covA_k4']['fvu']:.2f} pca {c['pca_k4']['fvu']:.2f} "
        f"(k8 own {c['own_k8']['fvu']:.2f} rot {c['rot_k8']['fvu']:.2f}) | k4 loss recovered own {c['own_k4']['rec']:.2f} rot {c['rot_k4']['rec']:.2f} covA {c['covA_k4']['rec']:.2f} pca {c['pca_k4']['rec']:.2f} | "
        f"topic tracking (best |corr| per topic) own {res['topic_best_corr_own']:.2f} rot {res['topic_best_corr_rot']:.2f} | by family, k4 unexplained own/rot: " + " ".join(f"{k} {v['own']:.2f}/{v['rot']:.2f}" for k, v in res["families_k4"].items()) + f" | usage by type {json.dumps({k: round(v, 2) for k, v in res['usage_by_type'].items()})} | {time.time() - t0:.0f}s")
log(summ); record(f"e449b_writtenlm_{variant}", res, summ)
