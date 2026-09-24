"""e434: does the network keep what is said in its own words? e432 found that the next block carries a displacement
along the top principal directions forward whole but erases 25-50% of a random one. Here the same measurement is made
word by word and compared with how the self-description uses each word.
Per model, middle depth, 2 sequences x 256 tokens, typical positions (sinks excluded):
- Every probe direction w is added at block L with a fixed size s (a quarter of the median centred-state norm).
- Retention r(w) = mean over positions of <delta_{L+1}, w> / s, where delta is the change of block L+1's output.
  1 means carried unchanged; below 1 the block writes against it (erasure); above 1 it amplifies.
  Also measured at L+3, along with the off-direction response.
Probes:
- own words: MLP write rows of blocks 0..L, in three strata by their use in 16-word self-descriptions of 6 other
  sequences: the 128 most used, 128 random used, 128 random never used;
- the same 384 words rotated (same Gram matrix);
- 384 directions drawn from the states' covariance (covX);
- principal directions 1-40 of the states.
Relations across own words: r against usage, the word's variance share (w'Cw / tr C), its share in M and its block.
Pre-registered:
- median retention of the most-used words exceeds that of never-used words, rotated words and covX directions;
- across own words, r rises with usage after controlling for the variance share (partial Spearman above 0.2)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
name = sys.argv[1]; model, tok, fam = load_model(name); arch = Arch(model, fam); L = arch.NB // 2; D = arch.D
for p in model.parameters(): p.requires_grad_(False)
EA = torch.load("/workspace/wdd/eval_ids_all.pt", weights_only=False)[name]; E = EA["eval_ids"]; del EA
b1, b3 = min(L + 1, arch.NB - 1), min(L + 3, arch.NB - 1)
# usage of each MLP row in 16-word descriptions of 6 sequences (typical positions)
xs = block_states(model, arch, E[:6].to(DEV), [L], chunk=3)[L]; ok = ~sinkmask(xs); X = xs[ok]; mu = X.mean(0); Xc = X - mu
C = torch.cov(Xc.T.double(), correction=0).float(); trC = C.trace()
A, lab = build_dictionary(arch, blocks=list(range(L + 1))); typ, blk = lab["type"].to(DEV), lab["block"].to(DEV)
sel, _, _ = omp(Xc, A, 16, batch=256, record_err=False); use = torch.bincount(sel.flatten(), minlength=A.shape[0]).float(); del sel
mlp = torch.nonzero(typ == T_MLP)[:, 0]; um = use[mlp]; g = torch.Generator(device=DEV).manual_seed(0)
top = mlp[um.topk(128).indices]; used = mlp[um > 0]; used = used[~torch.isin(used, top)]; unused = mlp[um == 0]
rnd_used = used[torch.randperm(used.numel(), device=DEV, generator=g)[:128]]; rnd_unused = unused[torch.randperm(unused.numel(), device=DEV, generator=g)[:128]]
own_idx = torch.cat([top, rnd_used, rnd_unused]); W_own = A[own_idx]; del A
U40 = torch.linalg.eigh(C.double())[1].flip(-1)[:, :40].float().T.contiguous()
W_rot = rotate(W_own, seed=7); W_cov = gauss_like(384, C, seed=3)
probes = torch.cat([W_own, W_rot, W_cov, U40]); groups = ["top"] * 128 + ["used"] * 128 + ["unused"] * 128 + ["rot"] * 384 + ["covX"] * 384 + [f"pc{i + 1}" for i in range(40)]
# retention by finite differences, 16 probes per pass (batch replicated)
ids = E[6:8, :256].to(DEV); x0 = block_states(model, arch, ids, [L, b3], chunk=2); base = block_states(model, arch, ids, [b1], chunk=2)[b1]
xl = x0[L]; okp = ~sinkmask(xl); s = 0.25 * (xl[okp] - xl[okp].mean(0)).norm(dim=-1).median()
NP = 16; r1, r3, off1 = [], [], []
def run_batch(W):
    n = W.shape[0]; ib = ids.repeat(n, 1); Xn = xl.repeat(n, 1, 1)
    add = (W[:, None, None, :] * s).expand(n, ids.shape[0], 1, D).reshape(n * ids.shape[0], 1, D) * okp.repeat(n, 1)[..., None].float()
    Xn = Xn + add; cap = {}
    def st(m, i, o, b):
        cap[b] = out_of(o)[:, 1:].detach().float()
        if b == b3: raise Stop
    hs = [arch.layers[L].register_forward_hook(replace_hook(Xn))] + [arch.layers[b].register_forward_hook(lambda m, i, o, b=b: st(m, i, o, b)) for b in sorted({b1, b3})]
    try:
        with torch.no_grad(): model(ib)
    except Stop: pass
    finally: [h.remove() for h in hs]
    out = []
    for b, ref in ((b1, base), (b3, x0[b3])):
        d = (cap[b] - ref.repeat(n, 1, 1)).view(n, ids.shape[0], -1, D); m_ = okp[None, ..., None].float()
        along = ((d * W[:, None, None, :]).sum(-1, keepdim=True) * m_).sum((1, 2, 3)) / (m_.sum() * s)
        offd = (((d - along[:, None, None, None] * s * W[:, None, None, :]).norm(dim=-1, keepdim=True) * m_).sum((1, 2, 3)) / (m_.sum() * s))
        out.append((along, offd))
    return out
for s0 in range(0, probes.shape[0], NP):
    (a1, o1), (a3, o3) = run_batch(probes[s0:s0 + NP]); r1.append(a1); r3.append(a3); off1.append(o1)
r1, r3, off1 = torch.cat(r1), torch.cat(r3), torch.cat(off1)
G = {}
for i, gname in enumerate(groups): G.setdefault(gname if not gname.startswith("pc") else ("pc1_8" if int(gname[2:]) <= 8 else ("pc9_16" if int(gname[2:]) <= 16 else "pc17_40")), []).append(i)
res = dict(model=name, level=L, blocks=[b1, b3], step=s.item(), groups={})
for gname, ix in G.items():
    ix = torch.tensor(ix, device=DEV); res["groups"][gname] = dict(n=len(ix), r1_median=r1[ix].median().item(), r1_mean=r1[ix].mean().item(), r3_median=r3[ix].median().item(), off1_median=off1[ix].median().item())
# relations across the 384 own words
rk = lambda v: v.argsort().argsort().float()
def spear(a, b): return torch.corrcoef(torch.stack([rk(a), rk(b)]))[0, 1].item()
def partial(a, b, c):
    ra, rb, rc = rk(a), rk(b), rk(c); rab, rac, rbc = [torch.corrcoef(torch.stack([p, q]))[0, 1] for p, q in ((ra, rb), (ra, rc), (rb, rc))]
    return ((rab - rac * rbc) / ((1 - rac ** 2) * (1 - rbc ** 2)).sqrt()).item()
ro = r1[:384]; uo = use[own_idx]; vs = ((W_own @ C) * W_own).sum(-1) / trC; U8 = U40[:8]; ms = (W_own @ U8.T).pow(2).sum(-1); bo = blk[own_idx].float()
res["own_relations"] = dict(spearman_r_usage=spear(ro, uo), spearman_r_varshare=spear(ro, vs), spearman_r_Mshare=spear(ro, ms), spearman_r_block=spear(ro, bo),
                            spearman_usage_varshare=spear(uo, vs), partial_r_usage_given_varshare=partial(ro, uo, vs))
rot_vs = ((W_rot @ C) * W_rot).sum(-1) / trC; res["rot_relations"] = dict(spearman_r_varshare=spear(r1[384:768], rot_vs))
gg = res["groups"]; rel = res["own_relations"]
res["checks"] = dict(top_over_unused=gg["top"]["r1_median"] > gg["unused"]["r1_median"], top_over_rot=gg["top"]["r1_median"] > gg["rot"]["r1_median"],
                     top_over_covX=gg["top"]["r1_median"] > gg["covX"]["r1_median"], partial_usage_over_02=rel["partial_r_usage_given_varshare"] > 0.2)
summ = (f"{name} L{L}->L{b1} (L{b3}): median retention " + " ".join(f"{k} {v['r1_median']:.2f} ({v['r3_median']:.2f})" for k, v in gg.items())
        + f" | own words: Spearman r~usage {rel['spearman_r_usage']:+.2f}, r~variance share {rel['spearman_r_varshare']:+.2f}, r~M-share {rel['spearman_r_Mshare']:+.2f}, r~block {rel['spearman_r_block']:+.2f}, "
        f"usage~variance share {rel['spearman_usage_varshare']:+.2f}, partial r~usage | variance share {rel['partial_r_usage_given_varshare']:+.2f} | rotated words r~variance share {res['rot_relations']['spearman_r_varshare']:+.2f} | checks {json.dumps(res['checks'])}")
log(summ); record(f"e434_kept_{name}", res, summ)
