"""e510: which directions does the learned contraction spare? e509 found that a random direction of the chord's size,
subtracted from what an MLP sees, elicits a contraction of -0.8 to -1.7 from the small writes, and the chord itself
a third to a half of that. The chord is the position's own large writes: native atoms. Is the sparing about the
atoms (the model's write structure) or about being on the state cloud's manifold (any high-variance direction)? The
discriminating control is a random direction with the state cloud's covariance (no provenance, the same variance),
as in e497; the second question is whether the atoms used as native words are spared more than unused ones.
Per block b (NB/4, NB/2, 3NB/4) and typical position: the residual entering the block's second layer norm h; the
MLP's directional gain along a unit direction d, <MLP(LN(h + eps d)) - MLP(LN(h)), d> / eps, with eps a fraction
(0.1, and 1.0) of the centred state norm at that position, the median over positions, then over the directions of
a set; and the norm of the response over eps. Sets of 128 directions: MLP rows of earlier blocks chosen at random,
the 128 rows most used as native words at that block (16-word OMP over the centred states), 128 rows never used,
token embeddings, head-basis atoms, random directions, covariance-matched random directions, and the state cloud's
top 64 principal directions.
Reported: the median gain by set and size; the sparing ratios native rows over random, native rows over
covariance-matched, used rows over unused; and across Pythia's checkpoints whether the sparing emerges with the
contraction.
Pre-registered (honest guesses):
- at the end of training native MLP rows are contracted less than random directions in all three models (gain ratio
  below 0.7 at the middle block, size 0.1) (0.6);
- native MLP rows are contracted less than covariance-matched directions too (ratio below 0.85) (0.4);
- the rows used as words are contracted less than unused rows (ratio below 0.85) (0.5);
- over Pythia's training the random-direction gain and the sparing ratio move together, both emerging between
  steps 512 and 16000 (0.5).
Arguments: name [revision]."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; rev = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] not in ("none", "main") else None
model, tok, fam = load_model(name, revision=rev); arch = Arch(model, fam); NB = arch.NB; D = arch.D; DFF = arch.DFF; ND = 128; NPC = 64; SIZES = (0.1, 1.0); BATCH = 16; K = 16
blocks = sorted({NB // 4, NB // 2, (3 * NB) // 4}); LB = max(blocks); ids = eval_ids(name)[:8, :256].to(DEV); B_, T_ = ids.shape; CH = 4
def ln2(b): return arch.layers[b].ln_2 if fam == "gpt2" else arch.layers[b].post_attention_layernorm
def mlp(b): return arch.layers[b].mlp
H = {b: [] for b in blocks}
def mk(b):
    def pre(m, args): H[b].append(args[0].detach().float()[:, 1:].reshape(-1, D)); return None
    return pre
hs = [ln2(b).register_forward_pre_hook(mk(b)) for b in blocks]
try: block_states(model, arch, ids, [LB], chunk=CH)
finally: [h.remove() for h in hs]
H = {b: torch.cat(v) for b, v in H.items()}
g = torch.Generator(device=DEV).manual_seed(0)
def gains(b, Hk, xn, dirs, s):
    """median over positions of the MLP's directional gain along each unit direction, and of the response norm over eps"""
    base = mlp(b)(ln2(b)(Hk)).float(); out_g, out_n = [], []
    for i in range(0, dirs.shape[0], BATCH):
        dd = dirs[i:i + BATCH]; eps = (s * xn)[None, :, None]; y = mlp(b)(ln2(b)(Hk[None] + eps * dd[:, None, :])).float() - base[None]
        out_g.append(((y * dd[:, None, :]).sum(-1) / eps[:, :, 0]).median(1).values); out_n.append((y.norm(dim=-1) / eps[:, :, 0]).median(1).values)
    return torch.cat(out_g), torch.cat(out_n)
res = dict(model=name, revision=rev, blocks=blocks, by_block={})
for b in blocks:
    Hb = H[b]; keep = ~sinkmask(Hb); Hk = Hb[keep]; Hc = Hk - Hk.mean(0); xn = Hc.norm(dim=-1).clamp_min(1e-6); N = Hk.shape[0]
    A, lab = build_dictionary(arch, blocks=list(range(b))); Au = unitr(A); del A; typ = lab["type"].to(DEV); m = Au.shape[0]
    sel, _, _ = omp(Hc, Au, K, batch=1024, record_err=False); usage = torch.bincount(sel.reshape(-1), minlength=m)
    mlp_atoms = torch.nonzero(typ == T_MLP)[:, 0]; order = usage[mlp_atoms].argsort(descending=True); used = mlp_atoms[order[:ND]]; unused_pool = mlp_atoms[usage[mlp_atoms] == 0]
    pick = lambda pool, n: pool[torch.randperm(pool.numel(), generator=g, device=DEV)[:n]]
    C = torch.cov(Hc.T.double(), correction=0); L = torch.linalg.cholesky(C + 1e-6 * torch.eye(D, device=DEV, dtype=torch.float64)).float(); ev, evec = torch.linalg.eigh(C); pca = evec[:, -NPC:].T.float().flip(0)
    sets = {"mlp_rows_random": Au[pick(mlp_atoms, ND)], "mlp_rows_used": Au[used], "mlp_rows_unused": Au[pick(unused_pool, ND)], "token_embeddings": Au[pick(torch.nonzero(typ == T_TOK)[:, 0], ND)], "head_bases": Au[pick(torch.nonzero(typ == T_ATT)[:, 0], ND)],
            "random": unitr(torch.randn(ND, D, generator=g, device=DEV)), "covariance_matched": unitr(torch.randn(ND, D, generator=g, device=DEV) @ L.T), "pca_top64": unitr(pca)}
    out = dict(n_positions=N, n_atoms=m, n_unused_rows=int(unused_pool.numel()), share_rows_used=float((usage[mlp_atoms] > 0).float().mean()), gain={}, response_norm={})
    with torch.no_grad():
        for s in SIZES:
            out["gain"][str(s)] = {}; out["response_norm"][str(s)] = {}
            for nm, dirs in sets.items():
                gg, nn = gains(b, Hk, xn, dirs, s); out["gain"][str(s)][nm] = float(gg.median()); out["response_norm"][str(s)][nm] = float(nn.median())
    gs = out["gain"]["0.1"]; out["ratios"] = {s: dict(native_over_random=out["gain"][s]["mlp_rows_random"] / out["gain"][s]["random"] if out["gain"][s]["random"] != 0 else None, native_over_covariance=out["gain"][s]["mlp_rows_random"] / out["gain"][s]["covariance_matched"] if out["gain"][s]["covariance_matched"] != 0 else None,
                                       used_over_unused=out["gain"][s]["mlp_rows_used"] / out["gain"][s]["mlp_rows_unused"] if out["gain"][s]["mlp_rows_unused"] != 0 else None, covariance_over_random=out["gain"][s]["covariance_matched"] / out["gain"][s]["random"] if out["gain"][s]["random"] != 0 else None) for s in out["gain"]}
    res["by_block"][b] = out; r1 = out["ratios"]["0.1"]
    log(f"{name}{' ' + rev if rev else ''} block {b} ({N} positions, {m} atoms, rows used as words {out['share_rows_used']:.2f}): directional gain at size 0.1: " + ", ".join(f"{k} {v:+.3f}" for k, v in gs.items()) + " | at size 1.0: " + ", ".join(f"{k} {v:+.3f}" for k, v in out["gain"]["1.0"].items())
        + f" | ratios at 0.1: native/random {r1['native_over_random']}, native/covariance {r1['native_over_covariance']}, used/unused {r1['used_over_unused']}, covariance/random {r1['covariance_over_random']}")
    del Au; torch.cuda.empty_cache()
mid = res["by_block"][NB // 2]["ratios"]["0.1"]; Bk = res["by_block"]; fmt = lambda v: "n/a" if v is None else f"{v:.2f}"
res["checks"] = dict(native_spared_vs_random=mid["native_over_random"] is not None and 0 <= mid["native_over_random"] < 0.7, native_spared_vs_covariance=mid["native_over_covariance"] is not None and 0 <= mid["native_over_covariance"] < 0.85, used_spared_vs_unused=mid["used_over_unused"] is not None and 0 <= mid["used_over_unused"] < 0.85)
summ = (f"{name}{' ' + rev if rev else ''}: by block " + " | ".join(f"{b}: gain at size 0.1 random {o['gain']['0.1']['random']:+.3f}, covariance-matched {o['gain']['0.1']['covariance_matched']:+.3f}, PCA top 64 {o['gain']['0.1']['pca_top64']:+.3f}, MLP rows random {o['gain']['0.1']['mlp_rows_random']:+.3f}, used {o['gain']['0.1']['mlp_rows_used']:+.3f}, unused {o['gain']['0.1']['mlp_rows_unused']:+.3f}, token embeddings {o['gain']['0.1']['token_embeddings']:+.3f}, head bases {o['gain']['0.1']['head_bases']:+.3f}; at size 1.0 random {o['gain']['1.0']['random']:+.3f}, covariance-matched {o['gain']['1.0']['covariance_matched']:+.3f}, MLP rows random {o['gain']['1.0']['mlp_rows_random']:+.3f}, used {o['gain']['1.0']['mlp_rows_used']:+.3f}; ratios at 0.1 native/random {fmt(o['ratios']['0.1']['native_over_random'])}, native/covariance {fmt(o['ratios']['0.1']['native_over_covariance'])}, used/unused {fmt(o['ratios']['0.1']['used_over_unused'])}, covariance/random {fmt(o['ratios']['0.1']['covariance_over_random'])}" for b, o in Bk.items())
        + f" | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e510_anisotropy_{name}{'_' + rev if rev else ''}", res, summ)
