"""e508: why does the chord predict the maximum better than the profile? e506 found the centred chord's maximum
within 1-13% of the observed in GPT-2 and its argmax the winning atom in a third to a half of states; e507 found the
same chord only a partial predictor of the whole profile (calibrated at 0.75-0.87 on its own top 200, R^2 0.54-0.66).
One explanation needs no structure: every projection is chord plus residual, and the winner is selected where the
residual happened to help, so conditioning on winning makes the chord look better than it is marginally. That is an
extreme-value selection effect, and it has a null: keep the chord, make the residual exchangeable across atoms, and
see whether the winner's identity, the maximum's level and the winner's residual come out the same.
Per typical state: p the centred chord's projection on every atom, o the observed, r = (o - p) sign(p) the residual
along the chord's sign (positive reinforces). Observed: the winner (argmax |o|), its chord rank (rank of |p| among
all atoms), the winner's standardised residual among the chord's top 200, and the maximum's level over the chord's.
Null: r permuted across atoms, once over all atoms and once within the chord's top 200 (the rest kept), the winner
of |p| + r_perm and the same quantities; four permutations. Reported per block: the share of states whose winner has
chord rank 1, at most 10, at most 100 and at most 1000, observed and under both nulls; the per-atom probability of
winning by chord-rank bin (1, 2-10, 11-100, 101-1000, beyond) observed and null; the median chord rank of the winner;
the winner's standardised residual observed and null; the observed maximum over the null maximum; and, for the
chord's top atom, its observed over predicted projection against the same at the winner.
Setup: a model at a checkpoint (arguments), blocks NB/4, NB/2, 3NB/4; 8 x 256 evaluation tokens, typical positions.
Pre-registered (honest guesses):
- the exchangeable-residual null (within the top 200) reproduces the observed share of winners at chord rank 1
  within 0.1 in GPT-2 at the middle block (0.5);
- the null maximum is within 10% of the observed at GPT-2's middle block (0.5);
- the winner's standardised residual is positive and above the null's in GPT-2 (the residual specifically reinforces
  the winner beyond selection) (0.4).
Arguments: name [revision]."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; rev = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] not in ("none", "main") else None
model, tok, fam = load_model(name, revision=rev); arch = Arch(model, fam); NB = arch.NB; D = arch.D; DFF = arch.DFF; NW = 64; TOPS = 200; NPERM = 4
blocks = sorted({NB // 4, NB // 2, (3 * NB) // 4}); LB = max(blocks); ids = eval_ids(name)[:8, :256].to(DEV); B_, T_ = ids.shape
acts = {b: [] for b in range(LB + 1)}
def mk(b):
    def pre(m, a): acts[b].append(a[0].detach().float()[:, 1:].reshape(-1, DFF)); return None
    return pre
hs = [arch.mlp_lin(b).register_forward_pre_hook(mk(b)) for b in range(LB + 1)]
try: S_ = block_states(model, arch, ids, blocks, chunk=4)
finally: [h.remove() for h in hs]
A_all = {b: torch.cat(acts[b]) for b in range(LB + 1)}; del acts; rown = {b: arch.wdir(b).float().norm(dim=-1) for b in range(LB + 1)}
tokens = ids[:, 1:].reshape(-1); positions = torch.arange(1, T_, device=DEV).repeat(B_); has_pos = len(arch.emb) > 1
g = torch.Generator(device=DEV).manual_seed(0); BINS = [(1, 1), (2, 10), (11, 100), (101, 1000)]
res = dict(model=name, revision=rev, blocks=blocks, by_block={})
for b in blocks:
    X = S_[b].reshape(-1, D); keep = ~sinkmask(X); Xk = X[keep]; Xc = Xk - Xk.mean(0); U = unitr(Xc); xn = Xc.norm(dim=-1); N = U.shape[0]
    A, lab = build_dictionary(arch, blocks=list(range(b + 1))); anorm = A.float().norm(dim=-1); Au = unitr(A); del A; m = Au.shape[0]
    typ = lab["type"].to(DEV); blk = lab["block"].to(DEV); idx = lab["index"].to(DEV)
    gidx = torch.full((b + 1, DFF), -1, device=DEV, dtype=torch.long)
    for bb in range(b + 1): mm = (typ == T_MLP) & (blk == bb); gidx[bb, idx[mm]] = torch.nonzero(mm)[:, 0]
    Cled = torch.cat([A_all[bb][keep] * rown[bb][None] for bb in range(b + 1)], 1); top = Cled.abs().topk(NW, dim=1); writers = gidx.reshape(-1)[top.indices]; wcoef = Cled.gather(1, top.indices)
    mt = typ == T_TOK; tok_lut = torch.full((int(idx[mt].max()) + 1,), -1, device=DEV, dtype=torch.long); tok_lut[idx[mt]] = torch.nonzero(mt)[:, 0]; tatom = tok_lut[tokens[keep]]
    writers = torch.cat([writers, tatom[:, None]], 1); wcoef = torch.cat([wcoef, anorm[tatom][:, None]], 1)
    if has_pos:
        mp = typ == T_POS; pos_lut = torch.full((int(idx[mp].max()) + 1,), -1, device=DEV, dtype=torch.long); pos_lut[idx[mp]] = torch.nonzero(mp)[:, 0]; patom = pos_lut[positions[keep]]
        writers = torch.cat([writers, patom[:, None]], 1); wcoef = torch.cat([wcoef, anorm[patom][:, None]], 1)
    assert bool((writers >= 0).all())
    W64r = torch.einsum("nk,nkd->nd", wcoef, Au[writers]); Pch = (W64r - W64r.mean(0)) / xn[:, None]
    acc = {k: [] for k in ("rank_obs", "rank_null_all", "rank_null_top", "z_obs", "z_null_top", "max_obs", "max_null_all", "max_null_top", "ratio_top_atom", "ratio_winner", "pmax")}
    for s in range(0, N, 128):
        O = (U[s:s + 128] @ Au.T); P = (Pch[s:s + 128] @ Au.T); Pa = P.abs(); Oa = O.abs(); r = (O - P) * torch.sign(P); n = O.shape[0]
        prank = Pa.argsort(1, descending=True).argsort(1) + 1                                                              # chord rank of every atom, 1 = largest
        win = Oa.argmax(1); acc["rank_obs"].append(prank.gather(1, win[:, None])[:, 0]); acc["max_obs"].append(Oa.max(1).values); acc["pmax"].append(Pa.max(1).values)
        topset = Pa.topk(TOPS, 1).indices; rt = r.gather(1, topset); mu, sd = rt.mean(1, keepdim=True), rt.std(1, keepdim=True).clamp_min(1e-9)
        acc["z_obs"].append(((r.gather(1, win[:, None]) - mu) / sd)[:, 0])
        ptop = Pa.argmax(1); acc["ratio_top_atom"].append(Oa.gather(1, ptop[:, None])[:, 0] / Pa.gather(1, ptop[:, None])[:, 0].clamp_min(1e-9)); acc["ratio_winner"].append(Oa.gather(1, win[:, None])[:, 0] / Pa.gather(1, win[:, None])[:, 0].clamp_min(1e-9))
        for _ in range(NPERM):
            perm = torch.argsort(torch.rand(n, m, generator=g, device=DEV), dim=1); rp = r.gather(1, perm); On = Pa + rp; wn = On.argmax(1)
            acc["rank_null_all"].append(prank.gather(1, wn[:, None])[:, 0]); acc["max_null_all"].append(On.max(1).values)
            permt = torch.argsort(torch.rand(n, TOPS, generator=g, device=DEV), dim=1); r2 = r.clone(); r2.scatter_(1, topset, rt.gather(1, permt)); On2 = Pa + r2; wn2 = On2.argmax(1)
            acc["rank_null_top"].append(prank.gather(1, wn2[:, None])[:, 0]); acc["max_null_top"].append(On2.max(1).values); acc["z_null_top"].append(((r2.gather(1, wn2[:, None]) - mu) / sd)[:, 0])
    ac = {k: torch.cat(v) for k, v in acc.items()}
    def shares(rk): return dict(rank1=float((rk == 1).float().mean()), le10=float((rk <= 10).float().mean()), le100=float((rk <= 100).float().mean()), le1000=float((rk <= 1000).float().mean()), median_rank=float(rk.float().median()))
    def per_atom(rk):
        out = {}
        for lo, hi in BINS: out[f"{lo}-{hi}"] = float(((rk >= lo) & (rk <= hi)).float().mean() / (hi - lo + 1))
        out["beyond"] = float((rk > 1000).float().mean() / (m - 1000)); return out
    o = dict(n_states=N, n_atoms=m, observed=shares(ac["rank_obs"]), null_all_atoms=shares(ac["rank_null_all"]), null_top200=shares(ac["rank_null_top"]),
             per_atom_win_probability=dict(observed=per_atom(ac["rank_obs"]), null_top200=per_atom(ac["rank_null_top"]), floor_only=1 / m),
             winner_residual_z=dict(observed=float(ac["z_obs"].median()), null_top200=float(ac["z_null_top"].median())),
             max_observed_over_null=dict(all_atoms=float(ac["max_obs"].mean() / ac["max_null_all"].mean()), top200=float(ac["max_obs"].mean() / ac["max_null_top"].mean())), max_observed_over_chord_max=float(ac["max_obs"].mean() / ac["pmax"].mean()),
             observed_over_predicted=dict(at_chord_top_atom=float(ac["ratio_top_atom"].median()), at_winner=float(ac["ratio_winner"].median())))
    res["by_block"][b] = o
    log(f"{name}{' ' + rev if rev else ''} block {b} ({N} states, {m} atoms): winner at chord rank 1 / within 10 / within 100 / within 1000: observed {o['observed']['rank1']:.2f}/{o['observed']['le10']:.2f}/{o['observed']['le100']:.2f}/{o['observed']['le1000']:.2f} (median rank {o['observed']['median_rank']:.0f}), null within the top 200 {o['null_top200']['rank1']:.2f}/{o['null_top200']['le10']:.2f}/{o['null_top200']['le100']:.2f}/{o['null_top200']['le1000']:.2f} (median {o['null_top200']['median_rank']:.0f}), null over all atoms {o['null_all_atoms']['rank1']:.2f}/{o['null_all_atoms']['le10']:.2f}/{o['null_all_atoms']['le100']:.2f}/{o['null_all_atoms']['le1000']:.2f} | per-atom win probability by chord rank, observed / null: " + ", ".join(f"{k} {v:.4f}/{o['per_atom_win_probability']['null_top200'][k]:.4f}" for k, v in o["per_atom_win_probability"]["observed"].items()) + f" (floor only {1 / m:.6f}) | winner's standardised residual observed {o['winner_residual_z']['observed']:+.2f}, null {o['winner_residual_z']['null_top200']:+.2f} | observed max over null max {o['max_observed_over_null']['top200']:.2f} (all-atom null {o['max_observed_over_null']['all_atoms']:.2f}), over the chord's max {o['max_observed_over_chord_max']:.2f} | observed over predicted at the chord's top atom {o['observed_over_predicted']['at_chord_top_atom']:.2f}, at the winner {o['observed_over_predicted']['at_winner']:.2f}")
    del Au, Cled, W64r, Pch; torch.cuda.empty_cache()
mid = res["by_block"][NB // 2]; Bk = res["by_block"]
res["checks"] = dict(null_reproduces_rank1_within_0_1=abs(mid["observed"]["rank1"] - mid["null_top200"]["rank1"]) < 0.1, null_max_within_10pct=0.9 <= mid["max_observed_over_null"]["top200"] <= 1.1,
                     winner_residual_reinforces_beyond_null=mid["winner_residual_z"]["observed"] > 0 and mid["winner_residual_z"]["observed"] > mid["winner_residual_z"]["null_top200"])
summ = (f"{name}{' ' + rev if rev else ''}: by block " + " | ".join(f"{b}: winner at chord rank 1 / within 10 / within 100 observed {o['observed']['rank1']:.2f}/{o['observed']['le10']:.2f}/{o['observed']['le100']:.2f}, null (residual permuted within the chord's top 200) {o['null_top200']['rank1']:.2f}/{o['null_top200']['le10']:.2f}/{o['null_top200']['le100']:.2f}, null (permuted over all atoms) {o['null_all_atoms']['rank1']:.2f}/{o['null_all_atoms']['le10']:.2f}/{o['null_all_atoms']['le100']:.2f}; winner's standardised residual observed {o['winner_residual_z']['observed']:+.2f} vs null {o['winner_residual_z']['null_top200']:+.2f}; observed max over null max {o['max_observed_over_null']['top200']:.2f}, over the chord's max {o['max_observed_over_chord_max']:.2f}; observed over predicted at the chord's top atom {o['observed_over_predicted']['at_chord_top_atom']:.2f} vs at the winner {o['observed_over_predicted']['at_winner']:.2f}" for b, o in Bk.items())
        + f" | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e508_chordwins_{name}{'_' + rev if rev else ''}", res, summ)
