"""e442: self-description across the light cone. sd_common replaces every position's state at once, so loss recovered
mixes two paths:
- self: what a position's state does for its own next-token prediction;
- broadcast: what later positions read from it through attention.
e438 found that half of what the huge directions do goes through attention patterns. Here the 16-word descriptions
(own words and their rotation, OMP over the dictionary of blocks 0..L) and the mean state are spliced at a random tenth
of the typical positions per pass (10 disjoint passes covering them all; the rest exact; sinks exact).
- self damage = loss change at the replaced positions minus the change at the others in the same pass;
- broadcast damage = the change at the other positions, divided by the replaced fraction (per unit of context
  replaced).
Recovery on each path = 1 - damage(description) / damage(mean state). The all-positions splice checks additivity.
Pre-registered: the own-over-rotation advantage is larger on the broadcast path than on the self path in at least three
of five models."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; model, tok, fam = load_model(name); arch = Arch(model, fam); L = arch.NB // 2; D = arch.D
ev = eval_ids(name)[:4].to(DEV); B, T = ev.shape
xe = block_states(model, arch, ev, [L], chunk=2)[L]; ne_ = ~sinkmask(xe); X = xe[ne_]; mu = X.mean(0); Xc = X - mu
A, lab = build_dictionary(arch, blocks=list(range(L + 1)))
desc = {}
for vn, V in (("own", A), ("rot", rotate(A, seed=7))):
    sel, _, _ = omp(Xc, V, 16, batch=256, record_err=False); cof, _ = refit(Xc, V, sel); desc[vn] = mu + torch.einsum("nk,nkd->nd", cof, V[sel]); del sel, cof
desc["mean"] = mu[None].expand_as(X); del A
sp = Splicer(model, arch, ev, L); J = 10; g = torch.Generator(device=DEV).manual_seed(0)
tidx = torch.nonzero(ne_); perm = tidx[torch.randperm(tidx.shape[0], device=DEV, generator=g)]; parts = perm.chunk(J)
pos_of = torch.full((B, T - 1), -1, dtype=torch.long, device=DEV); pos_of[ne_] = torch.arange(X.shape[0], device=DEV)
acc = {vn: dict(self_=0.0, bc=0.0) for vn in desc}; f = 1.0 / J
for pj in parts:
    m = torch.zeros(B, T - 1, dtype=torch.bool, device=DEV); m[pj[:, 0], pj[:, 1]] = True
    lin, lout = sp.lossmask(m), sp.lossmask(ne_ & ~m)
    for vn, Xd in desc.items():
        Xn = xe.clone(); Xn[m] = Xd[pos_of[m]]; d = sp.run(Xn)["loss"] - sp.c["loss"]
        acc[vn]["self_"] += (d[lin].mean() - d[lout].mean()).item() / J; acc[vn]["bc"] += d[lout].mean().item() / (f * J)
alls = {}
for vn, Xd in desc.items():
    Xn = xe.clone(); Xn[ne_] = Xd; alls[vn] = (sp.run(Xn)["loss"] - sp.c["loss"])[sp.lossmask(ne_)].mean().item()
rec = lambda vn, k: 1 - acc[vn][k] / max(acc["mean"][k], 1e-9)
res = dict(model=name, level=L, damage=acc, all_positions=alls, recovery={vn: dict(self_=rec(vn, "self_"), broadcast=rec(vn, "bc"), all=1 - alls[vn] / max(alls["mean"], 1e-9)) for vn in ("own", "rot")})
Rr = res["recovery"]; adv_self, adv_bc = Rr["own"]["self_"] - Rr["rot"]["self_"], Rr["own"]["broadcast"] - Rr["rot"]["broadcast"]
res["advantage"] = dict(self_=adv_self, broadcast=adv_bc, all=Rr["own"]["all"] - Rr["rot"]["all"]); res["checks"] = dict(broadcast_advantage_larger=adv_bc > adv_self)
summ = (f"{name} L{L}: mean-state damage self {acc['mean']['self_']:+.3f} nats, broadcast per unit context {acc['mean']['bc']:+.3f} (sum {acc['mean']['self_'] + acc['mean']['bc']:+.3f} vs all-positions {alls['mean']:+.3f}) | "
        f"recovery self own {Rr['own']['self_']:.2f} rot {Rr['rot']['self_']:.2f}, broadcast own {Rr['own']['broadcast']:.2f} rot {Rr['rot']['broadcast']:.2f}, all own {Rr['own']['all']:.2f} rot {Rr['rot']['all']:.2f} | "
        f"own-over-rotation advantage self {adv_self:+.2f} broadcast {adv_bc:+.2f} | checks {json.dumps(res['checks'])}")
log(summ); record(f"e442_lightcone_{name}", res, summ)
