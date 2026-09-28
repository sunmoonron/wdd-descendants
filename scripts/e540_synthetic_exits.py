"""e540: does the one-way gate appear without training? e538b found entries to be crossings of the floor and exits
to be losses of usage with the projection kept; e539 reads the competition in Pythia. Here the same analysis runs in
e537's synthetic worlds (8192 unit rows diffusing on the sphere against a fixed Gaussian cloud with Pythia-410m's
block-12 covariance at step 8000, sixteen steps): isotropic diffusion (no memory, no pull); the pull along the cloud's
covariance (0.6 sigma, no memory); and memory with the pull (the world closest to Pythia). Recorded per world:
e538b's profiles, hysteresis, the own-versus-cutoff decomposition at lost and gained positions, and the takeover
audit, exactly as in e539.
Pre-registered (honest guesses):
- isotropic diffusion shows no gate: its leavers' median S at k=0 is under the floor and fewer than half are over it (0.6);
- the pull gives the gate: leavers over the floor at k=0 in 0.7 or more, S changing by less than 0.1 across an exit (0.6);
- in the pull world the exits are displacements, the cutoff moving more than the row's own projection at most lost positions (0.5);
- the takeover is as diffuse as in Pythia and the takers are at chance cosine to the leaver in every world (0.7).
Arguments: name."""
import sys, os, json as _json, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ex_common import *
from lr_common import eval_ids
name = sys.argv[1]; B = 12; K = 16; T = 16; R = 8192; N = 2000; COS_STEP = 0.96
idsB = eval_ids(name)[:8, :256].to(DEV); gen = torch.Generator(device=DEV).manual_seed(0)
model, tok, fam = load_model(name, revision="step8000"); arch = Arch(model, fam); D = arch.D
for p in model.parameters(): p.requires_grad_(False)
X = block_states(model, arch, idsB, [B], chunk=4)[B].reshape(-1, D); keep = ~sinkmask(X); Xk = X[keep]; Xc8 = Xk - Xk.mean(0); C = (Xc8.T @ Xc8 / Xc8.shape[0]).double(); del model, X; torch.cuda.empty_cache()
ev, V = torch.linalg.eigh(C); ev = ev.clamp_min(0); Cn = (C / ev.max()).float()
cloud = unitr(((torch.randn(N, D, device=DEV, generator=gen, dtype=torch.float64) * ev.sqrt()[None]) @ V.T).float()); sigma = math.sqrt(2 * (1 - COS_STEP))
def noise(g): return unitr(torch.randn(R, D, device=DEV, generator=g))
WORLDS = {"isotropic": dict(pull=0.0, rho=0.0), "pull": dict(pull=0.6, rho=0.0), "memory_pull": dict(pull=0.6, rho=0.3)}
res = dict(model=name, worlds={})
for wname, cfg in WORLDS.items():
    g = torch.Generator(device=DEV).manual_seed(1); W = unitr(torch.randn(R, D, device=DEV, generator=g)); d = noise(g); H = []; Ws = []
    for t in range(T):
        H.append(stats(cloud, W, K)); Ws.append(W.cpu())
        eta = noise(g); d = unitr(cfg["rho"] * d + math.sqrt(1 - cfg["rho"] ** 2) * eta); step = d - (d * W).sum(1, keepdim=True) * W; step = unitr(step) * sigma
        if cfg["pull"]: pull = W @ Cn; pull = pull - (pull * W).sum(1, keepdim=True) * W; step = step + cfg["pull"] * sigma * unitr(pull)
        W = unitr(W + step)
    res["worlds"][wname] = full_analysis(H, lambda t, Ws=Ws: Ws[t], f"world {wname}", name); del H; torch.cuda.empty_cache()
g_ = lambda x: -9 if x is None else x; iso, pl = res["worlds"]["isotropic"], res["worlds"]["pull"]
res["checks"] = dict(isotropic_no_gate=g_(iso["profiles"]["exits_aligned"][0]["S"]) < 1.0 and g_(iso["profiles"]["fractions"]["leavers_over_floor_k0"]) < 0.5, pull_gate=g_(pl["profiles"]["fractions"]["leavers_over_floor_k0"]) >= 0.7 and abs(g_(pl["profiles"]["delta_S_exit"])) < 0.1,
                     pull_exit_is_displacement=g_(pl["decomposition"]["exits"]["cutoff_moved_more"]) >= 0.5, takers_at_chance_everywhere=all(abs(g_(o["takeover"]["takers"]["cos_to_leaver"])) < 0.1 for o in res["worlds"].values()))
f2 = lambda x: "n/a" if x is None else f"{x:.2f}"
summ = f"{name}: " + " | ".join(f"{w}: leavers' S at k=-1/0/+2 {f2(o['profiles']['exits_aligned'][-1]['S'])}/{f2(o['profiles']['exits_aligned'][0]['S'])}/{f2(o['profiles']['exits_aligned'][2]['S'])}, over the floor at k=0 {f2(o['profiles']['fractions']['leavers_over_floor_k0'])}, S change across exits {f2(o['profiles']['delta_S_exit'])} / entries {f2(o['profiles']['delta_S_entry'])}, entrants' S at -1/0/+2 {f2(o['profiles']['entries_aligned'][-1]['S'])}/{f2(o['profiles']['entries_aligned'][0]['S'])}/{f2(o['profiles']['entries_aligned'][2]['S'])}; at lost positions own {f2(o['decomposition']['exits']['own_before'])}->{f2(o['decomposition']['exits']['own_after'])} against cutoff {f2(o['decomposition']['exits']['cut_before'])}->{f2(o['decomposition']['exits']['cut_after'])} (cutoff moved more at {f2(o['decomposition']['exits']['cutoff_moved_more'])}); takers: effective over lost {f2(o['takeover']['takers']['eff_over_lost'])}, established/entrant/non-word {f2(o['takeover']['takers']['share_established'])}/{f2(o['takeover']['takers']['share_entrant'])}/{f2(o['takeover']['takers']['share_nonword'])}, cosine {f2(o['takeover']['takers']['cos_to_leaver'])}; hysteresis matched difference at 1.1-1.2 {f2(o['hysteresis']['1.1-1.2']['matched_difference'])}" for w, o in res["worlds"].items()) + f" | checks {_json.dumps(res['checks'])}"
log(summ); record(f"e540_synthetic_exits_{name}", res, summ)
