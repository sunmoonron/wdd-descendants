"""e701 (sessions 157 and 158): when the brake forms. e695 found the crowd's net sum opposing a neuron's sixteen largest
inputs off its class and not at its core, in the final models. Here the same net ledger is taken at one block across a
model's checkpoints, for one fixed sample of the final model's neurons: at each checkpoint the neuron's class is its own
top 1 percent, and the measures are the sixteen largest inputs' share of the net drive at the core and in the bottom half
of positions, the crowd's net sum over those inputs there (signed), the share of the 1 to 5 percent shell adjacent to the
core in ledger space, and the one-token share of the top twenty firings. Session 157 ran Pythia-160m (three seeds, block
6); session 158 adds OLMo-1B-0724 (SwiGLU, blocks 4, 8 and 12). Arguments: --model M --block B [--smoke]."""
import argparse
from c11 import *
F_ = torch.nn.functional
ap = argparse.ArgumentParser(); ap.add_argument("--model", default="pythia160"); ap.add_argument("--block", type=int, default=6); ap.add_argument("--smoke", action="store_true"); A_ = ap.parse_args(); SMOKE = A_.smoke
name = A_.model; B = A_.block; TAG = f"e701_{name}_b{B}" + ("_smoke" if SMOKE else ""); T = 256; L_ = lambda s: log(TAG, s)
PY = {"pythia160": [1000, 2000, 3000, 4000, 6000, 8000, 12000, 16000, 32000, 64000, None], "pythia160s1": [1000, 2000, 4000, 8000, 16000, None], "pythia160s2": [1000, 2000, 4000, 8000, 16000, None]}
OL = ["step1000-tokens2B", "step2000-tokens4B", "step4000-tokens8B", "step8000-tokens16B", "step16000-tokens33B", None]
STEPS = (PY[name] if name in PY else OL); STEPS = ([STEPS[0], STEPS[3], None] if SMOKE else STEPS); keyname = lambda s: "final" if s is None else (str(s) if isinstance(s, int) else s.split("-")[0][4:]); stepnum = lambda s: 1454000.0 if (s is None and name.startswith("olmo")) else (143000.0 if s is None else float(keyname(s)))
revname = lambda s: None if s is None else (f"step{s}" if isinstance(s, int) else s)
NWIN = 8 if SMOKE else 128; NNEU = 6 if SMOKE else 100; NSUB = 400
M = load(name); NB, D, DFF = M["NB"], M["D"], M["DFF"]; NATT = n_att_feeding(M, B); ids = windows_from(wikitext_ids(M, "train"), NWIN, T).to(DEV); N = NWIN * (T - 1); NMLP = B * DFF; NU = NMLP + NATT + 1; K1 = max(20, N // 100); GATED = M["fam"] == "gated"
def capture(M):
    UP, AT, EM, XM, AB = [], [], [], [], []
    with Capture(M, list(range(B + 1)), ["act", "att", "xin", "emb"]) as C:
        for s0 in range(0, NWIN, 8):
            run_base(M, ids[s0:s0 + 8])
            UP.append(torch.cat([C.cap[("act", b)][:, 1:].half() for b in range(B)], -1).reshape(-1, NMLP)); AT.append(torch.stack([C.cap[("att", b)][:, 1:].half() for b in range(NATT)], 2).reshape(-1, NATT, D))
            EM.append(C.cap[("emb", 0)][:, 1:].half().reshape(-1, D)); XM.append(C.cap[("xin", B)][:, 1:].float().reshape(-1, D)); AB.append(C.cap[("act", B)][:, 1:].half().reshape(-1, DFF))
    return dict(UP=torch.cat(UP), AT=torch.cat(AT), EM=torch.cat(EM), XM=torch.cat(XM), AB=torch.cat(AB))
def measure(M, NEU, gen):
    S = capture(M); Rd = read_w(M, B); RB = read_bias(M, B); key0 = "h" if "h" in Rd else "g"; RT = read_tilde(M, B, Rd[key0]); WUP = torch.cat([write_w(M, b).T for b in range(B)]); P_ = ln_params(M, B)
    SIG = (S["XM"].var(-1, unbiased=False) + P_["eps"]).sqrt() if P_["centre"] else (S["XM"].pow(2).mean(-1) + P_["eps"]).sqrt(); TOK = ids[:, 1:].reshape(-1); ab = S["AB"].float()
    def contrib(k, P):
        r = RT[k]; V = WUP @ r; s = SIG[P]; return torch.cat([S["UP"][P].float() * V[None] / s[:, None], (S["AT"][P].float() @ r) / s[:, None], ((S["EM"][P].float() @ r) / s)[:, None]], 1)
    def net(Cx): tp = Cx.abs().topk(16, dim=1); t16 = Cx.gather(1, tp.indices).sum(1); tot = Cx.sum(1); return float((t16 / tot.abs().clamp_min(1e-9) * torch.sign(tot)).median()), float(((tot - t16) * torch.sign(t16) / t16.abs().clamp_min(1e-9)).median())
    maxcos = lambda A, Bm: (unitr(A) @ unitr(Bm).T).max(1).values; out = []
    for k in NEU:
        a = ab[:, k]; order = a.argsort(descending=True); core = order[:K1]; s15 = order[K1:int(0.05 * N)]; null = order[int(0.5 * N):][torch.randperm(N - int(0.5 * N), device=DEV, generator=gen)[:NSUB]]
        Cc = contrib(k, core); Cn = contrib(k, null); C1 = contrib(k, s15[torch.randperm(s15.numel(), device=DEV, generator=gen)[:NSUB]])
        thr = float(maxcos(Cn, Cc).quantile(0.95)); adj = float((maxcos(C1, Cc) > thr).float().mean()); n16c, c16c = net(Cc); n16n, c16n = net(Cn); n161, c161 = net(C1)
        one20 = float(torch.bincount(TOK[order[:20]], minlength=M["V"]).max() / 20); onec = float(torch.bincount(TOK[core], minlength=M["V"]).max() / K1)
        out.append(dict(neuron=int(k), net_top16_core=n16c, crowd_over_top16_core=c16c, net_top16_null=n16n, crowd_over_top16_null=c16n, net_top16_s1_5=n161, crowd_over_top16_s1_5=c161, adjacent_s1_5=adj, one_token_top20=one20, one_token_core=onec, act_core=float(a[core].median()), act_max=float(a.max())))
    return out
M0 = M; ab0 = capture(M0)["AB"].float(); mx = ab0.max(0).values; a100 = ab0.topk(100, dim=0).values[-1]; floor_act = 1.0 if not GATED else float(mx.quantile(0.1)); cand = torch.nonzero((mx > floor_act) & (a100 >= 0.25 * mx))[:, 0].tolist(); NEU = sorted(random.Random(685 + B).sample(cand, min(NNEU, len(cand)))); L_(f"{len(NEU)} neurons fixed from the final model; {len(STEPS)} checkpoints")
RES = {}
for step in STEPS:
    gen = torch.Generator(device=DEV).manual_seed(701); Mc = M0 if step is None else load(name, revision=revname(step)); rows = measure(Mc, NEU, gen); key = keyname(step)
    RES[key] = dict(rows=rows, summary={m: med([r[m] for r in rows]) for m in rows[0] if m != "neuron"}); s = RES[key]["summary"]
    L_(f"step {key}: crowd over the 16 wires core {s['crowd_over_top16_core']:+.2f}, shell 1-5 {s['crowd_over_top16_s1_5']:+.2f}, bottom half {s['crowd_over_top16_null']:+.2f}; net top-16 share core {s['net_top16_core']:.2f} null {s['net_top16_null']:.2f}; adjacent 1-5 {s['adjacent_s1_5']:.2f}; one-token top20 {s['one_token_top20']:.2f}")
    if step is not None: del Mc; torch.cuda.empty_cache()
keys = list(RES); steps_num = [stepnum(s) for s in STEPS]; brake = [RES[k]["summary"]["crowd_over_top16_null"] for k in keys]; ncore = [RES[k]["summary"]["net_top16_core"] for k in keys]
def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); rx = x.argsort().argsort(); ry = y.argsort().argsort(); return float(np.corrcoef(rx, ry)[0, 1])
first = next((k for k in keys if RES[k]["summary"]["crowd_over_top16_null"] < -0.5), None)
V = dict(T1_no_brake_at_1000=bool(abs(RES[keys[0]]["summary"]["crowd_over_top16_null"]) < 0.3), T2_brake_by_8000=bool(first is not None and first != "final" and float(first) <= 8000), T3_brake_strengthens_monotonically=bool(spearman(steps_num, brake) < -0.8), T4_core_net_share_rises=bool(spearman(steps_num, ncore) > 0.5), T5_brake_in_final=bool(RES["final"]["summary"]["crowd_over_top16_null"] < -0.5))
res = dict(model=M0["hf"], block=B, steps=keys, n_neurons=len(NEU), checkpoints=RES, first_brake_step=first, spearman_brake=spearman(steps_num, brake), spearman_core_share=spearman(steps_num, ncore), verdicts=V)
summ = (f"[{name} block {B}] {len(NEU)} neurons over {len(keys)} checkpoints: the crowd's net sum over the sixteen wires in the bottom half (signed) " + ", ".join(f"{k} {RES[k]['summary']['crowd_over_top16_null']:+.2f}" for k in keys) + "; at the core " + ", ".join(f"{k} {RES[k]['summary']['crowd_over_top16_core']:+.2f}" for k in keys) +
        "; the sixteen wires' net share at the core " + ", ".join(f"{k} {RES[k]['summary']['net_top16_core']:.2f}" for k in keys) + f"; first checkpoint with the brake under -0.5: {first}; Spearman of the brake with the step {res['spearman_brake']:.2f}; verdicts {V}")
save(TAG, res, summ)
