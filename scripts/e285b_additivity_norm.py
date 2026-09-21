"""e285b: the additivity prediction of the transport theory, designed cleanly. Single images of 16 random unit
directions are measured at the natural amplitude s (one pass, directions spread over the tokens); the linear
prediction for any combination is the matching linear combination of these. Three families of injections of the
same random directions: (A) m directions each at s (total norm sqrt(m) s), (B) m directions each at s/sqrt(m)
(total norm s), (C) one direction at sqrt(m) s (total norm sqrt(m) s, m = 1). The theory says the error against
the linear prediction is a function of the total norm, not of the number of superposed writes: error_A(m) should
match error_C(m), and error_B(m) should stay at the m = 1 floor."""
import sys, os, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from desc_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 12; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; b = 2; D = c.D; run = make_runner(model, arch, c, ids_seq, [L], NT)
led, tn, tc = dominant(model, arch, c, ids_seq, b); S0 = run(); typ = typical_mask(S0[L]); pool = torch.nonzero(typ)[:, 0]; s = tc.abs().median(); torch.manual_seed(2); W16 = unit(torch.randn(16, D, device=DEV))
def image(vec, amp):
    inj = torch.zeros(NT, D, device=DEV); inj[pool] = amp * vec[None]; S2 = run(inject=inj, inject_block=b + 1); return (S2[L] - S0[L])[pool].mean(0) / amp
a = torch.randint(0, 16, (len(pool),), device=DEV); inj = torch.zeros(NT, D, device=DEV); inj[pool] = s * W16[a]; S2 = run(inject=inj, inject_block=b + 1); img = (S2[L] - S0[L])[pool]; single = torch.stack([img[a == q].mean(0) for q in range(16)]) / s
err = lambda act, lin: ((act - lin).norm() / act.norm()).item(); out = {}
for m in (1, 2, 4, 8, 16):
    v = W16[:m].sum(0); lin = single[:m].sum(0); actA = image(v, s); actB = image(v, s / math.sqrt(m)); actC = image(W16[0], math.sqrt(m) * s)
    out[m] = dict(total_norm_A=(s * v.norm() / s).item(), A_each_at_s=err(actA, lin), B_constant_total=err(actB, lin), C_one_at_sqrt_m=err(actC, single[0]), cosA=(unit(actA[None]) @ unit(lin[None]).T).item(), cosB=(unit(actB[None]) @ unit(lin[None]).T).item(), cosC=(unit(actC[None]) @ unit(single[0][None]).T).item())
log(f"{tag}: relative error against the linear prediction, m = 1/2/4/8/16: (A) m writes each at s, total norm growing: " + "/".join(f"{out[m]['A_each_at_s']:.2f}" for m in out) + " | (B) m writes at constant total norm s: " + "/".join(f"{out[m]['B_constant_total']:.2f}" for m in out) + " | (C) one write at the same total norm as A: " + "/".join(f"{out[m]['C_one_at_sqrt_m']:.2f}" for m in out) + " | cosines A " + "/".join(f"{out[m]['cosA']:.2f}" for m in out) + " B " + "/".join(f"{out[m]['cosB']:.2f}" for m in out) + " C " + "/".join(f"{out[m]['cosC']:.2f}" for m in out))
record(f"e285b_additivity_{tag}", dict(model=tag, b=b, L=L, per_m={str(k): v for k, v in out.items()}), "error A (growing norm, m writes) " + "/".join(f"{out[m]['A_each_at_s']:.2f}" for m in out) + " | B (constant norm, m writes) " + "/".join(f"{out[m]['B_constant_total']:.2f}" for m in out) + " | C (growing norm, one write) " + "/".join(f"{out[m]['C_one_at_sqrt_m']:.2f}" for m in out))
