"""e408: a functional description without activations. e400 chose words under the network's Fisher metric, which needs
activations and gradients. Can the weights alone say which directions of the state matter? The reader metric: the
sum over every module that reads the middle-depth state downstream (each later block's attention input and MLP
input, and the unembedding) of the Gram matrix of its read directions (norm gain folded in, centred), one vote per
module; also the unembedding alone (the direct logit readout). Pythia-410m (argument: revision), middle depth, 8
sequences: the share of each metric's trace on the states' top-8 principal directions (0.015 for the Fisher metric at
the end, e400), the metrics' agreement (normalised Frobenius cosine), and words chosen under the Euclidean, reader,
unembedding and Fisher metrics for the own vocabulary, a rotation of it, and Gaussian words with the states' covariance.
Pre-registered: (1) the reader metric puts far less of its trace on the huge directions than their variance share;
(2) the reader-metric pursuit lifts the own words at k 4 and 16 by at least half of what the Fisher pursuit adds;
(3) the own-minus-rotation gap under the reader metric is at least half its Euclidean size."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sd_common import *
rev = sys.argv[1] if len(sys.argv) > 1 else "step143000"
c = Cache("pythia410"); ev = c.s["eval_ids"][:8].to(DEV); fit = c.s["eval_ids"][8:16].to(DEV); KS = [4, 8, 16, 32, 64]
model, tok, fam = load_model("pythia410", revision=rev); arch = Arch(model, fam); L = arch.NB // 2; own, _ = build_dictionary(arch, blocks=list(range(L + 1)))
GF = fisher_gram(model, arch, fit, L); GR, GU = reader_gram(model, arch, list(range(L + 1, arch.NB)))
lv = Level(model, arch, ev, L); fl = Level(model, arch, fit, L); SX = (fl.Xc.T @ fl.Xc) / fl.Xc.shape[0]; del fl
evX, UX = torch.linalg.eigh(SX.double()); UM = UX[:, -8:].float()
cosF = lambda A_, B_: ((A_ / A_.trace()) * (B_ / B_.trace())).sum().item() / ((A_ / A_.trace()).norm() * (B_ / B_.trace()).norm()).item()
res = dict(rev=rev, level=L, k=KS, gap=lv.gap, var_share_top8=(evX[-8:].sum() / evX.sum()).item(),
           trace_share_top8=dict(fisher=share_on(GF, UM), readers=share_on(GR, UM), unembed=share_on(GU, UM)),
           metric_cos=dict(readers_fisher=cosF(GR, GF), unembed_fisher=cosF(GU, GF), readers_unembed=cosF(GR, GU)), cells={})
log(f"{rev} top-8 PCs: variance {res['var_share_top8']:.2f}; trace share fisher {res['trace_share_top8']['fisher']:.3f} readers {res['trace_share_top8']['readers']:.3f} unembed {res['trace_share_top8']['unembed']:.3f}; "
    f"metric cosines readers-fisher {res['metric_cos']['readers_fisher']:.2f} unembed-fisher {res['metric_cos']['unembed_fisher']:.2f}")
S = dict(euc=None, readers=metric_sqrt(GR), unembed=metric_sqrt(GU), fisher=metric_sqrt(GF))
for nm, V in [("own", own), ("rot7", rotate(own, seed=7)), ("covX", gauss_like(own.shape[0], SX, seed=2))]:
    for mt, Sm in S.items():
        res["cells"][f"{nm}:{mt}"] = describe(lv, V, KS, S=Sm); log(f"{rev} {nm} {mt}: rec k4..64 {fmt(res['cells'][f'{nm}:{mt}'], KS)}")
g = lambda n, k=16: res["cells"][n][str(k)]["rec"]
lift = lambda mt, k: (g(f"own:{mt}", k) - g("own:euc", k)) / max(g("own:fisher", k) - g("own:euc", k), 1e-9)
res["reader_lift_share"] = {k: lift("readers", k) for k in (4, 16)}; res["unembed_lift_share"] = {k: lift("unembed", k) for k in (4, 16)}
summ = (f"{rev} k16 own/rot/covX: " + " | ".join(f"{mt} {g(f'own:{mt}'):.2f}/{g(f'rot7:{mt}'):.2f}/{g(f'covX:{mt}'):.2f}" for mt in S)
        + f" | reader-metric share of the Fisher lift k4 {res['reader_lift_share'][4]:.2f} k16 {res['reader_lift_share'][16]:.2f}; unembed {res['unembed_lift_share'][4]:.2f}/{res['unembed_lift_share'][16]:.2f}"
        + f" | top-8 trace shares F {res['trace_share_top8']['fisher']:.3f} R {res['trace_share_top8']['readers']:.3f} U {res['trace_share_top8']['unembed']:.3f} (variance {res['var_share_top8']:.2f})")
log(summ); record(f"e408_weightfunc_pythia410_{rev}", res, summ)
