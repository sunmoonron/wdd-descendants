"""e330: does equivalence survive amplitude? Pairs of atoms equivalent at the natural amplitude (from 256 exact
coordinates) are re-injected at 4x and at 8x; the pair cosine of their coordinates and of their logit effects at each
amplitude, against random pairs. Equivalence that breaks with amplitude marks the quotient as a linear-regime object."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quot_common import *
tag = sys.argv[1]; S = Setup(tag, levels=[0]); L = S.L; S = Setup(tag, levels=[L]); d = 16; nat = S.natural(); F = nat["F"][L]; tr, te = S.halves(len(S.idx)); Fc = F - F[tr].mean(0, keepdim=True); Z, Bv = logit_scores(nat["dl"], tr); P = pls(Fc[tr], Z[tr], d)
torch.manual_seed(0); atoms = torch.randperm(S.DFF, device=DEV)[:256]; V = S.W2[atoms]; out = {}; pairs = None; rnd = None
for al in (1.0, 4.0, 8.0):
    img = torch.zeros(256, S.D, device=DEV); Fl = torch.zeros(256, Bv.shape[1], device=DEV); cnt = torch.zeros(256, device=DEV)
    for p in range(3):
        r = S.inject_family(V, S.b + 1, amp=al * S.s_inj, seed=p); img.index_add_(0, r["a"], r["F"][L]); Fl.index_add_(0, r["a"], r["dl"] @ Bv); cnt.index_add_(0, r["a"], torch.ones(len(r["a"]), device=DEV))
    img = img / cnt[:, None].clamp_min(1); Fl = Fl / cnt[:, None].clamp_min(1); z = img @ P
    if pairs is None:
        zc = unit(z) @ unit(z).T; pc = (V @ V.T).abs(); zc.fill_diagonal_(-1); pairs = [(i, int(zc[i].argmax())) for i in range(256) if zc[i].max() >= zc.max(1).values.quantile(0.9) and pc[i, int(zc[i].argmax())] < 0.3]; rnd = [(i, int(torch.randint(0, 256, (1,)).item())) for i in range(len(pairs))]
    out[al] = dict(coordinate_equivalent=float(torch.tensor([(unit(z[i][None]) @ unit(z[j][None]).T).item() for i, j in pairs]).median()), coordinate_random=float(torch.tensor([(unit(z[i][None]) @ unit(z[j][None]).T).item() for i, j in rnd]).median()), effect_equivalent=float(torch.tensor([(unit(Fl[i][None]) @ unit(Fl[j][None]).T).item() for i, j in pairs]).median()), effect_random=float(torch.tensor([(unit(Fl[i][None]) @ unit(Fl[j][None]).T).item() for i, j in rnd]).median()))
log(f"{tag} ({len(pairs)} pairs equivalent at 1x): amplitude -> coordinate cos equivalent/random, effect cos equivalent/random :: " + " ; ".join(f"{al}x: {v['coordinate_equivalent']:.2f}/{v['coordinate_random']:.2f}, {v['effect_equivalent']:.2f}/{v['effect_random']:.2f}" for al, v in out.items()))
record(f"e330_ampequiv_{tag}", dict(model=tag, L=L, n_pairs=len(pairs), per_amplitude={str(k): v for k, v in out.items()}), " ".join(f"{al}x:{v['coordinate_equivalent']:.2f}/{v['coordinate_random']:.2f},{v['effect_equivalent']:.2f}/{v['effect_random']:.2f}" for al, v in out.items()))
