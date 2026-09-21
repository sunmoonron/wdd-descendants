"""e230: can the descendant signature be read from the STATE, without the intervention? Build a descendant
dictionary at level L: for each neuron dominant at >= 20 tokens in block b (b = 2, 3), the unit centroid of its
causal footprints (from one ablation run) on a training half of its tokens. On the held-out half: (i) nearest-
centroid identification of the source neuron from the CENTERED STATE at L (not the footprint), vs chance; (ii) the
same with the state's projection restricted to the span of the centroids (least squares over the descendant
dictionary, then argmax coefficient); (iii) the native WDD reading of the same neurons at L (dual@64 with the model's
own atoms) for comparison; (iv) the footprint-based identification as the ceiling. Split: a state-level descendant
dictionary exists (state identification well above chance and above native readability) vs the signature is only
visible under intervention."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 16; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; NB = c.NB; lab = c.d["lab"]; A = c.d["A"]
AL, labL = c.dictionary(L); mu = c.s["mu"][L + 1].to(DEV); S = AL.T @ AL; ev, V = torch.linalg.eigh(S); Winv = V @ torch.diag(1 / (ev + 1e-2 * ev[-1])) @ V.T
def run(b=None, neuron=None):
    st = {}; hs = [arch.layers[L].register_forward_hook(lambda m, i, o: st.__setitem__("H", (o[0] if isinstance(o, tuple) else o).detach().float().reshape(-1, c.D)))]
    if b is not None:
        def pre(m, inp):
            x = inp[0].clone(); flat = x.reshape(-1, c.DFF); flat[torch.arange(NT, device=DEV), neuron] = 0; return (flat.reshape(x.shape),)
        hs.append(arch.mlp_lin(b).register_forward_pre_hook(pre))
    model(ids_seq); [h.remove() for h in hs]; return st["H"]
H0 = run(); typ = typical_mask(H0); X = H0 - mu; sel = oneshot(X, AL, 64, whiten=Winv)[0]
def unit(v): return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
torch.manual_seed(0); res = {}
for b in (2, 3):
    st_a = {}; h = arch.mlp_lin(b).register_forward_pre_hook(lambda m, inp: st_a.__setitem__("a", inp[0].detach().float().reshape(-1, c.DFF))); model(ids_seq); h.remove()
    led = st_a["a"] * c.d["WN"][b].to(DEV)[None]; tn = led.abs().argmax(1); tc = torch.gather(led, 1, tn[:, None])[:, 0]; big = tc.abs() >= tc.abs().quantile(0.5); H1 = run(b, tn); F = H0 - H1
    idx = torch.nonzero(typ & big)[:, 0]; neur = tn[idx]; u, cnt = torch.unique(neur, return_counts=True); keep = u[cnt >= 20]
    if len(keep) < 3: continue
    m = torch.isin(neur, keep); idx = idx[m]; neur = neur[m]; lab_i = (neur[:, None] == keep[None, :]).float().argmax(1); split = torch.rand(len(idx), device=DEV) < 0.5; tr, te = idx[split], idx[~split]; ltr, lte = lab_i[split], lab_i[~split]
    cents = unit(torch.stack([F[tr][ltr == k].mean(0) if (ltr == k).any() else torch.zeros(c.D, device=DEV) for k in range(len(keep))]))
    # (i) state nearest centroid; (ii) least squares over the descendant dictionary; (iii) native reading; (iv) footprint ceiling
    Xte = X[te]; acc_state = ((unit(Xte) @ cents.T).argmax(1) == lte).float().mean().item()
    coef = torch.linalg.lstsq(cents.T[None].expand(len(te), -1, -1), Xte[:, :, None]).solution[:, :, 0]; acc_ls = (coef.argmax(1) == lte).float().mean().item()
    Fte = F[te]; acc_fp = ((unit(Fte) @ cents.T).argmax(1) == lte).float().mean().item()
    row = c.atom_index(L, torch.full_like(tn.cpu(), b), tn.cpu()).to(DEV); native = (sel[te] == row[te][:, None]).any(1).float().mean().item()
    # a matched control: centroids of the writes' own directions (the native atoms) used the same way on the state
    R = A[(lab["type"] == T_MLP) & (lab["block"] == b)].float().to(DEV); acc_atom = ((unit(Xte) @ R[keep].T).argmax(1) == lte).float().mean().item()
    res[b] = dict(n_neurons=int(len(keep)), n_test=int(len(te)), chance=1.0 / len(keep), state_centroid=acc_state, state_lstsq=acc_ls, state_native_atoms_argmax=acc_atom, native_dual_recall=native, footprint_ceiling=acc_fp)
    log(f"{tag} born b{b} ({len(keep)} neurons, {len(te)} test tokens, chance {1 / len(keep):.2f}): source neuron from the STATE via descendant centroids {acc_state:.2f}, via least squares {acc_ls:.2f}; via the native atoms (argmax cosine) {acc_atom:.2f}; native dual@64 reading {native:.2f} | ceiling from the footprint {acc_fp:.2f}")
import numpy as np
record(f"e230_descdict_{tag}", dict(model=tag, L=L, per_birth={str(k): v for k, v in res.items()}), f"state-level source identification via descendant centroids {np.mean([v['state_centroid'] for v in res.values()]):.2f} (least squares {np.mean([v['state_lstsq'] for v in res.values()]):.2f}; chance {np.mean([v['chance'] for v in res.values()]):.2f}) vs native atoms argmax {np.mean([v['state_native_atoms_argmax'] for v in res.values()]):.2f} and native dual reading {np.mean([v['native_dual_recall'] for v in res.values()]):.2f} | footprint ceiling {np.mean([v['footprint_ceiling'] for v in res.values()]):.2f}")
