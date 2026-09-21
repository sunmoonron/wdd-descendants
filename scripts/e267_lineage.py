"""e267: is a write's set of successors a stable lineage or a dense graph? The delta-ledger of blocks 3..L for the
removed dominant block-2 write (as in e261). For each candidate neuron: the top-20 compensators computed on two
disjoint halves of its tokens, and their Jaccard overlap (stability); across neurons: the number of 'hub' later
neurons that appear in the top-20 of at least half of the candidates, and the share of each neuron's compensating
energy carried by hubs; and the pairwise Jaccard of top-20 sets between different neurons vs the within-neuron
split overlap."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from desc_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 8; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; b = 2; D = c.D; DFF = c.DFF; later = list(range(b + 1, L + 1)); WN = {j: c.d["WN"][j].to(DEV) for j in later}
def run(neuron=None):
    st = {"a": {}}; hs = [arch.layers[L].register_forward_hook(lambda m, i, o: st.__setitem__("H", (o[0] if isinstance(o, tuple) else o).detach().float().reshape(-1, D)))]
    for j in later: hs.append(arch.mlp_lin(j).register_forward_pre_hook((lambda j_: lambda m, inp: st["a"].__setitem__(j_, inp[0].detach().float().reshape(-1, DFF)))(j)))
    if neuron is not None:
        def pre(m, inp):
            x = inp[0].clone(); flat = x.reshape(-1, DFF); flat[torch.arange(NT, device=DEV), neuron] = 0; return (flat.reshape(x.shape),)
        hs.append(arch.mlp_lin(b).register_forward_pre_hook(pre))
    model(ids_seq); [h.remove() for h in hs]; return st
led, tn, tc = dominant(model, arch, c, ids_seq, b); S0 = run(); typ = typical_mask(S0["H"]); big = tc.abs() >= tc.abs().quantile(0.5); S1 = run(tn); idx, lab_i, keep = classes(tn, typ & big, 20); K = len(keep); dC = torch.cat([(S1["a"][j] - S0["a"][j]) * WN[j][None] for j in later], 1); torch.manual_seed(0)
tops = []; overlaps = []; energies = []
for k in range(K):
    toks = idx[lab_i == k]; perm = toks[torch.randperm(len(toks), device=DEV)]; h1, h2 = perm[: len(perm) // 2], perm[len(perm) // 2:]; t1 = set(dC[h1].abs().mean(0).topk(20).indices.tolist()); t2 = set(dC[h2].abs().mean(0).topk(20).indices.tolist()); overlaps.append(len(t1 & t2) / len(t1 | t2)); tall = dC[toks].abs().mean(0).topk(20).indices; tops.append(set(tall.tolist())); energies.append((dC[toks] ** 2).mean(0))
from collections import Counter
cnt = Counter(x for s in tops for x in s); hubs = [x for x, n in cnt.items() if n >= K / 2]; hub_idx = torch.tensor(hubs, device=DEV, dtype=torch.long) if hubs else None; hub_share = [(e[hub_idx].sum() / e.sum()).item() for e in energies] if hubs else [0.0]
between = [len(tops[i] & tops[j]) / len(tops[i] | tops[j]) for i in range(K) for j in range(i + 1, K)]
import numpy as np
res = dict(K=K, within_split_jaccard=float(np.mean(overlaps)), between_neuron_jaccard=float(np.mean(between)), n_hubs=len(hubs), hub_energy_share=float(np.mean(hub_share)), n_later_neurons=int(dC.shape[1]))
log(f"{tag} (K {K}, {dC.shape[1]} later neurons): a neuron's top-20 compensators on two halves of its tokens overlap with Jaccard {res['within_split_jaccard']:.2f}; between different neurons {res['between_neuron_jaccard']:.2f} | hubs (in the top-20 of at least half the neurons): {res['n_hubs']}, carrying {res['hub_energy_share']:.2f} of a neuron's compensating energy")
record(f"e267_lineage_{tag}", dict(model=tag, b=b, L=L, **res), f"K {K}: within-neuron split Jaccard {res['within_split_jaccard']:.2f} vs between-neuron {res['between_neuron_jaccard']:.2f}; hubs {res['n_hubs']} carrying {res['hub_energy_share']:.2f} of compensating energy")
