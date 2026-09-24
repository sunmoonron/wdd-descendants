"""e408b: how close can a vocabulary chosen from the weights alone come to one learned from activations? GPT-2 small, the
state after block 6, 8 sequences, the all-ones component removed (as e403). Vocabularies of the SAE's size (24576):
a random subset of the own words, the own words that the downstream readers see most (largest norm under the reader
metric of e408, weights only), and the own words with the largest raw write norm (MLP rows by their output-weight
norm, embeddings by their norm, head directions by their singular value; weights only); plus the full own vocabulary,
Bloom's residual SAE decoder (trained on activations) and rotations of both. Words chosen under the Euclidean metric,
the reader metric (weights only) and, for the full own vocabulary and the SAE, the Fisher metric (activations).
Pre-registered: the reader-selected subset beats the random subset at k 16; with both the selection and the pursuit
weight-only, the own words close at least a third of the gap to the SAE at k 8."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sd_common import *
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
c = Cache("gpt2"); ev = c.s["eval_ids"][:8].to(DEV); fit = c.s["eval_ids"][8:16].to(DEV); KS = [4, 8, 16, 32, 64]
model, tok, fam = load_model("gpt2"); arch = Arch(model, fam); L = 6; blocks = list(range(L + 1))
A, lab = build_dictionary(arch, blocks=blocks)
nr = [arch.emb[0].detach().float().norm(dim=-1), arch.emb[1].detach().float().norm(dim=-1)]
for b in blocks:
    nr.append(arch.wdir(b).norm(dim=-1)); Wo = arch.wo(b)
    for h in range(arch.NH): nr.append(torch.linalg.svdvals(Wo[h * arch.HD:(h + 1) * arch.HD, :]))
    for v in [arch.attn_bias(b), arch.mlp_bias(b)]:
        if v is not None: nr.append(v.norm()[None])
nr = torch.cat([x.to(DEV) for x in nr]); assert nr.shape[0] == A.shape[0], (nr.shape, A.shape)
sae = load_file(hf_hub_download("jbloom/GPT2-Small-SAEs-Reformatted", f"blocks.{L + 1}.hook_resid_pre/sae_weights.safetensors"))
Wd = sae["W_dec"].float().to(DEV); Wd = Wd if Wd.shape[1] == arch.D else Wd.T; NS = Wd.shape[0]
GR, GU = reader_gram(model, arch, list(range(L + 1, arch.NB))); GF = fisher_gram(model, arch, fit, L)
Ac = unitr(A - A.mean(-1, keepdim=True)); SR = metric_sqrt(GR); SF = metric_sqrt(GF)
seen = (Ac @ SR).norm(dim=-1)   # how strongly the downstream readers see each unit word (weights only)
g0 = torch.Generator(device=DEV).manual_seed(0)
V = dict(own=A, own_rot=rotate(A, seed=7), sub_random=A[torch.randperm(A.shape[0], device=DEV, generator=g0)[:NS]],
         sub_read=A[seen.topk(NS).indices], sub_write=A[nr.topk(NS).indices], sae=Wd, sae_rot=rotate(Wd, seed=7))
lv = Level(model, arch, ev, L, dimcentre=True)
res = dict(level=L, k=KS, gap=lv.gap, n_sae=NS, cells={}, sub_read_types={int(t): (lab["type"].to(DEV)[seen.topk(NS).indices] == t).float().mean().item() for t in (T_TOK, T_POS, T_MLP, T_ATT)},
           sub_write_types={int(t): (lab["type"].to(DEV)[nr.topk(NS).indices] == t).float().mean().item() for t in (T_TOK, T_POS, T_MLP, T_ATT)})
plan = dict(own=["euc", "readers", "fisher"], own_rot=["euc", "readers"], sub_random=["euc", "readers"], sub_read=["euc", "readers"], sub_write=["euc", "readers"], sae=["euc", "readers", "fisher"], sae_rot=["euc", "readers"])
Sm = dict(euc=None, readers=SR, fisher=SF)
for nm, mts in plan.items():
    for mt in mts:
        res["cells"][f"{nm}:{mt}"] = describe(lv, V[nm], KS, S=Sm[mt]); log(f"gpt2 {nm} {mt}: rec k4..64 {fmt(res['cells'][f'{nm}:{mt}'], KS)}")
g = lambda n, k=16: res["cells"][n][str(k)]["rec"]
res["gap_closed_k8"] = (g("sub_read:readers", 8) - g("sub_random:euc", 8)) / max(g("sae:euc", 8) - g("sub_random:euc", 8), 1e-9)
summ = ("gpt2 L6 k8/k16: " + " | ".join(f"{n} {g(n, 8):.2f}/{g(n):.2f}" for n in res["cells"]) + f" | weight-only selection+pursuit closes {res['gap_closed_k8']:.2f} of the random-subset-to-SAE gap at k8"
        + " | sub_read types " + " ".join(f"{t}:{v:.2f}" for t, v in res["sub_read_types"].items()))
log(summ); record("e408b_weightvocab_gpt2", res, summ)
