"""e403: how does the network's own vocabulary compare with a vocabulary learned from its activations? GPT-2 small, the
state after block 6 (= the input of block 7), 8 sequences, the same OMP and splice for every vocabulary: the own WDD
vocabulary (embeddings, positions, MLP rows and head bases of blocks 0-6) and two rotations; a random subset of it of the
SAE's size (24576) and its rotation (equal dictionary size); the own MLP rows and their rotation; and the decoder rows of
Bloom's residual SAE for blocks.7.hook_resid_pre (24576 latents, trained on these activations) and its rotation. The
SAE lives in TransformerLens coordinates, where every write is centred over the model dimension; since every GPT-2
reader's LayerNorm removes the all-ones component, states and atoms are described with that component removed (it
cannot change the function). Pre-registered: the learned decoder is at least as good as the own vocabulary at every k;
reported, the fraction of the learned vocabulary's advantage over its rotation that the own vocabulary achieves."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sd_common import *
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
c = Cache("gpt2"); ev = c.s["eval_ids"][:8].to(DEV); KS = [4, 8, 16, 32, 64]
model, tok, fam = load_model("gpt2"); arch = Arch(model, fam); L = 6
A, lab = build_dictionary(arch, blocks=list(range(L + 1))); typ = lab["type"].to(DEV)
sae = load_file(hf_hub_download("jbloom/GPT2-Small-SAEs-Reformatted", f"blocks.{L + 1}.hook_resid_pre/sae_weights.safetensors"))
Wd = sae["W_dec"].float().to(DEV); Wd = Wd if Wd.shape[1] == arch.D else Wd.T; NS = Wd.shape[0]
lv = Level(model, arch, ev, L, dimcentre=True)
res = dict(level=L, k=KS, gap=lv.gap, n_sae=NS, n_own=A.shape[0], cells={})
g0 = torch.Generator(device=DEV).manual_seed(0); sub = A[torch.randperm(A.shape[0], device=DEV, generator=g0)[:NS]]
W = A[typ == T_MLP]
for nm, V in [("own", A), ("own_rot7", rotate(A, seed=7)), ("own_rot11", rotate(A, seed=11)), ("own_sub", sub), ("own_sub_rot7", rotate(sub, seed=7)),
              ("mlp", W), ("mlp_rot7", rotate(W, seed=7)), ("sae", Wd), ("sae_rot7", rotate(Wd, seed=7)), ("sae_rot11", rotate(Wd, seed=11))]:
    res["cells"][nm] = describe(lv, V, KS); log(f"gpt2 L{L} {nm}: rec k4..64 {fmt(res['cells'][nm], KS)}")
g = lambda n, k=16: res["cells"][n][str(k)]["rec"]
frac = {k: (g("own", k) - (g("own_rot7", k) + g("own_rot11", k)) / 2) / max(g("sae", k) - (g("sae_rot7", k) + g("sae_rot11", k)) / 2, 1e-9) for k in KS}
res["own_share_of_learned_advantage"] = frac
summ = (f"gpt2 L6 k8/k16: own {g('own', 8):.2f}/{g('own'):.2f} own_sub {g('own_sub', 8):.2f}/{g('own_sub'):.2f} mlp {g('mlp', 8):.2f}/{g('mlp'):.2f} sae {g('sae', 8):.2f}/{g('sae'):.2f} "
        f"| rot own {g('own_rot7', 8):.2f}/{g('own_rot7'):.2f} sae {g('sae_rot7', 8):.2f}/{g('sae_rot7'):.2f} | own share of the learned advantage over rotation k4..64 " + " ".join(f"{frac[k]:.2f}" for k in KS))
log(summ); record("e403_learnedref_gpt2", res, summ)
