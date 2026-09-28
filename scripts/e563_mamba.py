"""e563 (session 101): does a state-space model have a native vocabulary? Every word so far lives in a transformer, where
attention supplies a quarter of the state and the MLP rows the rest. Mamba-130m (24 blocks, width 768, no attention, no
MLP) writes to the residual stream through one matrix per block, the mixer's output projection (1536 rows). Same
instrument: states after block B on the Pile evaluation sequences (Mamba-130m uses the GPT-NeoX tokenizer, so the Pythia
ids serve unchanged), dictionary the output-projection rows of blocks 0..B, three dictionaries compared by FVU at K=16
and top-256 usage share, the 256 most used rows' extreme-value statistics against the rotated dictionary's, at B=6, 12
and 18 (a quarter, half and three quarters of the depth; state-spaces/mamba-370m-hf, 48 blocks of width 1024, is run by
the same script with the model name as its argument). Pythia-160m at block 6 is measured by the same code for scale. Pre-registered (probabilities are honest guesses):
 M1 (0.7) the native rows beat the rotated rows in FVU at K=16 at the middle block;
 M2 (0.5) the 256 most used rows sit over the floor at the median (S >= 1.2): words without attention or MLPs;
 M3 (0.55) the advantage is smaller than Pythia-160m's (the SSM's writes are gated mixtures, less row-like)."""
from s101_common import *
from transformers import MambaForCausalLM
t0 = time.time(); name = sys.argv[1] if len(sys.argv) > 1 else "state-spaces/mamba-130m-hf"; TAG = "" if name.endswith("130m-hf") else "_" + name.split("/")[-1].replace("-", "_")
model = MambaForCausalLM.from_pretrained(name, dtype=torch.float32).to(DEV).eval()
ids = pile_ids("pythia410"); res = dict(model=name, blocks={})
class Stop(Exception): pass
def mamba_states(B):
    cap = {}
    def hk(m, i, o): cap["x"] = (o[0] if isinstance(o, tuple) else o).detach().float(); raise Stop
    h = model.backbone.layers[B].register_forward_hook(hk)
    try:
        with torch.no_grad(): model(ids)
    except Stop: pass
    finally: h.remove()
    X = cap["x"][:, 1:].reshape(-1, cap["x"].shape[-1]); keep = ~sinkmask(X)
    A = unitr(torch.cat([model.backbone.layers[b].mixer.out_proj.weight.detach().float().T for b in range(B + 1)]))
    return unitr(X[keep] - X[keep].mean(0)), A, keep
NL = len(model.backbone.layers); BL = (NL // 4, NL // 2, 3 * NL // 4)
for B in BL:
    U, A, keep = mamba_states(B); c = dict_compare(U, A, K); c.pop("_st")
    c["n_positions"] = int(U.shape[0]); c["n_rows"] = int(A.shape[0]); c["sinks_dropped"] = int((~keep).sum())
    res["blocks"][B] = c; log(summarize_compare(f"Mamba block {B} ({U.shape[0]} positions, {A.shape[0]} rows)", c)); del U, A; torch.cuda.empty_cache()
del model; torch.cuda.empty_cache()
S = lm_states("pythia160"); cp = dict_compare(S["U"], S["A"], K); cp.pop("_st"); res["pythia160_block6"] = cp; log(summarize_compare("Pythia-160m block 6", cp))
m12, p6 = res["blocks"][BL[1]], cp; res["blocks"] = {str(k): v for k, v in res["blocks"].items()}; b6, b12, b18 = (res["blocks"][str(b)] for b in BL)
summ = (f"{name}: block {BL[1]} FVU own/rot/covA {m12['own_fvu']:.3f}/{m12['rot_fvu']:.3f}/{m12['covA_fvu']:.3f} (advantage {m12['advantage_fvu']:.3f}; Pythia-160m block 6 {p6['advantage_fvu']:.3f}), "
        f"top-256 usage share own/rot {m12['own_top256_usage_share']:.2f}/{m12['rot_top256_usage_share']:.2f} (Pythia {p6['own_top256_usage_share']:.2f}/{p6['rot_top256_usage_share']:.2f}); "
        f"Mamba words: median S {m12['words']['median_S']:.2f} (rotated {m12['rotated_words']['median_S']:.2f}), share over the floor {m12['words']['share_S_over_1']:.2f}, breadth {m12['words']['median_breadth']:.2f} (Pythia words median S {p6['words']['median_S']:.2f}, breadth {p6['words']['median_breadth']:.2f}); "
        f"rows over the floor anywhere {m12['words']['rows_over_floor_anywhere']} vs rotated {m12['rotated_words']['rows_over_floor_anywhere']} (Pythia {p6['words']['rows_over_floor_anywhere']} vs {p6['rotated_words']['rows_over_floor_anywhere']}); "
        f"blocks {BL[0]}/{BL[1]}/{BL[2]} advantage {b6['advantage_fvu']:.3f}/{b12['advantage_fvu']:.3f}/{b18['advantage_fvu']:.3f}, words' median S {b6['words']['median_S']:.2f}/{b12['words']['median_S']:.2f}/{b18['words']['median_S']:.2f} | {time.time() - t0:.0f}s")
log(summ); record(f"e563_mamba{TAG}", res, summ)
