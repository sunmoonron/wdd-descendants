"""e239: does the signature belong to the vector or to the downstream network? Pythia-410m final vs step 16000.
Block 2, level 6. Neurons dominant at >= 20 tokens in BOTH models; natural footprint centroids in each model.
Transplant the FINAL model's write vectors (median size) into the STEP-16000 model's stream at the block-3 input on
foreign tokens and classify the resulting footprints against the final model's centroids and against the
step-16000 model's own centroids; and the reverse. Also the cosine between the two checkpoints' write vectors."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from desc_common import *
c = Cache("pythia410"); NS = 16; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; b = 2; lv = 6; models = {}
for nm, rev in (("final", None), ("step16000", "step16000")):
    model, tok, fam = load_model(c.name, revision=rev); arch = Arch(model, fam); run = make_runner(model, arch, c, ids_seq, [lv], NT); W = arch.wdir(b).to(DEV); R = W / W.norm(dim=1, keepdim=True).clamp_min(1e-9)
    st = {}; h = arch.mlp_lin(b).register_forward_pre_hook(lambda m, inp: st.__setitem__("a", inp[0].detach().float().reshape(-1, arch.DFF))); model(ids_seq); h.remove(); led = st["a"] * W.norm(dim=1)[None]; tn = led.abs().argmax(1); tc = torch.gather(led, 1, tn[:, None])[:, 0]
    S0 = run(); S1 = run(b, tn); typ = typical_mask(S0[lv]); big = tc.abs() >= tc.abs().quantile(0.5); models[nm] = dict(model=model, arch=arch, run=run, R=R, led=led, tn=tn, tc=tc, F=S0[lv] - S1[lv], H0=S0[lv], mask=typ & big)
uA, cA = torch.unique(models["final"]["tn"][models["final"]["mask"]], return_counts=True); uB, cB = torch.unique(models["step16000"]["tn"][models["step16000"]["mask"]], return_counts=True); keep = torch.tensor(sorted(set(uA[cA >= 20].tolist()) & set(uB[cB >= 20].tolist())), device=DEV); K = len(keep)
def cents_of(nm):
    idx = torch.nonzero(models[nm]["mask"])[:, 0]; neur = models[nm]["tn"][idx]; m = torch.isin(neur, keep); idx = idx[m]; lab_i = (neur[m][:, None] == keep[None, :]).float().argmax(1); return centroids(models[nm]["F"][idx], lab_i, K), idx
def transplant(src, dst):
    S = models[src]; D = models[dst]; med = torch.stack([S["tc"][(S["tn"] == k) & S["mask"]].median() for k in keep]); typ = typical_mask(D["H0"]); foreign = torch.nonzero(typ)[:, 0]; torch.manual_seed(0); assign = torch.randint(0, K, (len(foreign),), device=DEV)
    act = S["led"] if src == dst else D["led"]; ok = ~(act[foreign].abs() >= 0.05 * act[foreign].abs().max(1, keepdim=True).values)[torch.arange(len(foreign), device=DEV), keep[assign]]; foreign, assign = foreign[ok], assign[ok]
    inj = torch.zeros(NT, c.D, device=DEV); inj[foreign] = med[assign][:, None] * S["R"][keep[assign]]; S2 = D["run"](inject=inj, inject_block=b + 1); return (S2[lv] - D["H0"])[foreign], assign
cF, _ = cents_of("final"); cS, _ = cents_of("step16000"); res = dict(K=K, chance=1.0 / K, write_vector_cosine=((models["final"]["R"][keep] * models["step16000"]["R"][keep]).sum(1)).median().item())
Ft, at = transplant("final", "step16000"); res["final_vectors_in_step16000_vs_final_centroids"] = accuracy(Ft, cF, at); res["final_vectors_in_step16000_vs_step16000_centroids"] = accuracy(Ft, cS, at)
Ft2, at2 = transplant("step16000", "final"); res["step16000_vectors_in_final_vs_step16000_centroids"] = accuracy(Ft2, cS, at2); res["step16000_vectors_in_final_vs_final_centroids"] = accuracy(Ft2, cF, at2)
res["centroid_cosine_final_vs_step16000"] = ((cF * cS).sum(1)).median().item()
log(f"pythia410 final vs step16000 ({K} shared neurons, chance {1 / K:.2f}): write vectors cosine across checkpoints {res['write_vector_cosine']:.2f}, descendant centroids cosine {res['centroid_cosine_final_vs_step16000']:.2f} | final vectors through the step-16000 network: classified by final centroids {res['final_vectors_in_step16000_vs_final_centroids']:.2f}, by step-16000 centroids {res['final_vectors_in_step16000_vs_step16000_centroids']:.2f} | step-16000 vectors through the final network: by step-16000 centroids {res['step16000_vectors_in_final_vs_step16000_centroids']:.2f}, by final centroids {res['step16000_vectors_in_final_vs_final_centroids']:.2f}")
record("e239_crosscheckpoint_pythia410", res, f"K {K} (chance {res['chance']:.2f}): vectors cos {res['write_vector_cosine']:.2f}, centroids cos {res['centroid_cosine_final_vs_step16000']:.2f}; final vectors in step-16000 net: final-centroids {res['final_vectors_in_step16000_vs_final_centroids']:.2f} vs step-centroids {res['final_vectors_in_step16000_vs_step16000_centroids']:.2f}; step vectors in final net: step-centroids {res['step16000_vectors_in_final_vs_step16000_centroids']:.2f} vs final-centroids {res['step16000_vectors_in_final_vs_final_centroids']:.2f}")
