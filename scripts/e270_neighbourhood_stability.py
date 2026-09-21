"""e270: is the descendant geometry itself stable? For the K candidates: their 3 nearest descendant neighbours at L
computed on two disjoint token halves (Jaccard), and (GPT-2, SmolLM2) on WikiText vs the Pile without refitting;
(Pythia) at the final checkpoint vs steps 64000 and 16000 for the shared candidates. Compared with the stability of
write-space neighbours (trivially 1 within a model; across checkpoints the write vectors change)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from func_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); b = 2; NS = 12
def geometry(model, arch, cache_for_tokens, ids_seq, keep_filter=None):
    NT = NS * CTX; lab = c.d["lab"]; A = c.d["A"]; run = make_runner_logits(model, arch, c, ids_seq, [L], NT, b); W = arch.wdir(b).to(DEV); WN = W.norm(dim=1); st = {}; h = arch.mlp_lin(b).register_forward_pre_hook(lambda m, inp: st.__setitem__("a", inp[0].detach().float().reshape(-1, arch.DFF))); model(ids_seq); h.remove(); led = st["a"] * WN[None]; tn = led.abs().argmax(1); tc = torch.gather(led, 1, tn[:, None])[:, 0]
    S0 = run(); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); S1 = run(tn); idx, lab_i, keep = classes(tn, typ & big, 20); F = S0[L] - S1[L]; return dict(F=F, idx=idx, lab_i=lab_i, keep=keep, Rw=unit(W / WN[:, None].clamp_min(1e-9)))
def nn3(G):
    G2 = G.clone(); G2.fill_diagonal_(-2); return G2.topk(3, dim=1).indices
def jacc(n1, n2): return float(sum(len(set(n1[i].tolist()) & set(n2[i].tolist())) / len(set(n1[i].tolist()) | set(n2[i].tolist())) for i in range(len(n1))) / len(n1))
model, tok, fam = load_model(c.name); arch = Arch(model, fam); g = geometry(model, arch, c, c.s["eval_ids"][:NS].to(DEV)); K = len(g["keep"]); torch.manual_seed(0); split = torch.rand(len(g["idx"]), device=DEV) < 0.5
def cents_of(gg, mask=None, keep=None):
    keep = gg["keep"] if keep is None else keep; idx, lab_i = gg["idx"], gg["lab_i"]
    if mask is not None: idx, lab_i = idx[mask], lab_i[mask]
    m = torch.isin(gg["keep"][lab_i], keep); idx, lab_i = idx[m], lab_i[m]; lab2 = (gg["keep"][lab_i][:, None] == keep[None, :]).float().argmax(1); return unit(centroids(gg["F"][idx], lab2, len(keep)))
C1, C2 = cents_of(g, split), cents_of(g, ~split); res = dict(K=K, token_halves_jaccard=jacc(nn3(C1 @ C1.T), nn3(C2 @ C2.T)), chance=3.0 / (K - 1))
if tag in ("gpt2", "smollm2"):
    cB = Cache(tag + "_pile"); gB = geometry(model, arch, c, cB.s["eval_ids"][:NS].to(DEV)); shared = torch.tensor(sorted(set(g["keep"].tolist()) & set(gB["keep"].tolist())), device=DEV)
    if len(shared) >= 5: CA, CB = cents_of(g, keep=shared), cents_of(gB, keep=shared); res["cross_corpus_jaccard"] = jacc(nn3(CA @ CA.T), nn3(CB @ CB.T)); res["cross_corpus_K"] = int(len(shared)); res["cross_corpus_chance"] = 3.0 / (len(shared) - 1)
if tag == "pythia410":
    for rev in ("step64000", "step16000"):
        m2, _, f2 = load_model(c.name, revision=rev); a2 = Arch(m2, f2); g2 = geometry(m2, a2, c, c.s["eval_ids"][:NS].to(DEV)); shared = torch.tensor(sorted(set(g["keep"].tolist()) & set(g2["keep"].tolist())), device=DEV)
        if len(shared) >= 5: CA, CB = cents_of(g, keep=shared), cents_of(g2, keep=shared); RA, RB = g["Rw"][shared], g2["Rw"][shared]; res[f"checkpoint_{rev}_jaccard"] = jacc(nn3(CA @ CA.T), nn3(CB @ CB.T)); res[f"checkpoint_{rev}_write_jaccard"] = jacc(nn3(RA @ RA.T), nn3(RB @ RB.T)); res[f"checkpoint_{rev}_K"] = int(len(shared))
        del m2; torch.cuda.empty_cache()
log(f"{tag} (K {K}, chance {res['chance']:.2f}): descendant 3-NN Jaccard across token halves {res['token_halves_jaccard']:.2f}" + (f" | across corpora {res['cross_corpus_jaccard']:.2f} (K {res['cross_corpus_K']}, chance {res['cross_corpus_chance']:.2f})" if 'cross_corpus_jaccard' in res else "") + "".join(f" | vs {rev}: descendant {res[f'checkpoint_{rev}_jaccard']:.2f}, write-space {res[f'checkpoint_{rev}_write_jaccard']:.2f} (K {res[f'checkpoint_{rev}_K']})" for rev in ("step64000", "step16000") if f"checkpoint_{rev}_jaccard" in res))
record(f"e270_stability_{tag}", dict(model=tag, b=b, L=L, **res), f"K {K}: token-halves Jaccard {res['token_halves_jaccard']:.2f} (chance {res['chance']:.2f})" + (f"; cross-corpus {res['cross_corpus_jaccard']:.2f}" if 'cross_corpus_jaccard' in res else "") + "".join(f"; vs {rev} descendant {res[f'checkpoint_{rev}_jaccard']:.2f} write {res[f'checkpoint_{rev}_write_jaccard']:.2f}" for rev in ("step64000", "step16000") if f"checkpoint_{rev}_jaccard" in res))
