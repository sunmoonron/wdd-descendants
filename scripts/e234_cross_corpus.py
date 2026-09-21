"""e234: does the descendant signature transfer across corpora without refitting? Corpus A = the WikiText cache,
corpus B = the Pile cache of the same model (GPT-2, SmolLM2). Block b = 2, level L, neurons dominant at >= 20
typical tokens in BOTH corpora. Centroids fitted on all A tokens and tested on all B tokens (and the reverse),
with within-corpus held-out accuracy for reference. Kill: cross-corpus identification near chance."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from desc_common import *
tag = sys.argv[1]; cA = Cache(tag); cB = Cache(tag + "_pile"); L = mid(cA); model, tok, fam = load_model(cA.name); arch = Arch(model, fam); NS = 16; b = 2; data = {}
for nm, cc in (("wikitext", cA), ("pile", cB)):
    ids_seq = cc.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; run = make_runner(model, arch, cA, ids_seq, [L], NT); led, tn, tc = dominant(model, arch, cA, ids_seq, b); S0 = run(); S1 = run(b, tn); typ = typical_mask(S0[L]); big = tc.abs() >= tc.abs().quantile(0.5); data[nm] = dict(F=S0[L] - S1[L], tn=tn, mask=typ & big)
def cls(nm, keep):
    idx = torch.nonzero(data[nm]["mask"])[:, 0]; neur = data[nm]["tn"][idx]; m = torch.isin(neur, keep); idx = idx[m]; neur = neur[m]; return idx, (neur[:, None] == keep[None, :]).float().argmax(1)
uA, cA_ = torch.unique(data["wikitext"]["tn"][data["wikitext"]["mask"]], return_counts=True); uB, cB_ = torch.unique(data["pile"]["tn"][data["pile"]["mask"]], return_counts=True); keep = torch.tensor(sorted(set(uA[cA_ >= 20].tolist()) & set(uB[cB_ >= 20].tolist())), device=DEV); K = len(keep)
iA, lA = cls("wikitext", keep); iB, lB = cls("pile", keep); FA = data["wikitext"]["F"][iA]; FB = data["pile"]["F"][iB]; torch.manual_seed(0); sA = torch.rand(len(iA), device=DEV) < 0.5; sB = torch.rand(len(iB), device=DEV) < 0.5
res = dict(K=K, nA=int(len(iA)), nB=int(len(iB)), chance=1.0 / K, within_wikitext=accuracy(FA[~sA], centroids(FA[sA], lA[sA], K), lA[~sA]), within_pile=accuracy(FB[~sB], centroids(FB[sB], lB[sB], K), lB[~sB]), wikitext_to_pile=accuracy(FB, centroids(FA, lA, K), lB), pile_to_wikitext=accuracy(FA, centroids(FB, lB, K), lA))
log(f"{tag}: {K} neurons shared (chance {1 / K:.2f}), tokens {len(iA)} / {len(iB)} | within-corpus held-out: wikitext {res['within_wikitext']:.2f}, pile {res['within_pile']:.2f} | cross-corpus without refitting: wikitext -> pile {res['wikitext_to_pile']:.2f}, pile -> wikitext {res['pile_to_wikitext']:.2f}")
record(f"e234_crosscorpus_{tag}", dict(model=tag, b=b, L=L, **res), f"K {K} (chance {res['chance']:.2f}): within wikitext {res['within_wikitext']:.2f}, within pile {res['within_pile']:.2f}; wikitext -> pile {res['wikitext_to_pile']:.2f}, pile -> wikitext {res['pile_to_wikitext']:.2f}")
