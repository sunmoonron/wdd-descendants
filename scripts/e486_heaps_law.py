"""e486: the growth of the vocabulary in use (Herdan 1960, Heaps 1978). In a text, the number of distinct words grows
as a power of the number of tokens read, V(T) ~ T^beta, with beta below 1: a finite vocabulary reused. Describing T
states with 16 native words each gives a "text" in the native vocabulary. Its Heaps exponent says whether the words are
a reused vocabulary (small beta) or a fresh set for every state (beta near 1), and how it compares with the rotated
dictionary (no provenance) and with the lexicon of the input text itself.
Setup: 32 x 512 evaluation tokens (sinks excluded), the middle block and the quarter block, dictionary up to the block;
T = 256, 512, ..., 16384 tokens in sequence order; V(T) = distinct words used in the 16-word descriptions of the first
T states; beta fitted on T >= 1024 by least squares in log-log; also the number of distinct words needed to cover half
and nine tenths of all selections at T max (a Zipf summary), and the share of the dictionary ever used.
Models (argument): the five.
Pre-registered (honest guesses):
- beta is below 0.8 for native words and closer to 1 for rotated words in all five (0.6);
- the native exponent is above the text's lexicon exponent (a state needs more new words than a text does) (0.6);
- half of all selections are covered by under 5% of the words used (0.6)."""
import sys, os, json as _json, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; model, tok, fam = load_model(name); arch = Arch(model, fam); K = 16; D = arch.D
E = eval_ids(name)[:32, :512].to(DEV); B_, T_ = E.shape; BLOCKS = sorted({arch.NB // 4, arch.NB // 2})
S_ = block_states(model, arch, E, BLOCKS, chunk=4)
TS = [256, 512, 1024, 2048, 4096, 8192, 16384]
def heaps(counts_seq, TS):
    """counts_seq: [Tmax, k] symbols in order; V(T) = distinct symbols among the first T rows"""
    V = []
    for T in TS: V.append(int(torch.unique(counts_seq[:T].reshape(-1)).numel()))
    xs = torch.tensor([math.log(T) for T in TS if T >= 1024]); ys = torch.tensor([math.log(v) for T, v in zip(TS, V) if T >= 1024])
    beta = float(((xs - xs.mean()) * (ys - ys.mean())).sum() / ((xs - xs.mean()) ** 2).sum()); return V, beta
res = dict(model=name, blocks=BLOCKS, T=TS, by_block={})
tokens = E[:, 1:].reshape(-1)
for b in BLOCKS:
    flat = S_[b].reshape(-1, D); keep = ~sinkmask(flat); Xc = flat[keep] - flat[keep].mean(0); toks = tokens[keep][:, None]
    A, lab = build_dictionary(arch, blocks=list(range(b + 1))); Au = unitr(A); Ar = unitr(rotate(A, seed=7)); m = Au.shape[0]; del A
    sel_n, _, _ = omp(Xc, Au, K, batch=1024, record_err=False); sel_r, _, _ = omp(Xc, Ar, K, batch=1024, record_err=False)
    out = {}
    for kind, S in (("native", sel_n), ("rotated", sel_r), ("lexicon", toks)):
        V, beta = heaps(S, TS); cnt = torch.bincount(S.reshape(-1)).sort(descending=True).values; cnt = cnt[cnt > 0]; cum = cnt.cumsum(0) / cnt.sum()
        out[kind] = dict(V=V, beta=beta, distinct=int(cnt.numel()), cover_half=int((cum < 0.5).sum()) + 1, cover_0_9=int((cum < 0.9).sum()) + 1, share_of_dictionary_used=(int(cnt.numel()) / m) if kind != "lexicon" else None)
    res["by_block"][b] = out
    log(f"{name} block {b} (m {m}): " + " | ".join(f"{k}: beta {v['beta']:.3f}, V(T) " + "/".join(str(x) for x in v["V"]) + f", half by {v['cover_half']} words, 0.9 by {v['cover_0_9']}" for k, v in out.items()))
mid = res["by_block"][arch.NB // 2]
res["checks"] = dict(native_beta_under_0_8_rotated_higher=mid["native"]["beta"] < 0.8 and mid["rotated"]["beta"] > mid["native"]["beta"], native_over_lexicon=mid["native"]["beta"] > mid["lexicon"]["beta"], half_under_5pct=mid["native"]["cover_half"] < 0.05 * mid["native"]["distinct"])
summ = (f"{name}: Heaps exponent beta (V ~ T^beta), " + " || ".join(f"block {b}: native {v['native']['beta']:.3f} (distinct {v['native']['distinct']} of {int(v['native']['distinct'] / max(v['native']['share_of_dictionary_used'], 1e-9))}, half of selections by {v['native']['cover_half']} words), rotated {v['rotated']['beta']:.3f} (distinct {v['rotated']['distinct']}), lexicon {v['lexicon']['beta']:.3f} (distinct {v['lexicon']['distinct']})" for b, v in res["by_block"].items())
        + f" | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e486_heaps_{name}", res, summ)
