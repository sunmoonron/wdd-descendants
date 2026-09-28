"""e572 (session 102): is a word more than a large row? e569b found that 256 random rows matched to the words' norms
describe the states at K=8 as well as the 256 words do. Here the rest of what a word is, for the words against
norm-matched non-word rows (each word paired with the unused row nearest in norm within its block) and against random
rows, on Pythia-410m (block 12), Pythia-160m (block 6) and Mamba-130m (block 12): the usage itself, S over the floor,
the count over the floor and the breadth, the context-set twin rate across models (410m words against 160m's words and
against 160m's norm-matched rows; Mamba against Pythia-410m by token positions), and the norm rank of the words among
all rows. Pre-registered (probabilities are honest guesses):
 W1 (0.7) the words' usage exceeds the norm-matched rows' by three times or more (usage is not norm);
 W2 (0.6) the words' S exceeds the norm-matched rows' S by 0.2 or more at the median, and the twin rate of the
    norm-matched rows across models is under half the words';
 W3 (0.6) the words sit in the top quarter of the norm distribution."""
from s101_common import *
from transformers import MambaForCausalLM
t0 = time.time(); JT = 0.25
def jac(P, Q):
    P = P.float(); Q = Q.float(); inter = P.T @ Q; return inter / (P.sum(0)[:, None] + Q.sum(0)[None] - inter).clamp_min(1)
def profile(U, A, norms, DFF, tag):
    st = stats(U, A, K); m = A.shape[0]; w = torch.nonzero(wordset(st["usage"]))[:, 0]; blk = torch.arange(m) // DFF; used = torch.zeros(m, dtype=torch.bool); used[w] = True; nm = []
    for r in w.tolist():
        cand = torch.nonzero(~used & (blk == blk[r]))[:, 0]; j = cand[(norms[cand] - norms[r]).abs().argmin()]; used[j] = True; nm.append(int(j))
    nm = torch.tensor(nm); g = torch.Generator().manual_seed(0); rnd = torch.nonzero(~used)[:, 0]; rnd = rnd[torch.randperm(rnd.numel(), generator=g)[:256]]
    nrank = norms.argsort(descending=True).argsort().float() / m
    out = {}
    for name, rows in (("words", w), ("norm_matched", nm), ("random", rnd)):
        out[name] = dict(usage=float(st["usage"][rows].median()), S=float(st["S"][rows].median()), share_over_floor=float((st["S"][rows] >= 1).float().mean()), cnt=float(st["cnt"][rows].median()), breadth=float((st["cnt75"][rows] / st["cnt"][rows].clamp_min(1)).median()), norm=float(norms[rows].median()), norm_rank=float(nrank[rows].median()), over=(st["ratio"][:, rows].float() > 1))
        o = out[name]; log(f"{tag} {name}: usage {o['usage']:.0f}, S {o['S']:.2f} (over the floor {o['share_over_floor']:.2f}), count {o['cnt']:.0f}, breadth {o['breadth']:.2f}, norm {o['norm']:.2f} (rank {o['norm_rank']:.2f})")
    out["usage_norm_corr"] = float(torch.corrcoef(torch.stack([st["usage"], norms]))[0, 1]); out["usage_norm_spearman"] = float(torch.corrcoef(torch.stack([st["usage"].argsort().argsort().float(), norms.argsort().argsort().float()]))[0, 1]); log(f"{tag}: usage-norm correlation {out['usage_norm_corr']:.2f} (rank {out['usage_norm_spearman']:.2f})")
    return out
S4 = lm_states("pythia410"); S1 = lm_states("pythia160"); keepc = S4["keep"] & S1["keep"]
for S_ in (S4, S1): S_["U"] = unitr(S_["X"][keepc] - S_["X"][keepc].mean(0))
P4 = profile(S4["U"], S4["A"], S4["norms"].cpu(), S4["arch"].DFF, "pythia410"); P1 = profile(S1["U"], S1["A"], S1["norms"].cpu(), S1["arch"].DFF, "pythia160"); del S1["model"], S4["model"]; torch.cuda.empty_cache()
model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf", dtype=torch.float32).to(DEV).eval(); ids = pile_ids("pythia410"); cap = {}
class Stop(Exception): pass
def hk(m, i, o): cap["x"] = (o[0] if isinstance(o, tuple) else o).detach().float(); raise Stop
h = model.backbone.layers[12].register_forward_hook(hk)
try:
    with torch.no_grad(): model(ids)
except Stop: pass
finally: h.remove()
Xm = cap["x"][:, 1:].reshape(-1, cap["x"].shape[-1]); Rm_ = torch.cat([model.backbone.layers[b].mixer.out_proj.weight.detach().float().T for b in range(13)]); keepm = ~sinkmask(Xm) & keepc; Um = unitr(Xm[keepm] - Xm[keepm].mean(0)); del model; torch.cuda.empty_cache()
PM = profile(Um, unitr(Rm_), Rm_.norm(dim=1).cpu(), 1536, "mamba130")
sub = keepm[keepc].cpu()
def twins(a, b, tag):
    best = jac(a["over"], b["over"]).max(1).values; r = dict(share=float((best >= JT).float().mean()), median=float(best.median())); log(f"twins {tag}: {r['share']:.2f} (median Jaccard {r['median']:.2f})"); return r
res = dict(pythia410={k: {q: v for q, v in d.items() if q != "over"} if isinstance(d, dict) else d for k, d in P4.items()}, pythia160={k: {q: v for q, v in d.items() if q != "over"} if isinstance(d, dict) else d for k, d in P1.items()}, mamba130={k: {q: v for q, v in d.items() if q != "over"} if isinstance(d, dict) else d for k, d in PM.items()}, twins={})
for a_, b_, tag in (("words", "words", "410m words -> 160m words"), ("norm_matched", "words", "410m norm-matched -> 160m words"), ("norm_matched", "norm_matched", "410m norm-matched -> 160m norm-matched"), ("random", "words", "410m random -> 160m words"), ("words", "norm_matched", "410m words -> 160m norm-matched")):
    res["twins"][tag] = twins(P4[a_], P1[b_], tag)
for a_, b_, tag in (("words", "words", "mamba words -> 410m words"), ("norm_matched", "words", "mamba norm-matched -> 410m words"), ("random", "words", "mamba random -> 410m words")):
    res["twins"][tag] = twins({"over": PM[a_]["over"]}, {"over": P4[b_]["over"][sub]}, tag)
w4, n4, r4 = P4["words"], P4["norm_matched"], P4["random"]; T = res["twins"]
summ = (f"Pythia-410m words vs norm-matched non-words vs random rows: usage {w4['usage']:.0f}/{n4['usage']:.0f}/{r4['usage']:.0f}, S {w4['S']:.2f}/{n4['S']:.2f}/{r4['S']:.2f} (over the floor {w4['share_over_floor']:.2f}/{n4['share_over_floor']:.2f}/{r4['share_over_floor']:.2f}), count {w4['cnt']:.0f}/{n4['cnt']:.0f}/{r4['cnt']:.0f}, norm rank {w4['norm_rank']:.2f}/{n4['norm_rank']:.2f}/{r4['norm_rank']:.2f}; usage-norm rank correlation {P4['usage_norm_spearman']:.2f} (160m {P1['usage_norm_spearman']:.2f}, Mamba {PM['usage_norm_spearman']:.2f}); "
        f"twins 410m->160m: words {T['410m words -> 160m words']['share']:.2f}, norm-matched rows {T['410m norm-matched -> 160m words']['share']:.2f} (to 160m's norm-matched {T['410m norm-matched -> 160m norm-matched']['share']:.2f}), random {T['410m random -> 160m words']['share']:.2f}; Mamba->410m words {T['mamba words -> 410m words']['share']:.2f}, norm-matched {T['mamba norm-matched -> 410m words']['share']:.2f}, random {T['mamba random -> 410m words']['share']:.2f}; "
        f"Mamba words/norm-matched S {PM['words']['S']:.2f}/{PM['norm_matched']['S']:.2f}, usage {PM['words']['usage']:.0f}/{PM['norm_matched']['usage']:.0f} | {time.time() - t0:.0f}s")
log(summ); record("e572_norm_matched_words", res, summ)
