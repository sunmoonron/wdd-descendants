"""e471: the life cycle of a native word over training. Does a neuron become important before, with, or after it becomes
a used word, and does its importance die when it falls out of use? From an external review's "birth ordering" and
"word death".
Pythia-410m at steps 1000, 2000, 4000, 8000, 16000, 32000, 64000 and the final checkpoint, on 8 x 256 evaluation
tokens. For every MLP neuron of blocks 0..L (L = 12):
- usage: its selection frequency in the 16-word native descriptions of the middle-depth states (the checkpoint's own
  dictionary);
- lexical selectivity: among the positions of its top 1% activations, the share of the most common token. This is a
  crude readability proxy; e446 showed that token proxies reward lexical units.
For a selected set of neurons:
- births and deaths: the 20 largest rises and 20 largest falls of usage between consecutive checkpoints;
- controls: 60 random neurons used at some checkpoint.
For these, the importance at every checkpoint: the loss change when the neuron's activation is zeroed at every
position.
Reported:
- for births, their importance percentile (within the selected set) at the checkpoint before and after the rise;
- lagged Spearman correlations across the selected neurons, importance(t) with the change of usage (t -> t+1) against
  usage(t) with the change of importance (t -> t+1): which leads;
- for deaths, the change of importance.
Pre-registered (honest guesses):
- importance leads usage (the first lagged correlation above the second) (0.4);
- changes of usage and of importance co-move within an interval (Spearman above 0.3) (0.4);
- dying words keep their importance (median importance change of deaths at most 0) (0.5)."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
REVS = ["step1000", "step2000", "step4000", "step8000", "step16000", "step32000", "step64000", "main"]
ids = eval_ids("pythia410")[:8, :256].to(DEV); ntok = ids.numel()
usage, selectivity, meta = [], [], None
for rev in REVS:
    model, tok, fam = load_model("pythia410", revision=None if rev == "main" else rev); arch = Arch(model, fam); L = arch.NB // 2; DFF = arch.DFF
    acts, cap = {}, {}
    def mk(b):
        def pre(m, a): acts[b] = a[0].detach().float().reshape(-1, DFF); return None       # must return None
        return pre
    hs = [arch.mlp_lin(b).register_forward_pre_hook(mk(b)) for b in range(L + 1)]
    hs.append(arch.layers[L].register_forward_hook(lambda m, i, o: cap.__setitem__("x", out_of(o).detach().float())))
    with torch.no_grad(): base = token_loss(model(ids).logits.float(), ids).mean().item()
    [h.remove() for h in hs]
    X = cap["x"][:, 1:].reshape(-1, arch.D); keep = ~sinkmask(X); Xc = X[keep] - X[keep].mean(0)
    A, lab = build_dictionary(arch, blocks=list(range(L + 1))); isM = (lab["type"] == T_MLP).to(DEV)
    sel, _, _ = omp(Xc, unitr(A), 16, batch=512, record_err=False)
    usage.append(torch.bincount(sel.flatten(), minlength=A.shape[0]).float()[isM] / Xc.shape[0]); del A, sel
    tokflat = ids.reshape(-1); sv = []
    for b in range(L + 1):
        a = acts[b]; k_ = max(1, int(0.01 * a.shape[0])); top = a.topk(k_, dim=0).indices                      # [k_, DFF]
        t_ = tokflat[top]; srt = t_.sort(0).values; same = (srt[1:] == srt[:-1]).float()
        # share of the most common token among each neuron's top positions (run length of the mode)
        runs = torch.zeros_like(srt, dtype=torch.float); runs[0] = 1
        for r in range(1, srt.shape[0]): runs[r] = torch.where(srt[r] == srt[r - 1], runs[r - 1] + 1, torch.ones_like(runs[r]))
        sv.append(runs.max(0).values / k_)
    selectivity.append(torch.cat(sv)); meta = dict(L=L, DFF=DFF)
    log(f"lifecycle {rev}: loss {base:.3f}, words used {int((usage[-1] > 0).sum())}"); del model, acts; torch.cuda.empty_cache()
U = torch.stack(usage); S = torch.stack(selectivity); L, DFF = meta["L"], meta["DFF"]; nR = len(REVS)
chosen = set()
for t in range(nR - 1):
    d = U[t + 1] - U[t]; chosen.update(d.topk(20).indices.tolist()); chosen.update((-d).topk(20).indices.tolist())
g = torch.Generator().manual_seed(0); used = torch.nonzero(U.sum(0) > 0)[:, 0].cpu()
chosen.update(used[torch.randperm(used.numel(), generator=g)[:60]].tolist()); chosen = sorted(chosen)
IMP = torch.zeros(nR, len(chosen))
for ti, rev in enumerate(REVS):
    model, tok, fam = load_model("pythia410", revision=None if rev == "main" else rev); arch = Arch(model, fam)
    with torch.no_grad(): base = token_loss(model(ids).logits.float(), ids).mean().item()
    for ci, n_ in enumerate(chosen):
        b, j = divmod(n_, DFF)
        def pre(m, a, j=j):
            x = a[0].clone(); x[..., j] = 0; return (x,) + tuple(a[1:])
        h = arch.mlp_lin(b).register_forward_pre_hook(pre)
        try:
            with torch.no_grad(): IMP[ti, ci] = token_loss(model(ids).logits.float(), ids).mean().item() - base
        finally: h.remove()
    log(f"lifecycle importance {rev}: median {IMP[ti].median().item():.5f}"); del model; torch.cuda.empty_cache()
def spear(x, y):
    rx, ry = x.argsort().argsort().double(), y.argsort().argsort().double(); rx, ry = rx - rx.mean(), ry - ry.mean(); return float((rx * ry).sum() / (rx.norm() * ry.norm()))
ch = torch.tensor(chosen); Uc, Sc = U[:, ch].cpu(), S[:, ch].cpu()
pct = lambda v: (v.argsort().argsort().float() / (v.numel() - 1))
res = dict(model="pythia410", revisions=REVS, n_selected=len(chosen), intervals=[])
for t in range(nR - 1):
    d = Uc[t + 1] - Uc[t]; births = d.topk(20).indices; deaths = (-d).topk(20).indices
    it = dict(interval=f"{REVS[t]}->{REVS[t + 1]}",
              births_importance_pct_before=float(pct(IMP[t])[births].median()), births_importance_pct_after=float(pct(IMP[t + 1])[births].median()),
              births_selectivity_pct_before=float(pct(Sc[t])[births].median()), births_selectivity_pct_after=float(pct(Sc[t + 1])[births].median()),
              deaths_importance_change=float((IMP[t + 1] - IMP[t])[deaths].median()), others_importance_change=float((IMP[t + 1] - IMP[t]).median()),
              lag_importance_to_usage=spear(IMP[t], Uc[t + 1] - Uc[t]), lag_usage_to_importance=spear(Uc[t], IMP[t + 1] - IMP[t]),
              lag_selectivity_to_usage=spear(Sc[t], Uc[t + 1] - Uc[t]), comove_usage_importance=spear(Uc[t + 1] - Uc[t], IMP[t + 1] - IMP[t]))
    res["intervals"].append(it)
    log(f"{it['interval']}: births importance pct {it['births_importance_pct_before']:.2f} -> {it['births_importance_pct_after']:.2f}, selectivity pct {it['births_selectivity_pct_before']:.2f} -> {it['births_selectivity_pct_after']:.2f} | "
        f"lag imp->use {it['lag_importance_to_usage']:+.2f} vs use->imp {it['lag_usage_to_importance']:+.2f} (sel->use {it['lag_selectivity_to_usage']:+.2f}) | co-move {it['comove_usage_importance']:+.2f} | deaths importance change {it['deaths_importance_change']:+.5f} (others {it['others_importance_change']:+.5f})")
I_ = res["intervals"]; m = lambda k: sum(x[k] for x in I_) / len(I_)
res["mean"] = {k: m(k) for k in ("lag_importance_to_usage", "lag_usage_to_importance", "lag_selectivity_to_usage", "comove_usage_importance", "births_importance_pct_before", "births_importance_pct_after")}
res["checks"] = dict(importance_leads=res["mean"]["lag_importance_to_usage"] > res["mean"]["lag_usage_to_importance"], comove_over_0_3=res["mean"]["comove_usage_importance"] > 0.3,
                     deaths_keep_importance=sorted(x["deaths_importance_change"] for x in I_)[len(I_) // 2] <= 0)
summ = (f"pythia410 word life cycle over {len(REVS)} checkpoints, {len(chosen)} neurons: mean over intervals: lag importance->usage {res['mean']['lag_importance_to_usage']:+.2f}, usage->importance {res['mean']['lag_usage_to_importance']:+.2f}, "
        f"selectivity->usage {res['mean']['lag_selectivity_to_usage']:+.2f}, co-move {res['mean']['comove_usage_importance']:+.2f}; births' importance percentile {res['mean']['births_importance_pct_before']:.2f} before -> {res['mean']['births_importance_pct_after']:.2f} after | "
        + " | ".join(f"{x['interval']}: births imp {x['births_importance_pct_before']:.2f}->{x['births_importance_pct_after']:.2f}, deaths dImp {x['deaths_importance_change']:+.4f}" for x in I_) + f" | checks {_json.dumps(res['checks'])}")
log(summ); record("e471_lifecycle_pythia410", res, summ)
