"""e593b (session 110): the twenty runs of e593 summarised per condition (four seeds each): the grokking step, the final
test accuracy, the advantage, and the share of the base run's eventual words that are words at the end, with the
share of those words that had been pruned."""
from s101_common import *
import glob
runs = {}
for f in sorted(glob.glob("/workspace/wdd/results/e593_prune_*.json")):
    d = json.load(open(f)); runs.setdefault(d["condition"], []).append(d)
res = {}
for c in ("none", "low", "random", "high", "low384"):
    L = runs.get(c, [])
    if not L: continue
    gs = [d["grok_step"] for d in L]; res[c] = dict(n=len(L), pruned=L[0]["n_frozen"], base_words_pruned=mean([d["base_words_in_frozen"] for d in L]), grok_steps=gs, grok_median=med([x for x in gs if x is not None]) if any(x is not None for x in gs) else None, share_grokked=mean([float(x is not None) for x in gs]), test_acc=mean([d["final"]["test_acc"] for d in L]), advantage=mean([d["final"]["advantage"] for d in L]), base_words_final=mean([d["base_words_final"] for d in L]))
    r = res[c]; log(f"{c}: {r['pruned']} pruned ({r['base_words_pruned']:.2f} of the base words); grok steps {gs} (median {r['grok_median']}, grokked {r['share_grokked']:.2f}); final test acc {r['test_acc']:.2f}, advantage {r['advantage']:.2f}, base words at the end {r['base_words_final']:.2f}")
f_ = lambda c, k: res[c][k] if c in res else None
summ = "recruitment-guided pruning at step 1500 (four seeds): grok step median none / low-S half / random half / high-S half / lowest 384: " + " / ".join(str(f_(c, "grok_median")) for c in ("none", "low", "random", "high", "low384")) + "; final test accuracy " + " / ".join(f"{f_(c, 'test_acc'):.2f}" if f_(c, 'test_acc') is not None else "n/a" for c in ("none", "low", "random", "high", "low384")) + "; the base words among the pruned " + " / ".join(f"{f_(c, 'base_words_pruned'):.2f}" if f_(c, 'base_words_pruned') is not None else "n/a" for c in ("none", "low", "random", "high", "low384")) + "; base words at the end " + " / ".join(f"{f_(c, 'base_words_final'):.2f}" if f_(c, 'base_words_final') is not None else "n/a" for c in ("none", "low", "random", "high", "low384"))
log(summ); record("e593b_pruning_summary", res, summ)
