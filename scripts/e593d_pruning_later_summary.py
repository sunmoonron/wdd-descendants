"""e593d (session 110): the runs of e593c (pruning at 3000 or 3500) summarised per condition (four seeds each): the grokking step, the final
test accuracy, the advantage, and the share of the base run's eventual words that are words at the end, with the
share of those words that had been pruned."""
from s101_common import *
import glob
runs = {}
for f in sorted(glob.glob("/workspace/wdd/results/e593c_prune_*.json")):
    d = json.load(open(f)); runs.setdefault(d["condition"], []).append(d)
res = {}
for c in ("low3000", "random3000", "high3000", "low3843000", "low3500", "random3500", "high3500"):
    L = runs.get(c, [])
    if not L: continue
    gs = [d["grok_step"] for d in L]; res[c] = dict(n=len(L), pruned=L[0]["n_frozen"], base_words_pruned=mean([d["base_words_in_frozen"] for d in L]), grok_steps=gs, grok_median=med([x for x in gs if x is not None]) if any(x is not None for x in gs) else None, share_grokked=mean([float(x is not None) for x in gs]), test_acc=mean([d["final"]["test_acc"] for d in L]), advantage=mean([d["final"]["advantage"] for d in L]), base_words_final=mean([d["base_words_final"] for d in L]))
    r = res[c]; log(f"{c}: {r['pruned']} pruned ({r['base_words_pruned']:.2f} of the base words); grok steps {gs} (median {r['grok_median']}, grokked {r['share_grokked']:.2f}); final test acc {r['test_acc']:.2f}, advantage {r['advantage']:.2f}, base words at the end {r['base_words_final']:.2f}")
f_ = lambda c, k: res[c][k] if c in res else None
CC = ("low3000", "random3000", "high3000", "low3843000", "low3500", "random3500", "high3500"); summ = "pruning at 3000 / 3500 (four seeds): grok step median low3000 / random3000 / high3000 / low384 at 3000 / low3500 / random3500 / high3500: " + " / ".join(str(f_(c, "grok_median")) for c in CC) + "; final test accuracy " + " / ".join(f"{f_(c, 'test_acc'):.2f}" if f_(c, 'test_acc') is not None else "n/a" for c in CC) + "; the base words among the pruned " + " / ".join(f"{f_(c, 'base_words_pruned'):.2f}" if f_(c, 'base_words_pruned') is not None else "n/a" for c in CC) + "; base words at the end " + " / ".join(f"{f_(c, 'base_words_final'):.2f}" if f_(c, 'base_words_final') is not None else "n/a" for c in CC)
log(summ); record("e593d_pruning_later_summary", res, summ)
