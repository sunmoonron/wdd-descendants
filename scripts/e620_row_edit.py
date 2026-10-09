"""e620 (session 118): parameter-identified concept editing. e477 found one MLP row of block 4 in 91% of the items' own
gender supports on Qwen2.5-0.5B. Here that row is found again by vote (OMP of each item's sign-aligned gender difference
on the native rows of blocks up to b, the most chosen atom per block) and then edited in the weights: its write column
zeroed, negated, doubled, and its inputs silenced, against eight random MLP rows of the same block. Scored on e473's
items (the argmax candidate: kept, gender flipped, generation flipped, both), the behavioural gender score (log-probability
of the gender-flipped candidate minus the item's own), and the loss on Pile windows. Pre-registered in e619_prereg.json:
E1 (0.3) negating the one row flips gender in a quarter of the items or more (random rows 0.05 or less); E2 (0.6) zeroing
it moves the gender score toward the opposite gender with a Pile loss change under 0.02 nats; E3 (0.5) its effect on the
gender score exceeds every one of the eight random rows'. Arguments: model [--smoke].
"""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "e473_word_algebra.py")).read()
exec(src[src.index("QUADS = {"): src.index("dif = {")], globals())
SMOKE = "--smoke" in sys.argv
if SMOKE: items = items[:16]; n = len(items)          # items, model, arch, L, Au, Ar, ends, run, scores, ...
KS = [4, 16]; BL = list(range(L + 1)); KMAX = 16
q_of = [it["q"] for it in items]; lg_of = [it["lg"] for it in items]
dif = {j: torch.stack([it["X"][j] - it["X"][0] for it in items], 1) for j in (1, 2)}          # [L+1, n, D]: 1 = gender, 2 = generation
# sign-aligned copies for the means: an item whose base word already has the attribute flipped (i & j) has a difference
# pointing the other way (queen -> king against king -> queen), so the group mean would cancel (v1's bug)
sgn = {j: torch.tensor([-1.0 if (it["i"] & j) else 1.0 for it in items], device=DEV) for j in (1, 2)}
dif_al = {j: dif[j] * sgn[j][None, :, None] for j in (1, 2)}

# ---- e620: the gender word by vote, then the weight edits
from wdd_common import omp as _omp
from datasets import load_dataset
mean = lambda v: sum(v) / max(len(v), 1)
VOTE = {}
for b in range(L + 1):
    X_b = dif_al[1][b]; sel, _, _ = _omp(X_b, Au[:ends[b]], 8, batch=256, record_err=False); cnt = torch.bincount(sel.reshape(-1), minlength=ends[b]); top = int(cnt.argmax()); VOTE[b] = (top, float(cnt[top]) / X_b.shape[0])
bg = max(VOTE, key=lambda b: VOTE[b][1]); g_atom, g_share = VOTE[bg]; keys = list(lab.keys()); log(f"vote: block {bg} atom {g_atom} in {g_share:.2f} of items' gender descriptions; label keys {keys}; " + ", ".join(f"{k}={lab[k][g_atom].item() if hasattr(lab[k], 'shape') else lab[k][g_atom]}" for k in keys))
typ = lab["type"][g_atom] if "type" in lab else None; row_j = int(lab["index"][g_atom]) if "index" in lab else int(lab["row"][g_atom]) if "row" in lab else None; blk_g = int(lab["block"][g_atom])
assert (typ in (2, "mlp", None)) and row_j is not None, (typ, row_j)
pile = load_dataset("NeelNanda/pile-10k", split="train"); buf, wins = [], []
for ex in pile:
    buf += tok(ex["text"])["input_ids"] + [tok.eos_token_id]
    while len(buf) >= 257 and len(wins) < (4 if SMOKE else 16): wins.append(buf[:257]); buf = buf[257:]
    if len(wins) >= (4 if SMOKE else 16): break
PW = torch.tensor(wins, device=DEV)
def pile_loss(m):
    tot = 0.0
    with torch.no_grad():
        for s0 in range(0, PW.shape[0], 8):
            x = PW[s0:s0 + 8]; lg = m(x).logits.float(); tot += float(torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), x[:, 1:].reshape(-1))) * x.shape[0]; del lg
    return tot / PW.shape[0]
def gender_scores():
    outcome = torch.zeros(4); sg = []
    for it in items:
        lp, _ = run(it["ids"], it["p"], it["cand"]); outcome[int(lp.argmax()) ^ it["i"]] += 1; sg.append(float(lp[it["i"] ^ 1] - lp[it["i"]]))
    return (outcome / n).tolist(), mean(sg)
mlp_of = lambda b: arch.layers[b].mlp
def edit(kind, b, j, saved):
    l = mlp_of(b)
    with torch.no_grad():
        if kind == "zero": l.down_proj.weight[:, j] = 0
        elif kind == "negate": l.down_proj.weight[:, j] = -saved["down"]
        elif kind == "double": l.down_proj.weight[:, j] = 2 * saved["down"]
        elif kind == "silence": l.gate_proj.weight[j, :] = 0; l.up_proj.weight[j, :] = 0
def restore(b, j, saved):
    l = mlp_of(b)
    with torch.no_grad(): l.down_proj.weight[:, j] = saved["down"]; l.gate_proj.weight[j, :] = saved["gate"]; l.up_proj.weight[j, :] = saved["up"]
def snapshot(b, j): l = mlp_of(b); return dict(down=l.down_proj.weight[:, j].clone(), gate=l.gate_proj.weight[j, :].clone(), up=l.up_proj.weight[j, :].clone())
out0, sg0 = gender_scores(); L0 = pile_loss(model); log(f"unedited: outcomes kept/gender/generation/both {out0}, gender score {sg0:+.3f}, pile loss {L0:.3f}")
res = dict(model=name, level=L, n_items=n, vote={str(b): dict(atom=a, share=s) for b, (a, s) in VOTE.items()}, gender_word=dict(block=blk_g, row=row_j, atom=g_atom, share=g_share), unedited=dict(outcomes=out0, gender_score=sg0, pile_loss=L0), edits={}, random_rows={})
sv = snapshot(blk_g, row_j)
for kind in ("zero", "negate", "double", "silence"):
    edit(kind, blk_g, row_j, sv); o, sg = gender_scores(); Lp = pile_loss(model); restore(blk_g, row_j, sv)
    res["edits"][kind] = dict(outcomes=o, gender_score=sg, gender_shift=sg - sg0, pile_loss_change=Lp - L0); log(f"edit {kind} on block {blk_g} row {row_j}: outcomes {o}, gender score {sg:+.3f} ({sg - sg0:+.3f}), pile loss {Lp - L0:+.4f}")
g_r = torch.Generator().manual_seed(11); DFFm = mlp_of(blk_g).down_proj.weight.shape[1]
for r in torch.randperm(DFFm, generator=g_r)[:(2 if SMOKE else 8)].tolist():
    if r == row_j: continue
    sv_r = snapshot(blk_g, r); out_r = {}
    for kind in ("zero", "negate"):
        edit(kind, blk_g, r, sv_r); o, sg = gender_scores(); Lp = pile_loss(model); restore(blk_g, r, sv_r); out_r[kind] = dict(outcomes=o, gender_score=sg, gender_shift=sg - sg0, pile_loss_change=Lp - L0)
    res["random_rows"][str(r)] = out_r
rz = [v["negate"]["outcomes"][1] for v in res["random_rows"].values()]; rs = [abs(v["zero"]["gender_shift"]) for v in res["random_rows"].values()]
e1 = res["edits"]["negate"]["outcomes"][1] >= 0.25 and (max(rz) if rz else 0) <= 0.05; e2 = res["edits"]["zero"]["gender_shift"] > 0 and abs(res["edits"]["zero"]["pile_loss_change"]) < 0.02; e3 = all(abs(res["edits"]["zero"]["gender_shift"]) > x for x in rs) if rs else False
res["verdicts"] = dict(E1=e1, E2=e2, E3=e3)
summ = (f"single-row edit ({name}, block {blk_g} row {row_j}, in {g_share:.2f} of items' gender descriptions; {n} items): unedited gender-flipped {out0[1]:.2f}, score {sg0:+.3f}; " + "; ".join(f"{k}: flipped {v['outcomes'][1]:.2f}, score shift {v['gender_shift']:+.3f}, loss {v['pile_loss_change']:+.4f}" for k, v in res["edits"].items()) + f"; random rows negated flip {max(rz) if rz else 0:.2f} at most, zeroed shift {max(rs) if rs else 0:.3f} at most; verdicts E1 {e1}, E2 {e2}, E3 {e3}")
log(summ); record(f"e620_row_edit_{name}" + ("_smoke" if SMOKE else ""), res, summ)
