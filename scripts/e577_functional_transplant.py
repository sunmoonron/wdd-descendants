"""e577 (session 103): does a transplanted direction carry computation, or only the label? e565 showed that copying a
later neuron's write direction into the earlier model makes the row a word (its S rises to the later value). Here the
next-token distribution of the step-8000 model at the entrants' classes (their over-the-floor positions at 16000) with
and without the transplant: the KL of the change, at the class positions against all other positions, and whether the
change moves the 8000 model toward the 16000 model there (the KL to the 16000 model's distribution before and after).
Transplants: the write direction at the old norm (Wd) for all entrants at once, the whole neuron (WRb), a random
Pile word's direction into the same neurons (a control), and the entrants' directions into random other neurons (the
label without the neuron's selectivity). Pre-registered (probabilities are honest guesses):
 T4 (0.7) the transplant's KL at the class positions is under 0.01 nats: the label travels, the computation does not;
 T5 (0.6) the whole neuron moves the 8000 model toward the 16000 model at the class positions by more than the
    direction alone, but by under a tenth of the gap."""
from s101_common import *
t0 = time.time(); B = 12; ids = pile_ids("pythia410")
m8, tok, fam = load_model("pythia410", revision="step8000"); m16, _, _ = load_model("pythia410", revision="step16000"); arch8, arch16 = Arch(m8, fam), Arch(m16, fam); DFF = arch8.DFF
S8 = lm_states("pythia410", B=B, ids=ids, model=m8); S16 = lm_states("pythia410", B=B, ids=ids, model=m16); keepc = S8["keep"] & S16["keep"]; kidx = torch.nonzero(keepc)[:, 0]
for S_ in (S8, S16): S_["U"] = unitr(S_["X"][keepc] - S_["X"][keepc].mean(0))
st8 = stats(S8["U"], S8["A"], K); st16 = stats(S16["U"], S16["A"], K); w8, w16 = wordset(st8["usage"]), wordset(st16["usage"]); ent = torch.nonzero(w16 & ~w8)[:, 0]
cls = (st16["ratio"][:, ent].float() > 1).any(1); classmask = torch.zeros(keepc.numel(), dtype=torch.bool); classmask[kidx[cls]] = True; other = torch.zeros(keepc.numel(), dtype=torch.bool); other[kidx[~cls]] = True
pm_c = torch.zeros(ids.shape, dtype=torch.bool, device=DEV); pm_c[:, 1:] = classmask.reshape(ids.shape[0], -1).to(DEV); pm_o = torch.zeros_like(pm_c); pm_o[:, 1:] = other.reshape(ids.shape[0], -1).to(DEV)
log(f"entrants {ent.numel()}, class positions {int(cls.sum())} of {int(keepc.sum())}")
def logp(model):
    with torch.no_grad(): return model(ids).logits.float().log_softmax(-1)
def kl(a, b, pm): return float(((a.exp() * (a - b)).sum(-1))[pm].mean())
L8, L16 = logp(m8), logp(m16); gap = dict(cls=kl(L16, L8, pm_c), other=kl(L16, L8, pm_o)); log(f"gap 16000 vs 8000: KL at the class positions {gap['cls']:.4f}, elsewhere {gap['other']:.4f}")
def transplant(rows, parts, src_rows=None, direction_only=False):
    saved = []; src_rows = rows if src_rows is None else src_rows
    with torch.no_grad():
        for r, s in zip(rows.tolist(), src_rows.tolist()):
            b, j = r // DFF, r % DFF; bs, js = s // DFF, s % DFF; l8, l16 = arch8.layers[b].mlp, arch16.layers[bs].mlp
            saved.append((b, j, l8.dense_4h_to_h.weight[:, j].clone(), l8.dense_h_to_4h.weight[j].clone(), l8.dense_h_to_4h.bias[j].clone()))
            if "W" in parts:
                src = l16.dense_4h_to_h.weight[:, js]; l8.dense_4h_to_h.weight[:, j] = src / src.norm() * l8.dense_4h_to_h.weight[:, j].norm() if direction_only else src
            if "R" in parts: l8.dense_h_to_4h.weight[j] = l16.dense_h_to_4h.weight[js]
            if "b" in parts: l8.dense_h_to_4h.bias[j] = l16.dense_h_to_4h.bias[js]
    def restore():
        with torch.no_grad():
            for b, j, wc, wr, bb in saved: arch8.layers[b].mlp.dense_4h_to_h.weight[:, j] = wc; arch8.layers[b].mlp.dense_h_to_4h.weight[j] = wr; arch8.layers[b].mlp.dense_h_to_4h.bias[j] = bb
    return restore
g = torch.Generator().manual_seed(0); pw = torch.nonzero(w16)[:, 0]; rnd_words = pw[torch.randint(0, pw.numel(), (ent.numel(),), generator=g)]; nonw = torch.nonzero(~w8 & ~w16)[:, 0]; rnd_neurons = nonw[torch.randperm(nonw.numel(), generator=g)[:ent.numel()]]
res = dict(n_entrants=int(ent.numel()), n_class_positions=int(cls.sum()), gap=gap, transplants={})
for tag, rows, parts, src, donly in (("direction", ent, "W", None, True), ("whole_neuron", ent, "WRb", None, False), ("random_word_direction", ent, "W", rnd_words, True), ("direction_into_random_neurons", rnd_neurons, "W", ent, True)):
    restore = transplant(rows, parts, src, donly); Lx = logp(m8); restore()
    r = dict(kl_change_class=kl(L8, Lx, pm_c), kl_change_other=kl(L8, Lx, pm_o), kl_to_16000_class_after=kl(L16, Lx, pm_c), kl_to_16000_other_after=kl(L16, Lx, pm_o))
    r["moved_toward_16000_class"] = gap["cls"] - r["kl_to_16000_class_after"]; r["moved_share_of_gap"] = r["moved_toward_16000_class"] / gap["cls"]; res["transplants"][tag] = r
    log(f"{tag}: change at the class positions KL {r['kl_change_class']:.5f} (elsewhere {r['kl_change_other']:.5f}); KL to 16000 at the class {gap['cls']:.4f} -> {r['kl_to_16000_class_after']:.4f} (moved {r['moved_toward_16000_class']:+.5f}, {r['moved_share_of_gap']:+.3f} of the gap)")
T = res["transplants"]
summ = (f"functional transplant into the 8000 model ({ent.numel()} entrants, {int(cls.sum())} class positions; gap to 16000 there {gap['cls']:.4f} nats): the direction alone changes the class positions by KL {T['direction']['kl_change_class']:.5f} (elsewhere {T['direction']['kl_change_other']:.5f}) and moves them toward 16000 by {T['direction']['moved_share_of_gap']:+.3f} of the gap; the whole neuron {T['whole_neuron']['kl_change_class']:.5f}, {T['whole_neuron']['moved_share_of_gap']:+.3f}; a random word's direction {T['random_word_direction']['kl_change_class']:.5f}, {T['random_word_direction']['moved_share_of_gap']:+.3f}; the entrants' directions into random neurons {T['direction_into_random_neurons']['kl_change_class']:.5f}, {T['direction_into_random_neurons']['moved_share_of_gap']:+.3f} | {time.time() - t0:.0f}s")
log(summ); record("e577_functional_transplant", res, summ)
