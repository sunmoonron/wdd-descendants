"""e422: are the native language's function words the super weights? e414 found that the most used native words point
into the states' few huge principal directions (29-47% of their norm) and carry little function on their own; e418/e419
found that at 7B those directions hold 97% of the variance and 0.3% of the local Fisher trace yet recover 29% of the
loss alone. The established lens: massive activations (Sun et al. 2024) written by super weights (Yu et al. 2024), in
gated models an early MLP neuron whose activation spikes and whose down-projection writes a massive residual channel.
Qwen2.5-7B, 8 sequences of natural text (position 0 excluded as everywhere): (1) the super neuron, the MLP neuron with
the largest activation at the earliest spiking block (0-7), and its super weight, the largest entry of its write
column; the massive residual channels (largest |x| at the middle depth); (2) at the middle depth (block 14): the
states' top-8 principal subspace M and its share on the massive channels; usage of every native word across 16-word
OMP descriptions; where the super neuron's write ranks, its share of norm in M, and the identity of the top-10 words.
Pre-registered: the super neuron's write is among the 10 most used native words and more than half of its norm lies in
M; M is concentrated on the massive channels."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ws_common import *
from sd_common import Level
name = sys.argv[1] if len(sys.argv) > 1 else "qwen7"; K = 16
model, tok, fam = load_bf16(name); arch = Arch(model, fam); L = arch.NB // 2
E = torch.load("/workspace/wdd/eval_ids_all.pt", weights_only=False)["qwen05"]["eval_ids"]; ev = E[:8].to(DEV)
# (1) super neuron: max |MLP activation| per neuron at blocks 0-7 (positions 1:)
mx = {}
hs = [arch.mlp_lin(b).register_forward_pre_hook((lambda b_: lambda m, a: mx.__setitem__(b_, torch.maximum(mx.get(b_, torch.zeros(a[0].shape[-1], device=DEV)), a[0][:, 1:].float().abs().amax((0, 1)))))(b)) for b in range(8)]
xmax = {}
hx = arch.layers[L].register_forward_hook(lambda m, i, o: xmax.__setitem__("x", torch.maximum(xmax.get("x", torch.zeros(arch.D, device=DEV)), (o[0] if isinstance(o, tuple) else o)[:, 1:].float().abs().amax((0, 1)))))
try:
    with torch.no_grad():
        for s in range(ev.shape[0]): model(ev[s:s + 1])
finally: [h.remove() for h in hs]; hx.remove()
blockmax = {b: (mx[b].max().item(), mx[b].median().item()) for b in range(8)}
sb = next((b for b in range(8) if blockmax[b][0] / max(blockmax[b][1], 1e-9) > 1000), max(range(8), key=lambda b: blockmax[b][0]))
sn = int(mx[sb].argmax()); col = arch.layers[sb].mlp.down_proj.weight[:, sn].float(); sc = int(col.abs().argmax())
chan_order = xmax["x"].argsort(descending=True); massive = chan_order[:4].tolist()
log(f"{name}: per-block max/median MLP activation " + " ".join(f"b{b} {blockmax[b][0]:.0f}/{blockmax[b][1]:.3f}" for b in range(8))
    + f" | super neuron block {sb} #{sn}, super weight down_proj[{sc},{sn}] = {col[sc].item():.3f}; massive channels at block {L}: {massive} (|x| {xmax['x'][chan_order[:4]].tolist()})")
# (2) middle-depth native vocabulary and usage
A, blk, typ, ends = lean_dictionary(arch, L); A = A[:ends[L]]
lv = Level(model, arch, ev, L); SX = (lv.Xc.T @ lv.Xc) / lv.Xc.shape[0]; ev_, UX = torch.linalg.eigh(SX.double()); UM = UX[:, -8:].float()
M_on_massive = UM[massive].pow(2).sum().item() / 8                     # share of M's squared norm on the 4 massive channels
V0 = arch.emb[0].shape[0]; DFF = arch.wdir(0).shape[0]; per_block = DFF + arch.NH * arch.HD
sw_idx = V0 + sb * per_block + sn                                       # the super neuron's word in the prefix-ordered vocabulary
sel, _, _ = omp(lv.Xc, A, K, batch=128, record_err=False); cnt = torch.bincount(sel.flatten(), minlength=A.shape[0]).float()
order = cnt.argsort(descending=True); rank_sw = int((order == sw_idx).nonzero()[0, 0]) + 1 if cnt[sw_idx] > 0 else None
mshare = lambda i: (A[i] @ UM).pow(2).sum().item()
top = [dict(idx=int(i), block=int(blk[i]), type=int(typ[i]), uses=int(cnt[i]), mshare=mshare(int(i)), is_super=int(i) == sw_idx,
            massive_share=A[i][massive].pow(2).sum().item()) for i in order[:10]]
res = dict(model=name, level=L, blockmax=blockmax, super_block=sb, super_neuron=sn, super_weight_coord=[sc, sn], super_weight=col[sc].item(),
           massive_channels=massive, top8_var=(ev_[-8:].sum() / ev_.sum()).item(), M_on_massive=M_on_massive, super_word_rank=rank_sw,
           super_word_uses=int(cnt[sw_idx]), super_word_mshare=mshare(sw_idx), super_word_massive_share=A[sw_idx][massive].pow(2).sum().item(),
           share_positions_using_super=(sel == sw_idx).any(-1).float().mean().item(), top_words=top)
for w in top: log(f"  top word block {w['block']} type {w['type']} uses {w['uses']} M-share {w['mshare']:.2f} massive-channel share {w['massive_share']:.2f}{' <- SUPER NEURON' if w['is_super'] else ''}")
summ = (f"{name}: super neuron b{sb}#{sn} (weight [{sc},{sn}] {col[sc].item():.2f}); its native word ranks {rank_sw} in usage at block {L} (used at {res['share_positions_using_super']:.2f} of positions), "
        f"M-share {res['super_word_mshare']:.2f}, massive-channel share {res['super_word_massive_share']:.2f} | top-8 PCs hold {res['top8_var']:.2f} of variance, {M_on_massive:.2f} of M on the 4 massive channels {massive} | "
        f"top-10 words' mean M-share {sum(w['mshare'] for w in top) / 10:.2f}")
log(summ); record(f"e422_superweight_{name}", res, summ)
