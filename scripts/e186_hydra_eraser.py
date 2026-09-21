"""e186: who erases when the eraser is gone. Clean vs ablate-MLP(b+1) forward passes; for dominant block-b writes
(b = 1..4), per-block direct MLP contributions to raw survival at level L in both runs, plus the non-MLP rest
(attention + embedding) as total minus MLP sum. Tests whether the cancellation moves to block b+2 (chain
compensation), spreads over all later blocks, or is taken over by attention."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); model, tok, fam = load_model(c.name); arch = Arch(model, fam); NS = 16; ids_seq = c.s["eval_ids"][:NS].to(DEV)
A, lab = c.dictionary(L); WN = [c.d["WN"][b].to(DEV) for b in range(L + 1)]; wd = [c.wdir_cpu(b).to(DEV) for b in range(L + 1)]; mb = [None if c.d["mlp_bias"][b] is None else c.d["mlp_bias"][b].to(DEV) for b in range(L + 1)]
def run(ablate_block=None):
    store = {"acts": {}, "H": None}; hs = []
    for b in range(L + 1): hs.append(arch.mlp_lin(b).register_forward_pre_hook((lambda b_: lambda m, inp: store["acts"].__setitem__(b_, inp[0].detach().float().reshape(-1, c.DFF)))(b)))
    hs.append(arch.layers[L].register_forward_hook(lambda m, i, o: store.__setitem__("H", (o[0] if isinstance(o, tuple) else o).detach().float().reshape(-1, c.D))))
    if ablate_block is not None: hs.append(arch.mlp_lin(ablate_block).register_forward_pre_hook(lambda m, inp: (torch.zeros_like(inp[0]),)))
    model(ids_seq); [h.remove() for h in hs]; return store["H"], store["acts"]
H0, acts0 = run(); typ = typical_mask(H0); rows = []
for b in [bb for bb in (1, 2, 3, 4) if bb + 2 <= L]:
    led_b = acts0[b] * WN[b][None]; tnv = led_b.abs().argmax(1); tcv = torch.gather(led_b, 1, tnv[:, None])[:, 0]; row = c.atom_index(L, torch.full_like(tnv.cpu(), b), tnv.cpu()).to(DEV); d = A[row]; big = typ & (tcv.abs() >= tcv.abs().quantile(0.5))
    prof = {}
    for nm, ab in (("clean", None), ("ablate_next", b + 1)):
        H1, acts1 = run(ab); tot = ((H1 * d).sum(1) / tcv)[big].mean().item(); per = []
        for bp in range(L + 1):
            w = acts1[bp] @ wd[bp]
            if mb[bp] is not None: w = w + mb[bp]
            v = ((w * d).sum(1) / tcv)[big]
            if bp == b: v = v - 1.0
            per.append(v.mean().item())
        prof[nm] = dict(total=tot, mlp=per, rest=tot - 1.0 - sum(per))
    later_clean = sum(prof["clean"]["mlp"][b + 2:]); later_abl = sum(prof["ablate_next"]["mlp"][b + 2:])
    rows.append(dict(birth=b, clean=prof["clean"], ablate_next=prof["ablate_next"], next_clean=prof["clean"]["mlp"][b + 1], plus2_clean=prof["clean"]["mlp"][b + 2], plus2_abl=prof["ablate_next"]["mlp"][b + 2], later_clean=later_clean, later_abl=later_abl, rest_clean=prof["clean"]["rest"], rest_abl=prof["ablate_next"]["rest"]))
    log(f"{tag} born b{b}: total {prof['clean']['total']:.2f}->{prof['ablate_next']['total']:.2f} | next-block direct {prof['clean']['mlp'][b + 1]:+.2f} (removed) | block b+2 {prof['clean']['mlp'][b + 2]:+.2f}->{prof['ablate_next']['mlp'][b + 2]:+.2f} | all blocks >= b+2: {later_clean:+.2f}->{later_abl:+.2f} | rest(att+emb) {prof['clean']['rest']:+.2f}->{prof['ablate_next']['rest']:+.2f} | own-block others {prof['clean']['mlp'][b]:+.2f}->{prof['ablate_next']['mlp'][b]:+.2f}")
record(f"e186_hydra_{tag}", dict(model=tag, L=L, rows=rows), " | ".join(f"b{r['birth']}: removed next {r['next_clean']:+.2f}; b+2 {r['plus2_clean']:+.2f}->{r['plus2_abl']:+.2f}; later sum {r['later_clean']:+.2f}->{r['later_abl']:+.2f}; rest {r['rest_clean']:+.2f}->{r['rest_abl']:+.2f}" for r in rows))
