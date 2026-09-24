"""e443: is the native vocabulary a union of per-block accents? e436 found that the middle block's own increment is
described by its own words only to second order (GPT-2 and SmolLM2: covA carries 0.43-0.65 of the own-over-rotation
gap, mix8 0.80-0.85). The state at the same depth is word-level (covA 0.02-0.28). Every earlier control was pooled over
blocks:
- covA drew Gaussian words with the second moment of the whole dictionary;
- mix8 mixed words of one family (MLP, attention, embedding) from all blocks.
If the state is a sum of block contributions, each second-order in its own block's words, a pooled control cannot
tell which block's subspace a component lies in, but a per-block control can.
Controls here, per model at middle depth (6 sequences, sinks exact), at k 4 and 16:
- covA_block: Gaussian words drawn with each (family, block) group's own second moment, same counts;
- mix8_block: signed sums of 8 words within each (family, block) group;
- the pooled covA and mix8 as before.
Arguments: model, optional revision (Pythia-410m checkpoints).
Pre-registered: covA_block recovers over half of the own-over-rotation gap at k 16 in at least four of five models;
the word-level character of the native vocabulary is then mostly which block wrote a component, not which word."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; rev = sys.argv[2] if len(sys.argv) > 2 else None
model, tok, fam = load_model(name, revision=rev); arch = Arch(model, fam); L = arch.NB // 2
ev = eval_ids(name)[:6].to(DEV)
A, lab = build_dictionary(arch, blocks=list(range(L + 1))); typ, blk = lab["type"].to(DEV), lab["block"].to(DEV)
fam_groups = [torch.nonzero(typ == t)[:, 0] for t in (T_TOK, T_POS, T_MLP, T_ATT, T_BIAS) if (typ == t).any()]
tb_groups = []
for t in (T_TOK, T_POS, T_MLP, T_ATT, T_BIAS):
    for b in torch.unique(blk[typ == t]).tolist():
        ix = torch.nonzero((typ == t) & (blk == b))[:, 0]
        if ix.numel(): tb_groups.append(ix)
covA_block = torch.empty_like(A)
for i, ix in enumerate(tb_groups):
    Ag = A[ix]; covA_block[ix] = gauss_like(ix.numel(), (Ag.T @ Ag) / ix.numel(), seed=100 + i)
V = dict(own=A, rot=rotate(A, seed=7), covA=gauss_like(A.shape[0], (A.T @ A) / A.shape[0], seed=1), covA_block=covA_block,
         mix8=mixtures(A, fam_groups, m=8, seed=4), mix8_block=mixtures(A, tb_groups, m=8, seed=5))
lv = NSLevel(model, arch, ev, L); cells = {vn: describe(lv, W, [4, 16]) for vn, W in V.items()}
rec = lambda vn, k: cells[vn][str(k)]["rec"]; gap = lambda k: rec("own", k) - rec("rot", k)
res = dict(model=name, rev=rev, level=L, n_groups=len(tb_groups), rec={vn: {k: rec(vn, k) for k in (4, 16)} for vn in V},
           shares={vn: {k: ((rec(vn, k) - rec("rot", k)) / gap(k) if abs(gap(k)) > 1e-3 else None) for k in (4, 16)} for vn in ("covA", "covA_block", "mix8", "mix8_block")})
S = res["shares"]; fm = lambda x: "n/a" if x is None else f"{x:.2f}"
res["checks"] = dict(covA_block_over_half=(S["covA_block"][16] or 0) > 0.5)
summ = (f"{name} {rev or 'final'} L{L} ({len(tb_groups)} groups): k4/k16 " + " ".join(f"{vn} {rec(vn, 4):.2f}/{rec(vn, 16):.2f}" for vn in V)
        + " | share of the own-over-rotation gap k16 (k4): " + " ".join(f"{vn} {fm(S[vn][16])} ({fm(S[vn][4])})" for vn in S) + f" | checks {json.dumps(res['checks'])}")
log(summ); record(f"e443_blockaccents_{name}_{rev or 'final'}", res, summ)
