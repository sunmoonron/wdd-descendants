"""e500: which kind of atom becomes a word, and when. e498 found that over Pythia's training the state's energy moves
from MLP writes (0.94 of the state at block 12 at step 256) to attention (0.75 at the end), while the words that form
are MLP rows (e389, at the end). So the vocabulary forms in the component that carries less and less of the state.
This run follows the composition of the 16-word native description through training: the share of words by type
(token embedding, MLP row, head basis), and the loss recovered by each type's atoms alone against the full dictionary
and the rotated one.
Setup: Pythia-410m at a checkpoint (argument), blocks 6, 12 and 18; 8 x 256 evaluation tokens, typical positions;
16-word OMP over the full dictionary up to the block, over the MLP rows only, the head bases only, the token embeddings
only, and the rotated full dictionary; loss recovered by splicing.
Arguments: name [revision]."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; rev = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] not in ("none", "main") else None
model, tok, fam = load_model(name, revision=rev); arch = Arch(model, fam); NB = arch.NB; D = arch.D; K = 16
blocks = sorted({NB // 4, NB // 2, (3 * NB) // 4}); ids = eval_ids(name)[:8, :256].to(DEV)
S_ = block_states(model, arch, ids, blocks, chunk=4)
res = dict(model=name, revision=rev, blocks=blocks, by_block={})
for L in blocks:
    sp = Splicer(model, arch, ids, L, chunk=2); X = S_[L]; flat = X.reshape(-1, D); keep = ~sinkmask(flat); mu = flat[keep].mean(0); Xc = flat[keep] - mu
    lm = sp.lossmask(keep.view(X.shape[0], -1)); base = sp.c["loss"][lm].mean().item(); ms = flat.clone(); ms[keep] = mu; mean_loss = sp.run(ms.view_as(X))["loss"][lm].mean().item()
    def rec(Xh): new = flat.clone(); new[keep] = mu + Xh; return (mean_loss - sp.run(new.view_as(X))["loss"][lm].mean().item()) / max(mean_loss - base, 1e-9)
    A, lab = build_dictionary(arch, blocks=list(range(L + 1))); typ = lab["type"].to(DEV); blk = lab["block"].to(DEV); Au = unitr(A); Ar = unitr(rotate(A, seed=7)); del A
    out = {}
    for kind, Dct in (("full", Au), ("mlp_only", Au[typ == T_MLP]), ("heads_only", Au[typ == T_ATT]), ("tokens_only", Au[typ == T_TOK]), ("rotated", Ar)):
        sel, _, _ = omp(Xc, Dct, K, batch=1024, record_err=False); cof, _ = refit(Xc, Dct, sel); out[kind] = dict(loss_recovered=rec(torch.einsum("nk,nkd->nd", cof, Dct[sel])), fvu=float(((Xc - torch.einsum("nk,nkd->nd", cof, Dct[sel])).pow(2).sum(1) / Xc.pow(2).sum(1)).median()))
        if kind == "full":
            t = typ[sel]; out["type_mix"] = dict(token=float((t == T_TOK).float().mean()), mlp=float((t == T_MLP).float().mean()), head=float((t == T_ATT).float().mean()), pos=float((t == T_POS).float().mean()))
            bb = blk[sel].float(); out["mlp_words_from_last_two_blocks"] = float(((t == T_MLP) & (bb >= L - 1)).float().sum() / (t == T_MLP).float().sum().clamp_min(1))
    res["by_block"][L] = out
    log(f"{name}{' ' + rev if rev else ''} block {L}: loss recovered full {out['full']['loss_recovered']:.2f}, MLP rows only {out['mlp_only']['loss_recovered']:.2f}, heads only {out['heads_only']['loss_recovered']:.2f}, tokens only {out['tokens_only']['loss_recovered']:.2f}, rotated {out['rotated']['loss_recovered']:.2f} | word types in the full description: token {out['type_mix']['token']:.2f} MLP {out['type_mix']['mlp']:.2f} head {out['type_mix']['head']:.2f}; MLP words from the last two blocks {out['mlp_words_from_last_two_blocks']:.2f}")
    del Au, Ar, sp; torch.cuda.empty_cache()
Bk = res["by_block"]
summ = f"{name}{' ' + rev if rev else ''}: by block " + " | ".join(f"{L}: recovered full/MLP/heads/tokens/rotated {Bk[L]['full']['loss_recovered']:.2f}/{Bk[L]['mlp_only']['loss_recovered']:.2f}/{Bk[L]['heads_only']['loss_recovered']:.2f}/{Bk[L]['tokens_only']['loss_recovered']:.2f}/{Bk[L]['rotated']['loss_recovered']:.2f}, types token/MLP/head {Bk[L]['type_mix']['token']:.2f}/{Bk[L]['type_mix']['mlp']:.2f}/{Bk[L]['type_mix']['head']:.2f}" for L in blocks)
log(summ); record(f"e500_typemix_{name}{'_' + rev if rev else ''}", res, summ)
