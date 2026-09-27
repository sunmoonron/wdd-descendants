"""e521: which earlier writes move the state toward a future word? Sessions 72-75 established that a row's largest
projection over the floor, S, predicts its entry into the vocabulary at the next checkpoint, and that neither the
row's own write, nor the position's chord, nor any part of the state, nor the composition of the projection explains
the prediction beyond S. The remaining question is what computation puts the state along a future word's direction.
This run attributes the alignment to its causal sources by transport: at an origin checkpoint, for each row that
will be a word at the next checkpoint (and for a non-entrant matched on S), at the position of its largest
projection, the gradient of <x_b(p), w_i> with respect to every earlier block's residual at every position, times the
increments there (the first-order descendant account of areas 06-07: d_j = c_j J w_j), gives the share of the
alignment attributed to every earlier MLP write, every attention output and the embeddings, at the same position and
at others. The attribution's linearity is checked against the projection, and the top-attributed writes are checked
by ablation (the write zeroed in a forward pass, the projection recomputed).
Per row: the attributed alignment from MLP writes at the same position and at other positions, from attention
outputs and from the embeddings; the direct part (the writes' own vectors at the position, which e519 measured) and
the transported remainder; the share from rows that are words at the origin; the participation ratio of the write
attributions; the top five attributed writes (position, block, whether a word) and their ablation effects; and, for
the unembedding hypothesis, the row's cosine with the unembedding's top directions. Entrants against matched
non-entrants, medians and AUCs.
Setup: Pythia-410m at the origin checkpoint (argument), block 12 (and 6); 8 x 256 evaluation tokens; the next
checkpoint's word set from e519's records.
Pre-registered (honest guesses), block 12:
- more than half of the entrants' attributed alignment comes from writes at other positions, transported by
  attention (0.5);
- the writes that push the state toward a future word are mostly rows that are already words (share above 0.5) (0.5);
- the direct part is under a third of the attribution; the rest is transport (0.5);
- the unembedding alignment does not separate entrants from matched non-entrants (AUC under 0.6) (0.6).
Arguments: name revision (the origin checkpoint)."""
import sys, os, json as _json, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; rev = sys.argv[2]; STEPS = ["step0", "step64", "step256", "step512", "step1000", "step2000", "step3000", "step4000", "step8000", "step16000", "step32000", "step64000", "main"]
assert rev in STEPS[:-1], rev; NEXTS = STEPS[STEPS.index(rev) + 1]; CDIR = f"/workspace/wdd/cache/e519_{name}"; NWORD = 256; NR = 160; TOPK = 5; BL = [12, 6]
ref = {s: torch.load(f"{CDIR}/{s}.pt") for s in (rev, NEXTS)}
model, tok, fam = load_model(name, revision=None if rev == "main" else rev); arch = Arch(model, fam); NB = arch.NB; D = arch.D; DFF = arch.DFF; ids = eval_ids(name)[:8, :256].to(DEV); B_, T_ = ids.shape
g = torch.Generator(device=DEV).manual_seed(0)
def auc(pos, neg):
    if pos.numel() < 5 or neg.numel() < 5: return None
    x = torch.cat([pos, neg]); r = x.argsort().argsort().double() + 1; npos, nneg = pos.numel(), neg.numel(); return float((r[:npos].sum() - npos * (npos + 1) / 2) / (npos * nneg))
WU = (model.embed_out.weight if hasattr(model, "embed_out") else model.lm_head.weight).detach().float(); U_top = torch.linalg.svd(WU - WU.mean(0), full_matrices=False).Vh[:64]           # the unembedding's top 64 directions
res = dict(model=name, origin=rev, horizon=NEXTS, by_block={})
for b in BL:
    # word sets, entrants and matched non-entrants
    u0, u1 = ref[rev][b]["usage"], ref[NEXTS][b]["usage"]; S0 = ref[rev][b]["max_over_floor"]["real"].float()
    W0 = set(torch.nonzero(u0 > 0)[:, 0][u0[u0 > 0].argsort(descending=True)[:NWORD]].tolist()); W1 = set(torch.nonzero(u1 > 0)[:, 0][u1[u1 > 0].argsort(descending=True)[:NWORD]].tolist())
    R = u0.numel(); in0 = torch.zeros(R, dtype=torch.bool); in0[list(W0)] = True; in1 = torch.zeros(R, dtype=torch.bool); in1[list(W1)] = True
    ent = torch.nonzero(~in0 & in1)[:, 0]; ent = ent[torch.randperm(ent.numel(), generator=torch.Generator().manual_seed(0))[:NR]]; pool = torch.nonzero(~in0 & ~in1)[:, 0]; taken = torch.zeros(R, dtype=torch.bool); matched = []
    for i in ent.tolist():
        cand = pool[~taken[pool]]; j = cand[(S0[cand] - S0[i]).abs().argmin()]; taken[j] = True; matched.append(int(j))
    matched = torch.tensor(matched); rows_sel = torch.cat([ent, matched]).to(DEV); is_ent = torch.cat([torch.ones(ent.numel()), torch.zeros(matched.numel())]).bool()
    # a forward pass with the graph kept, capturing the embedding output, every block's output, attention and MLP increments, and the MLP activations
    cap = dict(out={}, attn={}, mlp={}, act={})
    def h_out(bb):
        def f(m, i, o): cap["out"][bb] = o[0] if isinstance(o, tuple) else o
        return f
    def h_attn(bb):
        def f(m, i, o): cap["attn"][bb] = (o[0] if isinstance(o, tuple) else o).detach()
        return f
    def h_mlp(bb):
        def f(m, i, o): cap["mlp"][bb] = (o[0] if isinstance(o, tuple) else o).detach()
        return f
    def h_act(bb):
        def f(m, a): cap["act"][bb] = a[0].detach().float(); return None
        return f
    def h_emb(m, i, o): o.requires_grad_(True); cap["emb"] = o
    def h_stop(m, i, o): raise Stop
    layers = arch.layers; attn_mod = lambda bb: layers[bb].attention if fam == "neox" else (layers[bb].attn if fam == "gpt2" else layers[bb].self_attn); emb_mod = model.gpt_neox.embed_in if fam == "neox" else (model.transformer.wte if fam == "gpt2" else model.model.embed_tokens)
    hs = [layers[bb].register_forward_hook(h_out(bb)) for bb in range(b + 1)] + [attn_mod(bb).register_forward_hook(h_attn(bb)) for bb in range(b + 1)] + [layers[bb].mlp.register_forward_hook(h_mlp(bb)) for bb in range(b + 1)] + [arch.mlp_lin(bb).register_forward_pre_hook(h_act(bb)) for bb in range(b + 1)] + [emb_mod.register_forward_hook(h_emb), layers[b].register_forward_hook(h_stop)]
    with torch.enable_grad():
        for p in model.parameters(): p.requires_grad_(False)
        try: model(ids)
        except Stop: pass
        finally: [h.remove() for h in hs]
    xb = cap["out"][b].detach().float(); Xall = xb.reshape(-1, D); keep = ~sinkmask(Xall[:, :]); mu = Xall[keep].mean(0)
    Wd = {bb: arch.wdir(bb).float() for bb in range(b + 1)}
    A, lab = build_dictionary(arch, blocks=list(range(b + 1))); Au = unitr(A); del A; typ = lab["type"].to(DEV); blk = lab["block"].to(DEV); idx = lab["index"].to(DEV)
    gidx = torch.full((b + 1, DFF), -1, device=DEV, dtype=torch.long)
    for bb in range(b + 1): mm = (typ == T_MLP) & (blk == bb); gidx[bb, idx[mm]] = torch.nonzero(mm)[:, 0]
    rows_atoms = gidx.reshape(-1)[rows_sel]; Wsel = Au[rows_atoms]; del Au
    # each selected row's best position (largest centred projection over the kept positions, in units of the state norm)
    Xc = Xall - mu; xn = Xc.norm(dim=-1).clamp_min(1e-9); P = (Xc @ Wsel.T) / xn[:, None]; P[~keep] = 0; best = P.abs().argmax(0); bi, ti = best // T_, best % T_
    is_word0 = in0.to(DEV)
    quant = {k: [] for k in ("same_pos_mlp", "other_pos_mlp", "attention", "embedding", "total_attr", "projection", "direct_mlp", "transported_mlp", "word_share", "participation", "top_same_pos", "top_word", "top_block_gap", "abl_spearman", "abl_ratio", "unemb_cos")}
    for r in range(rows_sel.numel()):
        w = Wsel[r]
        with torch.enable_grad(): f = (cap["out"][b][bi[r], ti[r]].float() - mu) @ w; grads = torch.autograd.grad(f, [cap["emb"]] + [cap["out"][bb] for bb in range(b)], retain_graph=True)
        ge = grads[0].float(); G = {bb: grads[1 + bb].float() for bb in range(b)}
        emb_attr = float((cap["emb"].detach().float() * ge).sum())
        attn_attr = float(sum((cap["attn"][bb].float() * G[bb]).sum() for bb in range(b)))
        same, other, direct, wsum, allsum, tops = 0.0, 0.0, 0.0, 0.0, 0.0, []
        sq = 0.0; s1 = 0.0
        for bb in range(b):
            M = cap["act"][bb] * (G[bb] @ Wd[bb].T)                                                                         # [B, T, DFF] attribution of each write at each position
            samep = float(M[bi[r], ti[r]].sum()); tot = float(M.sum()); same += samep; other += tot - samep
            direct += float(cap["mlp"][bb][bi[r], ti[r]].float() @ w)
            wmask = is_word0.view(-1, DFF)[bb]; wsum += float(M[:, :, wmask].sum()); allsum += tot
            a = M.abs(); s1 += float(a.sum()); sq += float(a.pow(2).sum())
            v, ix = M.abs().reshape(-1).topk(TOPK); tops += [(float(M.reshape(-1)[k]), bb, int(k) // DFF, int(k) % DFF) for k in ix.tolist()]
        tops = sorted(tops, key=lambda t: -abs(t[0]))[:TOPK]
        # ablation of the top attributed writes: zero the neuron's activation at that position and block, recompute the projection
        effects = []
        for val, bb, pos, j in tops:
            def hk(m, a, bb=bb, pos=pos, j=j):
                a0 = a[0].clone(); a0.view(-1, DFF)[pos, j] = 0; return (a0,)
            cap2 = {}
            def cap_stop(m, i, o): cap2["o"] = (o[0] if isinstance(o, tuple) else o).detach(); raise Stop
            hh = [arch.mlp_lin(bb).register_forward_pre_hook(hk), layers[b].register_forward_hook(cap_stop)]
            try: model(ids)
            except Stop: pass
            finally: [h.remove() for h in hh]
            effects.append(float(f) - float((cap2["o"][bi[r], ti[r]].float() - mu) @ w))
        te = torch.tensor([t[0] for t in tops]); ef = torch.tensor(effects)
        quant["same_pos_mlp"].append(same); quant["other_pos_mlp"].append(other); quant["attention"].append(attn_attr); quant["embedding"].append(emb_attr); quant["total_attr"].append(same + other + attn_attr + emb_attr); quant["projection"].append(float(f))
        quant["direct_mlp"].append(direct); quant["transported_mlp"].append(same + other - direct); quant["word_share"].append(wsum / allsum if abs(allsum) > 1e-9 else float("nan")); quant["participation"].append((s1 ** 2) / sq if sq > 0 else float("nan"))
        quant["top_same_pos"].append(float(sum(1 for t in tops if t[2] == int(best[r])) / len(tops))); quant["top_word"].append(float(sum(1 for t in tops if bool(is_word0[t[1] * DFF + t[3]])) / len(tops))); quant["top_block_gap"].append(float(sum(b - t[1] for t in tops) / len(tops)))
        rt, re_ = te.argsort().argsort().double(), ef.argsort().argsort().double(); quant["abl_spearman"].append(float(((rt - rt.mean()) * (re_ - re_.mean())).sum() / ((rt - rt.mean()).norm() * (re_ - re_.mean()).norm()).clamp_min(1e-9))); quant["abl_ratio"].append(float(ef.sum() / te.sum()) if abs(float(te.sum())) > 1e-9 else float("nan"))
        quant["unemb_cos"].append(float((U_top @ w).norm()))
    Q = {k: torch.tensor(v) for k, v in quant.items()}; proj = Q["projection"].abs().clamp_min(1e-9); sgn = torch.sign(Q["projection"])
    share = lambda k: (Q[k] * sgn / proj)                                                                                  # each attribution as a share of the projection, signed along it
    out = dict(n_entrants=int(is_ent.sum()), n_matched=int((~is_ent).sum()), linearity=dict(median_total_over_projection=float((Q["total_attr"] * sgn / proj).median())), groups={}, auc_entrants_vs_matched={})
    for gname, gm in (("entrants", is_ent), ("matched", ~is_ent)):
        out["groups"][gname] = dict(same_position_mlp=float(share("same_pos_mlp")[gm].median()), other_positions_mlp=float(share("other_pos_mlp")[gm].median()), attention_outputs=float(share("attention")[gm].median()), embedding=float(share("embedding")[gm].median()), direct_mlp=float(share("direct_mlp")[gm].median()), transported_mlp=float(share("transported_mlp")[gm].median()),
                                    word_share=float(Q["word_share"][gm].nanmedian()), participation_ratio=float(Q["participation"][gm].nanmedian()), top5_same_position=float(Q["top_same_pos"][gm].mean()), top5_words=float(Q["top_word"][gm].mean()), top5_block_gap=float(Q["top_block_gap"][gm].mean()), ablation_spearman=float(Q["abl_spearman"][gm].nanmedian()), ablation_over_attribution=float(Q["abl_ratio"][gm].nanmedian()), unembedding_cos=float(Q["unemb_cos"][gm].median()), S=float(S0[rows_sel.cpu()][gm].median()))
    for k in ("other_pos_mlp", "attention", "transported_mlp"): out["auc_entrants_vs_matched"][k + "_share"] = auc(share(k)[is_ent], share(k)[~is_ent])
    for k in ("word_share", "participation", "unemb_cos"): out["auc_entrants_vs_matched"][k] = auc(Q[k][is_ent].nan_to_num(0), Q[k][~is_ent].nan_to_num(0))
    res["by_block"][b] = out; e_, m_ = out["groups"]["entrants"], out["groups"]["matched"]
    log(f"{name} {rev} block {b} ({out['n_entrants']} entrants, {out['n_matched']} matched non-entrants; attribution sums to {out['linearity']['median_total_over_projection']:.2f} of the projection): shares of the projection, entrants / matched: MLP writes at the same position {e_['same_position_mlp']:.2f}/{m_['same_position_mlp']:.2f}, at other positions {e_['other_positions_mlp']:.2f}/{m_['other_positions_mlp']:.2f}, attention outputs {e_['attention_outputs']:.2f}/{m_['attention_outputs']:.2f}, embedding {e_['embedding']:.2f}/{m_['embedding']:.2f}; direct writes {e_['direct_mlp']:.2f}/{m_['direct_mlp']:.2f}, transported {e_['transported_mlp']:.2f}/{m_['transported_mlp']:.2f}; share from current words {e_['word_share']:.2f}/{m_['word_share']:.2f}; participation ratio {e_['participation_ratio']:.0f}/{m_['participation_ratio']:.0f}; top-5 writes at the same position {e_['top5_same_position']:.2f}/{m_['top5_same_position']:.2f}, words {e_['top5_words']:.2f}/{m_['top5_words']:.2f}, block gap {e_['top5_block_gap']:.1f}/{m_['top5_block_gap']:.1f}; ablation Spearman {e_['ablation_spearman']:.2f}/{m_['ablation_spearman']:.2f}, ablation over attribution {e_['ablation_over_attribution']:.2f}/{m_['ablation_over_attribution']:.2f}; unembedding cosine {e_['unembedding_cos']:.2f}/{m_['unembedding_cos']:.2f} (AUC {out['auc_entrants_vs_matched']['unemb_cos']})")
    del cap, Wsel, P; torch.cuda.empty_cache()
o = res["by_block"][12]["groups"]["entrants"]; a12 = res["by_block"][12]["auc_entrants_vs_matched"]
res["checks"] = dict(other_positions_over_half=o["other_positions_mlp"] + o["attention_outputs"] > 0.5, words_push_words=o["word_share"] > 0.5, direct_under_third=o["direct_mlp"] < 1 / 3, unembedding_not_separating=(a12["unemb_cos"] or 0.5) < 0.6)
summ = (f"{name} {rev} to {NEXTS}: " + " | ".join(f"block {b}: entrants' projection from MLP writes at the same position {o['groups']['entrants']['same_position_mlp']:.2f}, other positions {o['groups']['entrants']['other_positions_mlp']:.2f}, attention outputs {o['groups']['entrants']['attention_outputs']:.2f}; direct {o['groups']['entrants']['direct_mlp']:.2f}, transported {o['groups']['entrants']['transported_mlp']:.2f}; from current words {o['groups']['entrants']['word_share']:.2f} (matched {o['groups']['matched']['word_share']:.2f}); participation {o['groups']['entrants']['participation_ratio']:.0f}; top-5 same position {o['groups']['entrants']['top5_same_position']:.2f}, words {o['groups']['entrants']['top5_words']:.2f}; ablation over attribution {o['groups']['entrants']['ablation_over_attribution']:.2f}; unembedding AUC {o['auc_entrants_vs_matched']['unemb_cos']}" for b, o in res["by_block"].items()) + f" | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e521_sources_{name}_{rev}", res, summ)
