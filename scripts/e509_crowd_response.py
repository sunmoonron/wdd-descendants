"""e509: is the crowd's contraction of the chord the network's response to the chord? e507 found the crowd of small
MLP writes runs along the chord with a gain of +1.4 at Pythia step 512 and -0.64 at the end (block 12), and -0.16 to
-0.46 at the end of GPT-2 and OLMo. Area 05 found every trained block answers a perturbation along any direction with
a linear gain of -0.16 to -0.42 (e194, e210), born as Pythia's warmup ends (e217). If the crowd's contraction is that
learned contraction acting on the model's own writes, then removing the chord from what each block's MLP sees should
remove the crowd's alignment with it: the crowd's gain along the chord should be, for the most part, a response.
Per position and block b' up to the state's block: the chord so far (the writers of blocks before b', plus the
embeddings) is subtracted from the input of block b''s second layer norm, so that its MLP sees the state without the
chord while attention is untouched; the MLP's activations are recorded with and without. The small writes' response
is the change in the rows that are not the chord's, times their write directions, summed over blocks; likewise the
chord's own rows' response, and the response to a random direction of the same norm as the chord so far at every
block (the generic contraction of area 05, measured the same way).
Reported per block, medians over typical positions: the crowd's actual gain along the centred chord's direction
(centred over positions as in e507, and raw), the response gain (the part of the crowd's alignment that vanishes
when the chord is removed), their ratio and Spearman across positions, the chord's own rows' response gain, the
random-direction response gain, the norm of the response over the norm of the crowd, and the response gain by block.
Setup: a model at a checkpoint (arguments), blocks NB/4, NB/2, 3NB/4; 8 x 256 evaluation tokens, typical positions.
Pre-registered (honest guesses):
- at the end of training the response accounts for more than half of the crowd's gain along the chord in GPT-2,
  Pythia and OLMo (0.5);
- the response gain is negative at the end and within 0.2 of zero at Pythia step 512, where the actual gain is
  positive: the early amplification is not a response (0.5);
- the response to a random direction of the same size is negative and smaller than the response to the chord (0.5).
Arguments: name [revision]."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; rev = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] not in ("none", "main") else None
model, tok, fam = load_model(name, revision=rev); arch = Arch(model, fam); NB = arch.NB; D = arch.D; DFF = arch.DFF; NW = 64
blocks = sorted({NB // 4, NB // 2, (3 * NB) // 4}); LB = max(blocks); ids = eval_ids(name)[:8, :256].to(DEV); B_, T_ = ids.shape; CH = 4
def ln2(b): return arch.layers[b].ln_2 if fam == "gpt2" else arch.layers[b].post_attention_layernorm
acts = {b: [] for b in range(LB + 1)}
def mk(b):
    def pre(m, a): acts[b].append(a[0].detach().float()[:, 1:].reshape(-1, DFF)); return None
    return pre
hs = [arch.mlp_lin(b).register_forward_pre_hook(mk(b)) for b in range(LB + 1)]
try: S_ = block_states(model, arch, ids, blocks, chunk=CH)
finally: [h.remove() for h in hs]
A_all = {b: torch.cat(acts[b]) for b in range(LB + 1)}; del acts; rown = {b: arch.wdir(b).float().norm(dim=-1) for b in range(LB + 1)}
tokens = ids[:, 1:].reshape(-1); positions = torch.arange(1, T_, device=DEV).repeat(B_); has_pos = len(arch.emb) > 1; Nall = B_ * (T_ - 1)
def acts_with_delta(bp, delta):
    """activations at block bp's down-projection (positions 1:) when delta [B, T-1, D] is subtracted from the input of its second layer norm"""
    out = []
    for s0 in range(0, B_, CH):
        cap = {}
        def pre_ln(m, args):
            h = args[0]; h2 = h.clone(); h2[:, 1:] = h2[:, 1:] - delta[s0:s0 + h.shape[0]].to(h.dtype); return (h2,)
        def pre_lin(m, a): cap["a"] = a[0].detach().float()[:, 1:].reshape(-1, DFF); return None
        def stop(m, i, o): raise Stop
        hh = [ln2(bp).register_forward_pre_hook(pre_ln), arch.mlp_lin(bp).register_forward_pre_hook(pre_lin), arch.layers[bp].register_forward_hook(stop)]
        try:
            with torch.no_grad(): model(ids[s0:s0 + CH])
        except Stop: pass
        finally: [h.remove() for h in hh]
        out.append(cap["a"])
    return torch.cat(out)
def spear(x, y):
    rx, ry = x.argsort().argsort().double(), y.argsort().argsort().double(); rx, ry = rx - rx.mean(), ry - ry.mean(); return float((rx * ry).sum() / (rx.norm() * ry.norm()).clamp_min(1e-9))
g = torch.Generator(device=DEV).manual_seed(0)
res = dict(model=name, revision=rev, blocks=blocks, by_block={})
for b in blocks:
    X = S_[b].reshape(-1, D); keep = ~sinkmask(X)
    A, lab = build_dictionary(arch, blocks=list(range(b + 1))); anorm = A.float().norm(dim=-1); Au = unitr(A); del A
    typ = lab["type"].to(DEV); blk = lab["block"].to(DEV); idx = lab["index"].to(DEV)
    gidx = torch.full((b + 1, DFF), -1, device=DEV, dtype=torch.long)
    for bb in range(b + 1): mm = (typ == T_MLP) & (blk == bb); gidx[bb, idx[mm]] = torch.nonzero(mm)[:, 0]
    Cled = torch.cat([A_all[bb] * rown[bb][None] for bb in range(b + 1)], 1); top = Cled.abs().topk(NW, dim=1); writers = gidx.reshape(-1)[top.indices]; wcoef = Cled.gather(1, top.indices); wblk = top.indices // DFF; widx = top.indices % DFF; del Cled
    mt = typ == T_TOK; tok_lut = torch.full((int(idx[mt].max()) + 1,), -1, device=DEV, dtype=torch.long); tok_lut[idx[mt]] = torch.nonzero(mt)[:, 0]; tatom = tok_lut[tokens]
    emb = anorm[tatom][:, None] * Au[tatom]
    if has_pos:
        mp = typ == T_POS; pos_lut = torch.full((int(idx[mp].max()) + 1,), -1, device=DEV, dtype=torch.long); pos_lut[idx[mp]] = torch.nonzero(mp)[:, 0]; patom = pos_lut[positions]; emb = emb + anorm[patom][:, None] * Au[patom]
    assert bool((writers >= 0).all())
    chord = torch.einsum("nk,nkd->nd", wcoef, Au[writers]) + emb                                                             # the chord at every position [Nall, D]
    rdir = unitr(torch.randn(Nall, D, generator=g, device=DEV))                                                              # one random direction per position, for the control
    S = torch.zeros(Nall, D, device=DEV); dS = torch.zeros_like(S); dB = torch.zeros_like(S); dR = torch.zeros_like(S); by_block = []
    for bp in range(b + 1):
        present = wblk < bp; so_far = torch.einsum("nk,nkd->nd", wcoef * present.float(), Au[writers]) + emb; nsf = so_far.norm(dim=-1, keepdim=True)
        a_base = A_all[bp]; a_mod = acts_with_delta(bp, so_far.view(B_, T_ - 1, D)); a_rnd = acts_with_delta(bp, (rdir * nsf).view(B_, T_ - 1, D))
        mask = torch.zeros(Nall, DFF, dtype=torch.bool, device=DEV); here = wblk == bp; mask[torch.nonzero(here)[:, 0], widx[here]] = True
        Wd = arch.wdir(bp); da = a_base - a_mod; dr = a_base - a_rnd
        s_bp = (a_base * (~mask)) @ Wd; ds_bp = (da * (~mask)) @ Wd; S += s_bp; dS += ds_bp; dB += (da * mask) @ Wd; dR += (dr * (~mask)) @ Wd; by_block.append(ds_bp)
    k = keep; ch = chord[k]; chc = ch - ch.mean(0); cn = chc.norm(dim=-1).clamp_min(1e-9); cd = chc / cn[:, None]; Sk = S[k]; Sc = Sk - Sk.mean(0)
    ga_c = (Sc * cd).sum(1) / cn; ga_r = (Sk * cd).sum(1) / cn; gr = (dS[k] * cd).sum(1) / cn; gb = (dB[k] * cd).sum(1) / cn; grand = (dR[k] * rdir[k]).sum(1) / cn
    o = dict(n_positions=int(k.sum()), gain_actual_centred=float(ga_c.median()), gain_actual_raw=float(ga_r.median()), gain_response=float(gr.median()), response_share=float(gr.median() / ga_c.median()) if abs(float(ga_c.median())) > 1e-6 else None,
             spearman_response_actual=spear(gr, ga_c), gain_chord_rows_response=float(gb.median()), gain_random_response=float(grand.median()), response_norm_over_crowd_norm=float((dS[k].norm(dim=-1) / Sk.norm(dim=-1).clamp_min(1e-9)).median()),
             chord_norm_over_state_norm=float((cn / (X[k] - X[k].mean(0)).norm(dim=-1)).median()), gain_response_by_block=[float(((v[k] * cd).sum(1) / cn).median()) for v in by_block])
    res["by_block"][b] = o
    log(f"{name}{' ' + rev if rev else ''} block {b} ({o['n_positions']} positions): crowd gain along the chord actual {o['gain_actual_centred']:+.2f} (raw {o['gain_actual_raw']:+.2f}), response {o['gain_response']:+.2f} (share {o['response_share'] if o['response_share'] is None else round(o['response_share'], 2)}, Spearman {o['spearman_response_actual']:.2f}); chord rows' own response {o['gain_chord_rows_response']:+.2f}; random-direction response {o['gain_random_response']:+.2f}; response norm over crowd norm {o['response_norm_over_crowd_norm']:.2f}; response by block " + " ".join(f"{v:+.2f}" for v in o["gain_response_by_block"]))
    del Au, chord, S, dS, dB, dR, by_block; torch.cuda.empty_cache()
mid = res["by_block"][NB // 2]; Bk = res["by_block"]
res["checks"] = dict(response_over_half_of_actual=all(o["response_share"] is not None and o["response_share"] > 0.5 and o["gain_actual_centred"] < 0 for o in Bk.values()), response_negative_at_mid=mid["gain_response"] < 0, random_negative_and_smaller=all(o["gain_random_response"] < 0 and abs(o["gain_random_response"]) < abs(o["gain_response"]) for o in Bk.values()))
fsh = lambda o: "n/a" if o["response_share"] is None else f"{o['response_share']:.2f}"
summ = (f"{name}{' ' + rev if rev else ''}: by block " + " | ".join(f"{b}: crowd gain along the chord actual {o['gain_actual_centred']:+.2f}, response {o['gain_response']:+.2f} (share {fsh(o)}, Spearman {o['spearman_response_actual']:.2f}), chord rows' response {o['gain_chord_rows_response']:+.2f}, random-direction response {o['gain_random_response']:+.2f}, response norm over crowd norm {o['response_norm_over_crowd_norm']:.2f}" for b, o in Bk.items())
        + f" | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e509_crowdresponse_{name}{'_' + rev if rev else ''}", res, summ)
