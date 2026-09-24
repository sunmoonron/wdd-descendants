"""e452: is the native vocabulary determined by the function a block computes? And is it written by training the writer
rows even when that function is held fixed? This is a function-preserving re-implementation, suggested by an external
review.
One MLP block is re-implemented. A new MLP of the same architecture is trained to reproduce the original block's
input-output map on the model's own inputs to that block (400k tokens of WikiText-2 train), then swapped into the model
in its place. The block is the middle one, L = NB // 2. Its output is the state the program describes, and its rows are
in that state's dictionary.
Variants, all on the same data, from GPT-2-style or Llama-style initialisation:
- fresh (seed 0) and fresh_s1 (seed 1): all weights trained;
- frozenW: the write rows (the MLP's output matrix) fixed at initialisation, the rest trained;
- frozenR: the read rows (the input matrices) fixed at initialisation, the write rows trained.
Output biases stay trainable. They are not rows.
Measured per variant, with the new MLP in place:
- function: the relative error of the block's output on held-out tokens, and the model's loss on the evaluation text
  (the change against the original model);
- words: at the output of block L (typical positions, sinks exact), the fraction of variance left unexplained by k = 4
  and 16 words drawn only from block L's MLP rows, against the same rows rotated. This is e444b's and e449b's family
  test. It is run for the new rows on the new model's states, and for the original rows on the original model's states.
  Crossed versions are also run: the original rows on the new states, and the new rows on the original states;
- the whole dictionary with the new rows: own against rotated at k = 16 (unexplained variance and loss recovered), and
  the share of the 16 words that are block L's MLP rows;
- identity: each new row's best absolute cosine with any original row (median, and the shares above 0.5 and 0.9), and
  the same between the two fresh seeds.
Models (argument): gpt2, smollm2.
Pre-registered (honest guesses):
- the fresh block matches the function within 0.1 nats (0.6);
- its rows are new vectors, with median best |cos| with the original rows below 0.5 (0.6);
- those rows are still words: they keep at least half of the original rows' advantage over rotation at k = 4 (0.55);
- frozen random write rows keep under a third of that advantage (0.6);
- frozen-reader blocks keep at least half of it (0.5)."""
import sys, os, math, copy, time, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
import torch.nn as nn
name = sys.argv[1]; model, tok, fam = load_model(name); arch = Arch(model, fam); D = arch.D; L = arch.NB // 2; t00 = time.time()
mlp = arch.layers[L].mlp; orig = {k: v.detach().clone() for k, v in mlp.state_dict().items()}
WR = "c_proj" if fam == "gpt2" else ("down_proj" if fam == "llama" else "dense_4h_to_h")
# ---- data: the block's MLP inputs and outputs
try:
    from datasets import load_dataset
    text = "\n\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"] if t.strip())
    allids = tok(text[:6_000_000], add_special_tokens=False, return_tensors="pt")["input_ids"][0]; src = "wikitext-2 train"
except Exception as e:
    log(f"datasets unavailable ({e}); falling back to the centring ids"); allids = torch.load("/workspace/wdd/eval_ids_all.pt", weights_only=False)[name]["cen_ids"].reshape(-1); src = "cen_ids"
TT = 256; nseq = min(1600, allids.numel() // TT); tr_ids = allids[:nseq * TT].view(nseq, TT).to(DEV)
ev = eval_ids(name)[:16].to(DEV)
def collect(ids, chunk=32):
    X, Y = [], []
    for s0 in range(0, ids.shape[0], chunk):
        cap = {}
        def mh(m, i, o): cap["x"] = i[0].detach(); cap["y"] = out_of(o).detach()
        def st(m, i, o): raise Stop
        hs = [mlp.register_forward_hook(mh), arch.layers[L].register_forward_hook(st)]
        try:
            with torch.no_grad(): model(ids[s0:s0 + chunk])
        except Stop: pass
        finally: [h.remove() for h in hs]
        X.append(cap["x"].reshape(-1, D).to(torch.bfloat16)); Y.append(cap["y"].reshape(-1, D).to(torch.bfloat16))
    return torch.cat(X), torch.cat(Y)
Xtr, Ytr = collect(tr_ids); Xev, Yev = collect(ev, chunk=4)
log(f"{name} L{L}: {Xtr.shape[0]} training tokens ({src}), {Xev.shape[0]} held-out ({time.time() - t00:.0f}s)")
def eval_loss():
    tl = []
    for s0 in range(0, ev.shape[0], 4):
        with torch.no_grad(): tl.append(token_loss(model(ev[s0:s0 + 4]).logits.float(), ev[s0:s0 + 4]))
    return torch.cat(tl).mean().item()
Q = torch.linalg.qr(torch.randn(D, D, generator=torch.Generator().manual_seed(11)))[0].to(DEV)
def fam_fvu(Xc, rows, ks=(4, 16)):
    R = unitr(rows); out = {}; tot = Xc.pow(2).sum()
    for vn, Dd in (("own", R), ("rot", R @ Q)):
        sel, _, _ = omp(Xc, Dd, max(ks), batch=1024, record_err=False)
        for k in ks: _, err = refit(Xc, Dd, sel[:, :k]); out[f"{vn}_k{k}"] = (err.sum() / tot).item()
    for k in ks: out[f"adv_k{k}"] = out[f"rot_k{k}"] - out[f"own_k{k}"]
    return out
def measure(tag):
    """function, words and dictionary measures for the MLP currently in place"""
    lv = NSLevel(model, arch, ev[:8], L); rows = arch.wdir(L)
    A, lab = build_dictionary(arch, blocks=list(range(L + 1))); isL = ((lab["type"] == T_MLP) & (lab["block"] == L)).to(DEV)
    own = describe(lv, A, [16]); rot = describe(lv, rotate(A, seed=7), [16])
    sel, _, _ = omp(lv.Xc, unitr(A), 16, batch=1024, record_err=False); selr, _, _ = omp(lv.Xc, unitr(rotate(A, seed=7)), 16, batch=1024, record_err=False)
    out = dict(eval_loss=eval_loss(), family=fam_fvu(lv.Xc, rows), dict_own_k16=own["16"], dict_rot_k16=rot["16"],
               share_blockL_rows_own=isL[sel].float().mean().item(), share_blockL_rows_rot=isL[selr].float().mean().item())
    del A, lab, sel, selr; torch.cuda.empty_cache()
    return out, lv.Xc, rows
def best_cos(R1, R2):
    S = unitr(R1) @ unitr(R2).T; b = S.abs().max(1).values
    return dict(median=b.median().item(), over_0_5=(b > 0.5).float().mean().item(), over_0_9=(b > 0.9).float().mean().item())
def make_student(seed, freeze):
    st = copy.deepcopy(mlp).float(); gen = torch.Generator().manual_seed(seed)
    for n_, p in st.named_parameters():
        wr = WR in n_
        with torch.no_grad():
            if p.dim() >= 2: p.copy_(torch.randn(p.shape, generator=gen) * (0.02 / math.sqrt(2 * arch.NB) if (fam == "gpt2" and wr) else 0.02))
            else: p.zero_()
        p.requires_grad_(not (p.dim() >= 2 and ((freeze == "W" and wr) or (freeze == "R" and not wr))))
    return st
def train(st, steps=6000, bs=4096, lr=1e-3):
    ps = [p for p in st.parameters() if p.requires_grad]; opt = torch.optim.Adam(ps, lr=lr); N = Xtr.shape[0]
    yv = Ytr.float().var(0).sum(); gen = torch.Generator(device=DEV).manual_seed(1)
    with torch.enable_grad():
        for s in range(steps):
            for gr in opt.param_groups: gr["lr"] = lr * min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * s / steps))
            idx = torch.randint(0, N, (bs,), device=DEV, generator=gen); x, y = Xtr[idx].float(), Ytr[idx].float()
            loss = (st(x) - y).pow(2).sum(-1).mean() / yv; opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    with torch.no_grad():
        rel = ((st(Xev.float()) - Yev.float()).pow(2).sum() / (Yev.float() - Yev.float().mean(0)).pow(2).sum()).item()
    return rel, loss.item()
# ---- the original block
res = dict(model=name, level=L, data=src, n_train=int(Xtr.shape[0]), variants={})
t_meas, Xc_t, rows_t = measure("original"); rows_t = rows_t.clone(); res["original"] = t_meas
log(f"{name} original: eval loss {t_meas['eval_loss']:.3f} | family k4 own/rot {t_meas['family']['own_k4']:.3f}/{t_meas['family']['rot_k4']:.3f} | dict k16 rec own {t_meas['dict_own_k16']['rec']:.2f} rot {t_meas['dict_rot_k16']['rec']:.2f} | share of block-L rows {t_meas['share_blockL_rows_own']:.3f}")
rows_by = {}
for vn, seed, fz in (("fresh", 0, None), ("fresh_s1", 1, None), ("frozenW", 0, "W"), ("frozenR", 0, "R")):
    t0 = time.time(); st = make_student(seed, fz); rel, lf = train(st)
    mlp.load_state_dict({k: v.to(orig[k].dtype) for k, v in st.state_dict().items()})
    m, Xc_s, rows_s = measure(vn); rows_by[vn] = rows_s.clone()
    m.update(rel_err_heldout=rel, final_train_loss=lf, d_eval_loss=m["eval_loss"] - t_meas["eval_loss"],
             cross_original_rows_on_new_states=fam_fvu(Xc_s, rows_t), cross_new_rows_on_original_states=fam_fvu(Xc_t, rows_s),
             identity_vs_original=best_cos(rows_s, rows_t), seconds=time.time() - t0)
    res["variants"][vn] = m; mlp.load_state_dict(orig); del st; torch.cuda.empty_cache()
    log(f"{name} {vn}: rel err {rel:.3f}, d loss {m['d_eval_loss']:+.3f} | family k4 own/rot {m['family']['own_k4']:.3f}/{m['family']['rot_k4']:.3f} (adv {m['family']['adv_k4']:+.3f}; original {t_meas['family']['adv_k4']:+.3f}) | "
        f"dict k16 rec own {m['dict_own_k16']['rec']:.2f} rot {m['dict_rot_k16']['rec']:.2f} | share block-L rows {m['share_blockL_rows_own']:.3f} | best |cos| with original rows median {m['identity_vs_original']['median']:.2f}")
res["identity_fresh_vs_fresh_s1"] = best_cos(rows_by["fresh"], rows_by["fresh_s1"])
V = res["variants"]; a0 = t_meas["family"]["adv_k4"]
res["checks"] = dict(fresh_within_0_1_nats=V["fresh"]["d_eval_loss"] < 0.1, fresh_rows_new=V["fresh"]["identity_vs_original"]["median"] < 0.5,
                     fresh_keeps_half_adv=V["fresh"]["family"]["adv_k4"] >= 0.5 * a0, frozenW_under_third=V["frozenW"]["family"]["adv_k4"] < a0 / 3,
                     frozenR_keeps_half=V["frozenR"]["family"]["adv_k4"] >= 0.5 * a0)
cell = lambda m: (f"rel {m['rel_err_heldout']:.3f} dL {m['d_eval_loss']:+.3f} | fam k4 own/rot {m['family']['own_k4']:.3f}/{m['family']['rot_k4']:.3f} k16 {m['family']['own_k16']:.3f}/{m['family']['rot_k16']:.3f} | "
                  f"dict k16 rec {m['dict_own_k16']['rec']:.2f}/{m['dict_rot_k16']['rec']:.2f} share {m['share_blockL_rows_own']:.3f} | cos-orig med {m['identity_vs_original']['median']:.2f} >0.9 {m['identity_vs_original']['over_0_9']:.2f} | "
                  f"orig rows on new states k4 own/rot {m['cross_original_rows_on_new_states']['own_k4']:.3f}/{m['cross_original_rows_on_new_states']['rot_k4']:.3f}")
summ = (f"{name} L{L} ({res['n_train']} tokens): original loss {t_meas['eval_loss']:.3f}, fam k4 own/rot {t_meas['family']['own_k4']:.3f}/{t_meas['family']['rot_k4']:.3f} k16 {t_meas['family']['own_k16']:.3f}/{t_meas['family']['rot_k16']:.3f}, "
        f"dict k16 rec {t_meas['dict_own_k16']['rec']:.2f}/{t_meas['dict_rot_k16']['rec']:.2f}, share {t_meas['share_blockL_rows_own']:.3f} || "
        + " || ".join(f"{vn}: {cell(m)}" for vn, m in V.items()) + f" || fresh vs fresh_s1 rows best |cos| median {res['identity_fresh_vs_fresh_s1']['median']:.2f} | checks {_json.dumps(res['checks'])} | {time.time() - t00:.0f}s")
log(summ); record(f"e452_reimplemented_{name}", res, summ)
