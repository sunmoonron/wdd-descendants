"""e452b: why do re-implemented rows keep only about half of the original rows' advantage (e452)? Two candidate
reasons:
- the original rows were shaped end to end by the language-modelling loss, while e452's rows were only fitted to the
  block's own outputs (a local objective);
- they simply had far more training.
The block (the middle MLP, as in e452) is re-implemented from a fresh initialisation (seed 0) by distillation, 6000
steps as in e452. It is then trained further in one of two ways:
- lm: 1500 more steps on the model's next-token loss (WikiText-2 train, batches of 8 x 256 tokens). Only this block's
  weights move; the rest of the network is frozen.
- distill: 6000 more distillation steps on the block's input-output pairs (a control for more training).
Measured before and after, as in e452: the loss change on the evaluation text, and the family test (the block's rows
alone against their rotation, k = 4 and 16). The families are compared with the original rows' advantage.
Models (argument): gpt2, smollm2.
Pre-registered (honest guesses):
- end-to-end training raises the advantage to at least three quarters of the original's (0.4);
- more distillation does not (0.6)."""
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
tr_lm = tr_ids
def lm_train(st, steps=1500, bs=8, lr=1e-4):
    """next-token training of this block only, with the student in place"""
    mlp.load_state_dict({k: v.to(orig[k].dtype) for k, v in st.state_dict().items()})
    for p in model.parameters(): p.requires_grad_(False)
    ps = [p for p in mlp.parameters()]; [p.requires_grad_(True) for p in ps]; opt = torch.optim.Adam(ps, lr=lr); gen = torch.Generator(device=DEV).manual_seed(2)
    with torch.enable_grad():
        for s_ in range(steps):
            for gr in opt.param_groups: gr["lr"] = lr * min(1.0, (s_ + 1) / 100) * 0.5 * (1 + math.cos(math.pi * s_ / steps))
            ids = tr_lm[torch.randint(0, tr_lm.shape[0], (bs,), device=DEV, generator=gen)]
            loss = token_loss(model(ids).logits.float(), ids).mean(); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    [p.requires_grad_(False) for p in ps]
    return loss.item()
a0, a0_16 = t_meas["family"]["adv_k4"], t_meas["family"]["adv_k16"]
for vn in ("lm", "distill"):
    t0 = time.time(); st = make_student(0, None); rel0, _ = train(st)
    mlp.load_state_dict({k: v.to(orig[k].dtype) for k, v in st.state_dict().items()}); before, _, _ = measure("before")
    if vn == "lm": lf = lm_train(st); rel = None
    else:
        rel, lf = train(st); mlp.load_state_dict({k: v.to(orig[k].dtype) for k, v in st.state_dict().items()})
    after, Xc_s, rows_s = measure(vn)
    with torch.no_grad(): rel_after = ((mlp(Xev.float().to(next(mlp.parameters()).dtype)).float() - Yev.float()).pow(2).sum() / (Yev.float() - Yev.float().mean(0)).pow(2).sum()).item()
    res["variants"][vn] = dict(before=before, after=after, rel_err_distilled=rel0, rel_err_after=rel_after, final_loss=lf,
                               share_of_original_adv_k4=dict(before=before["family"]["adv_k4"] / a0, after=after["family"]["adv_k4"] / a0),
                               share_of_original_adv_k16=dict(before=before["family"]["adv_k16"] / a0_16, after=after["family"]["adv_k16"] / a0_16),
                               d_eval_loss_before=before["eval_loss"] - t_meas["eval_loss"], d_eval_loss_after=after["eval_loss"] - t_meas["eval_loss"],
                               identity_vs_original=best_cos(rows_s, rows_t), seconds=time.time() - t0)
    mlp.load_state_dict(orig); del st; torch.cuda.empty_cache(); m = res["variants"][vn]
    log(f"{name} {vn}: dL {m['d_eval_loss_before']:+.3f} -> {m['d_eval_loss_after']:+.3f} | share of the original advantage k4 {m['share_of_original_adv_k4']['before']:.2f} -> {m['share_of_original_adv_k4']['after']:.2f}, k16 {m['share_of_original_adv_k16']['before']:.2f} -> {m['share_of_original_adv_k16']['after']:.2f} | block rel err {rel_after:.3f} | best |cos| with original rows {m['identity_vs_original']['median']:.2f}")
V = res["variants"]
res["checks"] = dict(lm_reaches_three_quarters=V["lm"]["share_of_original_adv_k4"]["after"] >= 0.75, distill_does_not=V["distill"]["share_of_original_adv_k4"]["after"] < 0.75)
summ = (f"{name} L{L}: original advantage k4 {a0:.3f} k16 {a0_16:.3f} || " + " || ".join(f"{vn}: dL {m['d_eval_loss_before']:+.3f} -> {m['d_eval_loss_after']:+.3f}, share of the original advantage k4 {m['share_of_original_adv_k4']['before']:.2f} -> {m['share_of_original_adv_k4']['after']:.2f} (k16 {m['share_of_original_adv_k16']['before']:.2f} -> {m['share_of_original_adv_k16']['after']:.2f}), block rel err {m['rel_err_after']:.3f}, best |cos| with original rows {m['identity_vs_original']['median']:.2f}" for vn, m in V.items())
        + f" | checks {_json.dumps(res['checks'])} | {time.time() - t00:.0f}s")
log(summ); record(f"e452b_endtoend_{name}", res, summ)
