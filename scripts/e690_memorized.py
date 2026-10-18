"""e690 (session 155): the writers of memorised text. The essay: memorisation is local and semantic (a Poe neuron
guides 'dreary'; an umbrella taught to be a bee rides existing routes) and sits as close to the output as it can;
Nanda: memorised random documents need superposition. EleutherAI's memorised-sequence lists for Pythia (32-token
prefixes whose next 32 tokens the model reproduces exactly, verified here) against two controls from Pile text: random
64-token windows, and low-loss windows the model does not reproduce. At each recited position the sixteen neurons with
the largest direct contribution to the recited token (e664b's localizer); per sequence their depth, the coherence of
each writer's own class on reference text (the entropy of the current token over its top 1% of positions) and whether
the writer fires on reference windows that share the sequence's content words (a z-score against random positions).
Arguments: --model pythia160|pythia410|pythia1b [--smoke]. Pre-registered in e690_prereg.json."""
import argparse, re
from c11 import *
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--smoke", action="store_true"); A_ = ap.parse_args(); SMOKE = A_.smoke
name = A_.model; TAG = f"e690_{name}" + ("_smoke" if SMOKE else ""); M = load(name); tok = M["tok"]; NB, D, DFF = M["NB"], M["D"], M["DFF"]; L_ = lambda s: log(TAG, s)
SIZE = {"pythia160": "160m", "pythia410": "410m", "pythia1b": "1b"}[name]; NMEM = 20 if SMOKE else 300; NREF = 32 if SMOKE else 256; TOPW = 16; rng = np.random.default_rng(690)
import datasets
dsm = datasets.load_dataset("EleutherAI/pythia-memorized-evals", split=f"duped.{SIZE}"); col = [c for c in dsm.column_names if isinstance(dsm[0][c], list)][0]; L_(f"memorised-evals duped.{SIZE}: {len(dsm)} rows, token column '{col}', columns {dsm.column_names}")
pick = rng.permutation(len(dsm))[: 4 * NMEM]; cand = torch.tensor([dsm[int(i)][col][:64] for i in pick if len(dsm[int(i)][col]) >= 64])
def reproduced(seqs):
    out = []
    for s0 in range(0, seqs.shape[0], 64):
        x = seqs[s0:s0 + 64].to(DEV); o = M["model"].generate(x[:, :32], max_new_tokens=32, do_sample=False, pad_token_id=tok.pad_token_id); out.append((o[:, 32:64] == x[:, 32:64]).all(1).cpu())
    return torch.cat(out)
rep = reproduced(cand); MEM = cand[rep][:NMEM]; L_(f"{int(rep.sum())} of {cand.shape[0]} candidates reproduced exactly; using {MEM.shape[0]}")
pile = pile_ids(M, ndocs=3000, skip=5000); PW = windows_from(pile, 3000 if not SMOKE else 200, 64, seed=690)
def losses(seqs):
    out = []
    for s0 in range(0, seqs.shape[0], 64): x = seqs[s0:s0 + 64].to(DEV); lg = M["model"](x).logits[:, 31:63].float().log_softmax(-1); out.append(-lg.gather(-1, x[:, 32:64, None])[..., 0].mean(1).cpu())
    return torch.cat(out)
pl = losses(PW); prep = reproduced(PW); low = torch.nonzero((pl <= pl.quantile(0.1)) & ~prep)[:, 0]; RAND = PW[torch.tensor(rng.choice(PW.shape[0], NMEM, replace=False))]; LOW = PW[low[torch.randperm(low.numel())[:NMEM]]]
L_(f"controls: {RAND.shape[0]} random windows (loss {float(pl.mean()):.2f}), {LOW.shape[0]} low-loss unreproduced windows (loss <= {float(pl.quantile(0.1)):.2f}; {int(prep.sum())} of {PW.shape[0]} Pile windows were reproduced)")
# ---------------- reference activations: every neuron at 65k positions; each neuron's class entropy
ref = windows_from(pile_ids(M, ndocs=2000, skip=8000), NREF, 256, seed=691).to(DEV); RA = []
with Capture(M, list(range(NB)), ["act"]) as C:
    for s0 in range(0, NREF, 8): run_base(M, ref[s0:s0 + 8]); RA.append(torch.cat([C.cap[("act", b)][:, 1:].half() for b in range(NB)], -1).reshape(-1, NB * DFF).cpu())
RA = torch.cat(RA); NR_ = RA.shape[0]; RCUR = ref[:, 1:].reshape(-1); K1 = max(10, NR_ // 100); RMU = torch.zeros(NB * DFF, device=DEV); RSD = torch.zeros(NB * DFF, device=DEV)   # RA stays in CPU RAM (13 to 17 GB for the larger models)
ENT = torch.zeros(NB * DFF); GW = [final_gain(M, write_w(M, b).T) for b in range(NB)]
for s in range(0, NB * DFF, 512):
    blkf = RA[:, s:s + 512].to(DEV).float(); RMU[s:s + 512] = blkf.abs().mean(0); RSD[s:s + 512] = blkf.abs().std(0).clamp_min(1e-6); top = blkf.topk(K1, dim=0).indices; del blkf
    for i in range(top.shape[1]):
        c = torch.bincount(RCUR[top[:, i]], minlength=M["V"]).float(); p = c[c > 0] / c.sum(); ENT[s + i] = float(-(p * p.log2()).sum())
L_(f"reference: {NR_} positions; class entropy median {float(ENT.median()):.2f} bits")
STOP = set("the of and to in a is that for it as was with be by on not he this are or his from at which but have an had they you were their one all we can her has there been if more when will would who so no".split())
def content_words(seq): return set(w for w in re.findall(r"[a-zA-Z]{4,}", tok.decode(seq.tolist()).lower()) if w not in STOP)
RWIN_WORDS = [content_words(ref[w]) for w in range(NREF)]
UMEAN = M["WU"].mean(0)
def analyse(seqs, label):
    rows = []
    for s0 in range(0, seqs.shape[0], 16):
        x = seqs[s0:s0 + 16].to(DEV); cap = {}
        with Capture(M, list(range(NB)), ["act"]) as C:
            h = M["fnorm"].register_forward_pre_hook(lambda m, a: cap.__setitem__("xl", a[0].float()))
            try: M["model"](x)
            finally: h.remove()
            acts = torch.stack([C.cap[("act", b)].float() for b in range(NB)], 2)   # [n, T, NB, DFF]
        xl = cap["xl"]; sig = (xl.var(-1, unbiased=False) + 1e-5).sqrt()
        for i in range(x.shape[0]):
            wr = collections.Counter(); blocks = []
            for t in range(32, 63):
                tgt = M["WU"][x[i, t + 1]] - UMEAN; c = torch.stack([acts[i, t, b] * (GW[b] @ tgt) for b in range(NB)]) / sig[i, t]; top = c.reshape(-1).topk(TOPW).indices.tolist()
                for u in top: wr[u] += 1
                blocks += [u // DFF for u in top]
            ws = [u for u, _ in wr.most_common(TOPW)]; cw = content_words(x[i]); fld = [w for w in range(NREF) if len(RWIN_WORDS[w] & cw) >= 3]
            if fld:
                P = torch.cat([torch.arange(w * 255, (w + 1) * 255) for w in fld]); fa = RA[:, ws][P].to(DEV).float().abs().mean(0); z = (fa - RMU[ws]) / (RSD[ws] / math.sqrt(P.numel())); zshare = float((z > 3).float().mean())
            else: zshare = None
            rows.append(dict(depth_median=med(blocks) / max(NB - 1, 1), n_distinct=len(wr), class_entropy=med([float(ENT[u]) for u in ws]), field_z_share=zshare, n_field_windows=len(fld)))
    S = dict(n=len(rows), depth_median=med([r["depth_median"] for r in rows]), distinct_median=med([r["n_distinct"] for r in rows]), class_entropy_median=med([r["class_entropy"] for r in rows]), field_z_share_mean=mean([r["field_z_share"] for r in rows]), field_windows_median=med([r["n_field_windows"] for r in rows]))
    L_(f"{label}: writers' depth median {S['depth_median']:.2f}, distinct writers per sequence {S['distinct_median']}, class entropy {S['class_entropy_median']:.2f} bits, share firing on the content field {S['field_z_share_mean']}"); return S, rows
S_mem, R_mem = analyse(MEM, "memorised"); S_rand, R_rand = analyse(RAND, "random"); S_low, R_low = analyse(LOW, "low-loss")
V = dict(M1_memorised_deeper_than_controls=bool(S_mem["depth_median"] - max(S_rand["depth_median"], S_low["depth_median"]) > 1 / max(NB - 1, 1)), M2_class_entropy_not_higher=bool(S_mem["class_entropy_median"] <= max(S_rand["class_entropy_median"], S_low["class_entropy_median"]) + 0.2),
         M3_field_not_lower=bool(S_mem["field_z_share_mean"] is not None and S_low["field_z_share_mean"] is not None and S_mem["field_z_share_mean"] >= S_low["field_z_share_mean"] - 0.05))
res = dict(model=M["hf"], n_blocks=NB, memorised=S_mem, random=S_rand, low_loss=S_low, verdicts=V, rows=dict(memorised=R_mem, random=R_rand, low_loss=R_low))
summ = f"[{name}] memorised ({S_mem['n']}) vs random ({S_rand['n']}) vs low-loss unreproduced ({S_low['n']}): writers' depth {S_mem['depth_median']:.2f} / {S_rand['depth_median']:.2f} / {S_low['depth_median']:.2f}; class entropy {S_mem['class_entropy_median']:.2f} / {S_rand['class_entropy_median']:.2f} / {S_low['class_entropy_median']:.2f} bits; writers on the content field {S_mem['field_z_share_mean']} / {S_rand['field_z_share_mean']} / {S_low['field_z_share_mean']}; distinct writers {S_mem['distinct_median']} / {S_rand['distinct_median']} / {S_low['distinct_median']}; verdicts {V}"
save(TAG, res, summ)
