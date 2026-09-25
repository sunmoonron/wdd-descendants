"""e459: do native words persist across generated tokens? This is the temporal lineage of a write, suggested by an
external review. Descendants were followed through depth. Here the axis is the autoregressive position.
Setup:
- 16 prompts (the first 64 tokens of evaluation sequences), each continued by 96 sampled tokens (temperature 0.8,
  fixed seed).
- At the middle depth, every generated position's state (centred, sinks excluded) is described with 16 of the model's
  own words and, as a control, 16 rotated words (same Gram matrix, no provenance).
- Only MLP-row words are scored, so that repeated tokens cannot recur through token-embedding words.
Measures:
- For lags 1, 2, 4, 8, 16 and 32, the lift is the probability that a word used at position t is used again at t + lag,
  divided by that word's overall usage rate. Own words are compared with rotated words, and with the cosine of the
  centred states at that lag.
- Regeneration: the share of words that drop out at t + 1 and are used again within the next 16 positions, also divided by the chance of a return at each word's own usage rate.
- The same measures on the prompts' own natural continuations, the next 96 tokens of the text, for comparison.
Models (argument): gpt2, qwen05.
Pre-registered (honest guesses):
- own words persist more than rotated words at lags of 4 or more (lift ratio above 1.2; 0.5);
- persistence is similar in generated and natural text (within 20%; 0.5)."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; model, tok, fam = load_model(name); arch = Arch(model, fam); L = arch.NB // 2; K = 16; P0, G = 64, 96
E = eval_ids(name); prompts = E[:16, :P0].to(DEV); natural = E[:16, :P0 + G].to(DEV)
gen_ = torch.Generator(device=DEV).manual_seed(0); seq = prompts.clone()
with torch.no_grad():
    for _ in range(G):
        lg = model(seq).logits[:, -1].float() / 0.8; nxt = torch.multinomial(torch.softmax(lg, -1), 1, generator=gen_); seq = torch.cat([seq, nxt], 1)
A, lab = build_dictionary(arch, blocks=list(range(L + 1))); isM = (lab["type"] == T_MLP).to(DEV); Au = unitr(A); Ar = unitr(rotate(A, seed=7)); del A
col = torch.full((Au.shape[0],), -1, dtype=torch.long, device=DEV); col[isM] = torch.arange(int(isM.sum()), device=DEV); NMC = int(isM.sum())   # MLP words only
cen = torch.load("/workspace/wdd/eval_ids_all.pt", weights_only=False)[name]["cen_ids"][:8].to(DEV)
ref = block_states(model, arch, cen, [L], chunk=2)[L]; mu = ref[~sinkmask(ref)].mean(0); del ref
LAGS = [1, 2, 4, 8, 16, 32]
def lineage(ids):
    X = block_states(model, arch, ids, [L], chunk=4)[L][:, P0 - 1:]                   # [B, G, D]: states at the continuation positions
    B, T_, _ = X.shape; ok = ~sinkmask(X.reshape(-1, arch.D)).view(B, T_); Xc = (X - mu).reshape(-1, arch.D)
    out = {}
    for vn, Dct in (("own", Au), ("rot", Ar)):
        sel, _, _ = omp(Xc, Dct, K, batch=512, record_err=False); sel = sel.view(B, T_, K)
        cs = col[sel]; use = torch.zeros(B, T_, NMC + 1, dtype=torch.bool, device=DEV); use.scatter_(2, torch.where(cs < 0, NMC, cs), True)
        use = use[..., :NMC] & ok[..., None]
        rate = use.float().sum((0, 1)) / ok.sum().clamp_min(1)                               # per-word usage rate
        lifts = {}
        for lag in LAGS:
            a, b = use[:, :-lag], use[:, lag:]; both = (a & b).float().sum((0, 1)); na = a.float().sum((0, 1))
            m = na > 0; p_again = both[m].sum() / na[m].sum(); p_base = (na[m] * rate[m]).sum() / na[m].sum()
            lifts[lag] = (p_again / p_base.clamp_min(1e-12)).item()
        drop = use[:, :-1] & ~use[:, 1:]; back = torch.zeros_like(drop)
        for j in range(2, 18):
            if j < T_: back[:, :T_ - j] |= use[:, j:]
        back = back[:, :drop.shape[1]]
        obs = (drop & back).float().sum().item() / drop.float().sum().clamp_min(1).item()
        exp = (drop.float() * (1 - (1 - rate).clamp(0, 1) ** 16)[None, None, :]).sum().item() / drop.float().sum().clamp_min(1).item()   # chance of a return at the word's own usage rate
        out[vn] = dict(lift=lifts, regeneration=obs, regeneration_expected=exp, regeneration_lift=obs / max(exp, 1e-12),
                       words_per_position=use.float().sum(-1)[ok].mean().item())
        del sel, use
    Z = Xc.view(B, T_, -1); Zn = Z / Z.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    out["state_cos"] = {lag: (Zn[:, :-lag] * Zn[:, lag:]).sum(-1)[ok[:, :-lag] & ok[:, lag:]].mean().item() for lag in LAGS}
    return out
res = dict(model=name, level=L, prompts=16, generated=G, generated_text=lineage(seq), natural_text=lineage(natural))
g, n = res["generated_text"], res["natural_text"]
res["checks"] = dict(own_over_rot_1_2_at_lag4plus=all(g["own"]["lift"][lag] > 1.2 * g["rot"]["lift"][lag] for lag in (4, 8, 16, 32)),
                     generated_close_to_natural=all(abs(g["own"]["lift"][lag] - n["own"]["lift"][lag]) <= 0.2 * n["own"]["lift"][lag] for lag in (4, 8, 16, 32)))
f_ = lambda o: (" ".join(f"lag{lag} {o['own']['lift'][lag]:.1f}/{o['rot']['lift'][lag]:.1f}" for lag in LAGS) + f" | regeneration own {o['own']['regeneration']:.2f} (x{o['own']['regeneration_lift']:.1f} its usage rate) rot {o['rot']['regeneration']:.2f} (x{o['rot']['regeneration_lift']:.1f}) | "
                f"MLP words per position own {o['own']['words_per_position']:.1f} rot {o['rot']['words_per_position']:.1f} | state cos " + " ".join(f"{o['state_cos'][lag]:.2f}" for lag in LAGS))
summ = f"{name} L{L}, lift of reuse own/rotated MLP words: generated: " + f_(g) + " || natural: " + f_(n) + f" | checks {_json.dumps(res['checks'])}"
log(summ); record(f"e459_lineage_{name}", res, summ)
