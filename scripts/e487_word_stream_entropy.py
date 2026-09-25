"""e487: the sequential entropy of the word stream (Shannon 1951, "Prediction and entropy of printed English"). Along a
sequence, each position's first native word makes a stream of symbols. How predictable is the next word from the
previous one, in bits, compared with the rotated dictionary's stream and with the text's own token stream? e459
measured word reuse across positions; this is the information-theoretic version, and it also asks how much of the
word stream is fixed by the current token (how lexical it is).
Setup: middle depth, 32 x 512 evaluation tokens in order (sinks skipped), the first native and first rotated word per
position; held-out cross-entropies (v2: the 1000 most frequent symbols and targets, smoothing tuned on sequences 20-23, fitted on 0-19,
scored on 24-31, as e485): H(w_t), H(w_t | w_{t-1}), H(w_t | tok_t), H(w_t | tok_{t-1}); the same for the token stream, H(tok_t) and
H(tok_t | tok_{t-1}); and a control with positions shuffled within sequences.
Models (argument): the five.
Pre-registered (honest guesses):
- the previous native word predicts the next at more bits than the previous rotated word predicts its next (0.6);
- the current token fixes the native word more than the previous word does (H(w|tok_t) < H(w|w_{t-1})) (0.7);
- the sequential information of the native stream is below the text's own (0.7)."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
from lr_common import eval_ids
name = sys.argv[1]; model, tok, fam = load_model(name); arch = Arch(model, fam); L = arch.NB // 2; D = arch.D
E = eval_ids(name)[:32, :512].to(DEV); B_, T_ = E.shape
X = block_states(model, arch, E, [L], chunk=4)[L]; flat = X.reshape(-1, D); keep = ~sinkmask(flat)
Xc = flat[keep] - flat[keep].mean(0); seq = torch.arange(B_, device=DEV)[:, None].expand(B_, T_ - 1).reshape(-1)[keep]; posn = torch.arange(1, T_, device=DEV)[None].expand(B_, T_ - 1).reshape(-1)[keep]
tokc = E[:, 1:].reshape(-1)[keep]; tokp = E[:, :-1].reshape(-1)[keep]
A, lab = build_dictionary(arch, blocks=list(range(L + 1))); Au = unitr(A); Ar = unitr(rotate(A, seed=7)); del A
wn = omp(Xc, Au, 1, batch=1024, record_err=False)[0][:, 0]; wr = omp(Xc, Ar, 1, batch=1024, record_err=False)[0][:, 0]
# consecutive pairs within a sequence (position t-1 -> t, both typical)
key = seq * T_ + posn; prev_key = key - 1; lut = torch.full((int(key.max()) + 2,), -1, device=DEV, dtype=torch.long); lut[key] = torch.arange(key.numel(), device=DEV)
pidx = lut[prev_key]; ok = pidx >= 0; cur_i = torch.nonzero(ok)[:, 0]; prv_i = pidx[ok]
def ce(sym_ctx, target, S=1000, V=1000):
    """v2: as e485's estimator; symbols and targets restricted to their 1000 most frequent values on the training split,
    smoothing tuned on a validation split; returns the held-out cross-entropy in bits"""
    sq = seq[cur_i]; tr = sq < 20; va = (sq >= 20) & (sq < 24); te = sq >= 24
    def compact(v, n):
        cnt = torch.bincount(v[tr], minlength=int(v.max()) + 1); top = cnt.topk(min(n, int((cnt > 0).sum()))).indices
        lut = torch.full((int(v.max()) + 1,), top.numel(), device=DEV, dtype=torch.long); lut[top] = torch.arange(top.numel(), device=DEV); return lut[v], top.numel()
    t, V_ = compact(target, V); marg = torch.bincount(t[tr], minlength=V_ + 1).float() + 1.0; marg = marg / marg.sum()
    if sym_ctx is None: return float(-torch.log2(marg[t[te]]).mean())
    s_, S_ = compact(sym_ctx, S); joint = torch.zeros(S_ + 1, V_ + 1, device=DEV); joint.index_put_((s_[tr], t[tr]), torch.ones(int(tr.sum()), device=DEV), accumulate=True); n_s = joint.sum(1, keepdim=True)
    def ce_a(alpha, mask): cond = (joint + alpha * marg[None]) / (n_s + alpha); return float(-torch.log2(cond[s_[mask], t[mask]]).mean())
    best = min([0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256], key=lambda a: ce_a(a, va)); return ce_a(best, te)
g = torch.Generator(device=DEV).manual_seed(0)
def shuffled(v):
    out = v.clone()
    for s in range(B_):
        m = seq[cur_i] == s; idx = torch.nonzero(m)[:, 0]; out[idx] = v[idx[torch.randperm(idx.numel(), generator=g, device=DEV)]]
    return out
res = dict(model=name, level=L, n_pairs=int(cur_i.numel()), streams={})
for kind, w in (("native", wn), ("rotated", wr), ("token", tokc)):
    tgt = w[cur_i]; H0 = ce(None, tgt); Hprev = ce(w[prv_i], tgt); Htok = ce(tokc[cur_i], tgt) if kind != "token" else None; Htokp = ce(tokp[cur_i], tgt); Hshuf = ce(shuffled(w[prv_i]), tgt)
    res["streams"][kind] = dict(H=H0, H_given_prev_word=Hprev, sequential_bits=H0 - Hprev, H_given_current_token=Htok, H_given_previous_token=Htokp, H_given_shuffled_prev=Hshuf, shuffled_bits=H0 - Hshuf)
    log(f"{name} {kind} stream: H {H0:.2f}, H|prev word {Hprev:.2f} (bits {H0 - Hprev:+.3f}; shuffled {H0 - Hshuf:+.3f}), H|current token {Htok if Htok is None else round(Htok, 2)}, H|previous token {Htokp:.2f}")
Sn, Sr, St = res["streams"]["native"], res["streams"]["rotated"], res["streams"]["token"]
res["checks"] = dict(native_sequential_over_rotated=Sn["sequential_bits"] > Sr["sequential_bits"], current_token_fixes_word_more_than_prev_word=Sn["H_given_current_token"] < Sn["H_given_prev_word"], native_sequential_below_text=Sn["sequential_bits"] < St["sequential_bits"])
summ = (f"{name} L{L}, {int(cur_i.numel())} consecutive pairs: word-stream entropy H and bits from the previous word (shuffled control): native {Sn['H']:.2f}, {Sn['sequential_bits']:+.3f} ({Sn['shuffled_bits']:+.3f}); rotated {Sr['H']:.2f}, {Sr['sequential_bits']:+.3f} ({Sr['shuffled_bits']:+.3f}); text tokens {St['H']:.2f}, {St['sequential_bits']:+.3f} | "
        f"native word given the current token {Sn['H_given_current_token']:.2f} (bits {Sn['H'] - Sn['H_given_current_token']:+.2f}), given the previous token {Sn['H_given_previous_token']:.2f}; rotated given the current token {Sr['H_given_current_token']:.2f} | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e487_stream_{name}", res, summ)
