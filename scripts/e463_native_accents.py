"""e463: do native words have "accents"? That is, is a token's grammatical role carried by which native words describe it,
or by the same words with different coefficients (an inflection)? Suggested by an external review.
Setup: twelve English words used as nouns or verbs (play, run, walk, drink, dance, swim, sleep, laugh, smile, jump, dream,
talk), each in 5 noun and 5 verb templates. At the middle depth, the state at the word's token is described with 16
native words (OMP, centred states); rotated words are the control.
Measured:
- support overlap (Jaccard) for the same token: within a role (noun with noun, verb with verb, different templates)
  against across roles; and different tokens in the same role;
- role decoding (noun against verb), leaving one word out (nearest centroid, cosine), from four inputs: which native
  words are used (binary), their coefficients, the rotated words' coefficients, and the raw state;
- the accent channel: for each token, the words used in both of its roles. Role decoding from their coefficients
  alone, leaving one word out, against decoding from the presence of the words used in one role only (the lexical
  channel); and the share of shared words whose mean coefficient changes sign between roles.
Models (argument): gpt2, qwen05.
Pre-registered (honest guesses):
- supports overlap more within a role than across roles for the same token (0.8);
- role is decodable from native supports alone at 0.8 or more, leaving one word out (0.6);
- the accent channel carries role (at least 0.7) (0.4);
- sign flips between roles are rare, under 10% of shared words (0.6)."""
import sys, os, json as _json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ma_common import *
WORDS = ["play", "run", "walk", "drink", "dance", "swim", "sleep", "laugh", "smile", "jump", "dream", "talk"]
NOUN = ["I liked the {} yesterday.", "The {} was great.", "We talked about the {}.", "That {} made me happy.", "She told me about her {}."]
VERB = ["We {} every day.", "They like to {} together.", "I will {} tomorrow.", "Please {} with me.", "You {} too much."]
name = sys.argv[1]; model, tok, fam = load_model(name); arch = Arch(model, fam); L = arch.NB // 2; K = 16
cen = torch.load("/workspace/wdd/eval_ids_all.pt", weights_only=False)[name]["cen_ids"][:8].to(DEV)
ref = block_states(model, arch, cen, [L], chunk=2)[L]; mu = ref[~sinkmask(ref)].mean(0); del ref
items = []
for wi, w in enumerate(WORDS):
    for role, T_ in (("noun", NOUN), ("verb", VERB)):
        for ti, t in enumerate(T_):
            s = t.format(w)
            enc = tok(s, add_special_tokens=False, return_offsets_mapping=True); a = s.index(w)
            pos = [i for i, (x0, x1) in enumerate(enc["offset_mapping"]) if x0 <= a < x1]
            if not pos or pos[0] == 0: continue
            ids = torch.tensor(enc["input_ids"], device=DEV)[None]; x = block_states(model, arch, ids, [L], chunk=1)[L][0][pos[0] - 1]
            items.append(dict(w=wi, role=0 if role == "noun" else 1, t=ti, x=x))
X = torch.stack([it["x"] for it in items]) - mu; wv = torch.tensor([it["w"] for it in items], device=DEV); rv = torch.tensor([it["role"] for it in items], device=DEV); n = len(items)
A, lab = build_dictionary(arch, blocks=list(range(L + 1))); Au = unitr(A); Ar = unitr(rotate(A, seed=7)); del A
def describe_(Dct):
    sel, cof, _ = omp(X, Dct, K, batch=256, record_err=False)
    C = torch.zeros(n, Dct.shape[0], device=DEV); C.scatter_(1, sel, cof); return sel, C
sel, C = describe_(Au); selr, Cr = describe_(Ar); B_ = (C != 0).float()
def jac(i, j):
    a, b = set(sel[i].tolist()), set(sel[j].tolist()); return len(a & b) / len(a | b)
within, across, diff_tok = [], [], []
for i in range(n):
    for j in range(i + 1, n):
        if wv[i] == wv[j] and rv[i] == rv[j]: within.append(jac(i, j))
        elif wv[i] == wv[j]: across.append(jac(i, j))
        elif rv[i] == rv[j]: diff_tok.append(jac(i, j))
mean = lambda v: sum(v) / max(len(v), 1)
def lowo(F):
    """role decoding leaving one word out: nearest role centroid by cosine"""
    F = F / F.norm(dim=-1, keepdim=True).clamp_min(1e-9); ok = 0
    for w in range(len(WORDS)):
        tr, te = wv != w, wv == w
        c0, c1 = F[tr & (rv == 0)].mean(0), F[tr & (rv == 1)].mean(0)
        pred = ((F[te] @ c1) > (F[te] @ c0)).long(); ok += int((pred == rv[te]).sum())
    return ok / n
res = dict(model=name, level=L, n=n, jaccard=dict(same_token_within_role=mean(within), same_token_across_roles=mean(across), different_token_same_role=mean(diff_tok)),
           decode=dict(native_support=lowo(B_), native_coefficients=lowo(C), rotated_coefficients=lowo(Cr), raw_state=lowo(X)))
# the accent channel against the lexical channel, per token
acc_F, lex_F, flips, shared_share = torch.zeros(n, C.shape[1], device=DEV), torch.zeros(n, C.shape[1], device=DEV), [], []
for w in range(len(WORDS)):
    m = wv == w; mn, mv = m & (rv == 0), m & (rv == 1)
    used_n, used_v = (B_[mn].sum(0) > 0), (B_[mv].sum(0) > 0); shared = used_n & used_v; only = used_n ^ used_v
    idx = torch.nonzero(m)[:, 0]; acc_F[idx] = C[idx] * shared.float(); lex_F[idx] = B_[idx] * only.float()
    if shared.any():
        cn, cv = C[mn][:, shared].mean(0), C[mv][:, shared].mean(0); flips.append(float(((cn * cv) < 0).float().mean()))
        e_sh = (C[idx][:, shared] ** 2).sum(); shared_share.append(float(e_sh / (C[idx] ** 2).sum()))
res["accent_channel"] = dict(decode_from_shared_word_coefficients=lowo(acc_F), decode_from_role_specific_word_presence=lowo(lex_F),
                             sign_flip_share=mean(flips), shared_words_energy_share=mean(shared_share))
J, Dd, Ac = res["jaccard"], res["decode"], res["accent_channel"]
res["checks"] = dict(within_over_across=J["same_token_within_role"] > J["same_token_across_roles"], support_decodes_0_8=Dd["native_support"] >= 0.8,
                     accent_carries_role=Ac["decode_from_shared_word_coefficients"] >= 0.7, sign_flips_rare=Ac["sign_flip_share"] < 0.1)
summ = (f"{name} L{L}, {len(WORDS)} noun/verb words x 10 templates (n {n}): Jaccard same token within role {J['same_token_within_role']:.2f}, across roles {J['same_token_across_roles']:.2f}, "
        f"different token same role {J['different_token_same_role']:.2f} | role decoding leaving one word out: native support {Dd['native_support']:.2f}, native coefficients {Dd['native_coefficients']:.2f}, "
        f"rotated {Dd['rotated_coefficients']:.2f}, raw state {Dd['raw_state']:.2f} | accent channel (coefficients of words shared by both roles) {Ac['decode_from_shared_word_coefficients']:.2f} "
        f"against lexical channel (presence of role-specific words) {Ac['decode_from_role_specific_word_presence']:.2f}; shared words carry {Ac['shared_words_energy_share']:.2f} of the description energy, sign flips {Ac['sign_flip_share']:.2f} | checks {_json.dumps(res['checks'])}")
log(summ); record(f"e463_accents_{name}", res, summ)
