"""e23: are the selected token-embedding atoms spurious, or attention copies? For each token-embedding atom in
the mid-layer OMP support: does it match the current token, the previous token, a token within the last 8 / 64
positions, any earlier token in the sequence, or nothing (spurious)? Compared to a null that draws random token
rows with the same unigram frequency. Also the coefficient sign and the position offset histogram of the matches."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = mid(c); X = c.X(L); typ = typical_mask(c.X(L, center=False)); A, lab = c.dictionary(L)
sel, cof, err = get_omp(c, L, A=A, X=X); ids_all = c.s["eval_ids"].to(DEV)               # [n_eval, CTX]
NT = c.NT; pos = torch.arange(NT, device=DEV) % CTX; seq = torch.arange(NT, device=DEV) // CTX
istok = (lab["type"].to(DEV)[sel] == T_TOK); tokid = sel.clone(); tokid[~istok] = -1        # token atoms: row == token id
cur = ids_all.reshape(-1)
def match_stats(tokid_):
    out = dict(n=0, current=0, prev=0, within8=0, within64=0, anywhere_before=0, after=0)
    for i in range(0, NT, 2048):
        t = tokid_[i:i + 2048]; m = t >= 0; out["n"] += int(m.sum())
        s_ = seq[i:i + 2048]; p_ = pos[i:i + 2048]; ctx = ids_all[s_]                       # [n, CTX]
        eq = (ctx[:, None, :] == t[:, :, None]) & m[:, :, None]                                # [n, k, CTX]
        offs = p_[:, None, None] - torch.arange(CTX, device=DEV)[None, None, :]                # position offset (>0 = before)
        out["current"] += int((eq & (offs == 0)).any(2).sum()); out["prev"] += int((eq & (offs == 1)).any(2).sum())
        out["within8"] += int((eq & (offs >= 1) & (offs <= 8)).any(2).sum()); out["within64"] += int((eq & (offs >= 1) & (offs <= 64)).any(2).sum())
        out["anywhere_before"] += int((eq & (offs >= 1)).any(2).sum()); out["after"] += int((eq & (offs < 0)).any(2).sum())
    return {k: (v / out["n"] if k != "n" else v) for k, v in out.items()}
real = match_stats(tokid)
# null: replace each selected token atom by a random token drawn from the unigram distribution of the centering slice
g = torch.Generator().manual_seed(0); cen = c.s["cen_ids"].reshape(-1)
draw = cen[torch.randint(0, len(cen), (int(istok.sum()),), generator=g)].to(DEV); null_tok = tokid.clone(); null_tok[istok] = draw
null = match_stats(null_tok)
res = dict(model=tag, L=L, frac_support_token_atoms=istok[typ].float().mean().item(), real=real, null=null)
# among matches to a previous position: sign of the coefficient (a copy should be positive) and the nearest offset histogram
signs = []; offs_hist = torch.zeros(9, device=DEV)   # 1..8 and >8
for i in range(0, NT, 2048):
    t = tokid[i:i + 2048]; m = t >= 0; s_ = seq[i:i + 2048]; p_ = pos[i:i + 2048]; ctx = ids_all[s_]
    eq = (ctx[:, None, :] == t[:, :, None]) & m[:, :, None]; offs = p_[:, None, None] - torch.arange(CTX, device=DEV)[None, None, :]
    before = eq & (offs >= 1); has = before.any(2)
    nearest = torch.where(before, offs, torch.full_like(offs, 10 ** 6)).min(2).values
    signs.append(cof[i:i + 2048][has] > 0); nb = nearest[has].clamp(max=9); offs_hist += torch.bincount(nb, minlength=10)[1:10].float()
res["copy_matches"] = dict(frac_positive_coef=torch.cat(signs).float().mean().item(), nearest_offset_hist=(offs_hist / offs_hist.sum()).tolist())
record(f"e23_copy_{tag}", res, f"token atoms {res['frac_support_token_atoms']:.2f} of support | match current {real['current']:.3f} (null {null['current']:.3f}) prev {real['prev']:.3f} ({null['prev']:.3f}) within8 {real['within8']:.3f} ({null['within8']:.3f}) within64 {real['within64']:.3f} ({null['within64']:.3f}) before {real['anywhere_before']:.3f} ({null['anywhere_before']:.3f}) | positive coef among copies {res['copy_matches']['frac_positive_coef']:.2f} offsets " + " ".join(f"{v:.2f}" for v in res['copy_matches']['nearest_offset_hist']))
