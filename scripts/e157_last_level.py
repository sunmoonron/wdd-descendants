"""e157: WDD at the output as a logit lens. At the last level (the raw final state, before the final norm), for
tied-embedding models the token atoms are unembedding rows: is the model's predicted next token's atom in the
k=64 support, is it the first token atom picked, and does the coefficient rank of token atoms track the logit
rank? For untied models the same with the unembedding rows appended as extra atoms."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); L = c.NB - 1; model, tok, fam = load_model(c.name); NS = 16; ids_seq = c.s["eval_ids"][:NS].to(DEV); NT = NS * CTX; ids = torch.arange(NT)
logits = model(ids_seq).logits.float().reshape(NT, -1); pred = logits.argmax(1); top5 = logits.topk(5, dim=1).indices
X = c.X(L)[ids]; typ = typical_mask(c.X(L, center=False)[ids]); A, lab = c.dictionary(L); typA = lab["type"].to(DEV)
WU = model.get_output_embeddings().weight.detach().float().to(DEV); tied = torch.allclose(WU[:100], c.d["emb"][0][:100].to(DEV), atol=1e-4)
if tied: Ad, tok_rows = A, torch.arange(c.n_tok_emb, device=DEV)
else: U = WU / WU.norm(dim=1, keepdim=True); Ad = torch.cat([A, U]); tok_rows = torch.arange(A.shape[0], A.shape[0] + WU.shape[0], device=DEV)
sel, cof, err = omp(X, Ad, 64); istok = sel >= tok_rows[0] if not tied else (typA[sel] == T_TOK)
pred_row = tok_rows[pred]; hitp = (sel == pred_row[:, None]).any(1); first_tok_pos = torch.where(istok.any(1), istok.float().argmax(1), torch.full((NT,), -1, device=DEV)); first_tok = sel.gather(1, first_tok_pos.clamp(min=0)[:, None])[:, 0]
first_is_pred = (first_tok == pred_row) & istok.any(1); first_in_top5 = torch.isin(first_tok - tok_rows[0], top5.flatten()) & istok.any(1)
first_in_top5 = torch.stack([(first_tok - tok_rows[0]) == top5[:, j] for j in range(5)], 1).any(1) & istok.any(1)
res = dict(model=tag, L=L, tied=bool(tied), pred_atom_in_support=hitp[typ].float().mean().item(), first_token_atom_is_pred=first_is_pred[typ].float().mean().item(), first_token_atom_in_top5=first_in_top5[typ].float().mean().item(), token_atoms_share=istok[typ].float().mean().item(), fvu64=fvu(err[:, 63], X, typ))
record(f"e157_lastlevel_{tag}", res, f"tied {tied} | predicted next token's atom in the k=64 support {res['pred_atom_in_support']:.2f} | first token atom picked = prediction {res['first_token_atom_is_pred']:.2f}, in top-5 {res['first_token_atom_in_top5']:.2f} | token atoms {res['token_atoms_share']:.2f} of support | fvu64 {res['fvu64']:.3f}")
