"""e53 (GPT-2): attention through the OV circuit, vectorized over heads. A head's write at position t is
sum_j alpha_tj (v_j W_O^h), a nonnegative combination of context-dependent value atoms that hidden states give
for free. (1) exactness check; (2) attention mass on {t, t-1, t-2, 0}; (3) alpha recovered from the write by
NNLS over the value atoms (projected gradient), correlation with the true alpha; (4) WDD with the dominant head's
value atoms (all j <= t) added to the static dictionary: is the head identified, how much of its write is captured."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = "gpt2"; c = Cache(tag); L = mid(c); model, tok, fam = load_model("gpt2"); arch = Arch(model, fam)
NS, NH, HD, D = c.n_eval, c.NH, c.HD, c.D; NTOK = 128
g = torch.Generator().manual_seed(0); pick_seq = torch.randint(0, NS, (NTOK,), generator=g); pick_pos = torch.randint(8, CTX, (NTOK,), generator=g)
A, lab = c.dictionary(L); typA, blkA, idxA = lab["type"].to(DEV), lab["block"].to(DEV), lab["index"].to(DEV)
X = c.X(L); flat = (pick_seq * CTX + pick_pos); Xs = X[flat.to(DEV)]
def qkv(b):
    l = arch.layers[b]; W = l.attn.c_attn.weight.detach().float(); bq = l.attn.c_attn.bias.detach().float()
    Hb = c.s["H"][b].view(NS, CTX, D).float().to(DEV); ln = torch.nn.functional.layer_norm(Hb, (D,), l.ln_1.weight.float(), l.ln_1.bias.float(), l.ln_1.eps)
    q = (ln @ W[:, :D] + bq[:D]).view(NS, CTX, NH, HD); k = (ln @ W[:, D:2 * D] + bq[D:2 * D]).view(NS, CTX, NH, HD); v = (ln @ W[:, 2 * D:] + bq[2 * D:]).view(NS, CTX, NH, HD)
    return q, k, v, arch.wo(b).view(NH, HD, D)
def nnls_batched(Bt, y, iters=300):
    """Bt [H, m, D], y [H, D] -> a [H, m] >= 0."""
    Lc = torch.linalg.eigvalsh(Bt @ Bt.transpose(1, 2)).max(1).values.clamp_min(1e-6)[:, None]; a = torch.zeros(Bt.shape[0], Bt.shape[1], device=DEV)
    for _ in range(iters): a = (a - torch.einsum("hmd,hd->hm", Bt, torch.einsum("hm,hmd->hd", a, Bt) - y) / Lc).clamp_min(0)
    return a
res = dict(model=tag, L=L, n_tokens=NTOK, blocks={}); allW = {}
for b in range(L + 1):
    q, k, v, Wo = qkv(b); rec, mass, corr, l1, sinkm = [], [], [], [], []
    for i in range(NTOK):
        s, t = pick_seq[i].item(), pick_pos[i].item()
        sc = torch.einsum("jhd,hd->hj", k[s, :t + 1], q[s, t]) / math.sqrt(HD); al = torch.softmax(sc, 1)                # [H, t+1]
        VA = torch.einsum("jhd,hde->hje", v[s, :t + 1], Wo)                                                              # [H, t+1, D]
        w = torch.einsum("hj,hje->he", al, VA); w_true = torch.stack([c.head_write(b, h, tokens=torch.tensor([s * CTX + t]))[0] for h in range(NH)]).to(DEV)
        rec.append(((w - w_true).norm(dim=1) / w_true.norm(dim=1).clamp_min(1e-6)).median().item())
        J = sorted(set([t, t - 1, t - 2, 0])); mass.append(al[:, J].sum(1)); sinkm.append(al[:, 0])
        a_hat = nnls_batched(VA, w_true); a_hat = a_hat / a_hat.sum(1, keepdim=True).clamp_min(1e-9)
        for h in range(NH): corr.append(torch.corrcoef(torch.stack([a_hat[h], al[h]]))[0, 1].nan_to_num(0).item())
        l1.append((a_hat - al).abs().sum(1))
        allW[(b, i)] = (al, VA, w_true)
    res["blocks"][b] = dict(recon_rel_err_med=float(np.median(rec)), local_mass_med=torch.cat(mass).median().item(), sink_mass_med=torch.cat(sinkm).median().item(), alpha_corr_med=float(np.median(corr)), alpha_corr_q25=float(np.quantile(corr, 0.25)), alpha_l1_med=torch.cat(l1).median().item())
    log(f"{tag} b{b}: recon {res['blocks'][b]['recon_rel_err_med']:.4f} | mass local {res['blocks'][b]['local_mass_med']:.2f} sink {res['blocks'][b]['sink_mass_med']:.2f} | alpha from write: corr med {res['blocks'][b]['alpha_corr_med']:.2f} q25 {res['blocks'][b]['alpha_corr_q25']:.2f} L1 {res['blocks'][b]['alpha_l1_med']:.2f}")
# (4) WDD with the dominant head's value atoms (all context positions) appended to the static dictionary
hits_static, hits_dyn, energy_dyn, n_dyn, dom_share = [], [], [], [], []
for i in range(NTOK):
    norms = torch.stack([allW[(b, i)][2].norm(dim=1) for b in range(L + 1)]); db, dh = divmod(int(norms.argmax()), NH); dom_share.append((norms.max() ** 2 / (norms ** 2).sum()).item())
    al, VA, w_true = allW[(db, i)]; VAh = VA[dh]; VAh = VAh / VAh.norm(dim=1, keepdim=True).clamp_min(1e-6)
    Ad = torch.cat([A, VAh]); sel, cof, err = omp(Xs[i:i + 1], Ad, 64); dyn = sel[0] >= A.shape[0]
    hits_dyn.append(dyn.any().item()); n_dyn.append(int(dyn.sum())); st = sel[0][~dyn]; hits_static.append(((typA[st] == T_ATT) & (blkA[st] == db) & (idxA[st] == dh)).any().item())
    if dyn.any():
        B_ = Ad[sel[0][dyn]]; coef = torch.linalg.lstsq(B_.T, w_true[dh][:, None]).solution[:, 0]; energy_dyn.append((1 - ((w_true[dh] - coef @ B_) ** 2).sum() / (w_true[dh] ** 2).sum()).item())
res["wdd_with_value_atoms"] = dict(dominant_head_share_of_att_energy_med=float(np.median(dom_share)), selected_static=float(np.mean(hits_static)), selected_dynamic=float(np.mean(hits_dyn)), n_value_atoms_selected_mean=float(np.mean(n_dyn)),
                                    energy_captured_med=float(np.median(energy_dyn)) if energy_dyn else None)
record("e53_ov_gpt2", res, " | ".join(f"b{b}: local {v['local_mass_med']:.2f} sink {v['sink_mass_med']:.2f} alpha-corr {v['alpha_corr_med']:.2f}" for b, v in res["blocks"].items()) + f" || dominant head selected: static {res['wdd_with_value_atoms']['selected_static']:.2f} -> with value atoms {res['wdd_with_value_atoms']['selected_dynamic']:.2f} ({res['wdd_with_value_atoms']['n_value_atoms_selected_mean']:.1f} atoms), head energy captured {res['wdd_with_value_atoms']['energy_captured_med']}")
