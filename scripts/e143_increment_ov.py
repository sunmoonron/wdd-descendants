"""e143 (GPT-2): complete increment attribution. For blocks 3..6 and 128 tokens: the block increment decomposed by
OMP (k=16) over (a) the block's MLP atoms + static head SVD atoms, (b) the block's MLP atoms + that token's OV
value atoms for all heads and all context positions (v_j W_O^h, j <= t). Metrics: increment FVU, the fraction of
the true attention write energy captured by the selected attention atoms, and dominant-MLP-write recall."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
c = Cache("gpt2"); model, tok, fam = load_model("gpt2"); arch = Arch(model, fam); NS, NH, HD, D = c.n_eval, c.NH, c.HD, c.D
g = torch.Generator().manual_seed(0); pick_seq = torch.randint(0, NS, (128,), generator=g); pick_pos = torch.randint(8, CTX, (128,), generator=g); tokid = pick_seq * CTX + pick_pos; rows = []
for b in (3, 4, 5, 6):
    l = arch.layers[b]; W = l.attn.c_attn.weight.detach().float(); bq = l.attn.c_attn.bias.detach().float(); Wo = arch.wo(b).view(NH, HD, D)
    Hb = c.s["H"][b].view(NS, CTX, D).float().to(DEV); ln = torch.nn.functional.layer_norm(Hb, (D,), l.ln_1.weight.float(), l.ln_1.bias.float(), l.ln_1.eps); v = (ln @ W[:, 2 * D:] + bq[2 * D:]).view(NS, CTX, NH, HD)
    Ab, labb = c.dictionary(b, blocks=[b], types=(T_MLP, T_ATT, T_BIAS)); Am = Ab[labb["type"].to(DEV) == T_MLP]; mlp_rows = torch.arange(Am.shape[0], device=DEV)
    led = c.ledger(b, blocks=[b])[b][tokid]; tn_ = led.abs().argmax(1).to(DEV)
    fv_s, fv_d, cap_s, cap_d, rec_s, rec_d = [], [], [], [], [], []
    for i in range(128):
        s, t = pick_seq[i].item(), pick_pos[i].item(); Dv = (c.s["H"][b + 1][tokid[i]].float() - c.s["H"][b][tokid[i]].float()).to(DEV)[None]
        att_true = c.s["ATT"][b][tokid[i]].float().to(DEV)
        VA = torch.einsum("jhd,hde->hje", v[s, :t + 1], Wo).reshape(-1, D); VA = VA / VA.norm(dim=1, keepdim=True).clamp_min(1e-6)   # [NH*(t+1), D]
        Adyn = torch.cat([Am, VA]); sd, cd, ed = omp(Dv, Adyn, 16); ss, cs, es = omp(Dv, Ab, 16)
        fv_s.append((es[0, 15] / (Dv ** 2).sum()).item()); fv_d.append((ed[0, 15] / (Dv ** 2).sum()).item())
        att_s = ss[0] >= Am.shape[0]; att_d = sd[0] >= Am.shape[0]
        for sel_, cof_, Ad_, m_, out in ((ss, cs, Ab, att_s, cap_s), (sd, cd, Adyn, att_d, cap_d)):
            if m_.any():
                B_ = Ad_[sel_[0][m_]]; coef = torch.linalg.lstsq(B_.T, att_true[:, None]).solution[:, 0]; out.append((1 - ((att_true - coef @ B_) ** 2).sum() / (att_true ** 2).sum()).item())
            else: out.append(0.0)
        rec_s.append((ss[0] == tn_[i]).any().item()); rec_d.append((sd[0] == tn_[i]).any().item())
    rows.append(dict(b=b, fvu16_static=float(np.mean(fv_s)), fvu16_ov=float(np.mean(fv_d)), att_energy_captured_static=float(np.median(cap_s)), att_energy_captured_ov=float(np.median(cap_d)), mlp_recall_static=float(np.mean(rec_s)), mlp_recall_ov=float(np.mean(rec_d))))
    log(f"b{b}: increment fvu16 static {rows[-1]['fvu16_static']:.3f} -> ov {rows[-1]['fvu16_ov']:.3f} | attention energy captured {rows[-1]['att_energy_captured_static']:.2f} -> {rows[-1]['att_energy_captured_ov']:.2f} | dominant MLP recall {rows[-1]['mlp_recall_static']:.2f} -> {rows[-1]['mlp_recall_ov']:.2f}")
record("e143_incov_gpt2", dict(rows=rows), " | ".join(f"b{r['b']}: fvu {r['fvu16_static']:.2f}->{r['fvu16_ov']:.2f} att-captured {r['att_energy_captured_static']:.2f}->{r['att_energy_captured_ov']:.2f} mlp-recall {r['mlp_recall_static']:.2f}->{r['mlp_recall_ov']:.2f}" for r in rows))
