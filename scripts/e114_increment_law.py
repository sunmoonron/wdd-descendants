"""e114: the prominence law at the increment level. Per block: the dominant MLP write's prominence inside the
block increment |D_c . d| / ||D_c|| vs the block dictionary's floor sqrt(2 ln m_b / d); predicted one-shot
identification P(prom > floor) vs measured one-shot and OMP (k=8) on the increment. No fitting."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); N = 4096; ids = sub(c.NT, N); rows = []
for b in range(c.NB):
    D_ = c.s["H"][b + 1][ids].float().to(DEV) - c.s["H"][b][ids].float().to(DEV); Dc = D_ - D_.mean(0); typ = typical_mask(c.X(b, center=False)[ids])
    Ab, labb = c.dictionary(b, blocks=[b], types=(T_MLP, T_ATT, T_BIAS)); mlp_rows = torch.nonzero(labb["type"] == T_MLP)[:, 0].to(DEV)
    led = c.ledger(b, blocks=[b])[b][ids]; tn_ = led.abs().argmax(1).to(DEV); row = mlp_rows[tn_]; d = Ab[row]
    prom = (Dc * d).sum(1).abs() / Dc.norm(dim=1); floor = math.sqrt(2 * math.log(Ab.shape[0]) / c.D)
    s1, _, _ = oneshot(Dc, Ab, 8); so, _, _ = omp(Dc, Ab, 8)
    rows.append(dict(b=b, m=Ab.shape[0], floor=floor, pred=(prom[typ] > floor).float().mean().item(), oneshot8=(s1 == row[:, None]).any(1)[typ].float().mean().item(), omp8=(so == row[:, None]).any(1)[typ].float().mean().item(), prom_med=prom[typ].median().item()))
    log(f"{tag} b{b}: floor {floor:.3f} prom med {rows[-1]['prom_med']:.2f} pred {rows[-1]['pred']:.2f} oneshot8 {rows[-1]['oneshot8']:.2f} omp8 {rows[-1]['omp8']:.2f}")
P = torch.tensor([r["pred"] for r in rows]); O = torch.tensor([r["oneshot8"] for r in rows]); M = torch.tensor([r["omp8"] for r in rows])
res = dict(model=tag, rows=rows, corr_pred_oneshot=torch.corrcoef(torch.stack([P, O]))[0, 1].item(), mae_oneshot=(P - O).abs().mean().item(), corr_pred_omp=torch.corrcoef(torch.stack([P, M]))[0, 1].item(), mae_omp=(P - M).abs().mean().item())
record(f"e114_inclaw_{tag}", res, f"blocks {len(rows)} | corr(pred, one-shot8) {res['corr_pred_oneshot']:.2f} MAE {res['mae_oneshot']:.3f} | corr(pred, omp8) {res['corr_pred_omp']:.2f} MAE {res['mae_omp']:.3f} | " + " ".join(f"b{r['b']}:{r['pred']:.2f}/{r['oneshot8']:.2f}" for r in rows))
