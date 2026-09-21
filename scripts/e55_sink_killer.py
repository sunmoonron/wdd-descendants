"""e55: who cancels the sink at the last block? For the sink states (norm > 10x median) at the second-to-last level,
the projection of the last block's attention write and MLP write onto the massive channel direction (the top
coordinate of the sink states), relative to the sink's projection; and the same for typical states."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdd_common import *
tag = sys.argv[1]; c = Cache(tag); out = dict(model=tag, levels={})
for L in range(c.NB - 3, c.NB):
    Xin = c.X(L - 1, center=False); Xout = c.X(L, center=False); typ = typical_mask(Xin); S = torch.nonzero(~typ)[:, 0]
    if len(S) == 0: out["levels"][L] = "no sinks at input"; continue
    ch = Xin[S].abs().mean(0).argmax().item(); d = torch.zeros(c.D, device=DEV); d[ch] = 1.0
    Sc = S.cpu(); att = c.s["ATT"][L][Sc].float().to(DEV); mlp = c.acts[L][Sc].float().to(DEV) @ c.wdir_cpu(L).to(DEV)
    if c.d["mlp_bias"][L] is not None: mlp = mlp + c.d["mlp_bias"][L].to(DEV)
    pin, pout = Xin[S] @ d, Xout[S] @ d; pa, pm = att @ d, mlp @ d
    # which MLP neurons of the last block do the cancelling (top-3 by mean contribution along d on sink tokens)
    contrib = c.acts[L][Sc].float().to(DEV) * c.wdir_cpu(L).to(DEV)[:, ch][None]; top = contrib.mean(0).abs().topk(3)
    out["levels"][L] = dict(n_sink=int(len(S)), channel=ch, in_med=pin.median().item(), out_med=pout.median().item(), att_med=pa.median().item(), mlp_med=pm.median().item(),
                            frac_removed=((pin - pout) / pin).median().item(), att_share_of_removal=(pa / (pin - pout).clamp_min(1e-6)).median().item() * -1, mlp_share_of_removal=(pm / (pin - pout).clamp_min(1e-6)).median().item() * -1,
                            top_mlp_neurons=[(int(i), float(contrib.mean(0)[i])) for i in top.indices.tolist()], sink_norm_ratio_in=(Xin[S].norm(dim=1).median() / Xin[typ].norm(dim=1).median()).item(), sink_norm_ratio_out=(Xout[S].norm(dim=1).median() / Xout[typ].norm(dim=1).median()).item())
    log(f"{tag} block {L}: sinks {len(S)} ch{ch} in {pin.median():.0f} out {pout.median():.0f} att {pa.median():.0f} mlp {pm.median():.0f} removed {out['levels'][L]['frac_removed']:.2f} (att {out['levels'][L]['att_share_of_removal']:.2f} mlp {out['levels'][L]['mlp_share_of_removal']:.2f}) norm ratio {out['levels'][L]['sink_norm_ratio_in']:.0f} -> {out['levels'][L]['sink_norm_ratio_out']:.0f}")
record(f"e55_sink_{tag}", out, " | ".join(f"blk {L}: " + (v if isinstance(v, str) else f"ch{v['channel']} removed {v['frac_removed']:.2f} att {v['att_share_of_removal']:.2f} mlp {v['mlp_share_of_removal']:.2f} ratio {v['sink_norm_ratio_in']:.0f}->{v['sink_norm_ratio_out']:.0f}") for L, v in out["levels"].items()))
