"""e709 (session 160): two clocks. A dictionary that is the same at every position has no time-scale; the model's
coordinates do. Along the sequence, the autocorrelation of each coordinate of the weak field (the weak write on its
sixteen directions), the strong field, the stream's own sixteen principal directions, sixteen random directions and the
sixty-four most active neurons' activations gives each an integral time-scale, a half-life and a slow fraction (the
variance kept by a thirty-two-token moving average); on Pile documents the jump of each coordinate across a document
boundary is compared with the jump over the same distance inside a document. Arguments: --model M --block B [--smoke].
Pre-registered in e707_e712_prereg.json."""
import argparse
from c11 import *
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--block", type=int, required=True); ap.add_argument("--smoke", action="store_true"); A_ = ap.parse_args(); SMOKE = A_.smoke
name = A_.model; B = A_.block; TAG = f"e709_{name}_b{B}" + ("_smoke" if SMOKE else ""); M = load(name); NB, D, DFF = M["NB"], M["D"], M["DFF"]; T = 512; L_ = lambda s: log(TAG, s)
NWIN = 8 if SMOKE else 64; NDOC = 100 if SMOKE else 600; LMAX = 128; NNEU = 64; Q = 0.25; MAV = 32; W = write_w(M, B)
ids = windows_from(wikitext_ids(M, "test"), NWIN, T).to(DEV); stream = pile_ids(M, ndocs=NDOC); EOS = M["tok"].eos_token_id
ep = (stream == EOS).nonzero()[:, 0]; ep = ep[(ep > T // 2) & (ep < len(stream) - T // 2 - 1)]; ep = ep[torch.linspace(0, len(ep) - 1, NWIN).long()]; pids = torch.stack([stream[p - T // 2:p - T // 2 + T] for p in ep.tolist()]).to(DEV)   # Pile windows centred on a document boundary
def capture(wins):
    X, AB = [], []
    with Capture(M, [B], ["x", "act"]) as C:
        for s0 in range(0, wins.shape[0], 8): run_base(M, wins[s0:s0 + 8]); X.append(C.cap[("x", B)].float()); AB.append(C.cap[("act", B)].float())
    return torch.cat(X), torch.cat(AB)
X, AB = capture(ids); XBAR = X.reshape(-1, D).mean(0); TAU = torch.cat([AB.reshape(-1, DFF)[:, c0:c0 + 512].abs().quantile(1 - Q, dim=0) for c0 in range(0, DFF, 512)])
def coords(X, AB):
    WK = AB.abs() < TAU; return dict(weak_field=(AB * WK) @ W.T, strong_field=(AB * ~WK) @ W.T, stream=X - XBAR)
CO = coords(X, AB); pcs = lambda Z: torch.linalg.eigh(Z.reshape(-1, D).T @ Z.reshape(-1, D))[1][:, -16:].contiguous()
SUB = {"weak_field": pcs(CO["weak_field"]), "strong_field": pcs(CO["strong_field"]), "stream_pcs": pcs(CO["stream"]), "random16": torch.linalg.qr(torch.randn(D, 16, device=DEV, generator=torch.Generator(device=DEV).manual_seed(709 + B)))[0]}
NEU = (AB.reshape(-1, DFF).abs() >= TAU).float().sum(0).topk(NNEU).indices
def series(X, AB):
    C_ = coords(X, AB); Z = {"weak_field": C_["weak_field"] @ SUB["weak_field"], "strong_field": C_["strong_field"] @ SUB["strong_field"], "stream_pcs": C_["stream"] @ SUB["stream_pcs"], "random16": C_["stream"] @ SUB["random16"], "neurons": AB[:, :, NEU]}; return Z
def clocks(Z):
    Zc = Z - Z.mean((0, 1)); var = Zc.pow(2).mean((0, 1)).clamp_min(1e-12); rho = torch.stack([(Zc[:, :-l] * Zc[:, l:]).mean((0, 1)) / var for l in range(1, LMAX + 1)], 1)   # [k, LMAX]
    neg = (rho <= 0).float().cumsum(1) > 0; tau_int = 1 + 2 * (rho * ~neg).sum(1); half = torch.where((rho < 0.5).any(1), (rho < 0.5).float().argmax(1) + 1, torch.full_like(tau_int, LMAX)).float()
    mav = torch.nn.functional.avg_pool1d(Zc.permute(0, 2, 1).reshape(-1, 1, Zc.shape[1]), MAV, stride=1).reshape(Zc.shape[0], Zc.shape[2], -1); slow = mav.pow(2).mean((0, 2)) / var
    return dict(tau_int=tau_int, half_life=half, slow_fraction=slow, rho1=rho[:, 0], rho16=rho[:, 15])
ZW = series(X, AB); CL = {g: clocks(Z) for g, Z in ZW.items()}
# document boundaries on the Pile stream: the jump over eight tokens across a boundary against inside documents
XP, ABP = capture(pids); ZP = series(XP, ABP); JUMP = {}
for g, Z in ZP.items():
    jb, jw = [], []
    for w in range(pids.shape[0]):
        e = (pids[w] == EOS).nonzero()[:, 0]; e = e[(e >= 9) & (e <= T - 10)]
        for p in e.tolist(): jb.append((Z[w, p + 8] - Z[w, p - 1]).pow(2))
        inside = torch.ones(T, dtype=torch.bool, device=DEV)
        for p in e.tolist(): inside[max(0, p - 10):p + 10] = False
        tt = torch.nonzero(inside[:T - 9] & inside[9:])[:, 0]
        if tt.numel(): jw.append((Z[w, tt + 9] - Z[w, tt]).pow(2).mean(0))
    JUMP[g] = (torch.stack(jb).mean(0) / torch.stack(jw).mean(0).clamp_min(1e-12)) if jb and jw else torch.full((Z.shape[-1],), float("nan"), device=DEV)
S_ = {g: dict(tau_int=med(CL[g]["tau_int"]), half_life=med(CL[g]["half_life"]), slow_fraction=med(CL[g]["slow_fraction"]), rho1=med(CL[g]["rho1"]), rho16=med(CL[g]["rho16"]), boundary_jump=med(JUMP[g])) for g in ZW}
ti = lambda g: S_[g]["tau_int"]
V = dict(C1_weak_field_3x_neurons=bool(ti("weak_field") >= 3 * ti("neurons")), C2_weak_field_2x_random=bool(ti("weak_field") >= 2 * ti("random16")), C3_weak_field_slower_than_strong=bool(ti("weak_field") >= 1.5 * ti("strong_field")), C4_boundary_jump_over_2=bool(S_["weak_field"]["boundary_jump"] > 2), C5_stream_pcs_slowest=bool(ti("stream_pcs") > ti("weak_field")))
res = dict(model=M["hf"], block=B, n_blocks=NB, windows=NWIN, T=T, lmax=LMAX, groups=list(ZW), summary=S_, per_coordinate={g: {k: v.tolist() for k, v in CL[g].items()} | {"boundary_jump": JUMP[g].tolist()} for g in ZW}, verdicts=V)
summ = (f"[{name} block {B}] integral time-scales (tokens): " + ", ".join(f"{g} {S_[g]['tau_int']:.1f} (half-life {S_[g]['half_life']:.0f}, slow fraction {S_[g]['slow_fraction']:.2f}, boundary jump {S_[g]['boundary_jump']:.2f})" for g in ZW) + f"; verdicts {V}")
save(TAG, res, summ)
