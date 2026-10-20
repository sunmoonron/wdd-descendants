"""The identification floor (e666's extreme-value calibration, verbatim), the native-row dictionary, and orthogonal matching pursuit in the Euclidean metric or in a supplied metric."""
import math, torch
from .model import DEV, write_w
def unitr(X): return X / X.norm(dim=-1, keepdim=True).clamp_min(1e-12)
def rotate(A, seed=7):
    g = torch.Generator(device="cpu").manual_seed(seed); Q, _ = torch.linalg.qr(torch.randn(A.shape[1], A.shape[1], generator=g)); return A @ Q.to(A.device)
def rows_dictionary(M, upto):
    """unit MLP write rows of blocks 0..upto [m, D], each row's block and neuron, and the row norms"""
    W = [write_w(M, b) for b in range(upto + 1)]; R = torch.cat([w.T for w in W]); blk = torch.cat([torch.full((w.shape[1],), b, dtype=torch.long) for b, w in enumerate(W)]); neu = torch.cat([torch.arange(w.shape[1]) for w in W])
    nr = R.norm(dim=1); return R / nr[:, None].clamp_min(1e-12), blk.to(DEV), neu.to(DEV), nr
def gabs(m, Tt=14.0, n=28001):
    t = torch.linspace(0, Tt, n, dtype=torch.float64); F = torch.special.erf(t / math.sqrt(2)); return float(torch.trapezoid(1 - F.pow(m), t))
def floor_calibration(U, A, seed=11):
    """U unit centred states [N, D]; A unit atoms [m, D]. The competitor maximum of a rotated copy, calibrated: returns what floor_of needs"""
    m = A.shape[0]; Ar = unitr(rotate(A, seed=seed)); CA = A.T @ A / m; CR = Ar.T @ Ar / m; gm = gabs(m); N = U.shape[0]
    s2r = ((U @ CR) * U).sum(1); Mr = torch.cat([(U[s:s + 256] @ Ar.T).abs().max(1).values for s in range(0, N, 256)])
    r_cal = float(Mr.mean() / (s2r.clamp_min(1e-12).sqrt() * gm).mean()); return dict(r_cal=r_cal, gm=gm, CA=CA, m=m)
def floor_of(U, cal): return (((U @ cal["CA"]) * U).sum(1)).clamp_min(1e-12).sqrt() * cal["gm"] * cal["r_cal"]
def omp(X, A, k=16, batch=256, ridge=1e-5):
    N, NA = X.shape[0], A.shape[0]
    sel = torch.zeros(N, k, dtype=torch.long, device=X.device); cof = torch.zeros(N, k, device=X.device); eye = torch.eye(k, device=X.device)
    for s in range(0, N, batch):
        x = X[s:s + batch]; n = x.shape[0]; r = x.clone(); S = torch.zeros(n, 0, dtype=torch.long, device=X.device); taken = torch.zeros(n, NA, dtype=torch.bool, device=X.device)
        for step in range(k):
            corr = (r @ A.T).abs_().masked_fill_(taken, -1.0); pick = corr.argmax(-1, keepdim=True); taken.scatter_(1, pick, True); S = torch.cat([S, pick], 1)
            As = A[S]; G = As @ As.transpose(1, 2) + ridge * eye[:step + 1, :step + 1]; c = torch.cholesky_solve(As @ x[:, :, None], torch.linalg.cholesky(G)); r = x - (c.transpose(1, 2) @ As)[:, 0]
        sel[s:s + n] = S; cof[s:s + n] = c[:, :, 0]; del taken, corr
    return sel, cof

def omp_metric(X, A, L, k=16, batch=256, ridge=1e-5):
    """OMP in the metric |L v|^2 (L is [D, D], for instance the square root of a Fisher plus lambda times the identity): the state X [N, D] and the atoms A [m, D] are whitened by L, the pursuit runs in the Euclidean metric there, and the reconstruction is mapped back to the original space. Returns the reconstruction [N, D] and the chosen atoms [N, k]."""
    Aw = A @ L.T; nrm = Aw.norm(dim=1).clamp_min(1e-12); Aw = Aw / nrm[:, None]; sel, cof = omp(X @ L.T, Aw, k=k, batch=batch, ridge=ridge); return torch.einsum("nk,nkd->nd", cof, (A / nrm[:, None])[sel]), sel
