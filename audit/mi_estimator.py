# -*- coding: utf-8 -*-
"""
mi_estimator.py  ——  Phase3 · continuous-variable mutual-information estimation (KSG / Kraskov k-NN, estimator 1)
===============================================================================
Provides a nonparametric estimate of I(X;Y) for the contract layer's "mutual-information leakage audit" (X,Y can be multi-dimensional continuous).

KSG estimator 1 (Kraskov-Stögbauer-Grassberger 2004):
    I(X;Y) = ψ(k) + ψ(N) − <ψ(n_x+1) + ψ(n_y+1)>
where, for each sample point, the k-th nearest-neighbor distance ε_i is found in the joint space (X,Y) using the Chebyshev (max) norm,
n_x/n_y = the number of points within ε_i of it in each marginal space. ψ=digamma.

Reliability note (INV-E, must be viewed honestly):
    - continuous MI estimation is **biased and has variance**; the smaller the sample size N and the higher the dimension, the larger the bias/variance.
    - for a **deterministic relation** (e.g. odom=truth), the true MI=+∞, and KSG returns a **finite large value** that grows with N (not actual infinity).
      So "full leak" manifests as MI far over the bound but not literally infinite — flagging RED on this basis is still valid.
    - this module **standardizes** each dimension before estimating (making the max norm comparable across dimensions); the return value is clamped to ≥0 (MI is non-negative).
"""

import numpy as np

try:
    from scipy.special import digamma
    from scipy.spatial import cKDTree
    _HAVE_SCIPY = True
except Exception:                       # fallback: use a simplified implementation when scipy is unavailable
    _HAVE_SCIPY = False


def _as2d(a):
    a = np.asarray(a, dtype=np.float64)
    return a.reshape(-1, 1) if a.ndim == 1 else a


def _standardize(a):
    """Per-dimension standardization (zero mean, unit variance); add tiny jitter to constant dimensions to avoid degeneracy."""
    mu = a.mean(axis=0, keepdims=True)
    sd = a.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (a - mu) / sd


def ksg_mi(X, Y, k=4, standardize=True, seed=0):
    """KSG estimate of I(X;Y) (nats). X:(N,dx), Y:(N,dy). Degenerate handling when N is insufficient or scipy is missing."""
    X, Y = _as2d(X), _as2d(Y)
    N = X.shape[0]
    if N <= k + 2:
        return 0.0
    if standardize:
        X, Y = _standardize(X), _standardize(Y)
    if not _HAVE_SCIPY:
        return _binned_mi(X, Y)        # fallback
    rng = np.random.default_rng(seed)
    # add tiny jitter to break tied distances (KSG is sensitive to ties)
    X = X + rng.normal(0, 1e-10, X.shape)
    Y = Y + rng.normal(0, 1e-10, Y.shape)
    XY = np.hstack([X, Y])
    tree = cKDTree(XY)
    dists, _ = tree.query(XY, k=k + 1, p=np.inf)   # includes itself, take the (k+1)-th = the k-th nearest neighbor
    eps = dists[:, -1]
    tx, ty = cKDTree(X), cKDTree(Y)
    nx = np.array([len(tx.query_ball_point(X[i], eps[i] - 1e-12, p=np.inf)) - 1
                   for i in range(N)])
    ny = np.array([len(ty.query_ball_point(Y[i], eps[i] - 1e-12, p=np.inf)) - 1
                   for i in range(N)])
    mi = digamma(k) + digamma(N) - np.mean(digamma(nx + 1) + digamma(ny + 1))
    return float(max(0.0, mi))


def _binned_mi(X, Y, bins=8):
    """Fallback: equal-frequency binned-histogram MI (used only when scipy is missing; larger bias)."""
    def disc(a):
        out = np.zeros(a.shape[0], dtype=np.int64)
        for j in range(a.shape[1]):
            q = np.quantile(a[:, j], np.linspace(0, 1, bins + 1)[1:-1])
            out = out * bins + np.digitize(a[:, j], q)
        return out
    xd, yd = disc(X), disc(Y)
    N = X.shape[0]
    mi = 0.0
    for xv in np.unique(xd):
        px = np.mean(xd == xv)
        for yv in np.unique(yd):
            py = np.mean(yd == yv)
            pxy = np.mean((xd == xv) & (yd == yv))
            if pxy > 0:
                mi += pxy * np.log(pxy / (px * py))
    return float(max(0.0, mi))


def increments(traj):
    """Trajectory (T,2) → per-step displacement increments (T-1,2)."""
    traj = np.asarray(traj, dtype=np.float64)
    return traj[1:] - traj[:-1]


if __name__ == "__main__":
    # self-check: a Gaussian channel with known MI Y = X + N(0,σ²), theory I=½log(1+Var(X)/σ²)
    rng = np.random.default_rng(1)
    N = 2000
    x = rng.normal(0, 1, (N, 1))
    for sigma in (0.3, 1.0):
        y = x + rng.normal(0, sigma, (N, 1))
        theo = 0.5 * np.log(1 + 1.0 / sigma ** 2)
        est = ksg_mi(x, y, k=4)
        print(f"σ={sigma}: KSG I={est:.3f} nats, theory ½log(1+1/σ²)={theo:.3f} nats")
    # deterministic Y=X: MI→∞, KSG returns a large finite value
    print(f"deterministic Y=X: KSG I={ksg_mi(x, x, k=4):.3f} nats (should be far larger than above, but finite)")
