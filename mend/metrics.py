"""The four benchmark metrics, unchanged from the published definitions.

All four are computed in m/s, because SPS carries an absolute constant and is
meaningless in normalised units.
"""
import numpy as np

from .config import PROBE_FRAC, SIGMA_GLOBAL


def rel_l2(p, y):
    """||pred - target||_2 / ||target||_2 over the scored elements."""
    return float(np.linalg.norm(p - y) / (np.linalg.norm(y) + 1e-12))


def tke_field(a):
    """0.5*(mean_t (u - mean_t u)^2 + mean_t (v - mean_t v)^2) over the window.

    Subtracting the temporal mean is what makes this blind to the mean field,
    and is the reason the mean and the fluctuation can be corrected separately.
    """
    f = a - a.mean(0, keepdims=True)
    return 0.5 * ((f[:, 0] ** 2).mean(0) + (f[:, 1] ** 2).mean(0))


def tke_err(p, y, m):
    return rel_l2(tke_field(p)[m], tke_field(y)[m])


def mvpe(p, y, m):
    """Time-averaged velocity profiles at probe columns behind the body, scored
    as a relative L2. Depends only on the window mean."""
    num = den = 0.0
    W = m.shape[1]
    for c in [int(W * f) for f in PROBE_FRAC]:
        col = m[:, c]
        if col.sum() < 4:
            continue
        pp = p[:, :, col, c].mean(0)
        yy = y[:, :, col, c].mean(0)
        num += float(((pp - yy) ** 2).sum())
        den += float((yy ** 2).sum())
    return float(np.sqrt(num / (den + 1e-12)))


def sps_window(lo, hi, y, m, e_dm, e_tke, e_mvpe):
    """Safe Prediction Score for one window, as published.

      inside   = (lower <= target <= upper)
      nil      = (upper - lower) / sigma_global
      pm_b     = e_b / (0.5 + e_b)                    for b in {dm, tke, mvpe}
      branch_b = mean over scored elements of (1 - pm_b) * exp(-nil), gated by
                 inside
      weighted = 0.5*branch_dm + 0.3*branch_tke + 0.2*branch_mvpe
      sps      = 100 * weighted

    (1 - pm_b) is a per-window scalar, so it factors out of the element mean and
    branch_b = (1 - pm_b) * Q with Q = mean[exp(-nil) * inside]. Q is computed
    once. Returns (sps, Q, coverage).
    """
    L, Hh, Y = lo[:, :, m], hi[:, :, m], y[:, :, m]
    inside = (Y >= L) & (Y <= Hh)
    nil = (Hh - L) / SIGMA_GLOBAL
    Q = float((np.exp(-nil) * inside).mean())
    pm = lambda e: e / (0.5 + e)
    weighted = Q * (0.5 * (1 - pm(e_dm))
                    + 0.3 * (1 - pm(e_tke))
                    + 0.2 * (1 - pm(e_mvpe)))
    return 100.0 * weighted, Q, float(inside.mean())
