"""The error budget.

Replace one statistic of a frozen forecast with the value measured from the
data, leave everything else alone, and re-score. The change is the error that
statistic owns, and no correction acting only on that statistic can recover
more. Costs a forward pass and no training.

Four statistics of increasing spatial resolution: the window mean, a single
scalar on the fluctuation, a gain per radial wavenumber band, and a gain per
grid point. The last matches the true energy everywhere and so zeroes the energy
error by construction; what it reports is the price in relative L2.
"""
import math

import numpy as np
import torch

from .config import u_of
from .metrics import mvpe, rel_l2, tke_err, tke_field
from .nets import DEV

ROWS = ["none (forecast as-is)", "window mean", "energy, one scalar",
        "energy, per wavenumber band", "energy, per grid point",
        "all (perfect forecast)"]


def _band_index(H, W, nbins):
    ky = np.fft.fftfreq(H)[:, None]
    kx = np.fft.rfftfreq(W)[None, :]
    kk = np.sqrt(kx ** 2 + ky ** 2)
    edges = np.exp(np.linspace(np.log(max(kk[kk > 0].min(), 1e-6)),
                               np.log(kk.max() * 1.01), nbins + 1))
    return np.clip(np.digitize(kk, edges) - 1, 0, nbins - 1)


def compute(bench, model, nbins=12, verbose=True):
    """Returns {row: (relL2, tke, mvpe)} averaged over the test windows."""
    H, W, K, HO = bench.H, bench.W, bench.K, bench.HO
    bi = _band_index(H, W, nbins)
    acc = {r: [] for r in ROWS}
    model.eval()

    for w in bench.test_w:
        x, y, m, key = bench.get_window(w)
        U = u_of(key[0])
        with torch.no_grad():
            p = model(torch.from_numpy(x.reshape(1, 2 * K, H, W)).to(DEV)).cpu().numpy()[0]
        p = (p.reshape(HO, 2, H, W) + x[-1][None]) * U
        y = y * U

        mp, my = p.mean(0), y.mean(0)
        fp, fy = p - mp, y - my
        Tp, Ty = tke_field(p), tke_field(y)

        F = np.fft.rfft2(fp, axes=(-2, -1))
        G = np.fft.rfft2(fy, axes=(-2, -1))
        Fb = F.copy()
        for c in range(2):
            ep = np.bincount(bi.ravel(), weights=(np.abs(F[:, c]) ** 2).sum(0).ravel(),
                             minlength=nbins)
            ey = np.bincount(bi.ravel(), weights=(np.abs(G[:, c]) ** 2).sum(0).ravel(),
                             minlength=nbins)
            Fb[:, c] *= np.sqrt(ey / (ep + 1e-30))[bi]

        scalar = math.sqrt(max(float((Tp[m] * Ty[m]).sum()
                                     / ((Tp[m] ** 2).sum() + 1e-30)), 0.0))
        variants = {
            "none (forecast as-is)": p,
            "window mean": my[None] + fp,
            "energy, one scalar": mp[None] + fp * scalar,
            "energy, per wavenumber band":
                mp[None] + np.fft.irfft2(Fb, s=(H, W), axes=(-2, -1)),
            "energy, per grid point":
                mp[None] + fp * np.sqrt(np.where(Tp > 1e-20, Ty / (Tp + 1e-30),
                                                 1.0))[None, None],
            "all (perfect forecast)": y,
        }
        for name, v in variants.items():
            acc[name].append((rel_l2(v[:, :, m], y[:, :, m]),
                              tke_err(v, y, m), mvpe(v, y, m)))

    out = {k: tuple(np.array(v).mean(0)) for k, v in acc.items()}
    if verbose:
        print(f"    {'statistic replaced':<30}{'Rel-L2':>9}{'TKE':>9}{'MVPE':>9}")
        for r in ROWS:
            a, b, c = out[r]
            print(f"    {r:<30}{a:>9.4f}{b:>9.4f}{c:>9.4f}")
    return out
