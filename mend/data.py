"""Loading the paired release, aligning the two clocks, and cutting windows.

Velocities are stored in each side's own units: the simulator is nondimensional
(U = L = 1), the measurement is in m/s. Models see both divided by the inflow
speed; scoring converts back, because SPS carries an absolute constant.
"""
import glob
import os
import re

import h5py
import numpy as np

from .config import BAD_REAL, Config, sim_per_real, u_of

_FNAME = re.compile(r"(\d+)_(\d+)\.h5$")


def _autodir(root, name):
    for pat in (f"{root}/**/{name}/**/*.h5", f"{root}/**/{name}/*.h5"):
        hits = glob.glob(pat, recursive=True)
        if hits:
            return os.path.dirname(sorted(hits)[0])
    return None


def _index(d):
    out = {}
    for fp in sorted(glob.glob(os.path.join(d, "*.h5"))):
        m = _FNAME.search(os.path.basename(fp))
        if m:
            out[(int(m.group(1)), int(m.group(2)))] = fp
    return out


class Bench:
    """Everything downstream code needs about the dataset, built once."""

    def __init__(self, cfg: Config, verbose=True):
        self.cfg = cfg
        self.H, self.W = cfg.H, cfg.W
        self.K, self.HO = cfg.K_IN, cfg.H_OUT

        sim_dir = _autodir(cfg.data_root, "train_sim")
        real_dir = _autodir(cfg.data_root, "train_real")
        if not (sim_dir and real_dir):
            raise FileNotFoundError(
                f"train_sim / train_real not found under {cfg.data_root!r}. "
                "Point --data-root or $REALPDE_ROOT at the RealPDEBench airfoil release.")
        self.sim_idx, self.real_idx = _index(sim_dir), _index(real_dir)

        paired = sorted((set(self.sim_idx) & set(self.real_idx)) - BAD_REAL)
        self.paired = paired[:cfg.N_CASES]

        # Split by Reynolds number, so no test condition is seen at any angle of
        # attack during training. Every reported number is transfer, not
        # interpolation.
        res = sorted({k[0] for k in self.paired})
        self.test_re, self.val_re = set(res[::5]), set(res[2::5])
        self.split = {"train": [], "val": [], "test": []}
        for k in self.paired:
            self.split[self._bucket(k)].append(k)

        self.data = {}
        for k in self.paired:
            real, vr = self._load_case(k, "real")
            sim, vs = self._load_case(k, "sim")
            self.data[k] = dict(real=real, sim=sim, mask=static_mask(vr, vs),
                                Re=k[0], aoa=k[1])

        self._sim_rs = {}
        self.val_w = self.windows(self.split["val"], cfg.STRIDE)
        self.test_w = self.windows(self.split["test"], cfg.STRIDE,
                                   cfg.EVAL_WIN * len(self.split["test"]))
        if verbose:
            print(f"{len(self.data)} cases | train {len(self.split['train'])} "
                  f"val {len(self.split['val'])} test {len(self.split['test'])} "
                  f"| test Re {sorted(self.test_re)}")
            print(f"windows: val {len(self.val_w)} test {len(self.test_w)}")

    def _bucket(self, k):
        if k[0] in self.test_re:
            return "test"
        return "val" if k[0] in self.val_re else "train"

    def _load_case(self, key, which):
        idx = self.sim_idx if which == "sim" else self.real_idx
        with h5py.File(idx[key], "r") as f:
            u = f["u"][:self.cfg.T_CAP].astype(np.float32)
            v = f["v"][:self.cfg.T_CAP].astype(np.float32)
        valid = (u != 0) | (v != 0)
        scale = 1.0 if which == "sim" else u_of(key[0])
        return np.stack([u, v], 1) / scale, valid

    # ---------------------------------------------------------------- windows
    def windows(self, keys, stride, cap=None, seed=1):
        out = [(k, t0) for k in keys
               for t0 in range(0, self.data[k]["real"].shape[0] - (self.K + self.HO) + 1,
                               stride)]
        if cap and len(out) > cap:
            r = np.random.RandomState(seed)
            out = [out[i] for i in r.choice(len(out), cap, False)]
        return out

    def get_window(self, w):
        k, t0 = w
        a = self.data[k]["real"]
        return (a[t0:t0 + self.K], a[t0 + self.K:t0 + self.K + self.HO],
                self.data[k]["mask"], k)

    # ------------------------------------------------------------- simulator
    def sim_resampled(self, key):
        """Simulator interpolated onto the camera's frame grid. Without this the
        two sides index different physical durations, by 0.7x to 5.0x."""
        if key in self._sim_rs:
            return self._sim_rs[key]
        s = self.data[key]["sim"]
        step = sim_per_real(key[0])
        n = int((s.shape[0] - 1) / step)
        t = np.arange(n) * step
        i0 = np.floor(t).astype(int)
        f = (t - i0).astype(np.float32)[:, None, None, None]
        nxt = np.minimum(i0 + 1, s.shape[0] - 1)
        self._sim_rs[key] = (s[i0] * (1 - f) + s[nxt] * f).astype(np.float32)
        return self._sim_rs[key]

    def sim_align(self, x, key):
        """Pick the resampled simulator frame closest to the last observed one
        and read the next HO frames off from there."""
        s = self.sim_resampled(key)
        m = self.data[key]["mask"]
        n = s.shape[0] - self.HO - 1
        if n <= 1:
            return np.repeat(x[-1][None], self.HO, 0)
        j = int(np.argmin(((s[:n][:, :, m] - x[-1][:, m][None]) ** 2).sum((1, 2))))
        return s[j + 1:j + 1 + self.HO]


def static_mask(vr, vs):
    """Points observed in every frame of both sides, eroded by one cell so the
    scored set does not touch the edge of the illuminated sheet."""
    m = vr.all(0) & vs.all(0)
    e = m.copy()
    e[1:-1, 1:-1] = (m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1]
                     & m[1:-1, :-2] & m[1:-1, 2:])
    return e
