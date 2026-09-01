"""One scorer for every method in the paper.

A method is a callable predict(x, key) -> (HO, ENS, 2, H, W) in normalised
units. Everything else -- unit conversion, interval level, aggregation -- happens
here, so no row of the results table can differ because of how it was scored.
"""
import json
import time

import numpy as np

from .config import u_of
from .metrics import mvpe, rel_l2, sps_window, tke_err

# The interval level is fitted per method on validation. SPS rewards intervals
# that are tight and still cover, so the best level is a property of a forecast;
# giving every method the same grid is what keeps the comparison fair.
Q_GRID = np.array([0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98])


def _bounds(S, q):
    return np.quantile(S, (1 - q) / 2, 1), np.quantile(S, (1 + q) / 2, 1)


def score_windows(bench, predict, wins, q):
    out = []
    for w in wins:
        x, y, m, key = bench.get_window(w)
        U = u_of(key[0])
        S = predict(x, key) * U           # -> m/s
        yp = y * U
        mu = S.mean(1)
        e_dm = rel_l2(mu[:, :, m], yp[:, :, m])
        e_tk = tke_err(mu, yp, m)
        e_mv = mvpe(mu, yp, m)
        lo, hi = _bounds(S, q)
        sps, _, cov = sps_window(lo, hi, yp, m, e_dm, e_tk, e_mv)
        out.append((e_dm, e_tk, e_mv, sps, cov))
    return np.array(out)


def fit_q(bench, predict, nwin=24):
    """Interval level that maximises SPS on validation."""
    sub = bench.val_w[:nwin]
    best_q, best_s = float(Q_GRID[0]), -1.0
    for q in Q_GRID:
        s = score_windows(bench, predict, sub, q)[:, 3].mean()
        if s > best_s:
            best_s, best_q = float(s), float(q)
    return best_q, best_s


class Results:
    """Per-method, per-seed records plus the aggregation used in the tables."""

    def __init__(self):
        self.r = {}

    def record(self, name, seed, res):
        self.r.setdefault(name, {})[seed] = res

    def __contains__(self, name):
        return name in self.r

    def agg(self, name):
        rows = list(self.r[name].values())
        f = lambda k: (float(np.mean([x[k] for x in rows])),
                       float(np.std([x[k] for x in rows])))
        return {k: f(k) for k in ("relL2", "tke", "mvpe", "sps", "coverage")}, len(rows)

    def table(self, order=None):
        order = order or list(self.r)
        print(f"\n{'method':<28}{'Rel-L2':>16}{'TKE':>16}{'MVPE':>16}{'SPS':>16}{'n':>4}")
        print("-" * 96)
        for n in order:
            if n not in self.r:
                continue
            a, k = self.agg(n)
            cell = lambda t: f"{t[0]:.4f}+-{t[1]:.4f}"
            print(f"{n:<28}{cell(a['relL2']):>16}{cell(a['tke']):>16}"
                  f"{cell(a['mvpe']):>16}{a['sps'][0]:>10.2f}+-{a['sps'][1]:<4.2f}{k:>4}")
        print("\nRel-L2 / TKE / MVPE: lower is better. SPS: higher is better (0-100).")
        print("mean +- standard deviation over seeds.")

    def save(self, path, extra=None):
        payload = {"results": {k: {str(s): v for s, v in d.items()}
                               for k, d in self.r.items()}}
        if extra:
            payload.update(extra)
        with open(path, "w") as f:
            json.dump(payload, f, indent=1, default=float)
        print("saved", path)

    @classmethod
    def load(cls, path):
        o = cls()
        with open(path) as f:
            blob = json.load(f)
        for name, per_seed in blob["results"].items():
            for s, v in per_seed.items():
                o.record(name, int(s), v)
        return o


def evaluate(bench, name, predict, results=None, seed=None, verbose=True,
             label=None):
    q, vs = fit_q(bench, predict)
    t0 = time.time()
    A = score_windows(bench, predict, bench.test_w, q)
    res = dict(relL2=float(A[:, 0].mean()), tke=float(A[:, 1].mean()),
               mvpe=float(A[:, 2].mean()), sps=float(A[:, 3].mean()),
               coverage=float(A[:, 4].mean()), q=q, val_sps=vs,
               seconds=time.time() - t0)
    if verbose:
        print(f"    {label or name:<30} relL2 {res['relL2']:.4f}  TKE {res['tke']:.4f}  "
              f"MVPE {res['mvpe']:.4f}  SPS {res['sps']:6.2f}   "
              f"(q={q:.2f}, cov {res['coverage']:.3f}, {res['seconds']:.0f}s)")
    if results is not None and seed is not None:
        results.record(name, seed, res)
    return res
