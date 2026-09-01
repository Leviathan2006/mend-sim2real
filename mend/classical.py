"""The three classical baselines.

Each corrects the simulator without learning a forecaster, so each is
deterministic given the data. The seed moves only the ensemble draw used to form
intervals: Rel-L2/TKE/MVPE come out with near-zero spread, SPS with a little.

  Kennedy-O'Hagan  y(x) = rho*m(x,theta) + delta(x) + eps   [K&OH 2001, sec. 4]
                   delta expanded on a POD basis            [Higdon et al. 2008]
  quantile map     simulator marginal transported onto the observed one
                                                            [Cannon et al. 2015]
  DMD              delay-embedded, fitted to the observed window alone
                                                            [Schmid 2010]
"""
import numpy as np

from .evaluate import evaluate


def det_ens(bench, p, sig, rng):
    """Deterministic forecast dressed with a per-point spread fitted on
    validation. Every deterministic method is dressed the same way, so interval
    construction is not a confound in the comparison."""
    cfg = bench.cfg
    noise = rng.randn(bench.HO, cfg.ENS, 2, bench.H, bench.W).astype(np.float32)
    return p[:, None] + sig[:, None] * noise


def fit_sigma(bench, mu_fn):
    se = np.zeros((bench.HO, 2, bench.H, bench.W))
    n = 0
    for w in bench.val_w:
        x, y, _, key = bench.get_window(w)
        se += (mu_fn(x, key) - y) ** 2
        n += 1
    return np.maximum(np.sqrt(se / max(n, 1)).astype(np.float32), 1e-5)


def run_classical(bench, results, name, mu_fn):
    sig = fit_sigma(bench, mu_fn)
    print(f"  == {name} ==")
    for s in bench.cfg.SEEDS:
        rng = np.random.RandomState(1000 + s)
        evaluate(bench, name, lambda x, k: det_ens(bench, mu_fn(x, k), sig, rng),
                 results=results, seed=s, label=f"{name} (seed {s})")


# ------------------------------------------------------------- quantile map
def fit_quantile_map(bench, nq=200, verbose=True):
    qs = np.linspace(0.001, 0.999, nq)
    qm = []
    for c in range(2):
        a = np.concatenate([bench.data[k]["sim"][::11, c][:, bench.data[k]["mask"]].ravel()
                            for k in bench.split["train"]])
        b = np.concatenate([bench.data[k]["real"][::11, c][:, bench.data[k]["mask"]].ravel()
                            for k in bench.split["train"]])
        qm.append((np.quantile(a, qs), np.quantile(b, qs)))
        if verbose:
            print(f"     ch{c}: sim {qm[-1][0][10]:+.3f}..{qm[-1][0][-10]:+.3f}"
                  f" | real {qm[-1][1][10]:+.3f}..{qm[-1][1][-10]:+.3f}")

    def mu_qm(x, key):
        p = bench.sim_align(x, key)
        o = np.empty_like(p)
        for c in range(2):
            o[:, c] = np.interp(p[:, c], qm[c][0], qm[c][1])
        return o
    return mu_qm


# ----------------------------------------------------- Kennedy-O'Hagan
def fit_koh(bench, n_pod=24, n_sim=32, verbose=True):
    """The discrepancy is a field, so it is expanded on a POD basis rather than
    modelled per point. The simulator state is projected on its own basis before
    the ridge, which keeps the normal equations at 33x33 instead of 16117^2."""
    H, W, HO = bench.H, bench.W, bench.HO
    koh_mask = np.ones((H, W), bool)
    for k in bench.paired:
        koh_mask &= bench.data[k]["mask"]
    if verbose:
        print(f"     common domain {int(koh_mask.sum())} px "
              f"({koh_mask.mean() * 100:.1f}% of grid)")

    resid, feat = [], []
    for k in bench.split["train"]:
        s = bench.sim_resampled(k)
        r = bench.data[k]["real"]
        n = min(len(s), len(r))
        for t in range(0, n, 9):
            resid.append((r[t] - s[t])[:, koh_mask])
            feat.append(s[t][:, koh_mask])
    Rm = np.stack(resid).reshape(len(resid), -1).astype(np.float64)
    Sm = np.stack(feat).reshape(len(feat), -1).astype(np.float64)

    rho = float((Rm * Sm).sum() / ((Sm ** 2).sum() + 1e-12) + 1.0)
    Dm = Rm - (rho - 1.0) * Sm
    dmean = Dm.mean(0)
    Dc = Dm - dmean
    _, sv, Vt = np.linalg.svd(Dc, full_matrices=False)
    Vt = Vt[:n_pod]
    smean = Sm.mean(0)
    Sc = Sm - smean
    _, ss, Vs = np.linalg.svd(Sc, full_matrices=False)
    Vs = Vs[:n_sim]

    Xf = np.concatenate([Sc @ Vs.T, np.ones((len(Sc), 1))], 1)
    G = Xf.T @ Xf
    lam = 1e-3 * float(np.trace(G)) / G.shape[0]
    Wko = np.linalg.solve(G + lam * np.eye(G.shape[0]), Xf.T @ (Dc @ Vt.T))
    if verbose:
        print(f"     rho = {rho:.4f} | {n_pod} discrepancy modes = "
              f"{float((sv[:n_pod] ** 2).sum() / (sv ** 2).sum()) * 100:.1f}% | "
              f"{n_sim} simulator modes = "
              f"{float((ss[:n_sim] ** 2).sum() / (ss ** 2).sum()) * 100:.1f}%")

    def mu_koh(x, key):
        s = bench.sim_align(x, key)
        flat = s[:, :, koh_mask].reshape(HO, -1).astype(np.float64) - smean
        co = np.concatenate([flat @ Vs.T, np.ones((HO, 1))], 1) @ Wko
        o = rho * s
        o[:, :, koh_mask] += (co @ Vt + dmean).reshape(HO, 2, -1).astype(np.float32)
        return o
    return mu_koh


# ------------------------------------------------------------------- DMD
def make_dmd(bench, delay=3, rank=8):
    """Delay embedding is what makes DMD usable on a short, noisy record.
    Growing modes are reflected onto the unit circle so a 20-step forecast
    cannot blow up."""
    K, HO = bench.K, bench.HO

    def mu_dmd(x, key):
        m = bench.data[key]["mask"]
        Xf = x[:, :, m].reshape(K, -1).T
        d = Xf.shape[0]
        Hk = np.concatenate([Xf[:, i:K - delay + i + 1] for i in range(delay)], 0)
        A, B = Hk[:, :-1], Hk[:, 1:]
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        r = min(rank, int((s > 1e-8 * s[0]).sum()))
        if r < 2:
            return np.repeat(x[-1][None], HO, 0)
        U, s, Vt = U[:, :r], s[:r], Vt[:r]
        ev, ew = np.linalg.eig(U.T @ B @ Vt.T @ np.diag(1.0 / s))
        ev = np.where(np.abs(ev) > 1.0, ev / np.abs(ev), ev)
        z = np.linalg.pinv(U @ ew) @ Hk[:, -1]
        out = np.repeat(x[-1][None], HO, 0).copy()
        for h in range(HO):
            z = ev * z
            out[h][:, m] = np.real(U @ ew @ z)[:d].reshape(2, -1)
        return out
    return mu_dmd


def run_all(bench, results):
    run_classical(bench, results, "quantile map", fit_quantile_map(bench))
    run_classical(bench, results, "Kennedy-O'Hagan", fit_koh(bench))
    run_classical(bench, results, "DMD", make_dmd(bench))
