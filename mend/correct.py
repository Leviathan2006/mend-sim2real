"""MEND: correcting a frozen forecast's mean and energy.

A forecast over a window splits as p(x,t) = m(x) + f(x,t). The profile metric
sees only m, the energy metric sees only f, and relative L2 sees both. The error
budget (see budget.py) says the mean carries a quarter of the pointwise error
and all of the profile error, and that per-point energy is the only granularity
at which the energy term is worth correcting.

So two auxiliary networks predict those two fields from the observed frames, and
their outputs adjust a frozen forecast. The forecaster's weights are never
touched, which is why the correction composes with any of the transfer methods.
"""
import time

import numpy as np
import torch
import torch.nn.functional as Fn

from .config import EPS
from .evaluate import evaluate, score_windows
from .metrics import tke_field
from .nets import AuxHead, DEV, loader, rel_loss, set_seed


def build_aux(bench, keys, source):
    """Windows as (input frames, window mean as a residual, log energy map)."""
    K, HO, H, W = bench.K, bench.HO, bench.H, bench.W
    X, YM, YT = [], [], []
    for k in keys:
        a = bench.data[k]["real"] if source == "real" else bench.sim_resampled(k)
        for t0 in range(0, a.shape[0] - (K + HO) + 1, bench.cfg.STRIDE):
            x = a[t0:t0 + K]
            fut = a[t0 + K:t0 + K + HO]
            X.append(x.reshape(2 * K, H, W))
            YM.append((fut.mean(0) - x[-1]).astype(np.float32))
            YT.append(np.log(tke_field(fut) + EPS).astype(np.float32)[None])
    return (torch.from_numpy(np.stack(X)), torch.from_numpy(np.stack(YM)),
            torch.from_numpy(np.stack(YT)))


def train_aux(tcfg, mcfg, model, dl_tr, dl_va, which, tag):
    """The mean head trains under a relative L2 loss, the energy head under a
    squared loss on the log map. The logarithm keeps the dynamic range usable:
    wake energy exceeds free-stream energy by more than an order of magnitude."""
    opt = torch.optim.AdamW(model.parameters(), lr=mcfg.AUX_LR, weight_decay=tcfg.WD)
    lossf = rel_loss if which == "mean" else Fn.mse_loss
    best, bad, best_state, e = 1e18, 0, None, 0
    for e in range(mcfg.AUX_EP):
        model.train()
        for xb, ym, yt in dl_tr:
            target = (ym if which == "mean" else yt).to(DEV)
            loss = lossf(model(xb.to(DEV)), target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        vl, n = 0.0, 0
        with torch.no_grad():
            for xb, ym, yt in dl_va:
                target = (ym if which == "mean" else yt).to(DEV)
                vl += lossf(model(xb.to(DEV)), target).item()
                n += 1
        vl /= max(n, 1)
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= tcfg.PATIENCE:
                break
    if best_state:
        model.load_state_dict(best_state)
    print(f"     {tag}: best val {best:.5f} after {e + 1} epochs")
    return model


def mend_correct(p, x, mhat, that, a, b, gclip):
    """Frozen point forecast -> corrected point forecast.

    a blends the predicted window mean into the forecast's own; b scales the
    fluctuation towards the predicted energy. Either at zero disables that term
    exactly, so each coefficient doubles as its own ablation.
    """
    mp = p.mean(0)
    fp = p - mp
    m_corr = (1 - a) * mp + a * (mhat + x[-1])
    if b > 0:
        Tp = 0.5 * ((fp[:, 0] ** 2).mean(0) + (fp[:, 1] ** 2).mean(0))
        g = np.clip(np.sqrt(np.clip(that, 0, None) / (Tp + EPS)), *gclip)
        fp = fp * (1.0 + b * (g - 1.0))[None, None]
    return m_corr[None] + fp


class Mend:
    """A trained pair of auxiliary heads plus the coefficient search."""

    def __init__(self, bench, tcfg, mcfg, aux_mean, aux_energy, rng_seed=12345):
        self.bench, self.tcfg, self.mcfg = bench, tcfg, mcfg
        self.aux_m, self.aux_t = aux_mean, aux_energy
        self.rng = np.random.RandomState(rng_seed)

    def _aux(self, xb):
        with torch.no_grad():
            mhat = self.aux_m(xb).cpu().numpy()[0]
            that = np.exp(self.aux_t(xb).cpu().numpy()[0, 0]) - EPS
        return mhat, that

    def _point(self, model, x, prob):
        b = self.bench
        xb = torch.from_numpy(x.reshape(1, 2 * b.K, b.H, b.W)).to(DEV)
        with torch.no_grad():
            if prob:
                p = model.sample(xb, b.cfg.ENS).mean(0).cpu().numpy()[0]
            else:
                p = model(xb).cpu().numpy()[0]
        mhat, that = self._aux(xb)
        return p.reshape(b.HO, 2, b.H, b.W) + x[-1][None], mhat, that

    def predictor(self, model, sigma, a, b_, prob=False):
        """The corrected point forecast dressed with the same per-point spread
        the deterministic baselines use, so intervals are identical across the
        whole table."""
        bn = self.bench
        sg = sigma.reshape(bn.HO, 2, bn.H, bn.W)
        ens = bn.cfg.ENS

        def predict(x, key):
            p, mhat, that = self._point(model, x, prob)
            pc = mend_correct(p, x, mhat, that, a, b_, self.mcfg.GCLIP)
            noise = self.rng.randn(bn.HO, ens, 2, bn.H, bn.W).astype(np.float32)
            return pc[:, None] + sg[:, None] * noise
        return predict

    def sigma_for(self, model, a, b_, prob):
        """Per-point spread of the corrected forecast, fitted on validation.
        Fitting on validation and not on test is what keeps the interval honest."""
        bn = self.bench
        se = np.zeros((bn.HO, 2, bn.H, bn.W))
        n = 0
        for w in bn.val_w:
            x, y, _, _ = bn.get_window(w)
            p, mhat, that = self._point(model, x, prob)
            pc = mend_correct(p, x, mhat, that, a, b_, self.mcfg.GCLIP)
            se += (pc - y) ** 2
            n += 1
        return np.maximum(np.sqrt(se / max(n, 1)).astype(np.float32), 1e-5)

    def select(self, model, prob, verbose=True):
        """Grid search over (a, b) on validation only. Returns the chosen pair,
        its spread, and the full sweep for the sensitivity figure."""
        rows, t0 = [], time.time()
        for a in self.mcfg.A_GRID:
            for b_ in self.mcfg.B_GRID:
                sg = self.sigma_for(model, a, b_, prob)
                pr = self.predictor(model, sg, a, b_, prob)
                s = score_windows(self.bench, pr,
                                  self.bench.val_w[:self.mcfg.FIT_WIN], 0.9)[:, 3].mean()
                rows.append(((a, b_), float(s), sg))
        (a, b_), best, sg = max(rows, key=lambda r: r[1])
        if verbose:
            print(f"   a={a} b={b_} (val SPS {best:.2f}, {time.time() - t0:.0f}s)")
        return (a, b_), sg, [(k, v) for k, v, _ in rows]


def train_heads(bench, tcfg, mcfg, seed):
    """Pretrain both heads on simulation, fine-tune on measurements."""
    K, H, W = bench.K, bench.H, bench.W
    cin = 2 * K + 2
    tr = build_aux(bench, bench.split["train"], "real")
    va = build_aux(bench, bench.split["val"], "real")
    sa = build_aux(bench, bench.split["train"] + bench.split["val"], "sim")
    c = int(len(sa[0]) * 0.9)
    dl_tr, dl_va = loader(tcfg, tr, True), loader(tcfg, va, False)
    dl_str = loader(tcfg, tuple(t[:c] for t in sa), True)
    dl_sva = loader(tcfg, tuple(t[c:] for t in sa), False)
    print(f"aux windows: real {len(tr[0])} train / {len(va[0])} val "
          f"| sim {len(sa[0])}")

    set_seed(seed)
    am = AuxHead(tcfg, cin, 2).to(DEV)
    train_aux(tcfg, mcfg, am, dl_str, dl_sva, "mean", "aux-mean pretrain")
    train_aux(tcfg, mcfg, am, dl_tr, dl_va, "mean", "aux-mean finetune")

    set_seed(seed)
    at = AuxHead(tcfg, cin, 1).to(DEV)
    train_aux(tcfg, mcfg, at, dl_str, dl_sva, "tke", "aux-energy pretrain")
    train_aux(tcfg, mcfg, at, dl_tr, dl_va, "tke", "aux-energy finetune")

    am.eval()
    at.eval()
    return am, at


def run_all(bench, results, tcfg, mcfg, keep, knobs=None, sweeps=None,
            ablations=True):
    """Wrap each kept base model on each seed, selecting coefficients on
    validation and scoring through the same evaluate() as every baseline."""
    knobs = {} if knobs is None else knobs
    sweeps = {} if sweeps is None else sweeps
    for seed in bench.cfg.SEEDS:
        print(f"\n{'=' * 70}\nMEND, seed {seed}\n{'=' * 70}")
        am, at = train_heads(bench, tcfg, mcfg, seed)
        m = Mend(bench, tcfg, mcfg, am, at)

        for base in mcfg.WRAP:
            if base not in keep or seed not in keep[base]:
                continue
            model = keep[base][seed].to(DEV)
            model.eval()
            prob = (base == "PNO")
            print(f"   {base}:")
            (a, b_), sg, sweep = m.select(model, prob)
            knobs[(base, seed)] = (a, b_)
            sweeps[(base, seed)] = sweep
            evaluate(bench, f"{base} + MEND", m.predictor(model, sg, a, b_, prob),
                     results=results, seed=seed,
                     label=f"{base} + MEND (seed {seed})")

            if ablations:
                for tag, aa, bb in ((f"{base} + MEND (mean only)", a, 0.0),
                                    (f"{base} + MEND (energy only)", 0.0, b_)):
                    if (aa, bb) == (a, b_):
                        continue
                    sga = m.sigma_for(model, aa, bb, prob)
                    evaluate(bench, tag, m.predictor(model, sga, aa, bb, prob),
                             results=results, seed=seed, verbose=False)
            keep[base][seed] = model.cpu()
        if DEV == "cuda":
            torch.cuda.empty_cache()
    return knobs, sweeps
