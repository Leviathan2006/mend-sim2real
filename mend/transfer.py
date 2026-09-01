"""The five learned sim-to-real methods.

All five start from the same simulator pretraining for a given seed. What
differs is only the constraint on the fine-tuning step:

  direct fine-tune  no constraint
  L2-SP             Omega(w) = (a/2)||w_S - w_S^0||^2 + (b/2)||w_Sbar||^2
                    Li, Grandvalet & Davoine, ICML 2018, eq. (3). The
                    architecture is unchanged between source and target, so S is
                    every parameter and the second term is empty.
  EWC               L(th) = L_B(th) + sum_i (lam/2) F_i (th_i - th*_i)^2
                    Kirkpatrick et al., PNAS 2017, eq. (3)
  PhysGuard         per-layer Gram-matrix gradient projection out of the leading
                    Fisher subspace estimated on simulation
  PNO               energy-score training with a reparametrised head
"""
import numpy as np
import torch

from .evaluate import evaluate
from .nets import DEV, FNO, energy_score, loader, rel_loss, set_seed, train


def build(bench, keys, source):
    """Windows as (input frames, residual from the last observed frame).

    The residual target matters: without it every net spends its capacity
    relearning the mean flow before it models anything that varies.
    """
    K, HO, H, W = bench.K, bench.HO, bench.H, bench.W
    X, Y = [], []
    for k in keys:
        a = bench.data[k]["real"] if source == "real" else bench.sim_resampled(k)
        for t0 in range(0, a.shape[0] - (K + HO) + 1, bench.cfg.STRIDE):
            X.append(a[t0:t0 + K].reshape(2 * K, H, W))
            Y.append((a[t0 + K:t0 + K + HO] - a[t0 + K - 1][None]).reshape(2 * HO, H, W))
    return torch.from_numpy(np.stack(X)), torch.from_numpy(np.stack(Y))


def to_pred(bench, model, sampler):
    """Wrap a model as evaluate()'s predict(x, key)."""
    K, HO, H, W = bench.K, bench.HO, bench.H, bench.W
    ens = bench.cfg.ENS

    def predict(x, key):
        xb = torch.from_numpy(x.reshape(1, 2 * K, H, W)).to(DEV)
        with torch.no_grad():
            S = sampler(model, xb)
        S = S.squeeze(1).reshape(ens, HO, 2, H, W).cpu().numpy()
        return np.transpose(S, (1, 0, 2, 3, 4)) + x[-1][None, None]
    return predict


def mos_sampler(bench, sig):
    """Model output statistics: a per-point spread fitted on validation. The
    deterministic methods and MEND all use this, so intervals are built the same
    way everywhere."""
    cout = 2 * bench.HO
    sg = torch.from_numpy(sig.reshape(1, cout, bench.H, bench.W)).to(DEV)
    ens = bench.cfg.ENS

    def f(m, xb):
        p = m(xb)
        return p[None] + sg[None] * torch.randn(ens, *p.shape, device=p.device)
    return f


def fit_sigma_t(bench, model):
    K, HO, H, W = bench.K, bench.HO, bench.H, bench.W
    se = np.zeros((2 * HO, H, W))
    n = 0
    with torch.no_grad():
        for w in bench.val_w:
            x, y, _, _ = bench.get_window(w)
            p = model(torch.from_numpy(x.reshape(1, 2 * K, H, W)).to(DEV)).cpu().numpy()[0]
            se += (p - (y - x[-1][None]).reshape(2 * HO, H, W)) ** 2
            n += 1
    return np.maximum(np.sqrt(se / max(n, 1)).astype(np.float32), 1e-5)


def score_det(bench, results, name, seed, model):
    evaluate(bench, name,
             to_pred(bench, model, mos_sampler(bench, fit_sigma_t(bench, model))),
             results=results, seed=seed, label=f"{name} (seed {seed})")


def run_all(bench, results, tcfg, keep=None):
    """Runs the five learned methods over every seed.

    keep, if given, collects the frozen models MEND later wraps.
    """
    K, HO, H, W = bench.K, bench.HO, bench.H, bench.W
    cin, cout = 2 * K + 2, 2 * HO

    tr_r = build(bench, bench.split["train"], "real")
    va_r = build(bench, bench.split["val"], "real")
    tr_s = build(bench, bench.split["train"] + bench.split["val"], "sim")
    dl_tr, dl_va = loader(tcfg, tr_r, True), loader(tcfg, va_r, False)
    c = int(len(tr_s[0]) * 0.9)
    dl_str = loader(tcfg, (tr_s[0][:c], tr_s[1][:c]), True)
    dl_sva = loader(tcfg, (tr_s[0][c:], tr_s[1][c:]), False)
    print(f"windows: real train {len(tr_r[0])} val {len(va_r[0])} "
          f"| simulator {len(tr_s[0])}")

    step_det = lambda m, x, y: rel_loss(m(x), y)
    mk = lambda prob=False: FNO(tcfg, cin, cout, prob=prob).to(DEV)

    for seed in bench.cfg.SEEDS:
        print(f"\n{'=' * 70}\nseed {seed}\n{'=' * 70}")
        set_seed(seed)

        m0 = mk()
        train(tcfg, m0, dl_str, dl_sva, step_det, tcfg.PRE_EP, "simulator pretraining")
        pre = {k: v.detach().clone() for k, v in m0.state_dict().items()}

        # ---- direct fine-tune
        m = mk(); m.load_state_dict(pre)
        train(tcfg, m, dl_tr, dl_va, step_det, tcfg.FT_EP, "direct fine-tune")
        score_det(bench, results, "direct fine-tune", seed, m)
        if keep is not None:
            keep.setdefault("direct fine-tune", {})[seed] = m.cpu()
            m.to(DEV)

        # ---- L2-SP
        m = mk(); m.load_state_dict(pre)
        anchor = {n: p.detach().clone() for n, p in m.named_parameters()}

        def step_l2sp(mm, x, y):
            pen = sum(((p - anchor[n]) ** 2).sum() for n, p in mm.named_parameters())
            return rel_loss(mm(x), y) + 0.5 * tcfg.L2SP_ALPHA * pen
        train(tcfg, m, dl_tr, dl_va, step_l2sp, tcfg.FT_EP, "L2-SP")
        score_det(bench, results, "L2-SP", seed, m)

        # ---- EWC
        mF = mk(); mF.load_state_dict(pre)
        fisher = {n: torch.zeros_like(p) for n, p in mF.named_parameters()}
        mF.eval()
        nb = 0
        for i, (xb, yb) in enumerate(dl_str):
            if i >= 30:
                break
            mF.zero_grad()
            step_det(mF, xb.to(DEV), yb.to(DEV)).backward()
            for n, p in mF.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2
            nb += 1
        for n in fisher:
            fisher[n] /= max(nb, 1)
        print(f"     mean diagonal Fisher "
              f"{np.mean([v.mean().item() for v in fisher.values()]):.3e}")
        m = mk(); m.load_state_dict(pre)
        anchor_e = {n: p.detach().clone() for n, p in m.named_parameters()}

        def step_ewc(mm, x, y):
            pen = sum((fisher[n] * (p - anchor_e[n]) ** 2).sum()
                      for n, p in mm.named_parameters())
            return rel_loss(mm(x), y) + 0.5 * tcfg.EWC_LAMBDA * pen
        train(tcfg, m, dl_tr, dl_va, step_ewc, tcfg.FT_EP, "EWC")
        score_det(bench, results, "EWC", seed, m)
        del mF, fisher
        if DEV == "cuda":
            torch.cuda.empty_cache()

        # ---- PhysGuard
        u_proj = _physguard_basis(tcfg, m0, tr_s, step_det, seed)
        m = mk(); m.load_state_dict(pre)

        def pg_hook(mm):
            for n, p in mm.named_parameters():
                if p.grad is None:
                    continue
                g = p.grad.detach().reshape(-1)
                U = u_proj[n]
                p.grad.copy_((g - tcfg.PG_ALPHA * (U @ (U.T @ g))).reshape(p.shape))
        train(tcfg, m, dl_tr, dl_va, step_det, tcfg.FT_EP, "PhysGuard",
              post_grad=pg_hook)
        score_det(bench, results, "PhysGuard", seed, m)
        if keep is not None:
            keep.setdefault("PhysGuard", {})[seed] = m.cpu()
        del u_proj
        if DEV == "cuda":
            torch.cuda.empty_cache()

        # ---- PNO
        set_seed(seed)
        mp = mk(prob=True)
        step_pno = lambda mm, x, y: energy_score(mm.sample(x, tcfg.MT), y)
        train(tcfg, mp, dl_str, dl_sva, step_pno, tcfg.PRE_EP, "PNO pretraining")
        train(tcfg, mp, dl_tr, dl_va, step_pno, tcfg.FT_EP, "PNO")
        evaluate(bench, "PNO",
                 to_pred(bench, mp, lambda mm, xb: mm.sample(xb, bench.cfg.ENS)),
                 results=results, seed=seed, label=f"PNO (seed {seed})")
        if keep is not None:
            keep.setdefault("PNO", {})[seed] = mp.cpu()
        del m0, mp
        if DEV == "cuda":
            torch.cuda.empty_cache()


def _physguard_basis(tcfg, m0, tr_s, step_det, seed):
    """Per-layer orthonormal basis of the leading gradient subspace on
    simulation. k is the smallest rank capturing PG_TAU of the Gram spectrum."""
    n_sim = len(tr_s[0])
    n = max(8, min(int(n_sim * tcfg.PG_FRAC), tcfg.PG_MAXK))
    sel = np.random.RandomState(seed).choice(n_sim, n, replace=False)
    m0.eval()
    names = [nm for nm, _ in m0.named_parameters()]
    sizes = {nm: p.numel() for nm, p in m0.named_parameters()}
    G = {nm: torch.zeros(n, sizes[nm]) for nm in names}
    for i, j in enumerate(sel):
        m0.zero_grad()
        step_det(m0, tr_s[0][j:j + 1].to(DEV), tr_s[1][j:j + 1].to(DEV)).backward()
        for nm, p in m0.named_parameters():
            if p.grad is not None:
                G[nm][i] = p.grad.detach().reshape(-1).cpu()
    m0.zero_grad()

    out, tot_k = {}, 0
    for nm in names:
        Gm = G[nm].to(DEV)
        Kg = Gm @ Gm.T
        V, Lam, _ = torch.linalg.svd(Kg)
        frac = torch.cumsum(Lam, 0) / (Lam.sum() + 1e-30)
        k = max(1, min(int(torch.searchsorted(frac, tcfg.PG_TAU).item()) + 1,
                       n, tcfg.PG_MAXK))
        out[nm] = (Gm.T @ V[:, :k]) / Lam[:k].clamp_min(1e-20).sqrt()[None]
        tot_k += k
        del Gm, Kg
    G.clear()
    n0 = names[0]
    chk = (out[n0].T @ out[n0] - torch.eye(out[n0].shape[1], device=DEV)).abs().max()
    print(f"     PhysGuard: {n} samples, ranks sum {tot_k}, "
          f"|U^T U - I| {chk.item():.1e}")
    return out
