"""The shared FNO backbone, the auxiliary head, losses, and the training loop.

Every learned method in the paper uses this backbone and one pretraining run per
seed, so the only thing that differs between them is how the measurements are
allowed to move the weights.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class SpectralConv2d(nn.Module):
    def __init__(self, ci, co, m1, m2):
        super().__init__()
        self.m1, self.m2 = m1, m2
        sc = 1.0 / (ci * co)
        self.w1 = nn.Parameter(sc * torch.randn(ci, co, m1, m2, 2))
        self.w2 = nn.Parameter(sc * torch.randn(ci, co, m1, m2, 2))

    def mul(self, a, w):
        return torch.einsum("bixy,ioxy->boxy", a, torch.view_as_complex(w.contiguous()))

    def forward(self, x):
        B, _, H, W = x.shape
        xf = torch.fft.rfft2(x.float())
        o = torch.zeros(B, self.w1.shape[1], H, W // 2 + 1,
                        dtype=torch.cfloat, device=x.device)
        m1, m2 = min(self.m1, H // 2), min(self.m2, W // 2 + 1)
        o[:, :, :m1, :m2] = self.mul(xf[:, :, :m1, :m2], self.w1[:, :, :m1, :m2])
        o[:, :, -m1:, :m2] = self.mul(xf[:, :, -m1:, :m2], self.w2[:, :, :m1, :m2])
        return torch.fft.irfft2(o, s=(H, W))


def _coord_cat(x):
    B, _, H, W = x.shape
    gy, gx = torch.meshgrid(torch.linspace(0, 1, H, device=x.device),
                            torch.linspace(0, 1, W, device=x.device), indexing="ij")
    return torch.cat([x, torch.stack([gx, gy], 0)[None].expand(B, -1, -1, -1)], 1)


class FNO(nn.Module):
    def __init__(self, tcfg, cin, cout, prob=False):
        super().__init__()
        self.prob = prob
        self.lift = nn.Conv2d(cin, tcfg.WIDTH, 1)
        self.sp = nn.ModuleList([SpectralConv2d(tcfg.WIDTH, tcfg.WIDTH,
                                                tcfg.MODES_Y, tcfg.MODES_X)
                                 for _ in range(tcfg.LAYERS)])
        self.w = nn.ModuleList([nn.Conv2d(tcfg.WIDTH, tcfg.WIDTH, 1)
                                for _ in range(tcfg.LAYERS)])
        self.p1 = nn.Conv2d(tcfg.WIDTH, tcfg.LIFT, 1)
        self.mu = nn.Conv2d(tcfg.LIFT, cout, 1)
        self.sg = nn.Conv2d(tcfg.LIFT, cout, 1) if prob else None

    def forward(self, x):
        h = self.lift(_coord_cat(x))
        for sp, w in zip(self.sp, self.w):
            h = Fn.gelu(sp(h) + w(h))
        h = Fn.gelu(self.p1(h))
        if self.prob:
            return self.mu(h), Fn.softplus(self.sg(h)) + 1e-6
        return self.mu(h)

    def sample(self, x, n):
        mu, sg = self(x)
        return mu[None] + sg[None] * torch.randn(n, *mu.shape, device=mu.device)


class AuxHead(nn.Module):
    """Same trunk as the forecaster, different weights. Two output channels for
    the window time-mean, one for its log energy map."""

    def __init__(self, tcfg, cin, cout):
        super().__init__()
        self.lift = nn.Conv2d(cin, tcfg.WIDTH, 1)
        self.sp = nn.ModuleList([SpectralConv2d(tcfg.WIDTH, tcfg.WIDTH,
                                                tcfg.MODES_Y, tcfg.MODES_X)
                                 for _ in range(tcfg.LAYERS)])
        self.w = nn.ModuleList([nn.Conv2d(tcfg.WIDTH, tcfg.WIDTH, 1)
                                for _ in range(tcfg.LAYERS)])
        self.p1 = nn.Conv2d(tcfg.WIDTH, tcfg.LIFT, 1)
        self.out = nn.Conv2d(tcfg.LIFT, cout, 1)

    def forward(self, x):
        h = self.lift(_coord_cat(x))
        for sp, w in zip(self.sp, self.w):
            h = Fn.gelu(sp(h) + w(h))
        return self.out(Fn.gelu(self.p1(h)))


def rel_loss(p, y):
    return ((p - y).flatten(1).norm(dim=1) / (y.flatten(1).norm(dim=1) + 1e-8)).mean()


def energy_score(S, y):
    """Unbiased energy score, matching the reference implementation released
    with the probabilistic neural operator."""
    M = S.shape[0]
    t1 = sum((S[i] - y).flatten(1).norm(dim=1) for i in range(M)) / M
    t2 = sum((S[i] - S[j]).flatten(1).norm(dim=1)
             for i in range(M) for j in range(M) if i != j)
    return (t1 - t2 / (2 * M * (M - 1))).mean()


def loader(tcfg, tensors, shuffle):
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*tensors),
        batch_size=tcfg.BATCH, shuffle=shuffle)


def train(tcfg, model, dl_tr, dl_va, step, epochs, tag, post_grad=None):
    """Early stopping on validation loss, best weights restored. post_grad runs
    between backward() and step(), which is where PhysGuard projects."""
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.LR, weight_decay=tcfg.WD)
    best, bad, best_state, e = 1e18, 0, None, 0
    for e in range(epochs):
        model.train()
        for xb, yb in dl_tr:
            loss = step(model, xb.to(DEV), yb.to(DEV))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if post_grad:
                post_grad(model)
            opt.step()
        model.eval()
        vl, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in dl_va:
                vl += step(model, xb.to(DEV), yb.to(DEV)).item()
                n += 1
        vl /= max(n, 1)
        if vl < best - 1e-5:
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


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
