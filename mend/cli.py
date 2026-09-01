"""Shared argument parsing for the scripts."""
import argparse
import os

from .config import Config, MendConfig, TrainConfig


def parse(desc):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--data-root", default=os.environ.get("REALPDE_ROOT", "data"),
                   help="directory containing train_sim/ and train_real/")
    p.add_argument("--out", default="runs", help="where results and tables go")
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--quick", action="store_true",
                   help="one seed, few windows, 1-2 epochs: a smoke test, not a result")
    a = p.parse_args()

    cfg = Config(data_root=a.data_root, out_dir=a.out, quick=a.quick)
    if a.seeds is not None:
        cfg.SEEDS = a.seeds
    tcfg = TrainConfig()
    if a.quick:
        tcfg.quicken()
    os.makedirs(cfg.out_dir, exist_ok=True)
    return a, cfg, tcfg, MendConfig()
