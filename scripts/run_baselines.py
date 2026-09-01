"""Eight baselines: three classical, five learned. Writes runs/baselines.json.

  python scripts/run_baselines.py --data-root /path/to/realpde
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from mend import Bench, Results
from mend.cli import parse
from mend import classical, transfer

ORDER = ["Kennedy-O'Hagan", "quantile map", "DMD",
         "direct fine-tune", "L2-SP", "EWC", "PhysGuard", "PNO"]


def main():
    args, cfg, tcfg, _ = parse(__doc__)
    bench = Bench(cfg)
    results = Results()

    classical.run_all(bench, results)
    keep = {}
    transfer.run_all(bench, results, tcfg, keep=keep)

    results.table(ORDER)
    results.save(os.path.join(cfg.out_dir, "baselines.json"),
                 extra={"config": cfg.as_dict()})
    torch.save({k: {s: m.state_dict() for s, m in d.items()} for k, d in keep.items()},
               os.path.join(cfg.out_dir, "frozen.pt"))
    print("saved", os.path.join(cfg.out_dir, "frozen.pt"))


if __name__ == "__main__":
    main()
