"""The error budget of Table 1, on one frozen fine-tuned operator.

  python scripts/run_budget.py --data-root /path/to/realpde
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mend import Bench
from mend.budget import compute
from mend.cli import parse
from mend.nets import DEV
from scripts.run_mend import load_frozen


def main():
    args, cfg, tcfg, _ = parse(__doc__)
    bench = Bench(cfg)
    keep = load_frozen(os.path.join(cfg.out_dir, "frozen.pt"), bench, tcfg)
    seed = sorted(keep["direct fine-tune"])[0]
    model = keep["direct fine-tune"][seed].to(DEV)
    print(f"error budget on direct fine-tune, seed {seed}")
    out = compute(bench, model)
    path = os.path.join(cfg.out_dir, "budget.json")
    with open(path, "w") as f:
        json.dump({k: list(map(float, v)) for k, v in out.items()}, f, indent=1)
    print("saved", path)


if __name__ == "__main__":
    main()
