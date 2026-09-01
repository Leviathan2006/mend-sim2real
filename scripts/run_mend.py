"""MEND on top of the frozen models from run_baselines.py.

Goes through the same evaluate() as every baseline, so the rows are comparable.

  python scripts/run_mend.py --data-root /path/to/realpde
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from mend import Bench, Results
from mend.cli import parse
from mend.correct import run_all
from mend.nets import FNO
from scripts.run_baselines import ORDER


def load_frozen(path, bench, tcfg):
    cin, cout = 2 * bench.K + 2, 2 * bench.HO
    blob = torch.load(path, map_location="cpu")
    keep = {}
    for name, per_seed in blob.items():
        for seed, state in per_seed.items():
            m = FNO(tcfg, cin, cout, prob=(name == "PNO"))
            m.load_state_dict(state)
            keep.setdefault(name, {})[int(seed)] = m.eval()
    return keep


def main():
    args, cfg, tcfg, mcfg = parse(__doc__)
    bench = Bench(cfg)
    base = os.path.join(cfg.out_dir, "baselines.json")
    if not os.path.exists(base):
        raise SystemExit("run scripts/run_baselines.py first")
    results = Results.load(base)
    keep = load_frozen(os.path.join(cfg.out_dir, "frozen.pt"), bench, tcfg)

    knobs, sweeps = run_all(bench, results, tcfg, mcfg, keep)

    order = ORDER + [f"{b} + MEND" for b in mcfg.WRAP if f"{b} + MEND" in results]
    results.table(order)
    results.save(os.path.join(cfg.out_dir, "all_results.json"),
                 extra={"config": cfg.as_dict(),
                        "knobs": {f"{k[0]}|{k[1]}": v for k, v in knobs.items()}})
    with open(os.path.join(cfg.out_dir, "sweeps.json"), "w") as f:
        json.dump({f"{k[0]}|{k[1]}": v for k, v in sweeps.items()}, f, indent=1)


if __name__ == "__main__":
    main()
