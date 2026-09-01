"""Emit the LaTeX tables from the saved results.

  python scripts/make_tables.py --out runs --tex paper/tables
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mend import Results
from mend.tables import budget_table, knobs_table, main_table
from scripts.run_baselines import ORDER


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="runs")
    p.add_argument("--tex", default="paper/tables")
    a = p.parse_args()

    res_path = os.path.join(a.out, "all_results.json")
    results = Results.load(res_path)
    with open(res_path) as f:
        blob = json.load(f)

    order = ORDER + [n for n in results.r if n.endswith("+ MEND")]
    main_table(results, order, os.path.join(a.tex, "table_main.tex"),
               "Eight baselines and MEND on the RealPDE airfoil pair, scored with "
               "the benchmark's own metrics. Mean $\\pm$ s.d. over 5 seeds.",
               "tab:main")

    bpath = os.path.join(a.out, "budget.json")
    if os.path.exists(bpath):
        with open(bpath) as f:
            budget_table(json.load(f), os.path.join(a.tex, "table_budget.tex"),
                         "Error budget. Each row replaces one degree of freedom of "
                         "the frozen forecast with its true value, bounding what any "
                         "method correcting that degree of freedom could achieve.",
                         "tab:budget")

    if "knobs" in blob:
        knobs = {tuple(k.split("|")): v for k, v in blob["knobs"].items()}
        knobs_table({(b, int(s)): v for (b, s), v in knobs.items()},
                    os.path.join(a.tex, "table_knobs.tex"),
                    "Coefficients selected on validation, per base model and seed.",
                    "tab:knobs")


if __name__ == "__main__":
    main()
