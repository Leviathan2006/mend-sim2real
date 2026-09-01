"""LaTeX emitters for the tables in the paper."""
import os

from .budget import ROWS

HEAD = (r"\begin{table}[t]" "\n" r"\centering" "\n" r"\small" "\n")
FOOT = r"\bottomrule" "\n" r"\end{tabular}" "\n"


def _cell(t, prec=4):
    return f"{t[0]:.{prec}f}\\,$\\pm$\\,{t[1]:.{prec}f}"


def main_table(results, order, path, caption, label, bold_suffix="+ MEND",
               show_n=False):
    cols = "lcccc" + ("c" if show_n else "")
    lines = [HEAD, f"\\begin{{tabular}}{{{cols}}}\n", "\\toprule\n",
             "Method & Rel-L2 $\\downarrow$ & TKE $\\downarrow$ & "
             "MVPE $\\downarrow$ & SPS $\\uparrow$"
             + (" & $n$" if show_n else "") + " \\\\\n", "\\midrule\n"]
    for name in order:
        if name not in results:
            continue
        a, n = results.agg(name)
        row = (f"{name} & {_cell(a['relL2'])} & {_cell(a['tke'])} & "
               f"{_cell(a['mvpe'])} & {_cell(a['sps'], 2)}")
        if show_n:
            row += f" & {n}"
        if bold_suffix and name.endswith(bold_suffix):
            row = "\\textbf{" + row.replace(" & ", "} & ", 1)
        lines.append(row + " \\\\\n")
    lines += [FOOT, f"\\caption{{{caption}}}\n", f"\\label{{{label}}}\n",
              "\\end{table}\n"]
    _write(path, lines)


def budget_table(budget, path, caption, label):
    lines = [HEAD, "\\begin{tabular}{lccc}\n", "\\toprule\n",
             "Statistic replaced & Rel-L2 $\\downarrow$ & TKE $\\downarrow$ & "
             "MVPE $\\downarrow$ \\\\\n", "\\midrule\n"]
    for r in ROWS:
        a, b, c = budget[r]
        lines.append(f"{r} & {a:.4f} & {b:.4f} & {c:.4f} \\\\\n")
    lines += [FOOT, f"\\caption{{{caption}}}\n", f"\\label{{{label}}}\n",
              "\\end{table}\n"]
    _write(path, lines)


def knobs_table(knobs, path, caption, label):
    lines = [HEAD, "\\begin{tabular}{llcc}\n", "\\toprule\n",
             "Base & Seed & $a$ & $b$ \\\\\n", "\\midrule\n"]
    for (base, seed), (a, b) in sorted(knobs.items()):
        lines.append(f"{base} & {seed} & {a} & {b} \\\\\n")
    lines += [FOOT, f"\\caption{{{caption}}}\n", f"\\label{{{label}}}\n",
              "\\end{table}\n"]
    _write(path, lines)


def _write(path, lines):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.writelines(lines)
    print("wrote", path)
