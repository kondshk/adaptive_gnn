#!/usr/bin/env python3
"""Generate fig_F2_oracle_gap (Fig. 6): mean-prior vs oracle BP-OSD LER.

Absolute LERs with Wilson 95% intervals against the per-qubit spread sigma,
one colour and marker per code at its own operating point. Lines join the
sigma values of one code only; codes are not joined, because code size and
operating point change together.

Data: results/tables_v2/oracle_mean.json (audit_tables_v2.py oracle).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
cells = json.loads((ROOT / "results" / "tables_v2" / "oracle_mean.json").read_text())["cells"]

CODES = [("72_12_6", "0.04", r"$[\![72,12,6]\!]$, $p=0.04$", "#0072B2", "o"),
         ("144_12_12", "0.06", r"$[\![144,12,12]\!]$, $p=0.06$", "#D55E00", "s"),
         ("288_12_18", "0.07", r"$[\![288,12,18]\!]$, $p=0.07$", "#009E73", "^")]
SIGMAS = ["0.25", "0.5", "1.0"]

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 8.5, "axes.linewidth": 0.7,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "legend.fontsize": 7, "legend.frameon": True,
    "legend.framealpha": 0.92, "legend.edgecolor": "#cccccc", "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05, "pdf.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(3.3, 3.1))
for i, (code, p, label, color, marker) in enumerate(CODES):
    x = [k + (i - 1) * 0.08 for k in range(len(SIGMAS))]
    for kind, ls, face in (("mean", "--", "white"), ("oracle", "-", color)):
        c = [cells[f"{code}_p{p}_s{s}"][kind] for s in SIGMAS]
        y = [v["ler"] for v in c]
        err = [[v["ler"] - v["ci"][0] for v in c], [v["ci"][1] - v["ler"] for v in c]]
        ax.errorbar(x, y, yerr=err, color=color, linestyle=ls, linewidth=1.3, marker=marker,
                    markersize=5.5, markerfacecolor=face, markeredgecolor=color, markeredgewidth=1.0,
                    capsize=2, elinewidth=0.8, zorder=5)

ax.set_yscale("log")
ax.set_xticks(range(len(SIGMAS)))
ax.set_xticklabels([rf"$\sigma={s}$" for s in SIGMAS])
ax.set_xlim(-0.4, len(SIGMAS) - 0.6)
ax.set_ylim(4e-3, 0.25)
ax.set_ylabel("Logical error rate")
ax.set_title("Mean-prior vs. oracle BP-OSD", pad=4)
ax.grid(axis="y", which="major", color="#e6e6e6", linewidth=0.5, zorder=0)

code_handles = [Line2D([], [], color=c, marker=m, linestyle="-", markersize=5, label=l) for _, _, l, c, m in CODES]
style_handles = [Line2D([], [], color="#444444", marker="o", markerfacecolor="white", linestyle="--",
                        markersize=5, label="mean prior"),
                 Line2D([], [], color="#444444", marker="o", linestyle="-", markersize=5, label="oracle (true rates)")]
leg1 = ax.legend(handles=code_handles, loc="lower left", handlelength=1.6, borderpad=0.4, labelspacing=0.3)
ax.add_artist(leg1)
ax.legend(handles=style_handles, loc="upper right", handlelength=1.8, borderpad=0.4, labelspacing=0.3)

out = Path(__file__).parent / "fig_F2_oracle_gap.pdf"
fig.savefig(out)
plt.close(fig)
print(f"Saved {out}")
