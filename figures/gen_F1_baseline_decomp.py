#!/usr/bin/env python3
"""Generate fig_F1_baseline_decomp sized for a single column (~3.3 in) in
a two-column quantumarticle.  Ratio labels are placed inside each bar as
centred white text so they never overlap.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

# --- Data: results/tables_v2/decomposition.json (audit_tables_v2.py decomposition) ---
_res = json.loads((Path(__file__).resolve().parent.parent / "results" / "tables_v2"
                   / "decomposition.json").read_text())
_keys = ["bp_flooding_ms100", "bp_serial_ms100", "bposd_serial_ms100"]
labels = ["Flooding BP", "+ Serial\nschedule", "+ OSD-CS-10"]
lers   = [_res["results"][k]["ler"] for k in _keys]
ci_lo  = [l - _res["results"][k]["ci"][0] for l, k in zip(lers, _keys)]
ci_hi  = [_res["results"][k]["ci"][1] - l for l, k in zip(lers, _keys)]
colors = ["#6BAED6", "#3182BD", "#2C5F7A"]
ratios = [None, lers[0] / lers[1], lers[1] / lers[2]]

# Style: 8-9 pt fonts, column-width figure (3.3 in)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 8.5,
    "axes.linewidth": 0.6,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "pdf.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(3.3, 2.7))   # matches \columnwidth in a4 twocolumn

x = np.arange(len(labels))
bars = ax.bar(x, lers, color=colors, width=0.55,
              edgecolor="black", linewidth=0.6)
ax.errorbar(x, lers, yerr=[ci_lo, ci_hi], fmt="none",
            ecolor="black", capsize=3, capthick=0.7, linewidth=0.7)

ax.set_yscale("log")
ax.set_ylim(5e-5, 2e-2)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Logical Error Rate")
ax.set_title(
    rf"$[\![288, 12, 18]\!]$, $p={_res['p']}$, $\eta=20$, {_res['shots'] // 1000}k shots",
    pad=4,
)

# Ratio labels inside bars (white, centred)
for i, ratio in enumerate(ratios):
    if ratio is None:
        continue
    bx = bars[i].get_x() + bars[i].get_width() / 2
    by = bars[i].get_height()
    ax.text(bx, (by * 5e-5) ** 0.5, f"{ratio:.1f}×",
            ha="center", va="center",
            fontsize=8, fontweight="bold", color="white")

# Save with name that matches \includegraphics{fig_F1_baseline_decomp} in LaTeX
out = Path(__file__).parent / "fig_F1_baseline_decomp.pdf"
fig.savefig(out)
plt.close(fig)
print(f"Saved {out}")
