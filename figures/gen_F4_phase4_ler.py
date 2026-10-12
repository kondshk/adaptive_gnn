#!/usr/bin/env python3
"""Generate fig_F4_phase4_ler (Fig. 5) — column width (3.3 in).

Serial BP-OSD with Wilson 95% intervals; GNN+OSD as the mean over the five
training seeds with the seed range as error bar. A point is marked when every
seed is significantly worse than BP-OSD after Holm correction over the 20
McNemar tests of the table (4 error rates x 5 seeds).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

# --- Data: results/tables_v2/gnn_vs_bposd.json (audit_tables_v2.py gnn_vs_bposd) ---
_res = json.loads((Path(__file__).resolve().parent.parent / "results" / "tables_v2"
                   / "gnn_vs_bposd.json").read_text())
_pts = sorted(_res["points"].items(), key=lambda kv: float(kv[0]))
ps          = np.array([float(k) for k, _ in _pts])
bp_osd_ler  = np.array([v["bposd"]["ler"] for _, v in _pts])
gnn_osd_ler = np.array([np.mean([g["ler"] for g in v["gnn"].values()]) for _, v in _pts])

n, z = _res["shots"], 1.96
def wilson(k):
    ph = k / n; den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den; h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return c - h, c + h

lo, hi = wilson(np.array([v["bposd"]["failures"] for _, v in _pts]))
bp_ci  = [bp_osd_ler - lo, hi - bp_osd_ler]
g_all  = np.array([[g["ler"] for g in v["gnn"].values()] for _, v in _pts])
gnn_ci = [gnn_osd_ler - g_all.min(1), g_all.max(1) - gnn_osd_ler]

# Holm step-down over all 20 tests; a point is "worse" if every seed is
# significantly worse (n01 > n10) after adjustment.
tests = [(i, g["mcnemar_bposd_vs_gnn"]) for i, (_, v) in enumerate(_pts) for g in v["gnn"].values()]
order = sorted(range(len(tests)), key=lambda j: tests[j][1]["p"])
adj, run = [0.0] * len(tests), 0.0
for r, j in enumerate(order):
    run = max(run, min(1.0, (len(tests) - r) * tests[j][1]["p"])); adj[j] = run
worse_mask = np.array([all(adj[j] < 0.05 and tests[j][1]["n01"] > tests[j][1]["n10"]
                           for j in range(len(tests)) if tests[j][0] == i) for i in range(len(_pts))])

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":         8,
    "axes.labelsize":    9,
    "axes.titlesize":    8.5,
    "axes.linewidth":    0.7,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.top":         True,
    "ytick.right":       True,
    "xtick.major.size":  3,
    "ytick.major.size":  3,
    "lines.linewidth":   1.4,
    "lines.markersize":  5.5,
    "legend.fontsize":   7.5,
    "legend.frameon":    True,
    "legend.framealpha": 0.92,
    "legend.edgecolor":  "#cccccc",
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.08,
    "pdf.fonttype":      42,
})

fig, ax = plt.subplots(figsize=(3.3, 3.3))

ax.errorbar(ps, bp_osd_ler, yerr=bp_ci,
            fmt="o-", color="#0055D4", capsize=3, capthick=0.8,
            linewidth=1.4, markersize=5.5,
            markeredgewidth=0.6, markeredgecolor="white",
            label="Serial BP-OSD", zorder=5)
ax.errorbar(ps, gnn_osd_ler, yerr=gnn_ci,
            fmt="s-", color="#E69F00", capsize=3, capthick=0.8,
            linewidth=1.4, markersize=5.5,
            markeredgewidth=0.6, markeredgecolor="white",
            label="GNN + OSD", zorder=5)

# Red × at statistically-worse points
ax.scatter(ps[worse_mask], gnn_osd_ler[worse_mask],
           marker="x", color="#B03030", s=80, linewidths=2.2, zorder=8)

ax.set_yscale("log")
ax.set_ylim(2e-3, 4e-1)
ax.set_xlim(0.014, 0.057)
ax.set_xlabel(r"Physical error rate $p$")
ax.set_ylabel("Logical error rate")
ax.set_title(rf"$[\![72,12,6]\!]$, $\eta=20$, serial BP-OSD, {n // 1000}k shots, {len(_res['checkpoints'])} seeds", pad=4)

# Legend upper LEFT
ax.legend(loc="upper left", handlelength=1.5, handletextpad=0.4)

# Annotation: placed in the LOWER-RIGHT quadrant of the log plot
# (below the two × marks, to the right) — far from the upper-left legend.
ax.annotate(
    r"$\times$ = every seed worse" "\n" r"(McNemar, Holm-adjusted $p<0.05$)",
    xy=(ps[worse_mask][-1], gnn_osd_ler[worse_mask][-1]),
    xytext=(0.047, 0.0055),
    fontsize=7,
    color="#B03030",
    ha="center",
    va="bottom",
    arrowprops=dict(arrowstyle="->", color="#B03030",
                    lw=0.8, shrinkA=4, shrinkB=4),
    bbox=dict(boxstyle="round,pad=0.25", fc="white",
              ec="#B03030", alpha=0.92, lw=0.7),
    zorder=9,
)

out = Path(__file__).parent / "fig_F4_phase4_ler.pdf"
fig.savefig(out)
plt.close(fig)
print(f"Saved {out}")
