#!/usr/bin/env python3
"""Generate fig_F3_drift_adaptation (Fig. 7) from results/drift/drift.json.

LER of serial BP-OSD under per-qubit OU drift against the volatility v, for
five per-qubit rate estimates. Oracle, stale, mean and empir: pooled LER over
20 streams with Wilson 95% intervals. GNN: mean over five training seeds,
error bars give the range over seeds. Produced by audit_drift.py eval.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
d = json.loads((ROOT / "results" / "drift" / "drift.json").read_text())
vols = np.array(d["vols"], dtype=float)
res = [d["results"][str(v)] for v in d["vols"]]
gnn_keys = [f"gnn_s{s}" for s in d["seeds"]]

SERIES = [  # key, label, colour, marker  (Okabe-Ito; validated palette)
    ("oracle", "Oracle (true rates)", "#009E73", "^"),
    ("stale", "Stale (rates at $t=0$)", "#D55E00", "s"),
    ("mean", "Mean prior", "#CC79A7", "D"),
    ("empir", "Empir (frequency, $W=32$)", "#E69F00", "v"),
]

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 8.5, "axes.linewidth": 0.7,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "legend.fontsize": 7, "legend.frameon": True,
    "legend.framealpha": 0.92, "legend.edgecolor": "#cccccc", "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05, "pdf.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(3.3, 3.6))
for key, label, color, marker in SERIES:
    y = np.array([r[key]["ler"] for r in res])
    lo = np.array([r[key]["ci"][0] for r in res]); hi = np.array([r[key]["ci"][1] for r in res])
    ax.errorbar(vols, y, yerr=[y - lo, hi - y], color=color, marker=marker, linewidth=1.3, markersize=5,
                markeredgecolor="white", markeredgewidth=0.6, capsize=2, elinewidth=0.8, label=label, zorder=5)
g = np.array([[r[k]["ler"] for k in gnn_keys] for r in res])
gm = g.mean(1)
ax.errorbar(vols, gm, yerr=[gm - g.min(1), g.max(1) - gm], color="#0072B2", marker="o", linewidth=1.3,
            markersize=5, markeredgecolor="white", markeredgewidth=0.6, capsize=2, elinewidth=0.8,
            label="GNN estimator (5 seeds)", zorder=6)

ax.set_yscale("log")
ax.set_xticks(vols)
ax.set_xticklabels([f"{v:g}" for v in vols])
ax.set_xlim(vols.min() - 0.04, vols.max() + 0.04)
ax.set_xlabel(r"Volatility $v$ (per-shot log-rate innovation s.d.)")
ax.set_ylabel("Logical error rate")
ax.set_title(r"$[\![72,12,6]\!]$, $p=0.04$, OU drift, serial BP-OSD", pad=4)
ax.grid(axis="y", which="major", color="#e6e6e6", linewidth=0.5, zorder=0)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, handlelength=1.6, borderpad=0.4,
          labelspacing=0.3, columnspacing=1.0)

out = Path(__file__).parent / "fig_F3_drift_adaptation.pdf"
fig.savefig(out)
plt.close(fig)
print(f"Saved {out}")
