#!/usr/bin/env python3
"""Generate fig_F6_circuit_oracle: circuit-level oracle gap on [[72,12,6]].

Reads results/circuit_v2/oracle_*.json (written by audit_circuit_v2.py), so
the figure always matches the stored paired outcomes. Column width (3.3 in).
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
recs = {}
for f in sorted((ROOT / "results" / "circuit_v2").glob("oracle_*.json")):
    for v in json.loads(f.read_text()).values():
        recs[v["p_mean"]] = v
ps = sorted(recs)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 8, "axes.linewidth": 0.7,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "xtick.direction": "in", "ytick.direction": "in",
    "ytick.right": True, "legend.fontsize": 7.5, "legend.frameon": True,
    "legend.framealpha": 0.92, "legend.edgecolor": "#cccccc", "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.08, "pdf.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(3.3, 3.0))
x = np.arange(len(ps))
floor = 1e-5
series = [("mean", "Mean-prior BP-OSD", "#0072B2", "o", -0.12),
          ("oracle", "Per-qubit oracle BP-OSD", "#E69F00", "s", 0.12)]
for key, label, color, marker, dx in series:
    for i, p in enumerate(ps):
        r = recs[p]
        k, lo, hi = r[f"{key}_failures"], *r[f"{key}_ci"]
        if k == 0:  # no failures: show the 95% upper bound
            ax.plot(x[i] + dx, hi, marker="v", ms=6, mfc="white", mec=color, mew=1.2, ls="none")
            continue
        ler = r[f"{key}_ler"]
        ax.errorbar(x[i] + dx, ler, yerr=[[ler - lo], [hi - ler]], fmt=marker, ms=5,
                    color=color, mec="white", mew=0.6, ecolor="#555555", elinewidth=0.8,
                    capsize=2.5, label=label if i == len(ps) - 1 else None)

def p_label(pv):
    if pv >= 1e-3:
        return f"$p={pv:.2g}$"
    e = math.floor(math.log10(pv))
    return rf"$p={pv / 10 ** e:.1f}\times10^{{{e}}}$"


for i, p in enumerate(ps):
    r = recs[p]; m = r["mcnemar_mean_vs_oracle"]
    top = max(r["mean_ci"][1], r["oracle_ci"][1])
    txt = ("no failures" if r["mean_failures"] + r["oracle_failures"] == 0 else
           f"{m['n10']}/{m['n01']}\n{p_label(m['p'])}")
    ax.text(x[i], top * 1.35, txt, ha="center", va="bottom", fontsize=7, color="#444444")

ax.set_yscale("log")
ax.set_ylim(floor, 0.1)
ax.set_xlim(-0.5, len(ps) - 0.5)
ax.set_xticks(x)
ax.set_xticklabels([rf"$p_\mathrm{{mean}}={p:g}$" for p in ps])
ax.set_ylabel("Logical error rate")
ax.set_title(r"$[\![72,12,6]\!]$, 6 rounds, $\eta=20$, log-normal $\sigma=1$" "\n"
             r"labels: discordant pairs $n_{10}/n_{01}$ and McNemar $p$", pad=4, fontsize=7)
ax.grid(axis="y", which="major", color="#e6e6e6", lw=0.5)
ax.set_axisbelow(True)
ax.plot([], [], marker="v", ms=6, mfc="white", mec="#555555", mew=1.2, ls="none",
        label="95% upper bound (0 failures)")
ax.legend(loc="lower right", handlelength=1.2, handletextpad=0.4, borderpad=0.5)

out = Path(__file__).parent / "fig_F6_circuit_oracle.pdf"
fig.savefig(out)
fig.savefig(out.with_suffix(".png"), dpi=200)
plt.close(fig)
print(f"Saved {out}")
