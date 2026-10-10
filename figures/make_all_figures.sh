#!/usr/bin/env bash
# Regenerate every figure in the paper (paper_v4/STAR_GNN_BB_v4.tex).
#
# Usage (from the repository root):
#     bash figures/make_all_figures.sh [output_dir]
#
# Output defaults to paper_v4/figures/. Requires Python with matplotlib and
# numpy, and pdflatex with TikZ for the three diagrams.
#
# Data figures read the stored per-shot results in results/; they do not
# re-run any simulation. The scripts that produced those results are listed in
# figures/README.md.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/paper_v4/figures}"
mkdir -p "$OUT"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cd "$ROOT"

echo "== TikZ diagrams (Figs. 1-3)"
for f in fig_tannergnn_arch fig_interleaved_pipeline fig_experimental_pipeline; do
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$TMP" "docs/$f.tex" > "$TMP/$f.out"
    cp "$TMP/$f.pdf" "$OUT/$f.pdf"
    echo "   $f.pdf"
done

echo "== Data figures (Figs. 4-8)"
python figures/gen_F1_baseline_decomp.py      # Fig. 4, results/tables_v2/decomposition.json
python figures/gen_F4_phase4_ler.py           # Fig. 5, results/tables_v2/gnn_vs_bposd.json
python figures/gen_F2_oracle_vs_distance.py   # Fig. 6, results/tables_v2/oracle_mean.json
python figures/gen_F3_drift_adaptation.py     # Fig. 7, values from the original drift run (no stored data)
for f in fig_F1_baseline_decomp fig_F4_phase4_ler fig_F2_oracle_vs_distance fig_F3_drift_adaptation; do
    cp "figures/$f.pdf" "$OUT/$f.pdf"
done
# Fig. 8, results/circuit_v2/oracle_v4_*.json (min-sum scaling 0.8, mean-anchored rates)
python figures/gen_F6_circuit_oracle.py "oracle_v4_*.json" "$OUT/fig_F6_circuit_oracle.pdf" "p"

echo "Done: 8 figures in $OUT"
