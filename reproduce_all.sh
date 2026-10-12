#!/usr/bin/env bash
# Re-run every experiment behind the paper from scratch, then check that the
# regenerated outcomes reproduce every results table of the manuscript and
# redraw the figures. Run from the repository root in the environment of
# requirements-lock.txt. Takes several hours on a 4-core CPU.
#
#     bash reproduce_all.sh 2>&1 | tee reproduce.log
set -euo pipefail
cd "$(dirname "$0")"

step() { echo; echo "== $(date '+%H:%M:%S') $*"; }

step "Table 4 data, baselines and the five GNN checkpoints (Tables 2, 4)"
python audit_headline_v2.py --make_data --baselines --seed 0 --seed 1 --seed 2 --seed 3 --seed 4 --summary

step "Table 3, Fig. 3, invariance checks (Sec. 5.1-5.2)"
python audit_tables_v2.py bposd_configs
python audit_tables_v2.py decomposition
python audit_tables_v2.py invariance

step "GNN+OSD vs BP-OSD (Table 6, Fig. 4)"
python audit_tables_v2.py gnn_vs_bposd

step "Paired failure sets, components, failure classes (Tables 5, 7, 8)"
python audit_failure_sets.py
python audit_components.py
python audit_mechanism.py

step "Gradient audit and capacity test (Sec. 5.3)"
python audit_gradient.py
python audit_capacity.py

step "Code-capacity oracle gap (Table 9, Fig. 5)"
python audit_tables_v2.py oracle

step "Drift experiment (Table 10, Fig. 6)"
python audit_drift.py train
python audit_drift.py eval

step "Circuit-level oracle gap (Tables 11-12, Fig. 7)"
python audit_circuit_v2.py --oracle --p 0.001 0.003 --profiles 20 --shots_per_profile 500 --scaling 0.8 --anchor mean --tag _v4_paper_points
python audit_circuit_v2.py --oracle --p 0.01 --profiles 40 --shots_per_profile 500 --scaling 0.8 --anchor mean --tag _v4_p010

step "Check every results table of the manuscript against the regenerated outcomes"
python make_tables.py --check paper/STAR_GNN_BB_v5.tex

step "Redraw the figures"
bash figures/make_all_figures.sh
step "done"
