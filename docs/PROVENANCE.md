# Where each result in the paper comes from

Numbers refer to [`paper/STAR_GNN_BB_v5.tex`](../paper/STAR_GNN_BB_v5.tex).
Every experiment script writes a JSON summary together with the per-shot
failure vectors (`*_failures.npz` or `*.npy`).
[`make_tables.py`](../make_tables.py) recomputes every results table from
those per-shot vectors (not from the JSON summaries), and
[`paper/build_v5.py`](../paper/build_v5.py) inserts its rows into the
manuscript, so no table value is typed by hand. `python make_tables.py
--check paper/STAR_GNN_BB_v5.tex` confirms that every generated row appears
in the manuscript.

All experiments use one failure criterion,
[`gnn_pipeline/decoding_failure.py`](../gnn_pipeline/decoding_failure.py): a
shot fails if the estimate does not reproduce the syndrome, or reproduces it
and flips a logical operator; the X and Z components are combined by OR.

| Paper item | Script (command) | Stored output |
|---|---|---|
| Table 1, codes | [`codes/code_registry.py`](../codes/code_registry.py), [`codes/codes_q.py`](../codes/codes_q.py) | |
| Table 2, model, training, reference decoder; Fig. 2 | [`audit_headline_v2.py`](../audit_headline_v2.py) (`run_seed`), `SERIAL_MS`/`OSD_CS10` in [`audit_tables_v2.py`](../audit_tables_v2.py) | |
| Table 3, Fig. 3, decomposition numbers (Sec. 5.1) | `python audit_tables_v2.py bposd_configs`, `... decomposition` | [`results/tables_v2/`](../results/tables_v2/) |
| Invariance checks (Sec. 5.2) | `python audit_tables_v2.py invariance` | [`results/tables_v2/invariance.json`](../results/tables_v2/invariance.json) |
| Table 4 (five seeds) | `python audit_headline_v2.py --make_data --baselines --seed 0 ... --seed 4 --summary` | [`results/headline_v2/`](../results/headline_v2/) |
| Table 5, X/Z components (Sec. 5.3) | `python audit_components.py` | [`results/components/`](../results/components/) |
| Gradient audit, capacity test (Sec. 5.3) | `python audit_gradient.py`, `python audit_capacity.py` | [`results/diagnostics/`](../results/diagnostics/) |
| Table 6, Fig. 4 | `python audit_tables_v2.py gnn_vs_bposd` | [`results/tables_v2/gnn_vs_bposd*`](../results/tables_v2/) |
| Table 7, failure sets (Sec. 5.4) | `python audit_failure_sets.py` | [`results/failure_sets/`](../results/failure_sets/) |
| Table 8, failure classes and symmetric configurations (Sec. 5.4) | `python audit_mechanism.py` | [`results/mechanism/`](../results/mechanism/) |
| Table 9, Fig. 5, oracle gap (Sec. 5.5) | `python audit_tables_v2.py oracle` | [`results/tables_v2/oracle_mean*`](../results/tables_v2/) |
| Table 10, Fig. 6, drift (Sec. 5.6) | `python audit_drift.py train`, `python audit_drift.py eval` | [`results/drift/`](../results/drift/) |
| Tables 11-12, Fig. 7, circuit level (Sec. 5.7) | `python audit_circuit_v2.py --oracle ...` (two runs, see [`reproduce_all.sh`](../reproduce_all.sh)); noise model in [`astra_stim/biased_noise.py`](../astra_stim/biased_noise.py), circuit in [`astra_stim/qldpc_circuit.py`](../astra_stim/qldpc_circuit.py), DEM in [`gnn_pipeline/dem_decoder.py`](../gnn_pipeline/dem_decoder.py) | [`results/circuit_v2/`](../results/circuit_v2/) |
| Fig. 1 and Fig. 8 (appendix), diagrams | TikZ sources in [`docs/`](./) | |

[`reproduce_all.sh`](../reproduce_all.sh) runs every command above in order
and then checks the manuscript tables; [`figures/make_all_figures.sh`](../figures/make_all_figures.sh)
redraws all eight figures from the stored results.

## Code paths behind the method description

- **Training graph (Sec. 3.2, Table 2, Fig. 2).** `audit_headline_v2.py`,
  `run_seed`: the loss is `focal_loss(sigmoid(-(prior + d)), b.y)`, i.e.
  the focal loss of q = sigma(-(l + Delta)) against the sampled Z error; BP
  is not in the loss. After each epoch, `score("val_p04")` decodes the
  corrected priors with 10-iteration flooding BP and the epoch with the
  fewest validation failures is kept.
- **Prior feature (Sec. 3.1).** `Split.llr = log((1 - p_values) / p_values)`
  with the per-shot `p_values` of the data file; `Harness.bp` passes the same
  prior (plus Delta) to both CSS components.
- **Loss target (Sec. 3.2).** `Harness.graphs` sets `y = z_errors` of the
  shot; the loss compares bit by bit and is not degeneracy-aware.
- **BP decoder.** `gnn_pipeline/bp_decoder.py`, `MinSumBPDecoder`: min-sum
  with scaling 0.8, messages clamped to +-20, early stop only when every
  shot of the batch has converged, so on a test set it runs the full 10
  (or 100) iterations.
- **Neural-BP initialisation (Appendix A).** `MinSumBPDecoder._bp_weight`:
  `softplus(raw) / ln 2`, so `raw = 0` gives weight 1 exactly.
- **FiLM input (Appendix A).** `gnn_pipeline/gnn_model.py`, `TannerGNN.forward`
  uses `data.p_value`, the shot's physical error rate, set in
  `gnn_pipeline/dataset.py`.

## Reading the numbers off the stored output

- **Table 4.** BP rows: `results/headline_v2/test_p0*_bp_fail.npy`; GNN
  rows: `seed*_test_p0*_gnn_fail.npy`; BP-100 and BP-OSD rows:
  `results/failure_sets/test_p0*_failures.npz` (`bp100`, `bposd`). The
  non-converged counts in the text are `bp_syndrome_mismatch` and
  `gnn_syndrome_mismatch` in `seed*.json`.
- **Table 5.** `results/components/test_p04_failures.npz`: `<decoder>_z`,
  `<decoder>_x`, `<decoder>_any`; `gnn_bp10_zonly_s*` is the correction
  applied to the Z component only.
- **Table 6.** `results/tables_v2/gnn_vs_bposd_failures.npz`; Holm
  correction over its 20 tests in `make_tables.py`.
- **Table 7.** `results/failure_sets/test_p04_failures.npz` and
  `test_p04.json` (`gnn_fixes`, `gnn_osd_changes`).
- **Table 8.** `results/mechanism/test_p04.json`: `bp10.classes`,
  `gnn.s*.fixed`, `by_symmetric`, `symmetric_all`.
- **Table 9.** `results/tables_v2/oracle_mean_failures.npz`, cells
  `<code>_p<p>_s<sigma>_{mean,oracle}`.
- **Table 10.** `results/drift/drift_failures.npz` and `drift.json`
  (`per_stream_failures`, `sign_tests`); estimator checkpoints
  `results/drift/estimator_seed*.pt`.
- **Table 12.** `results/circuit_v2/oracle_v4_*_failures.npz`; per-profile
  sign test from the 500-shot blocks of the p = 0.01 run.

## Data

- [`data/headline_v2/`](../data/headline_v2/): training (seeds 1001, 1002),
  validation (2001) and test (3001, 3002) sets of Table 4, written by
  `audit_headline_v2.py --make_data`. Tables 5-8 and the gradient audit use
  the same test sets.
- [`data/big72_test_p04.npz`](../data/big72_test_p04.npz): the fixed
  syndromes of the capacity test (its first 1,000 shots).
- The other experiments draw fresh data from fixed seeds stored in each JSON
  (`data_seed`, `data_seeds`) or in the script (drift streams: 9000 + 100 v_index + stream).

## Version history

- [`paper/reviewed/STAR_GNN_BB_reviewed.tex`](../paper/reviewed/STAR_GNN_BB_reviewed.tex)
  is the version reviewed in October 2026 (rebuilt from the v4 build script
  and checked against the PDF that was sent).
  [`paper/build_v5.py`](../paper/build_v5.py) applies every v5 change to it;
  [`docs/REVISION_v5.md`](REVISION_v5.md) lists them per review item.
- Earlier versions (v2 to v4, and files used only by them) are in the git
  history; `paper_v4/` is at commit `72f2cf4`.
