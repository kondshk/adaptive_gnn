# Where each result in the paper comes from

Table, figure and section numbers refer to the final paper,
[`paper_v4/STAR_GNN_BB_v4.tex`](../paper_v4/STAR_GNN_BB_v4.tex). Every
experiment script writes a JSON summary together with the per-shot failure
vectors (`*_failures.npz` or `*.npy`), so each logical error rate and each
McNemar count can be recomputed from the raw outcomes. Commands are run from
the repository root.

All experiments use one failure criterion,
[`gnn_pipeline/decoding_failure.py`](../gnn_pipeline/decoding_failure.py): a
shot fails if the estimate does not reproduce the syndrome, or reproduces it
and flips a logical operator; the X and Z components are combined by OR.

| Paper item | Script (command) | Stored output |
|---|---|---|
| Table 1, codes | [`codes/code_registry.py`](../codes/code_registry.py), [`codes/codes_q.py`](../codes/codes_q.py) | |
| Table 2, model / training / reference decoder | constants in [`audit_headline_v2.py`](../audit_headline_v2.py); `SERIAL_MS`, `OSD_CS10` in [`audit_tables_v2.py`](../audit_tables_v2.py) | |
| Table 3, BP / BP-OSD configurations on [[288,12,18]] | `python audit_tables_v2.py bposd_configs` | [`results/tables_v2/bposd_configs.json`](../results/tables_v2/bposd_configs.json) |
| Fig. 4 and the 8.8× / 5.4× / 47× decomposition (Sec. 5.1) | `python audit_tables_v2.py decomposition` | [`results/tables_v2/decomposition.json`](../results/tables_v2/decomposition.json) |
| Invariance checks (Sec. 5.2) | `python audit_tables_v2.py invariance` | [`results/tables_v2/invariance.json`](../results/tables_v2/invariance.json) |
| Table 4, GNN-BP vs. flooding BP, five seeds | `python audit_headline_v2.py --make_data --baselines --seed 0 --seed 1 --seed 2 --seed 3 --seed 4 --summary` | [`results/headline_v2/`](../results/headline_v2/) |
| Gradient audit (Sec. 5.3) | `python audit_gradient.py` | [`results/diagnostics/gradient_audit.json`](../results/diagnostics/gradient_audit.json) |
| Capacity test (Sec. 5.3) | `python audit_capacity.py` | [`results/diagnostics/capacity_test.json`](../results/diagnostics/capacity_test.json) |
| Table 5, Fig. 5, GNN+OSD vs. serial BP-OSD | `python audit_tables_v2.py gnn_vs_bposd` | [`results/tables_v2/gnn_vs_bposd.json`](../results/tables_v2/gnn_vs_bposd.json) |
| Table 6 and the failure-set text (Sec. 5.4) | `python audit_failure_sets.py` | [`results/failure_sets/`](../results/failure_sets/) |
| Table 7, Fig. 6, code-capacity oracle gap | `python audit_tables_v2.py oracle` | [`results/tables_v2/oracle_mean.json`](../results/tables_v2/oracle_mean.json) |
| Table 8, Fig. 7, drift (Sec. 5.6) | no script in this repository | values recorded in [`figures/gen_F3_drift_adaptation.py`](../figures/gen_F3_drift_adaptation.py) |
| Table 9, Fig. 8, circuit-level oracle gap | `python audit_circuit_v2.py --oracle ...`, two runs (see [`figures/README.md`](../figures/README.md)) | [`results/circuit_v2/`](../results/circuit_v2/) |
| Figs. 1–3, diagrams | TikZ sources in [`docs/`](./) | |

`figures/make_all_figures.sh` redraws all eight figures from the stored
results (see [`figures/README.md`](../figures/README.md)).

## Reading the numbers off the stored output

- **Table 3.** `results.<config>`: `ler`, `ci` (95% Wilson), `failures`.
- **Sec. 5.1 / Fig. 4.** `results.<config>.ler`; the paired counts 1138/3
  and 118/0 are `mcnemar.<a>_vs_<b>.n10/n01`.
- **Sec. 5.2.** `identical_shots` per decoder (2,000 shots each); the
  PyTorch decoder (`torch_minsum_flooding10`) changes 18 decisions on
  [[288,12,18]], sum-product (`ldpc_ps_serial100`) changes 25 and 43.
- **Table 4.** BP rows: `baselines.json` (`bp10`, `bp100`). GNN rows:
  `summary.json` (`gnn_ler_mean`, `gnn_ler_std`, `reduction_mean`,
  `reduction_std`, `n10`, `n01`, `p_max`). Serial BP-OSD row:
  `baselines.json` (`bposd_serial_cs10`), the same decoder and failures as in
  `results/failure_sets/`. The non-converged counts in the text (1,740 for
  BP, 838–975 with the GNN) are `bp_syndrome_mismatch` and
  `gnn_syndrome_mismatch` in `seed*.json`. The checkpoints are
  `seed*_best.pt`.
- **Sec. 5.3.** Gradient audit: `summary.ratio_min/max` (0.63–2.19) and
  `summary.mean_delta_min/max` (0.54–1.01), one entry per checkpoint under
  `trained`. Capacity test: `bp_events` (105) and the epoch-5 entry of
  `epoch_results` (93 failures, n10/n01 = 15/3, p = 0.0095).
- **Table 5.** `points.<p>.bposd.ler` and `points.<p>.gnn.<seed>.ler`; a
  seed counts as worse when `mcnemar_bposd_vs_gnn.p < 0.05` and `n01 > n10`.
- **Table 6.** `test_p04.json`: `describe` (failures, non-converged shots),
  `overlaps` (`p_a_given_b` gives the 93%, 100% and 87–88% rows),
  `gnn_fixes` (313–370 shots fixed; the 91–93% and 31–36% rows) and
  `gnn_osd_changes` (newly failed / rescued). The 96% at p = 0.06 is
  `overlaps["bp10|bposd"].p_a_given_b` in `test_p06.json`.
- **Table 7.** `cells.<code>_p<p>_s<sigma>`: `gap_vs_mean`, with
  `mean.failures` in parentheses and `mcnemar_mean_vs_oracle.p`.
- **Table 9.** `oracle_v4_paper_points.json` (p = 0.001, 0.003) and
  `oracle_v4_p010.json` (p = 0.01): `mean_ler`, `oracle_ler`,
  `mcnemar_mean_vs_oracle`, and `profiles_mean_worse` (38 of 40 profiles).

## Data

- [`data/headline_v2/`](../data/headline_v2/): the training (seeds 1001,
  1002), validation (2001) and test (3001, 3002) sets of Table 4, written by
  `audit_headline_v2.py --make_data`. The failure-set analysis and the
  gradient audit use the same test sets.
- [`data/big72_test_p04.npz`](../data/big72_test_p04.npz): the fixed
  syndromes of the capacity test (its first 1,000 shots).
- The other experiments draw fresh data inside the script from fixed seeds,
  which are stored in each JSON (`data_seed` / `data_seeds`).

## Changes from the reviewed draft (v2)

The tracked-changes PDF,
[`paper_v4/STAR_GNN_BB_v2_to_v4_tracked_changes.pdf`](../paper_v4/STAR_GNN_BB_v2_to_v4_tracked_changes.pdf),
shows every text change, and
[`paper_v4/build_v4_from_v2.py`](../paper_v4/build_v4_from_v2.py) applies
them to the v2 source one by one. The numbers changed for these reasons:

- **Failure criterion.** One explicit definition (above) is now used by
  every experiment.
- **Table 4.** Re-run with separately seeded training, validation and test
  data, the checkpoint chosen on the validation set, and five training
  seeds: 15.4% (one seed) became 12.5% ± 0.6%.
- **Table 3, Fig. 4.** Re-run with 100,000 identical shots per
  configuration and explicit decoder settings; the overall decomposition
  factor is about 47× (v2: about 40×).
- **Sec. 5.2.** The clamped PyTorch decoder and sum-product BP are not
  exactly invariant; the text now says so.
- **Table 5, Fig. 5.** Re-run with the five Table 4 checkpoints and 20,000
  shots per point.
- **Table 6.** New: the paired failure-set analysis.
- **Table 7, Fig. 6.** The per-qubit rates are mean-anchored,
  p_i = p·exp(σz − σ²/2), as Sec. 5.5 now states.
- **Table 9, Fig. 8.** The memory circuit now has first-round and final
  detectors. The v2 circuit lacked them, and that is why it showed no
  oracle gap; with complete detectors the gap is 55% at p = 0.01.
- **Sec. 5.3.** The gradient audit was re-run on the five Table 4
  checkpoints.
- **Removed.** The interleaved-training figure and the single-shot
  rate-estimator comparison.
- **Unchanged.** Table 8 and Fig. 7 (drift) keep the v2 values; the code
  that produced them is not in this repository.

Files used for the v2 results but not by the final paper were removed; they
remain in the git history (commit `aed361a`).
