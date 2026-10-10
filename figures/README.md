# Figures in the paper

Every figure in `paper_v4/STAR_GNN_BB_v4.tex` can be rebuilt with

```bash
bash figures/make_all_figures.sh            # writes to paper_v4/figures/
```

| Paper figure | File | Source | Data |
|---|---|---|---|
| Fig. 1 — TannerGNN architecture | `fig_tannergnn_arch.pdf` | `docs/fig_tannergnn_arch.tex` (TikZ) | — (matches `gnn_pipeline/gnn_model.py`) |
| Fig. 2 — Interleaved GNN-BP pipeline | `fig_interleaved_pipeline.pdf` | `docs/fig_interleaved_pipeline.tex` (TikZ) | — |
| Fig. 3 — Training and evaluation pipeline | `fig_experimental_pipeline.pdf` | `docs/fig_experimental_pipeline.tex` (TikZ) | — |
| Fig. 4 — Baseline decomposition, [[288,12,18]] | `fig_F1_baseline_decomp.pdf` | `figures/gen_F1_baseline_decomp.py` | `results/tables_v2/decomposition.json` |
| Fig. 5 — GNN+OSD vs. serial BP-OSD | `fig_F4_phase4_ler.pdf` | `figures/gen_F4_phase4_ler.py` | `results/tables_v2/gnn_vs_bposd.json` |
| Fig. 6 — Oracle calibration gap | `fig_F2_oracle_vs_distance.pdf` | `figures/gen_F2_oracle_vs_distance.py` | `results/tables_v2/oracle_mean.json` |
| Fig. 7 — Drift adaptation | `fig_F3_drift_adaptation.pdf` | `figures/gen_F3_drift_adaptation.py` | values hard-coded from the original drift run; no stored data (see `docs/PROVENANCE.md`) |
| Fig. 8 — Circuit-level oracle gap | `fig_F6_circuit_oracle.pdf` | `figures/gen_F6_circuit_oracle.py "oracle_v4_*.json" <out> "p"` | `results/circuit_v2/oracle_v4_*.json` |

## Where the data come from

| Data | Produced by |
|---|---|
| `results/tables_v2/decomposition.json` | `python audit_tables_v2.py decomposition` |
| `results/tables_v2/gnn_vs_bposd.json` | `python audit_tables_v2.py gnn_vs_bposd` (uses `results/headline_v2/seed*_best.pt`) |
| `results/tables_v2/oracle_mean.json` | `python audit_tables_v2.py oracle --anchor mean` |
| `results/circuit_v2/oracle_v4_paper_points.json` | `python audit_circuit_v2.py --oracle --p 0.001 0.003 --profiles 20 --shots_per_profile 500 --scaling 0.8 --anchor mean --tag _v4_paper_points` |
| `results/circuit_v2/oracle_v4_p010.json` | `python audit_circuit_v2.py --oracle --p 0.01 --profiles 40 --shots_per_profile 500 --scaling 0.8 --anchor mean --tag _v4_p010` |
| `results/headline_v2/` (Table 4 and the GNN checkpoints) | `python audit_headline_v2.py --make_data --baselines --seed 0 ... --seed 4 --summary` |

The data figures read stored results only; the commands above re-run the
simulations from scratch. `docs/PROVENANCE.md` maps every number in the paper
to its source.
