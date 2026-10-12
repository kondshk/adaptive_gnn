# Figures in the paper

Every figure in `paper/STAR_GNN_BB_v5.tex` can be rebuilt with

```bash
bash figures/make_all_figures.sh            # writes to paper/figures/
```

| Paper figure | File | Source | Data |
|---|---|---|---|
| Fig. 1 — TannerGNN architecture | `fig_tannergnn_arch.pdf` | `docs/fig_tannergnn_arch.tex` (TikZ) | — (matches `gnn_pipeline/gnn_model.py`) |
| Fig. 2 — Training and evaluation pipeline | `fig_experimental_pipeline.pdf` | `docs/fig_experimental_pipeline.tex` (TikZ) | — (matches `audit_headline_v2.py`, `run_seed`) |
| Fig. 3 — Baseline decomposition, [[288,12,18]] | `fig_F1_baseline_decomp.pdf` | `figures/gen_F1_baseline_decomp.py` | `results/tables_v2/decomposition.json` |
| Fig. 4 — GNN+OSD vs. serial BP-OSD | `fig_F4_phase4_ler.pdf` | `figures/gen_F4_phase4_ler.py` | `results/tables_v2/gnn_vs_bposd.json` |
| Fig. 5 — Mean-prior vs. oracle BP-OSD | `fig_F2_oracle_gap.pdf` | `figures/gen_F2_oracle_gap.py` | `results/tables_v2/oracle_mean.json` |
| Fig. 6 — Drift experiment | `fig_F3_drift_adaptation.pdf` | `figures/gen_F3_drift_adaptation.py` | `results/drift/drift.json` |
| Fig. 7 — Circuit-level oracle gap | `fig_F6_circuit_oracle.pdf` | `figures/gen_F6_circuit_oracle.py "oracle_v4_*.json" <out> "p"` | `results/circuit_v2/oracle_v4_*.json` |
| Fig. 8 (Appendix A) — Interleaved GNN-BP pipeline | `fig_interleaved_pipeline.pdf` | `docs/fig_interleaved_pipeline.tex` (TikZ) | — |

The data figures read stored results only. The commands that produce those
results are in `reproduce_all.sh` and `docs/PROVENANCE.md`. No figure applies
smoothing; error bars are Wilson 95% intervals or, for averages over training
seeds, the range over seeds, as stated in each caption.
