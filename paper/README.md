# Revised paper (v3)

`STAR_GNN_BB_v3.tex` is `STAR_GNN_BB_v2.tex` with every file in
`docs/revision/` applied. `references.bib` is `docs/references.bib`, with
all cited keys present and DOIs added; the new or corrected entries are also
in `docs/revision/bib_additions.bib`.

`figures/` holds the figures regenerated in this revision:

| File | Status |
|---|---|
| `fig_F1_baseline_decomp.pdf` | regenerated from `results/tables_v2/decomposition.json` |
| `fig_F2_oracle_vs_distance.pdf` | regenerated from `results/tables_v2/oracle_mean.json` |
| `fig_F4_phase4_ler.pdf` | regenerated from `results/tables_v2/gnn_vs_bposd.json` |
| `fig_F6_circuit_oracle.pdf` | new; replaces `fig_F6_circuit_null` |
| `fig_interleaved_pipeline.pdf` | Figure 2, larger fonts |
| `fig_F3_drift_adaptation.pdf` | unchanged original (kept at the authors' request) |

Two figures exist only in the Overleaf project and are not here:
`fig_tannergnn_arch` and `fig_experimental_pipeline`. `fig_F5_interleaved_training` was rebuilt, did not reproduce, and has been
removed from the paper. The drift experiment (Table 7,
`fig_F3_drift_adaptation`) is kept at the authors' request with its original
numbers; it has no source code in the repo (see docs/PROVENANCE.md). To build, upload the `.tex`, the `.bib` and
the figures above into the Overleaf project next to those three.

Builds cleanly with `pdflatex`, `bibtex`, `pdflatex`, `pdflatex` (quantumarticle
class, 13 pages): no errors, no BibTeX warnings, no undefined references,
no overfull boxes.
