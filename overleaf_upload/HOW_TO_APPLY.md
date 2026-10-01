# Applying the v3 revision in Overleaf

1. Main file: open your main `.tex` in Overleaf, select all, and paste the
   contents of `STAR_GNN_BB_v3.tex` over it (or upload it and set it as the
   main document in Menu > Main document).
2. Bibliography: replace `references.bib` with the one in this folder.
3. Figures: upload these five, overwriting the old files with the same name:
   - `fig_F1_baseline_decomp.pdf`    (Fig. F1, baseline decomposition, regenerated)
   - `fig_F2_oracle_vs_distance.pdf` (Fig. F2, oracle gap, regenerated)
   - `fig_F4_phase4_ler.pdf`         (Fig. F4, GNN+OSD vs BP-OSD, regenerated)
   - `fig_F6_circuit_oracle.pdf`     (circuit-level oracle gap, new file name)
   - `fig_interleaved_pipeline.pdf`  (Fig. 2, larger fonts)
4. Delete from the project (no longer used):
   `fig_F5_interleaved_training` and `fig_F6_circuit_null`.
5. Keep your existing `fig_tannergnn_arch`, `fig_experimental_pipeline` and
   `fig_F3_drift_adaptation` (unchanged).
6. Recompile (Overleaf runs BibTeX automatically). Expect 13 pages.

`STAR_GNN_BB_v2_to_v3_tracked_changes.pdf` shows every edit against the
version with Nithin's comments (red = removed, blue = added). Its figure
boxes are placeholders; it is for reading the text changes only.
