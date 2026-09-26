# Paper revision files

Paste-ready LaTeX for revising `STAR_GNN_BB_v2.tex`, one file per step of the
review. Every number comes from a committed script and stored per-shot
outcomes; `docs/PROVENANCE.md` maps each one to its source.

| File | Step | Replaces |
|---|---|---|
| `logical_failure_definition.tex` | 1 | new "Logical failure" paragraph in the metrics section |
| `headline_v2.tex` | 2 | headline protocol, loss, Table 4 and result text |
| `circuit_v2.tex` | 3 | circuit-level section, Table 8, Fig. F6 caption |
| `tables_v2.tex` | 4 | Tables 3, 5, 6; F1, invariance, oracle-noise and capacity-test text |
| `failure_sets.tex` | 5 | new failure-set table and text; contributions bullet |
| `writing_v3.tex` | 6 | title, abstract, contributions, Table 1, noise models, Table 2 and loss, discussion, limitations, conclusion |
| `bib_additions.bib` | 6 | entries to add to or replace in the Overleaf bibliography |

Where two files touch the same passage, the later step wins (for example the
Table 2 and loss text in `writing_v3.tex` supersedes the protocol paragraph in
`headline_v2.tex`).
