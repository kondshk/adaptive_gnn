# GNN-Augmented BP Architectures for QLDPC Decoding

Code, data and results for the paper *GNN-Augmented BP Architectures for QLDPC
Decoding* (Konduru, Chedalla, Raveendran). An earlier version of this work was
presented as a poster at QEC 2026 (8th International Quantum Error Correction
Conference, Santa Barbara, CA, June 2026; submission #195).

## Authors

**Jothiradithya (Sai) Konduru** · Paradise Valley High School
**Anish Chedalla** · Paradise Valley High School
**Dr. Nithin Raveendran** · University of Arizona (STAR Lab / QEC Labs) · nithin@arizona.edu

## What the paper shows

On bivariate bicycle (BB) codes under $Z$-biased noise ($\eta=20$):

- **A GNN gain over its training decoder.** A Tanner-graph GNN trained through
  10-iteration flooding BP reduces that decoder's logical error rate on
  [[72,12,6]] by 12.5% ± 0.6% at p = 0.04 and 10.6% ± 0.7% at p = 0.06 (five
  training seeds, leakage-free held-out data).
- **The gain does not carry over to a stronger decoder.** The same corrections
  do not improve serial min-sum BP-OSD; a paired failure-set analysis shows the
  two decoders' failure sets are nested and the GNN mostly repairs shots BP-OSD
  already decodes.
- **Decoder configuration matters.** On [[288,12,18]], BP schedule and OSD
  post-processing alone change the logical error rate by about 47×.
- **Per-qubit calibration is informative; a global noise scalar is not.**
  Uniform rescaling of the channel LLRs cannot change min-sum BP's decisions,
  whereas revealing true per-qubit error rates reduces BP-OSD's logical error
  rate by 65–88% at code capacity and by 55% under circuit-level noise.

## Repository layout

| Path | Contents |
|---|---|
| `paper_v4/` | Paper source (`STAR_GNN_BB_v4.tex`, `references.bib`, `figures/`), the tracked-changes PDF against the reviewed draft, and `build_v4_from_v2.py`, which applies every revision to that draft |
| `gnn_pipeline/` | TannerGNN model, min-sum BP, decoding-failure criterion, data generation, DEM decoder, training (`train_unified.py`) |
| `astra_stim/` | Stim memory circuits for CSS codes (with first-round and final detectors) and biased / per-qubit circuit noise |
| `codes/` | Bivariate bicycle code construction and the code registry |
| `audit_*.py` | The scripts behind the paper's tables and figures (below) |
| `results/` | Stored per-shot outcomes and JSON summaries read by the tables and figures |
| `data/` | Datasets used by the headline experiment and the capacity / gradient diagnostics |
| `figures/` | Figure scripts and `make_all_figures.sh` |
| `docs/` | TikZ sources of Figs. 1–3 and `PROVENANCE.md`, which maps each reported number to its source |
| `tests/` | Unit tests |

## Reproducing the paper

Each script writes its results to `results/`, including the per-shot failure
vectors, so every number can be recomputed from the raw outcomes.
[`docs/PROVENANCE.md`](docs/PROVENANCE.md) shows where each number in the
paper sits in those files. Run the commands from the repository root.

| Paper item | Script and command | Results |
|---|---|---|
| Table 3, Fig. 4, invariance checks (Sec. 5.2) | [`audit_tables_v2.py`](audit_tables_v2.py): `bposd_configs`, `decomposition`, `invariance` | [`results/tables_v2/`](results/tables_v2/) |
| Table 4 (headline, five seeds); Table 2 configuration | [`audit_headline_v2.py`](audit_headline_v2.py): `--make_data --baselines --seed 0 --seed 1 --seed 2 --seed 3 --seed 4 --summary` | [`results/headline_v2/`](results/headline_v2/) |
| Gradient audit and capacity test (Sec. 5.3) | [`audit_gradient.py`](audit_gradient.py), [`audit_capacity.py`](audit_capacity.py) | [`results/diagnostics/`](results/diagnostics/) |
| Table 5, Fig. 5 | [`audit_tables_v2.py`](audit_tables_v2.py): `gnn_vs_bposd` | [`results/tables_v2/gnn_vs_bposd.json`](results/tables_v2/gnn_vs_bposd.json) |
| Table 6 (paired failure sets) | [`audit_failure_sets.py`](audit_failure_sets.py) | [`results/failure_sets/`](results/failure_sets/) |
| Table 7, Fig. 6 (oracle gap) | [`audit_tables_v2.py`](audit_tables_v2.py): `oracle` | [`results/tables_v2/oracle_mean.json`](results/tables_v2/oracle_mean.json) |
| Table 9, Fig. 8 (circuit level) | [`audit_circuit_v2.py`](audit_circuit_v2.py): `--oracle`, two runs listed in [`figures/README.md`](figures/README.md) | [`results/circuit_v2/`](results/circuit_v2/) |
| All figures | [`figures/make_all_figures.sh`](figures/make_all_figures.sh) | [`paper_v4/figures/`](paper_v4/figures/) |

The five GNN checkpoints of Table 4 are stored in `results/headline_v2/`;
Tables 5 and 6 and the gradient audit load them.

The drift experiment (Sec. 5.6, Table 8, Fig. 7) reports values from an
earlier run whose code is not part of this repository; Fig. 7 is drawn from
those recorded values.

## Setup

```bash
pip install -e .            # numpy, scipy, stim, torch, torch-geometric, ldpc, matplotlib
pytest tests                # unit tests
```

Figures 1–3 need `pdflatex` with TikZ. All experiments run on CPU.

## Acknowledgments

We thank Dr. Nithin Raveendran and the UArizona STAR Lab / QEC Labs for their
guidance and support. The QEC 2026 poster was sponsored by Google Quantum AI.

## Contact

**Jothiradithya (Sai) Konduru** — sai.konduru10@gmail.com
**Anish Chedalla** — anishchedalla@gmail.com
