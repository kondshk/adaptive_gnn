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

- **A GNN gain over the decoder it is developed with.** A Tanner-graph GNN
  trained with a per-qubit focal loss, and whose checkpoint is selected with
  10-iteration flooding BP, reduces that decoder's logical error rate on
  [[72,12,6]] by 12.5% ± 0.6% at p = 0.04 and 10.6% ± 0.7% at p = 0.06 (five
  training seeds, held-out data). The whole gain is in the Z component.
- **The gain does not carry over to a stronger decoder.** The same
  corrections never improve serial min-sum BP-OSD. The two decoders' failure
  sets overlap strongly (93% of BP-OSD failures are BP failures), and the GNN
  mostly repairs shots on which BP converges slowly and which BP-OSD already
  decodes.
- **A failure the GNN cannot repair.** Weight-three Z errors that cover half
  of a weight-six stabilizer defeat BP, 100-iteration BP and GNN-BP on every
  test shot, and BP-OSD on none; the bitwise training loss cannot teach the
  network to break that tie.
- **Decoder configuration matters.** On [[288,12,18]], BP schedule and OSD
  post-processing alone change the logical error rate by about 47×.
- **Per-qubit calibration is informative; a global noise scalar is not.**
  Uniform rescaling of the channel LLRs cannot change min-sum BP's decisions,
  whereas revealing true per-qubit error rates reduces BP-OSD's logical error
  rate by 11–88% at code capacity and by 55% under circuit-level noise.
- **Tracking drift needs no learning here.** In a fully specified synthetic
  drift experiment, a stale per-qubit calibration loses its value as soon as
  the rates move, and a simple frequency estimator of the rates beats a
  learned GNN estimator on every stream at every volatility.

## Repository layout

| Path | Contents |
|---|---|
| `paper/` | Manuscript (`STAR_GNN_BB_v5.tex`, `references.bib`, `figures/`), the reviewed version it was built from (`reviewed/`), `build_v5.py`, which applies every revision to that version, and the tracked-changes PDF |
| `audit_*.py` | One script per experiment (table below) |
| `make_tables.py` | Rebuilds every results table from the stored per-shot outcomes and checks it against the manuscript |
| `reproduce_all.sh` | Re-runs every experiment, checks the tables and redraws the figures |
| `results/` | Per-shot outcomes and JSON summaries for every table and figure, and the trained GNN checkpoints |
| `data/` | Datasets of the headline experiment and the capacity test |
| `gnn_pipeline/` | TannerGNN model, min-sum BP, failure criterion, data generation, DEM decoder |
| `astra_stim/` | Stim memory circuits and the biased / per-qubit circuit noise |
| `codes/` | Bivariate bicycle code construction |
| `figures/` | Figure scripts and `make_all_figures.sh` |
| `docs/` | TikZ sources of the diagrams, `PROVENANCE.md` (where each number comes from) and `REVISION_v5.md` (what changed for the review) |
| `tests/` | Unit tests |

## Reproducing the paper

Each script writes its results to `results/`, including the per-shot failure
vectors. Run the commands from the repository root;
[`reproduce_all.sh`](reproduce_all.sh) runs all of them in order and then
checks every results table of the manuscript:

```bash
bash reproduce_all.sh --stored-checkpoints   # exact: evaluates the committed GNN checkpoints
bash reproduce_all.sh                        # full: also retrains every GNN
```

GNN training on CPU is not bit-reproducible across thread configurations, so
the full run reproduces the GNN results only up to seed-to-seed variation;
with `--stored-checkpoints` every table is reproduced exactly.

| Paper item | Script | Results |
|---|---|---|
| Tables 2 and 4 (five GNN seeds) | [`audit_headline_v2.py`](audit_headline_v2.py) `--make_data --baselines --seed 0 ... --seed 4 --summary` | [`results/headline_v2/`](results/headline_v2/) |
| Table 3, Fig. 3, invariance checks (Sec. 5.1–5.2) | [`audit_tables_v2.py`](audit_tables_v2.py) `bposd_configs`, `decomposition`, `invariance` | [`results/tables_v2/`](results/tables_v2/) |
| Table 5 (X/Z components) | [`audit_components.py`](audit_components.py) | [`results/components/`](results/components/) |
| Gradient audit, capacity test (Sec. 5.3) | [`audit_gradient.py`](audit_gradient.py), [`audit_capacity.py`](audit_capacity.py) | [`results/diagnostics/`](results/diagnostics/) |
| Table 6, Fig. 4 (GNN+OSD vs. BP-OSD) | [`audit_tables_v2.py`](audit_tables_v2.py) `gnn_vs_bposd` | [`results/tables_v2/`](results/tables_v2/) |
| Table 7 (paired failure sets) | [`audit_failure_sets.py`](audit_failure_sets.py) | [`results/failure_sets/`](results/failure_sets/) |
| Table 8 (failure classes) | [`audit_mechanism.py`](audit_mechanism.py) | [`results/mechanism/`](results/mechanism/) |
| Table 9, Fig. 5 (oracle gap) | [`audit_tables_v2.py`](audit_tables_v2.py) `oracle` | [`results/tables_v2/`](results/tables_v2/) |
| Table 10, Fig. 6 (drift) | [`audit_drift.py`](audit_drift.py) `train`, `eval` | [`results/drift/`](results/drift/) |
| Tables 11–12, Fig. 7 (circuit level) | [`audit_circuit_v2.py`](audit_circuit_v2.py) `--oracle` (two runs, see `reproduce_all.sh`) | [`results/circuit_v2/`](results/circuit_v2/) |
| All results tables | [`make_tables.py`](make_tables.py) (`--check paper/STAR_GNN_BB_v5.tex`) | — |
| All figures | [`figures/make_all_figures.sh`](figures/make_all_figures.sh) | [`paper/figures/`](paper/figures/) |

[`docs/PROVENANCE.md`](docs/PROVENANCE.md) maps every table, figure and
in-text number to its script, stored output and code path.

## Setup

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements-lock.txt   # exact versions used for the paper
pytest tests                           # unit tests
```

The diagrams need `pdflatex` with TikZ. All experiments run on CPU.

## Acknowledgments

We thank Dr. Nithin Raveendran and the UArizona STAR Lab / QEC Labs for their
guidance and support. The QEC 2026 poster was sponsored by Google Quantum AI.

## Contact

**Jothiradithya (Sai) Konduru** — sai.konduru10@gmail.com
**Anish Chedalla** — anishchedalla@gmail.com
