# PI-GNN for Shisper Glacier GLOF hazard forecasting

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21761944.svg)](https://doi.org/10.5281/zenodo.21761944)

Reference implementation for **“Physics-Informed Multimodal Graph Neural Network
(PI-GNN) for Spatio-Temporal Propagation of Glacier-Lake Instability and GLOF
Hazard Forecasting: Shisper Glacier, Karakoram.”**

The code fuses Sentinel-2 reflectance patches, ITS_LIVE velocity sequences, and
MODIS thermal sequences. It constructs a directed, flow-aware graph; applies
edge-aware message passing; predicts node-level instability and a six-step risk
trajectory; and adds risk-flux-continuity, temperature–elevation, and
downstream-flow penalties during training.

## Frozen manuscript release

The manuscript-linked reproducibility release is:

- **Version:** `v1.0.0-causal30`
- **Protocol:** `CAUSAL_30DAY_GLOF_FORECAST_V1_1`
- **GitHub release:** `v1.0.0-causal30`
- **Release commit:** `0e3f1586723b85d763dcab8dd31d3740f7c23613`
- **Version-specific DOI:** [10.5281/zenodo.21761944](https://doi.org/10.5281/zenodo.21761944)

The frozen protocol uses 27 training anchors from 2019–2020, seven validation
anchors from 2021, and nine held-out anchors from 2022. The five optimization
seeds are `7, 21, 42, 84, 123`. The nine held-out anchors are successive
forecast opportunities around one correlated 2022 drainage sequence, not nine
independent events.

Verify the archived evidence without downloading the 10.5 GB raw imagery:

```bash
python -m pip install -r requirements/causal30-release.txt
python scripts/verify_causal30_release.py \
  --evidence evidence/causal30_v1
python scripts/reproduce_tables.py \
  --evidence evidence/causal30_v1
python scripts/reproduce_figures.py \
  --evidence evidence/causal30_v1
pytest -q tests/test_causal30_release.py
```

The separate 125-scene rerun is not part of `causal30_v1` and must not be pooled
with this release.

## What is included

- PI-GNN and CNN-only, temporal-GRU, and plain-GNN comparison models
- frozen chronological `causal30_v1` release configuration and evidence
- native PyTorch message passing (PyTorch Geometric is not required)
- preprocessing and validation for the documented `.npz` interchange format
- classification, segmentation-overlap, calibration, and trajectory metrics
- exact Shapley attribution over the three input modalities
- a small synthetic quick test and unit tests
- scripts for experiments, evaluation, explanations, verification, and result reproduction

## Installation

Python 3.10 or newer is required. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

GPU execution is optional. To select a particular CUDA build, install PyTorch
using the command generated at <https://pytorch.org/get-started/locally/> before
installing this project.

## Software quick test

The following command creates a tiny synthetic dataset, trains PI-GNN for two
epochs, evaluates held-out synthetic regions, and verifies Shapley efficiency:

```bash
python scripts/quick_test.py
```

A successful run ends with `QUICK TEST PASSED` and writes
`outputs/quick_test/quick_test_report.json`.

> **Important:** the quick-test data and metrics are synthetic. They verify that
> the software executes; they are not manuscript results and must not be used as
> evidence about Shisper Glacier.

Run the repository-wide automated tests with:

```bash
pytest -q
```

## Raw data and full reconstruction

The multimodal upstream data described by GLOFNet are publicly linked from the
[GLOFNet publication](https://doi.org/10.1109/ICoDT269104.2025.11360730).
Source-specific preparation notes are in [docs/data.md](docs/data.md). Raw
imagery is not duplicated because of its size and upstream licensing.

The manuscript protocol is defined in
[`configs/causal30_v1.yaml`](configs/causal30_v1.yaml). Generic development and
training utilities remain available under `scripts/`, but manuscript claims and
reported values must be tied to the frozen `causal30_v1` evidence and not to the
synthetic quick test or the separate 125-scene rerun.

## Exact modality attribution

For a trained PI-GNN checkpoint:

```bash
python scripts/explain.py \
  --manifest data/processed/manifest.jsonl \
  --checkpoint outputs/pi_gnn/best_model.pt \
  --output outputs/pi_gnn/shapley.npz
```

With three modalities, the code evaluates all eight coalitions and computes
exact Shapley values rather than a sampling approximation.

## Scientific interpretation boundaries

- The held-out unit is one correlated 2022 drainage sequence.
- Five seeds quantify optimization variability, not independent-event uncertainty.
- Physics ablations are exploratory.
- The release does not establish operational forecasting performance.
- Cross-glacier generalization is not established.
- No claim of physical causality or general PI-GNN superiority is supported.

## Repository layout

```text
configs/                 experiment and frozen release configurations
docs/                    data schema and method-to-code map
evidence/                frozen causal30_v1 evidence as individual files
scripts/                 training, evaluation, verification, and reproduction
src/pignn_glof/          model, graph, physics, training, and metrics
tests/                   repository and release-specific tests
```

## License and citation

The source code is released under the [BSD 3-Clause License](LICENSE).
Machine-readable citation metadata are provided in [`CITATION.cff`](CITATION.cff).
For the manuscript-linked version, cite:

> Kaushik, P., Kukreja, V., & Jain, E. (2026). *PI-GNN for Shisper Glacier GLOF
> hazard forecasting* (Version v1.0.0-causal30) [Computer software]. Zenodo.
> https://doi.org/10.5281/zenodo.21761944

For a security issue, do not open a public issue containing sensitive data.
Contact the corresponding author, Vinay Kukreja, through the affiliation listed
in the manuscript.
