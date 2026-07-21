# PI-GNN for Shisper Glacier GLOF hazard forecasting

Reference implementation for **“Physics-Informed Multimodal Graph Neural Network
(PI-GNN) for Spatio-Temporal Propagation of Glacier-Lake Instability and GLOF
Hazard Forecasting: Shisper Glacier, Karakoram.”**

The code fuses Sentinel-2 reflectance patches, ITS_LIVE velocity sequences, and
MODIS thermal sequences. It constructs a directed, flow-aware graph; applies
edge-aware message passing; predicts node-level instability and a six-step risk
trajectory; and adds risk-flux-continuity, temperature–elevation, and
downstream-flow penalties during training.

## What is included

- PI-GNN and CNN-only, temporal-GRU, and plain-GNN comparison models
- deterministic region-level train/validation/test splitting to limit leakage
- native PyTorch message passing (PyTorch Geometric is not required)
- preprocessing and validation for the documented `.npz` interchange format
- classification, segmentation-overlap, calibration, and trajectory metrics
- exact Shapley attribution over the three input modalities
- a small synthetic quick test and unit tests
- scripts for full experiments, evaluation, explanations, and result summaries

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

## Quick test

The following command creates a tiny synthetic dataset, trains PI-GNN for two
epochs, evaluates the held-out regions, and verifies Shapley efficiency:

```bash
python scripts/quick_test.py
```

A successful run ends with `QUICK TEST PASSED` and writes
`outputs/quick_test/quick_test_report.json`.

> **Important:** the quick-test data and its metrics are synthetic. They verify
> that the software works; they are not manuscript results and must not be used
> as evidence about Shisper Glacier.

Run the automated tests with:

```bash
pytest -q
```

## Data and full experiment

The multimodal source data described by GLOFNet are publicly linked from the
[GLOFNet publication](https://doi.org/10.1109/ICoDT269104.2025.11360730). Download access and the
source-specific preparation notes are in [docs/data.md](docs/data.md). The raw
imagery is not duplicated in this repository because of its size and upstream
licensing.

After arranging each graph snapshot in the documented interchange format:

```bash
python scripts/prepare_data.py --input data/standardized --output data/processed
python scripts/train.py \
  --config configs/paper.yaml \
  --manifest data/processed/manifest.jsonl \
  --variant pi_gnn \
  --output outputs/pi_gnn
```

Run every comparison model and create a summary table:

```bash
python scripts/run_experiments.py \
  --config configs/paper.yaml \
  --manifest data/processed/manifest.jsonl \
  --output outputs/comparison
python scripts/summarize_runs.py --input outputs/comparison
```

Run the full, single-term, and no-physics ablations used for the Table 4
workflow:

```bash
python scripts/run_physics_ablations.py \
  --config configs/paper.yaml \
  --manifest data/processed/manifest.jsonl \
  --output outputs/physics_ablations
```

The complete paper configuration uses three graph layers, 128 hidden units,
AdamW (`learning_rate=0.001`, `weight_decay=0.0001`), a six-step forecast, and
the physics weights 0.50/0.30/0.20. Every setting is explicit in
[`configs/paper.yaml`](configs/paper.yaml).

## Exact modality attribution

For a trained PI-GNN checkpoint:

```bash
python scripts/explain.py \
  --manifest data/processed/manifest.jsonl \
  --checkpoint outputs/pi_gnn/best_model.pt \
  --output outputs/pi_gnn/shapley.npz
```

By default, this explains all held-out test snapshots. With three modalities,
the code evaluates all eight coalitions and therefore computes exact Shapley
values rather than a sampling approximation. It reports normalized mean
absolute contributions and the percentage of nodes dominated by each modality.

## Reproducibility boundaries

The repository fixes random seeds and records the resolved configuration,
training history, best checkpoint, predictions, and metrics for each run.
Hardware, upstream data revisions, and nondeterministic GPU kernels can still
produce small numerical differences. Exact manuscript table reproduction
requires the same prepared GLOFNet snapshot manifest used by the authors; the
synthetic quick test cannot reproduce those values.

## Repository layout

```text
configs/                 paper experiment configuration
docs/                    data schema and method-to-code map
scripts/                 command-line workflows
src/pignn_glof/          model, graph, physics, training, and metrics
tests/                   unit tests
```

## License and citation

The source code is released under the [BSD 3-Clause License](LICENSE). Cite the
accompanying manuscript and the archived software release; machine-readable
author metadata are provided in [`CITATION.cff`](CITATION.cff).

For a security issue, do not open a public issue containing sensitive data.
Contact the corresponding author, Vinay Kukreja, through the affiliation listed
in the manuscript.
