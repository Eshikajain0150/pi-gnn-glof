# Manuscript method-to-code map

| Manuscript component | Implementation |
|---|---|
| Sentinel-2 CNN encoder | `src/pignn_glof/models/encoders.py::OpticalEncoder` |
| ITS_LIVE and MODIS GRUs | `src/pignn_glof/models/encoders.py::SequenceEncoder` |
| Directed physics graph | `src/pignn_glof/data/graph.py::build_graph` |
| Edge-aware message passing | `src/pignn_glof/models/layers.py::PhysicsMessagePassing` |
| Node hazard head | `src/pignn_glof/models/pignn.py::PIGNN` |
| Six-step forecast head | `src/pignn_glof/models/pignn.py::AutoregressiveTrajectoryHead` |
| Physics regularization | `src/pignn_glof/physics.py` |
| Exact modality Shapley values | `src/pignn_glof/attribution.py` |
| Region-level splitting/training | `src/pignn_glof/data/loader.py`, `training.py` |
| Physics ablation workflow | `scripts/run_physics_ablations.py` |
| Metrics and result export | `src/pignn_glof/metrics.py`, `experiment.py` |

## Graph edge attributes

For a directed candidate edge from node `i` to node `j`, the repository stores:

1. row-normalized connection weight;
2. distance divided by the configured distance scale;
3. cosine alignment between local flow and the `i -> j` direction; and
4. downhill slope divided by the configured slope scale.

The physics graph raises the affinity of close, aligned, downhill neighbors. The
plain-GNN ablation uses the same top-k machinery with distance-only affinity.

## Physics objective

The total training objective combines binary cross entropy, trajectory mean
squared error, and a weighted physics penalty. The three physics terms are:

- net divergence of the edge-weighted risk flux;
- mismatch with a standardized warm-and-low elevation driver; and
- loss of risk along aligned downhill propagation edges.

Default component weights are 0.50, 0.30, and 0.20, with an overall physics
multiplier of 0.40. These are auditable in `configs/paper.yaml`.

Test-partition conformity is reported as the arithmetic mean of the three
unweighted component losses. `scripts/run_physics_ablations.py` changes one
component weight at a time and uses the distance-only plain GNN for the
no-physics comparison.
