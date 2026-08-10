# causal30_v1 claim ledger

Status: **frozen**

## Supported claims
- **C1** — A leakage-controlled 30-day Shisper benchmark used 2019-2020 for training, 2021 for validation, and 2022 for testing.
- **C2** — CNN was the strongest and most stable primary model for temporal discrimination and calibration across five initialization seeds.
  - `cnn_brier`: 0.1110 ± 0.0286
  - `cnn_pr_auc`: 1.0000 ± 0.0000
  - `full_pi_brier`: 0.4121 ± 0.2023
  - `full_pi_pr_auc`: 0.8343 ± 0.1777
- **C3** — The full composite PI-GNN was initialization-sensitive and did not consistently outperform non-physics baselines.
- **C4** — The post-hoc continuity-only diagnostic was the most balanced physics-informed configuration for F1 and event-node localization.
  - `brier`: 0.1402 ± 0.0543
  - `f1`: 0.8933 ± 0.0710
  - `pr_auc`: 0.9200 ± 0.1095
  - `rank`: 2.5500 ± 1.0518
- **C5** — The temperature-elevation-only diagnostic was highly unstable for spatial localization.
  - `brier`: 0.3476 ± 0.1480
  - `rank`: 20.5250 ± 33.0534

## Required disclosures
- Stage 11 is explicitly post-hoc.
- Mean ± sample SD refers to initialization seeds, not independent events.
- The 2022 test set is one event sequence with temporally correlated anchors.
- Unknown node-years were not treated as observed negatives.
- Claims are limited to local Shisper forecasting and localization.

## Prohibited claims
- The full PI-GNN outperformed all baselines.
- The continuity-only ablation is an independently validated final model.
- Five seeds are five independent GLOF events.
- The model estimates general glacier-wide GLOF probability.
- The results establish multi-glacier generalization.
- The model is operationally deployment-ready.
- Statistical significance was established from five dependent seeds.

## Evidence firewall
- Only `causal30_v1` is primary manuscript evidence.
- The separate 125-scene rerun is excluded from all release aggregates and cannot be used as confirmation.
- The nine held-out anchors represent one correlated 2022 drainage sequence.
- Five-seed dispersion is optimization variability, not independent-event uncertainty.
