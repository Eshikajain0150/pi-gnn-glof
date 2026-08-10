# causal30_v1 + calendar-control claim ledger

Status: **release candidate; scientific values frozen, publication identifiers pending**

## Supported claims

- **C1 — Frozen causal benchmark.** The benchmark uses 27 training anchors from
  2019-2020, seven validation anchors from 2021 and nine held-out anchors from
  2022, with 14 positive and 29 negative retained anchors.
- **C2 — Learned-model comparison.** Among the four learned primary models, the
  optical CNN had the strongest and most stable ranking and the lowest mean
  Brier score across five initialization seeds (`PR-AUC 1.0000 ± 0.0000`,
  `Brier 0.1110 ± 0.0286`).
- **C3 — Calendar confounding.** Scalar and harmonic day-of-year logistic
  controls also achieved `PR-AUC 1.0000` and `ROC-AUC 1.0000` on the held-out
  sequence. The test labels are seasonally ordered (five negatives followed by
  four positives), so perfect ranking is compatible with calendar position
  alone.
- **C4 — Descriptive differences beyond ranking.** The scalar and harmonic
  controls had test Brier scores of `0.2414` and `0.2988`, respectively, and
  validation-threshold F1 of `0.6667`, compared with CNN mean Brier `0.1110`
  and mean F1 `0.8342`. The calendar controls first issued true-positive alerts
  at 11 days lead, whereas responding CNN seeds first alerted at 27 days lead.
  These are descriptive single-sequence comparisons, not significance tests.
- **C5 — Full PI-GNN boundary.** Full PI-GNN did not consistently outperform
  simpler controls (`PR-AUC 0.8343 ± 0.1777`, `Brier 0.4121 ± 0.2023`).
- **C6 — Exploratory physics diagnostics.** Continuity-only regularization was
  the most balanced post-hoc reduced prior (`PR-AUC 0.9200 ± 0.1095`,
  `Brier 0.1402 ± 0.0543`, `F1 0.8933 ± 0.0710`, spatial rank
  `2.550 ± 1.052`). Temperature-elevation-only regularization was spatially
  unstable (rank `20.525 ± 33.053`).

## Required disclosures

- The nine test anchors are successive forecast opportunities around one 2022
  drainage episode, not nine independent events.
- Five seeds quantify optimization variability, not event-level uncertainty.
- The calendar controls are single deterministic fits and do not have a seed
  distribution.
- Unknown nodes were masked rather than converted to negatives.
- Physics ablations were post-hoc mechanism diagnostics.
- The separate 125-scene rerun is excluded from all release claims.

## Prohibited claims

- Perfect CNN PR-AUC proves a uniquely optical or event-specific precursor.
- The learned models outperform calendar position in held-out ranking.
- The full PI-GNN is superior to all simpler controls.
- The continuity-only ablation is an independently validated final model.
- The nine anchors or five seeds are independent GLOF events.
- Statistical significance, operational readiness, glacier-wide calibration or
  cross-glacier generalization has been established.
