# Leakage-controlled calendar-control protocol

## Scientific question

Can calendar position alone reproduce the apparent discrimination of the held-out
2022 anchor sequence?

## Frozen evidence boundary

- Protocol: `CAUSAL_30DAY_GLOF_FORECAST_V1_1` (`causal30_v1`).
- Retained anchors: 43 total; 27 train (2019-2020), 7 validation (2021), 9 test (2022).
- Models and scalers are fit on training anchors only.
- Alert thresholds are selected on validation anchors only by maximum F1, with
  the highest threshold selected under exact ties.
- The nine test anchors are evaluated once after model and threshold freezing.
- The separate 125-scene rerun is excluded.

## Controls

1. Training-prevalence constant probability.
2. Logistic regression using scalar day of year.
3. Logistic regression using standardized sine and cosine day-of-year features.

The logistic controls use `C=1`, the deterministic `liblinear` solver and
`random_state=42`. Their purpose is confounding diagnosis, not replacement of
the primary learned models.

## Required interpretation

The five negative test anchors precede four positive anchors in one monotonic
seasonal sequence. Consequently, a strictly increasing calendar score can
obtain perfect PR-AUC and ROC-AUC without observing imagery. Perfect CNN ranking
therefore cannot be attributed uniquely to optical precursors. Comparisons of
Brier score, validation-selected threshold behaviour, false alarms and alert
lead time remain descriptive and are bounded by the same single correlated
event sequence.
