# Data preparation and interchange format

## Source data

The study uses three public Earth-observation streams assembled for Shisper
Glacier in GLOFNet:

1. Sentinel-2 surface-reflectance patches: B02, B03, B04, B08, B11, and B12.
2. ITS_LIVE surface-velocity observations, represented as `vx`, `vy`, and speed.
3. MODIS thermal observations, represented as a temporal temperature sequence.

The GLOFNet authors provide a public-data link in their
[preprint](https://arxiv.org/abs/2510.10546). At the time this repository was
prepared, that link resolved to:

<https://drive.google.com/drive/folders/191x2uwFRzgd2CMfqpqdVw0UrT5YZYjHN>

Users are responsible for observing the terms of the original Copernicus,
NASA, and ITS_LIVE products. Keep downloaded files outside version control.

## One snapshot per file

The model consumes one compressed NumPy file (`.npz`) per spatio-temporal graph
snapshot. Every array is `float32` and shares the first node dimension `N`:

| Array | Required shape | Meaning |
|---|---:|---|
| `optical` | `[N, 6, H, W]` | Sentinel-2 patch in the band order above |
| `velocity` | `[N, Tv, 3]` | `vx`, `vy`, and speed sequence |
| `thermal` | `[N, Tt, 1]` | MODIS temperature/anomaly sequence |
| `coords` | `[N, 2]` | projected x/y node centroids in metres |
| `elevation` | `[N]` | node elevation in metres |
| `flow` | `[N, 2]` | local horizontal flow vector |
| `labels` | `[N]` | binary instability target |
| `trajectory` | `[N, 6]` | optional six-step risk target |

Coordinates must use a projected coordinate reference system because graph
distance is evaluated in metres. Missing/invalid values must be masked or
imputed before creating a snapshot. Do not encode missing values as zero unless
zero is physically meaningful.

Each source file may have a same-stem JSON sidecar:

```json
{
  "region_id": "shisper-zone-03",
  "timestamp": "2024-07-15"
}
```

`region_id` determines the leakage-safe split. Snapshots with the same region
are kept entirely in train, validation, or test. If no sidecar exists, the
filename stem becomes the region ID, which is suitable only when every file is
an independent region.

## Preparation command

Put standardized `.npz` files and sidecars in `data/standardized`, then run:

```bash
python scripts/prepare_data.py \
  --input data/standardized \
  --output data/processed
```

The command validates the required arrays, normalizes reflectance independently
by band, copies the prepared snapshots, and creates `manifest.jsonl`. Pass
`--keep-optical-scale` if reflectance is already normalized with a fixed
training-set transform.

Reusable helpers in `pignn_glof.data.preprocess` also provide velocity-magnitude
calculation, monthly thermal anomalies, and short-gap temporal interpolation.

## Label provenance

GLOF observations are rare. Store the provenance of observed, augmented, and
proxy labels in the data catalogue used to produce the snapshots. The model
does not silently synthesize labels. The only label-generating routine in this
repository is `generate_synthetic_dataset`, and its outputs are explicitly
marked `synthetic: true` in the manifest.
