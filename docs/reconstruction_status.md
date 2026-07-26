# Genuine reconstruction checkpoint — 26 July 2026

This page records the verified stopping point of the full GLOFNet reconstruction so that work resumes without treating synthetic data or manuscript values as reconstructed evidence.

## Completed and verified

- Public code repository, README, documentation, paper configuration, tests, and synthetic quick test are available.
- The synthetic quick test is a software check only and is not evidence for manuscript Tables 3–6.
- The public GLOFNet source folder and its linked source-specific folders were inspected.
- The released MODIS table was downloaded and audited.
- Official MODIS and ITS_LIVE recovery/audit workflows were prepared separately from model training.
- The linked Sentinel-2 imagery folder was identified as:
  `https://drive.google.com/drive/folders/15XLo8OANjsXmCToLA_9JnQjmWQhaoIMK`
- A partial optical probe successfully downloaded genuine dated GeoTIFFs, but the ordinary strict `gdown --folder` call stopped at Google Drive's 50-file folder-listing limit.
- Scientific PI-GNN training has **not** started on a reconstructed full multimodal manifest.

## Approved continuation

The author approved the approximately 10.5 GB imagery download and full reconstruction/rerun.

## Immediate next stage

Run the Stage 2C optical recovery and scientific-readiness audit in a fresh Kaggle session with Internet enabled. The Stage 2C workflow uses `gdown.download_folder(..., remaining_ok=True)`, preserves existing downloads, inventories every recovered file, records SHA-256 hashes, inspects raster metadata and radiometry, and builds a scene-by-band availability matrix.

Expected audit output:

`/kaggle/working/glofnet_stage2c_optical_audit.zip`

This stage must finish before optical conversion, multimodal synchronization, leakage-safe labels/splits, graph snapshot generation, or model training.

## Non-negotiable evidence rule

Do not report reconstructed PI-GNN metrics until all of the following exist and pass validation:

1. Complete optical inventory and scene coverage audit.
2. Source-pinned velocity and thermal sequences on a defensible common timeline.
3. Explicit observed/augmented/proxy label provenance.
4. Region-held-out split manifest with no leakage.
5. Schema-valid `.npz` graph snapshots and immutable manifest hashes.
6. Fresh training, comparison, ablation, forecast, calibration, and exact Shapley outputs.
7. Table and figure values regenerated directly from retained run artifacts.

Until then, values printed in the manuscript remain claims awaiting exact-manifest rerun verification.
