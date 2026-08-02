# causal30_v1 release candidate

## Scope

This release candidate synchronizes the public code repository with the frozen manuscript evidence. It replaces the generic random split used by the original demonstration configuration with the chronological 2019–2020/2021/2022 protocol.

## Verification

```bash
python scripts/materialize_causal30_archive.py
python scripts/verify_causal30_release.py --archive evidence/causal30_v1_release.zip
python scripts/reproduce_tables.py --archive evidence/causal30_v1_release.zip --output reproduced/tables
python scripts/reproduce_figures.py --archive evidence/causal30_v1_release.zip --output reproduced/figures
pytest -q
```

The compact verification path does not require the 10.5 GB raw imagery. A full training rerun requires the prepared graph snapshots reconstructed from the public upstream products.

## Scientific boundaries

- Test data consist of nine correlated anchors around one 2022 drainage sequence.
- Five seeds quantify optimization variability, not independent-event uncertainty.
- Physics ablations are exploratory.
- No operational, cross-glacier, causal-superiority or general-superiority claim is supported.
- The separate 125-scene rerun is not part of this evidence package.

## Release completion gate

A final immutable tag and archival DOI must be created only after CI passes and a fresh-environment verification succeeds.
