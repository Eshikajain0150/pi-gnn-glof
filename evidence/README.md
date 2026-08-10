# causal30_v1 evidence

The manuscript evidence is committed as individual uncompressed files. No
compressed archive is stored in this repository.

- `causal30_v1/` — frozen model evidence: the 43-anchor manifest, protocol,
  source provenance, five-seed primary metrics, seven graph/physics ablation
  configurations, held-out event-node predictions, principal figure and table
  inputs, per-file checksums and claim ledger.
- `causal30_v1_calendar/` — calendar-control extension evidence.

Integrity is established per file. Verify with:

```bash
cd evidence/causal30_v1 && sha256sum -c checksums.sha256 && cd ../..
cd evidence/causal30_v1_calendar && sha256sum -c checksums.sha256 && cd ../..
```

Reproduce the released verification, tables and figures with:

```bash
python scripts/verify_causal30_release.py --evidence evidence/causal30_v1
python scripts/reproduce_tables.py  --evidence evidence/causal30_v1
python scripts/reproduce_figures.py --evidence evidence/causal30_v1
python scripts/verify_calendar_release.py --evidence evidence/causal30_v1_calendar
pytest -q
```

The evidence does not duplicate the 10.5 GB raw imagery or the full all-node
tensors; those are reconstructed through the documented full-data workflow. The
separate 125-scene rerun is excluded and must not be pooled with `causal30_v1`.
