# causal30_v1 evidence archive

The files in `archive_parts/` reconstruct `causal30_v1_release.zip`, which is the frozen, checksum-protected manuscript evidence package. It contains the exact 43-anchor manifest, protocol, five-seed primary metrics, seven ablation configurations, held-out event-node and all-node predictions, figure/table inputs, source provenance, checksums, and claim ledger.

Verify without downloading the 10.5 GB raw imagery:

```bash
python scripts/materialize_causal30_archive.py
python scripts/verify_causal30_release.py --archive evidence/causal30_v1_release.zip
python scripts/reproduce_tables.py --archive evidence/causal30_v1_release.zip
python scripts/reproduce_figures.py --archive evidence/causal30_v1_release.zip
```

The separate 125-scene rerun is not part of this archive and must not be pooled with `causal30_v1`.
