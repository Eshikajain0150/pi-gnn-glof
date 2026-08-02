# causal30_v1 evidence archive

`causal30_v1_release.zip` is the checksum-protected compact manuscript evidence package. It contains the exact 43-anchor manifest, protocol, source provenance, five-seed primary metrics, seven graph/physics ablation configurations, held-out event-node predictions, principal figure/table inputs, internal checksums, and claim ledger.

Place the ZIP directly at `evidence/causal30_v1_release.zip`, then run:

```bash
python scripts/verify_causal30_release.py --archive evidence/causal30_v1_release.zip
python scripts/reproduce_tables.py --archive evidence/causal30_v1_release.zip
python scripts/reproduce_figures.py --archive evidence/causal30_v1_release.zip
pytest -q
```

The expected archive SHA-256 is:

`0fb298684852b69a3f7b03bbb6748adc59aa6d64b4436235f41e71a36a9e9aaa`

The compact archive does not duplicate the 10.5 GB raw imagery or full all-node tensors. Those are reconstructed through the documented full-data workflow. The separate 125-scene rerun is excluded and must not be pooled with `causal30_v1`.
