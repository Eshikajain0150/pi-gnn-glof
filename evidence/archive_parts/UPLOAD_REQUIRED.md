# Evidence archive upload gate

The compact `causal30_v1_release.zip` was locally reconstructed and verified with SHA-256:

`0fb298684852b69a3f7b03bbb6748adc59aa6d64b4436235f41e71a36a9e9aaa`

Before this pull request can be marked ready for review, the base64 archive parts must be added to this directory. CI will then materialize the ZIP, verify all internal checksums, reproduce the principal tables and figures, and run the release tests.

Do not create a release tag or archival DOI while this gate remains open.
