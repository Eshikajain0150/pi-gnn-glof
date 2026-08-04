# GitHub + Zenodo publishing checklist

No password or access token should be shared. Use normal browser sign-in or
GitHub Desktop/CLI authentication on the repository owner's computer.

## Files to publish

- Repository tree: this folder.
- Release asset: `pi-gnn-glof-v1.1.0-causal30-calendar.zip`.
- Tag/version: `v1.1.0-causal30-calendar`.
- After tagging and DOI reservation: `RELEASE_ATTESTATION.json`.

## Mandatory identity checks

Before clicking Publish, GitHub and Zenodo must show exactly:

- title: `Leakage-Controlled Shisper Glacier 30-Day Lake-Drainage Benchmark with Calendar Controls`;
- authors in order: Pratham Kaushik, Vinay Kukreja, Eshika Jain;
- version/tag: `v1.1.0-causal30-calendar`;
- license: BSD-3-Clause;
- previous version DOI: `10.5281/zenodo.21761944`;
- the full tag commit SHA in the record description or attestation;
- identical release ZIP filename and SHA-256 on GitHub and Zenodo.

## Human-only confirmations

Publication must pause until all authors confirm final submission approval and
accept the AI-use declaration recorded in `AUTHOR_CONFIRMATION_REQUIRED.md`.
