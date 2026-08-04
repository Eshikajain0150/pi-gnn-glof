# Final publication gate

The scientific files and deterministic calendar results are complete. The
committed release archive deliberately does **not** contain its own full commit
SHA: a Git commit cannot contain its final hash without changing that hash.
Instead, the tag commit, DOI and artifact checksum are bound after tagging in a
small external `RELEASE_ATTESTATION.json`.

## Safe publication order

1. Upload/commit this exact repository tree and push it to the existing public
   repository.
2. Create tag `v1.1.0-causal30-calendar` on that commit. Record the full
   40-character tag commit SHA.
3. In the existing Zenodo record, choose **New version**, upload the unchanged
   `pi-gnn-glof-v1.1.0-causal30-calendar.zip`, set version
   `v1.1.0-causal30-calendar`, and reserve the new version DOI while the record
   is still a draft.
4. Create a draft GitHub release for the same tag and attach the exact same ZIP.
5. Generate the attestation without modifying the repository tree:

   ```bash
   python scripts/create_release_attestation.py      --commit FULL_40_CHARACTER_COMMIT_SHA      --doi 10.5281/zenodo.NEW_ID      --asset pi-gnn-glof-v1.1.0-causal30-calendar.zip      --output RELEASE_ATTESTATION.json
   ```

6. Attach `RELEASE_ATTESTATION.json` to the GitHub release and Zenodo draft.
7. Publish the GitHub release, then publish the Zenodo version. Do not overwrite
   `v1.0.0-causal30` or DOI `10.5281/zenodo.21761944`.
8. Download the ZIP independently from GitHub and Zenodo and verify both copies
   equal the SHA-256 in `RELEASE_ATTESTATION.json`.
9. Replace the two placeholders in the pre-release manuscript with the full
   commit SHA and published version DOI, then regenerate the final DOCX/PDF.

This order avoids circular identifiers while preserving an auditable exact
binding among tag, commit, DOI and release bytes.
