# Public Release Security Report

## Scope

The audit covered tracked, ignored, and untracked workspace paths; reachable source-history blobs;
the fresh public working tree and its complete reachable history; configuration, JSON, YAML, CSV,
shell, archive, image-metadata, and workflow surfaces. Scans searched for credential shapes,
authentication assignments, private keys, personal email addresses, home-directory paths, private
hosts and IP addresses, confidential venue/status wording, and generated archives.

## Tools And Findings

- Repository inspection: Git plumbing commands, `rg`, file inventories, and metadata inspection.
- Automated public scanner: `scripts/security_scan.py`, which prints categories and locations only.
- Optional scanners: gitleaks and trufflehog were not installed locally; gitleaks is configured as a
  pre-commit hook, and the repository-native scanner is enforced in CI.
- Source workspace history: 35 blobs contained absolute home paths; 7 blobs contained key-shaped
  test or credential material; 78 objects contained confidential publication material; 11 objects
  contained credential-assignment shapes; and 12 objects contained personal email addresses.

## Resolution

The source history is excluded in full. The public repository uses a new history containing only
reviewed files. Provider-specific configuration, publication packaging, raw trajectories, patches,
upstream benchmark payloads, archives, local paths, personal metadata, and credential-shaped test
fixtures were removed or replaced. The framework PNG was regenerated without embedded metadata.

Remaining release blockers: none, conditional on the final clean working-tree/history scan and
remote CI succeeding. No secret value is recorded in this report.
