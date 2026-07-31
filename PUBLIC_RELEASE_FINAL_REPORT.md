# Public Release Final Report

## 1. Target

- Repository: `https://github.com/Rocky5502/HelloLoop_V1`
- Branch: `public-release-v1`
- Draft pull request: `https://github.com/Rocky5502/HelloLoop_V1/pull/1`
- Remote release-tree commit: `0110d655f4e37bc5c4b7c1fb760f832368fe0a9f`

## 2. Local Commits

- `59922d7` - secure public code structure
- `015980f` - framework and reproducibility documentation
- `03a7b55` - redacted data, figures, tables, schemas, and checksums
- `7cad509` - offline CI, reproduction, validation, and security tooling

The remote repository was empty. A minimal bootstrap was created on `main`, then the reviewed local
snapshot was uploaded through repository-scoped GitHub Git-object operations and committed on the
release branch. No force update or direct merge was performed.

## 3. Published And Excluded Material

Published: controller and uncertainty code, agent/evaluator interfaces, experiment sharding and
recovery, 316-row redacted run data, grouped signal predictions, identifier-only manifests,
analysis-ready tables, corrected figures, the framework image, schemas, documentation, and CI.

Excluded: source Git history, provider-specific endpoints, credentials, personal metadata, raw
prompts, patches, unredacted trajectories, provider request identifiers, hidden tests, gold
patches, upstream repositories and evaluator images, confidential correspondence, and private
research packaging. Credential-shaped test fixtures were reconstructed at runtime from inert
string fragments.

## 4. Verification

- Unit/offline integration tests: 105 passed, 10 optional dependency skips, 0 failed.
- Lint/format: Ruff passed; Git whitespace checks passed.
- Offline smoke: passed with 96 deterministic synthetic runs.
- Analysis reproduction: passed without a model key or API call.
- Artifact validation: 316 rows and 27 frozen checksums validated.
- Security: 0 findings in 133 public working-tree files and 158 reachable local Git objects.

Reproduction recovered 316 registered runs, 291 evaluated runs, 289 patches, 2 evaluated
abstentions, 141 resolutions, 10 agent failures, 11 infrastructure failures, 4 budget stops,
budget accounting of $132.2048925, pre-model reservations of $44.5984975, and token-derived
measured expenditure of $87.606395. Frozen Brier, ECE, and AUROC were independently recovered as
0.3520119574, 0.3172036082, and 0.4593380615; grouped full-vector AUROC was 0.4111583924.

## 5. License And Large Files

The license decision is resolved: original code is MIT licensed. Third-party components are not
vendored and retain their upstream terms; notices and pinned versions are documented. No file
requires Git LFS: the largest published file is the 340,629-byte framework PNG. Upstream benchmark
payloads are obtained separately with the explicit download helper.

## 6. Remote Status

The branch and draft pull request were verified remotely. Remote Actions were triggered by the
branch/PR publication and must finish before merge. No passing badge is displayed. The v1.0.0
release is not published because the draft PR has not been reviewed and merged.

## 7. Remaining Manual Actions

1. Review remote Actions and merge pull request 1 only if all checks pass.
2. Supply the verified public author list before replacing the generic citation author entry.
3. After merge, create the draft `v1.0.0` release with source archives, checksum manifest,
   reproduction instructions, and known limitations.
