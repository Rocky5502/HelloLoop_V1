# HelloLoop Public Release Plan

## Release Boundary

This repository is assembled as a clean public history from an explicit allowlist. The private
research repository and its history are not imported. Frozen numerical results are copied only
from the audited result set and are validated against machine-readable source tables.

## Source Repository Map

| Source area | Purpose | Public treatment |
| --- | --- | --- |
| `src/helloloop/` | Controllers, uncertainty proxies, adapters, logging, evaluation, and analysis | Publish after path, credential, and terminology scans |
| `tests/` | Unit and offline integration tests | Publish safe offline tests; exclude tests tied to confidential packaging |
| `configs/` | Loop, model, benchmark, and runtime configuration | Publish loop and offline configuration; sanitize provider routes and credentials |
| `manifests/` | Task identifiers and provenance locks | Publish identifier-only and provenance-safe manifests |
| `results/empirical/` | Frozen processed analysis outputs | Publish allowlisted analysis-ready tables only |
| `results/final_audit/` | Audited results and corrected diagnostics | Publish selected numerical tables and figures after terminology and identity scans |
| `results/runs/` and evaluator logs | Raw execution and provider artifacts | Exclude; publish redacted processed records and schemas instead |
| Manuscript and submission packages | Scientific writing and confidential workflow material | Exclude completely |
| Hosted workflow evidence | Runner, request, account, and infrastructure metadata | Exclude completely |

## Files Safe To Publish

- Original HelloLoop Python source required for offline use and reproducibility.
- Offline unit tests, synthetic fixtures, and schema validation.
- L0-L3 policy configurations and deterministic thresholds.
- Audited run registry fields needed for analysis after redaction.
- Processed uncertainty, cost, outcome, calibration, and sensitivity tables.
- Task identifiers and hashes where upstream redistribution is not required.
- Corrected publication figures and their source data.
- Documentation, contribution guidance, security policy, and CI configuration.

## Redactions Required

- Remove absolute local paths and machine-specific directories.
- Remove provider request identifiers, account metadata, private routes, and authentication data.
- Replace any credential-like test fixture with an unmistakably synthetic placeholder.
- Remove personal contact information not explicitly approved for public citation metadata.
- Strip image metadata and record aggregate trajectory redaction counts without revealing values.

## Excluded Material

- Anonymous or review-stage manuscript files and archives.
- Confidential correspondence, review discussions, scores, identifiers, and deadlines.
- Raw provider responses, billing records, headers, request IDs, and hosted runner logs.
- Hidden benchmark tests, gold patches, task hints, and restricted upstream dataset content.
- Generated binary archives already represented by source files and checksums.
- The source repository's Git history because it contains private metadata and excluded material.

## Third-Party Dependencies And Licenses

- HelloLoop original code: existing MIT license.
- mini-SWE-agent: installed as a pinned dependency; no upstream source vendored.
- SWE-bench: installed or downloaded by the user; hidden tests and gold patches are not included.
- OpenAI-compatible client, scientific Python, and testing tools: installed from package indexes;
  versions are bounded in `pyproject.toml` and documented in third-party notices.
- Benchmark task content is represented by identifiers and provenance metadata unless upstream
  redistribution terms are explicitly verified.

## Large-File Handling

Normal Git history will contain only lightweight source, documentation, CSV/JSON data, and
optimized figures. Raw trajectories, evaluator bundles, and generated release archives remain
outside Git. A checksum manifest records every published artifact. Future large redacted bundles
should be attached to a GitHub Release rather than committed directly.

## Planned Repository Structure

- Root project metadata, governance, security, citation, and environment files.
- `src/helloloop/` for reusable package code.
- `configs/` for public policy and offline experiment configuration.
- `scripts/` for setup, smoke testing, data download, reproduction, validation, and security scan.
- `tests/` for API-free unit and integration coverage.
- `data/` for manifests, schemas, processed records, and redaction documentation.
- `results/` for locked processed results, regenerated outputs, tables, figures, and checksums.
- `docs/` for architecture, reproduction, data dictionary, limitations, and inventories.
- `.github/` for offline CI, issue templates, and pull-request checks.

## Acceptance Tests And Scans

1. Verify the target remote and clean release history.
2. Install from the declared Python environment in a fresh virtual environment.
3. Run unit tests, Ruff, offline smoke tests, schema checks, and artifact validation.
4. Reproduce analysis without network access or API credentials.
5. Compare locked numerical claims with audited source tables.
6. Validate all checksums and ensure no large non-LFS blob is present.
7. Scan the complete clean Git history and working tree for secrets, identities, private paths,
   confidential terminology, and restricted content.
8. Review the complete staged diff deliberately before any push.
9. Push only after every release gate reports PASS.
