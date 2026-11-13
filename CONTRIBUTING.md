# Contributing

1. Open an issue describing the bug, reproduction problem, documentation change, or research extension.
2. Create a focused branch and install `.[analysis,dev]` on Python 3.11 or newer.
3. Run `make check` before opening a pull request.
4. Document data provenance, new dependencies, behavior changes, and any changed empirical claim.

Never commit credentials, private endpoints, benchmark gold patches, hidden tests, unredacted
trajectories, or personal data. New result files must be machine-generated from documented inputs;
synthetic outputs must remain explicitly labeled as synthetic.
