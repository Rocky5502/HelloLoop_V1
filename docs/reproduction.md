# Reproduction Guide

## Analysis-only

Use Python 3.11 or newer. From a clean clone:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[analysis,dev]"
make check
```

`make reproduce` reads only `data/processed/run_level.csv` and
`results/processed/grouped_cv_predictions.csv`. It uses a deterministic seed, performs no network
operation, and writes `results/generated/summary.json`, CSV tables, and PNG figures. Re-running
without `--clean` fails if generated output exists; `make reproduce-clean` explicitly replaces it.

Expected headline outputs are 316 registered runs, 291 evaluated runs, 289 patches, 2 evaluated
abstentions, 141 resolutions, 10 agent failures, 11 infrastructure failures, and 4 budget stops.

## Synthetic smoke test

`make smoke` writes to a temporary directory and requires no credentials, Docker, or external data.
It verifies the expected 96 deterministic synthetic runs and policy comparison output.

## Full experiment

Full runs are optional and are not executed by CI. They require Docker, sufficient local disk and
memory, upstream SWE-bench data/evaluator assets, mini-SWE-agent, and a user-selected model route.

```bash
python -m pip install -e ".[agent,data,evaluation,models]"
python scripts/download_third_party_data.py --help
cp .env.example .env
helloloop model-preflight --config configs/models/example_model.yaml
helloloop run-experiment --experiment pilot --limit 1 --cost-cap-usd 1
```

Set secrets only in the shell environment or a local ignored `.env`; the package does not load or
commit `.env` automatically. Verify model identifier, context limit, and current prices before a
run. The runner enforces configured global and per-run caps, but local provider billing remains the
operator's responsibility. Official evaluation uses the separately installed upstream harness.

## Docker

Follow the upstream SWE-bench Docker instructions for the pinned revision in
`THIRD_PARTY_NOTICES.md`. Expect substantial image, disk, and memory requirements. The public CI
does not download images or run Docker.
