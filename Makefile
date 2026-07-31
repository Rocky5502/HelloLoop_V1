PYTHON ?= python

.PHONY: setup test lint smoke reproduce reproduce-clean validate security check

setup:
	./scripts/setup_environment.sh

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

smoke:
	$(PYTHON) scripts/run_smoke_test.py

reproduce:
	$(PYTHON) scripts/reproduce_analysis.py

reproduce-clean:
	$(PYTHON) scripts/reproduce_analysis.py --clean

validate:
	$(PYTHON) scripts/validate_artifacts.py

security:
	$(PYTHON) scripts/security_scan.py

check: test lint smoke reproduce-clean validate security
