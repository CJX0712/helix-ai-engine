.PHONY: setup test run serve clean

VENV = .venv
PIP = $(VENV)/Scripts/pip
PY = $(VENV)/Scripts/python

setup:
	python -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -r requirements.lock
	$(PIP) install -e .[dev]

test:
	$(PY) -m pytest -q

run:
	$(PY) -m helix.cli.main rag

serve:
	$(PY) -m helix.cli.main serve

clean:
	rm -rf $(VENV) __pycache__ .pytest_cache
