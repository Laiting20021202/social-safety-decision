PYTHON ?= /usr/bin/python3.10
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

.PHONY: setup audit install-dev download-data download-models validate-data demo-local demo-dataset demo-cpu demo-cuda demo-jetson demo-ros2 test test-unit test-integration lint typecheck experiment-smoke experiment-full render-videos report clean status

setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

audit:
	$(PY) scripts/validate_environment.py

install-dev: setup
	cd apps/web && npm install

download-data:
	$(PY) scripts/download_data.py

download-models:
	$(PY) scripts/download_models.py

validate-data:
	$(PY) scripts/download_data.py --list-only

demo-local:
	$(PY) -m uvicorn services.dataset_service.app:app --host 0.0.0.0 --port 8000

demo-dataset:
	docker compose --profile dataset-demo up --build

demo-cpu:
	docker compose -f compose.yaml -f compose.cpu.yaml --profile cpu-demo up --build

demo-cuda:
	docker compose -f compose.yaml -f compose.cuda.yaml --profile cuda-full up --build

demo-jetson:
	docker compose -f compose.yaml -f compose.jetson.yaml --profile jetson build

demo-ros2:
	docker compose -f compose.yaml -f compose.ros2.yaml --profile ros2 up --build

test: test-unit test-integration

test-unit:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PY) -m pytest tests/unit

test-integration:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PY) -m pytest tests/integration

lint:
	$(PY) -m ruff check .

typecheck:
	$(PY) -m mypy packages services scripts

experiment-smoke:
	$(PY) scripts/run_smoke_experiment.py --scenarios 1 --formal false

experiment-full:
	$(PY) scripts/run_smoke_experiment.py --formal true

render-videos:
	$(PY) scripts/render_video.py

report:
	$(PY) scripts/generate_report.py

clean:
	rm -rf $(VENV) .pytest_cache .ruff_cache .mypy_cache apps/web/node_modules apps/web/dist

status:
	$(PY) scripts/validate_environment.py
