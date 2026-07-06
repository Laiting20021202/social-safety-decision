PYTHON ?= /usr/bin/python3.10
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python
RUN_PY := $(shell if [ -x "$(PY)" ]; then printf "%s" "$(PY)"; else command -v python3; fi)
DEMO_SCENARIO ?= 101_Spot_1_155
DATASET_SERVICE_URL ?= http://localhost:8000
CUDA_COMPOSE := docker compose -f compose.yaml -f compose.cuda.yaml --profile cuda-full
TRACKER_REPRO_OUTPUT ?= outputs/sam31_reproduction

.PHONY: setup audit install-dev doctor check-model-access download-data download-models validate-data precompute-scenario precompute-demo demo-local demo-dataset demo-cpu demo-cuda demo-jetson demo-ros2 stop logs status clean-cache clear-gpu-jobs tracker-repro tracker-smoke road-tracker-smoke vqa-smoke demo-real e2e-demo test test-unit test-integration lint typecheck experiment-smoke experiment-full render-videos report clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

audit:
	$(PY) scripts/validate_environment.py

install-dev: setup
	cd apps/web && npm install

doctor:
	bash scripts/gpu_doctor.sh

check-model-access:
	$(RUN_PY) scripts/check_model_access.py

download-data:
	$(PY) scripts/download_data.py

download-models:
	$(RUN_PY) scripts/download_models.py

precompute-scenario:
	@test -n "$(SCENARIO)" || (echo "Usage: make precompute-scenario SCENARIO=<scenario_id>" >&2; exit 2)
	$(RUN_PY) scripts/precompute_scenario.py --scenario "$(SCENARIO)" --dataset-service-url "$(DATASET_SERVICE_URL)"

precompute-demo:
	$(RUN_PY) scripts/precompute_scenario.py --scenario "$(DEMO_SCENARIO)" --dataset-service-url "$(DATASET_SERVICE_URL)"

validate-data:
	$(PY) scripts/download_data.py --list-only

demo-local:
	$(PY) -m uvicorn services.dataset_service.app:app --host 0.0.0.0 --port 8000

demo-dataset:
	docker compose --profile dataset-demo up --build

demo-cpu:
	docker compose -f compose.yaml -f compose.cpu.yaml --profile cpu-demo up --build

demo-cuda:
	$(CUDA_COMPOSE) up --build

stop:
	$(CUDA_COMPOSE) down

logs:
	$(CUDA_COMPOSE) logs -f --tail=200

status:
	bash scripts/gpu_doctor.sh
	$(CUDA_COMPOSE) ps

clean-cache:
	rm -rf outputs/hf_cache outputs/dataset_cache precomputed
	docker volume rm social-safety-amr_hf_cache social-safety-amr_dataset_cache || true

clear-gpu-jobs:
	curl -fsS -X POST http://localhost:8020/cache/empty || true

tracker-repro:
	@test -n "$(VIDEO)" || (echo "Usage: make tracker-repro VIDEO=/cache/dataset/imported_videos/files/<video_id>.mp4" >&2; exit 2)
	$(CUDA_COMPOSE) run --rm --no-deps sam3-service python scripts/reproduce_sam31_tracking.py --video "$(VIDEO)" --output-dir "$(TRACKER_REPRO_OUTPUT)"

tracker-smoke:
	@test -n "$(VIDEO)" || (echo "Usage: make tracker-smoke VIDEO=/cache/dataset/imported_videos/files/<video_id>.mp4" >&2; exit 2)
	$(CUDA_COMPOSE) run --rm --no-deps sam3-service python scripts/reproduce_sam31_tracking.py --video "$(VIDEO)" --max-frames 30 --output-dir "$(TRACKER_REPRO_OUTPUT)/smoke"

road-tracker-smoke:
	@echo "Road tracker smoke is blocked until SAM3.1 cross-frame propagation succeeds on a real continuous MP4." >&2
	@exit 2

vqa-smoke:
	@echo "Temporal VQA smoke is not implemented yet; geometry output is not treated as VQA." >&2
	@exit 2

demo-real:
	@test -n "$(VIDEO)" || (echo "Usage: make demo-real VIDEO=/absolute/path/to/real_continuous.mp4" >&2; exit 2)
	curl -fsS -X POST "$(DATASET_SERVICE_URL)/videos/import" -H "Content-Type: application/json" -d '{"path":"$(VIDEO)","dataset_name":"Local Continuous Video"}'

e2e-demo:
	RUN_FORMAL_GPU_TESTS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(RUN_PY) -m pytest -q tests/e2e tests/integration -m "gpu or model_access or slow"

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
