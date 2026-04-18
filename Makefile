.PHONY: train test serve report lint format clean setup

# ── Setup ──────────────────────────────────────────────
setup:
	python -m venv .venv
	.venv\Scripts\pip install -r requirements.txt

# ── Training ───────────────────────────────────────────
train:
	python experiments/run_all.py

train-quick:
	python experiments/run_all.py --quick

# ── Testing ────────────────────────────────────────────
test:
	pytest tests/ -v

# ── API Server ─────────────────────────────────────────
serve:
	uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload

# ── Code Quality ───────────────────────────────────────
lint:
	ruff check src/ tests/ experiments/
	black --check src/ tests/ experiments/

format:
	ruff check --fix src/ tests/ experiments/
	black src/ tests/ experiments/

# ── Report ─────────────────────────────────────────────
report:
	@echo "Report: report.md"
	@echo "Outputs: outputs/"

# ── Docker ─────────────────────────────────────────────
docker-build:
	docker build -t self-pruning-nn .

docker-run:
	docker run -p 8000:8000 self-pruning-nn

# ── Cleanup ────────────────────────────────────────────
clean:
	rm -rf outputs/ data/ __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
