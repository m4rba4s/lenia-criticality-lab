.PHONY: install lint test reproduce-small clean

install:
	pip install -e .[dev]

lint:
	ruff check .

test:
	pytest tests/

reproduce-small:
	@echo "Running minimal deterministic reproduction pipeline..."
	python scripts/run_experiment.py --seed 42 --steps 100 --artifact-dir ./artifacts
	@echo "Artifacts generated in ./artifacts"

clean:
	rm -rf .pytest_cache
	rm -rf src/__pycache__ tests/__pycache__
	rm -rf build/ dist/ *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
