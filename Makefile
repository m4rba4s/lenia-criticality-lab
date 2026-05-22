.PHONY: install lint test reproduce-small clean

install:
	pip install -e .[dev]

lint:
	ruff check .

test:
	pytest tests/

reproduce-small:
	@echo "Running minimal deterministic reproduction pipeline..."
	MPLCONFIGDIR=/tmp/lenia-mplconfig MPLBACKEND=Agg \
	python scripts/run_experiment.py \
		--experiment phase_diagram \
		--resolution 1 \
		--grid-size 32 \
		--workers 1 \
		--no-lyapunov \
		--serial \
		--output ./artifacts
	@echo "Artifacts generated in ./artifacts"

clean:
	rm -rf .pytest_cache
	rm -rf build/ dist/ *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
