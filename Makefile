.PHONY: install lint test reproduce-small clean

install:
	pip install -e .[dev]

lint:
	ruff check .

test:
	pytest tests/

reproduce-small:
	MPLCONFIGDIR=/tmp/lenia-mplconfig MPLBACKEND=Agg \
	python scripts/run_experiment.py \
		--experiment phase_diagram \
		--resolution 1 \
		--grid-size 32 \
		--workers 1 \
		--no-lyapunov \
		--serial \
		--output ./artifacts

clean:
	rm -rf .pytest_cache
	rm -rf src/__pycache__ tests/__pycache__
	rm -rf build/ dist/ *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
