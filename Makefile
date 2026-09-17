.PHONY: run test lint clean docker export-web sync-web export-qsvm benchmark

# Website checkout that consumes the browser model exports (override: make sync-web WEB=...)
WEB ?= ../website

run:
	python -m classifiers

# Retrain the browser-served linear models through the real plugins/Trainer and
# write provenance-stamped weights to exports/web/ (drift-checked in CI).
export-web:
	python -m classifiers.web_export

# Copy the canonical exports into the website checkout's model directory.
sync-web:
	cp exports/web/*.json $(WEB)/public/classifiers/models/

# Derive the QSVM paper-recreation weights (closed-form, from the notebook's
# recorded solution) into exports/web/ (drift-checked in CI; ship via sync-web).
export-qsvm:
	python -m classifiers.qsvm_export

# Measure every model's accuracy (seeded, with a 95% interval) into
# exports/benchmarks.json — the numbers the MODELS.md files are held to.
# Add SLOW=--slow for the MNIST CNN-backbone models (minutes each).
benchmark:
	python tools/benchmark.py $(SLOW)

test:
	python -m pytest tests/ -v

lint:
	ruff check .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache/ .coverage htmlcov/

docker:
	docker compose up --build
