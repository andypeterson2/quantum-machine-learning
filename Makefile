.PHONY: run test lint clean docker export-web sync-web export-site

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

# Export the QSVM notebook as CSP-clean HTML into the website checkout
# (needs the notebook venv: make -C notebooks/qsvm-iris venv).
export-site:
	$(MAKE) -C notebooks/qsvm-iris export-site

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
