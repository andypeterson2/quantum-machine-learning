.PHONY: run test lint clean docker

run:
	python -m classifiers

test:
	python -m pytest tests/ -v

lint:
	ruff check classifiers/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache/ .coverage htmlcov/

docker:
	docker compose up --build
