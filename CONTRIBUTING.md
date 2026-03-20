# Contributing

## Development Setup

```bash
# Clone the repository
git clone https://github.com/andypeterson2/quantum-machine-learning.git
cd quantum-machine-learning

# Install dependencies
pip install -r requirements.txt
pip install pytest ruff

# Run the application
python -m classifiers
```

## Running Tests

```bash
# Run all tests
make test

# Or directly
python -m pytest tests/ -v
```

## Code Style

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting:

```bash
make lint
```

## Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add quantum circuit visualization to results page
fix: correct accuracy calculation for batch predictions
docs: add architecture diagram to README
test: add endpoint tests for /predict route
chore: pin dependency versions in requirements.txt
```

## Adding a New Dataset

1. Create a subpackage under `classifiers/datasets/`
2. Implement `DatasetPlugin` ABC
3. Register the plugin in `__init__.py`
4. Add model architectures implementing `BaseModel`
5. Add tests in `tests/`

See the [README](README.md#adding-a-new-dataset) for a detailed example.
