# Changelog

## [0.2.0] - 2026-03-19

### Added
- CI/CD pipeline via GitHub Actions (tests, linting, Docker build)
- Comprehensive test suite (223 tests covering all modules)
- Iris dataset plugin and model tests
- Quadratic/Polynomial layer and model architecture tests
- Model persistence save/load/validation tests
- Plugin registry discovery tests
- Advanced training config tests (validation, regularization, distillation)
- Advanced route tests (export, disk ops, ablation, Iris routes)
- `docs/` directory with architecture, models, and API reference
- MIT License
- Makefile with run/test/lint/clean/docker targets
- `pyproject.toml` with project metadata and tool configuration
- `.dockerignore` to reduce Docker build context
- Mermaid architecture diagram in README
- CI badge, Python badge, and license badge in README
- Docker quick start section in README
- Tech stack section in README
- CHANGELOG.md

### Changed
- Expanded `.gitignore` to cover Python, IDE, OS, Docker, ML, and test artifacts
- Pinned all dependencies to exact versions in `requirements.txt`
- Fixed broken HTML template tests (frontend moved to website repo)

## [0.1.0] - Initial Release

### Added
- Classical CNN, Linear, and SVM classifiers for MNIST
- Quadratic and Polynomial feature-expansion models
- Optional Qiskit quantum-hybrid models (CNN + Linear)
- Iris dataset plugin with Linear, SVM, and PennyLane QVC models
- Plugin architecture for zero-config dataset addition
- Flask API with SSE streaming for training progress
- Interactive web frontend with canvas drawing
- Model persistence (save/load .pt checkpoints)
- Early stopping with validation monitoring
- Knowledge distillation
- Ensemble evaluation (majority-vote with logit tie-breaking)
- Per-layer ablation studies
- Docker containerization
- HTTPS support with self-signed dev certs
