# quantum-machine-learning

![CI](https://github.com/andypeterson2/quantum-machine-learning/actions/workflows/ci.yml/badge.svg)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

A Flask API that trains and scores classical and quantum-hybrid classifiers on three datasets: MNIST handwritten digits, Iris flower species, and BB84 eavesdropper detection on simulated quantum-key-distribution sessions. Every accuracy it reports carries a 95% Wilson interval and the sample count behind it.

The service has no UI and serves no HTML (`static_folder=None`); the portfolio portal calls it over the HTTP and SSE contract below, which the live-HTTP tests in `tests/contract/` hold in place. A dataset is a subpackage under `classifiers/datasets/` that registers itself at import.

## Measured accuracy

Each number is one seeded run at the plugin's default hyper-parameters, scored on the full test split with a 95% Wilson interval, recorded in `exports/benchmarks.json` by `make benchmark` and held there by `tests/test_model_docs.py`.

### MNIST

| Architecture | Description | Accuracy (95% CI, n) |
|-------------|-------------|-----------------|
| CNN (`MNISTNet`) | 2-layer ConvNet: Conv→ReLU→Conv→ReLU→Pool→FC→FC | 98.8% (98.6-99.0%, n=10,000) |
| Linear (`LinearNet`) | Logistic regression: Flatten→Linear(784→10) | 92.1% (91.5-92.6%, n=10,000) |
| SVM (`SVMNet`) | Linear layer + multi-class hinge loss | 91.6% (91.0-92.1%, n=10,000) |
| Quadratic (`MNISTQuadraticNet`) | CNN backbone + quadratic expansion layer | 98.2% (98.0-98.5%, n=10,000) |
| Polynomial (`MNISTPolynomialNet`) | CNN backbone + polynomial (log-linear-exp) layers | 98.0% (97.7-98.3%, n=10,000) |
| Qiskit-CNN (`QiskitCNN`) | CNN backbone + Qiskit quantum circuit layer | not measured\* |
| Qiskit-Linear (`QiskitLinear`) | Linear backbone + Qiskit quantum circuit layer | not measured\* |

\* The Qiskit models need `qiskit` and `qiskit-aer`, and only appear in the dataset's model types when those are installed. They sample a circuit per prediction, so scoring one over 10,000 samples takes hours, and they are not scored here.

### Iris

| Architecture | Description | Accuracy (95% CI, n) |
|-------------|-------------|-----------------|
| Linear (`IrisLinear`) | Single linear layer: Linear(4→3) | 90.0% (74.4-96.5%, n=30) |
| SVM (`IrisSVM`) | Linear layer + multi-class hinge loss | 96.7% (83.3-99.4%, n=30) |
| QVC (`IrisQVC`) | PennyLane variational classifier, 4 qubits, 2 layers | 83.3% (66.4-92.7%, n=30)\* |

The Iris test split is 30 samples, so one sample is 3.3 points and these three intervals overlap almost entirely. This split does not separate these architectures, whatever the point estimates suggest.

\* QVC needs `pennylane` installed, and only appears in the dataset's model types when it is.

### BB84

| Architecture | Description | Accuracy (95% CI, n) |
|-------------|-------------|-----------------|
| Linear (`BB84Linear`) | Single linear layer over the session features | 95.4% (93.2-96.9%, n=500) |
| SVM (`BB84SVM`) | Linear layer + multi-class hinge loss | 95.4% (93.2-96.9%, n=500) |
| QVC (`BB84QVC`) | PennyLane variational classifier, 2 qubits, 2 layers | 90.0% (87.1-92.3%, n=500)\* |

\* QVC needs `pennylane` installed, and only appears in the dataset's model types when it is.

### Knowledge distillation

Distilling the MNIST CNN into the linear student costs accuracy at the default blend: 92.05% for the student alone against 91.09% distilled, over 3 seeds, lower on every seed. Both arms return their final weights, so the comparison is the loss and nothing else. Per-seed numbers and intervals are in `exports/distillation.json` (`make distillation`).

## Paper recreations

`notebooks/qsvm-iris/` recreates the QSVM of Yang et al. (2019) as an executed notebook. Its solved decision rule ships to the portfolio site as browser weights through `make export-qsvm` (`classifiers/qsvm_export.py`), where the free parameters are chosen on a validation slice of the fit split and the held-out split takes no part.

`exports/hardware/` holds an IBM Quantum run of the optimized 4-qubit HHL circuit from arXiv:1909.11988 (Fig. 10), submitted and fetched by `tools/hardware_run.py`, with `tests/test_hardware_run.py` holding the stored result to what the scorer computes. On `ibm_marrakesh`, 8192 shots, transpiled to depth 18 with 4 two-qubit gates at optimization level 3, the measured distribution sat 0.0127 from ideal by Jensen-Shannon divergence in bits, and the success state came out at 49.87%. Error mitigation did not help (0.0211).

The paper's own optimized depth-7 circuit on `ibmqx2` reports 0.13, but Eq. 33 computes that in nats while `classifiers/hhl.py` uses base 2, so the two are not comparable as printed. Converted to the same base, this run is about 15× closer to ideal than the paper's — hardware seven years newer, on a shallower transpilation.

The quantum packages are optional and their versions differ by where the code runs, so each artifact records the stack that produced it under `provenance.versions`. That run was qiskit 2.3.0 with qiskit-ibm-runtime 0.45.1 on Python 3.12.1; the notebook pins qiskit 2.5.2 in `notebooks/qsvm-iris/requirements.txt`, and the image installs 2.5.2 from `requirements/linux/requirements.txt`. Install the recorded versions before re-running a submission, or the comparison moves under you.

## Quick start

### Docker

The container runs gunicorn against `classifiers.wsgi:app` on `$PORT` (default 8080, which the image `EXPOSE`s) and installs the quantum extras, so every model type is available.

```bash
git clone https://github.com/andypeterson2/quantum-machine-learning.git
cd quantum-machine-learning
docker build -t qml-classifiers .
docker run --rm -p 8080:8080 qml-classifiers
```

`CLASSIFIERS_PORT=8080 docker compose up --build` does the same through Compose, where `CLASSIFIERS_PORT` picks the host port; `make docker` is that with the default. Check it with `curl http://localhost:8080/health`.

### Local

```bash
pip install -r requirements.txt
```

That pulls `flask`, `flask-cors`, `mistune`, `torch`, `torchvision`, `numpy`, `Pillow`, `scikit-learn` and `gunicorn`, pinned to what the Intel-Mac dev machine can run. The Docker image installs current torch instead. For the quantum architectures:

```bash
pip install qiskit qiskit-aer   # MNIST Qiskit-CNN / Qiskit-Linear
pip install "pennylane<0.45"    # Iris QVC (0.45+ needs numpy>=2, which torch 2.2 cannot use)
```

```bash
CLASSIFIERS_PORT=5001 python -m classifiers
```

Without `CLASSIFIERS_PORT` the dev server takes a random free port and logs it. There is no `GET /` route:

```bash
curl http://localhost:5001/health
curl http://localhost:5001/api        # discovery index of every endpoint
```

MNIST downloads to `classifiers/data/` on first run (~11 MB). Iris comes bundled with scikit-learn, and BB84 sessions are simulated from fixed seeds, so neither needs anything on disk. MNIST arrives twice, from torchvision as 28×28 tensors and from openml as the flat `mnist_784` vectors the QSVM recreation needs; CI caches both before the tests run.

### Train, evaluate, predict

```bash
curl -N -X POST http://localhost:5001/d/mnist/train \
  -H "Content-Type: application/json" \
  -d '{"model_type": "CNN", "epochs": 3, "batch_size": 64, "lr": 0.001, "name": "My CNN"}'
```

Each streaming route has a `/sync` twin that returns the final result as JSON:

```bash
curl -X POST http://localhost:5001/d/mnist/train/sync \
  -H "Content-Type: application/json" \
  -d '{"model_type": "Linear", "epochs": 1}'

curl -X POST http://localhost:5001/d/mnist/evaluate/sync

curl -X POST http://localhost:5001/d/iris/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}}'
```

Trained models export to `./models/` as `.pt` checkpoints and come back later:

```bash
curl -X POST http://localhost:5001/d/mnist/models/My%20CNN/export
curl http://localhost:5001/d/mnist/models/disk
curl -X POST http://localhost:5001/d/mnist/models/disk/<filename>/load
```

## Adding a dataset

A dataset is three files under `classifiers/datasets/<name>/`: a `plugin.py` holding a `DatasetPlugin` subclass, a `models.py` with its architectures, and an `__init__.py` that calls `register_plugin()` on import. The registry discovers it from there.

```python
from classifiers.plugin_registry import register_plugin
from .plugin import FashionMNISTPlugin

register_plugin(FashionMNISTPlugin())
```

## API reference

Machine-readable schemas are in `tests/contract/schemas/`, and `GET /api` returns a live index of every endpoint.

| Method | Path | Body | Response |
|--------|------|------|----------|
| `GET` | `/health` | — | `{status, service, version, uptime_s, clients, timestamp}` |
| `GET` | `/api` | — | Discovery index: `{service, version, endpoints, streaming}` |
| `GET` | `/api/datasets` | — | `[{name, display_name, input_type}, ...]` |
| `GET` | `/api/datasets/<name>/config` | — | `{ui_config, model_types}` |
| `GET` | `/connect` | — | SSE stream: `welcome` (`client_id`, `heartbeat_interval`), then `ping` |
| `POST` | `/pong` | `{client_id}` | `204` or `404` |
| `POST` | `/disconnect` | `{client_id}` | `204` |

Dataset-scoped routes live under `/d/<dataset>/`:

| Method | Path | Body | Response |
|--------|------|------|----------|
| `POST` | `/d/<dataset>/train` | `{model_type, epochs, batch_size, lr, name, patience?, val_gap?, teacher?, distill_weight?, distill_temperature?}` | SSE stream |
| `POST` | `/d/<dataset>/train/sync` | same as `/train` | JSON: final training result |
| `POST` | `/d/<dataset>/evaluate` | `{}` | SSE stream |
| `POST` | `/d/<dataset>/evaluate/sync` | `{}` | `{results: {name: {accuracy, accuracy_ci, num_samples, avg_loss, per_class_accuracy, num_params}}}` |
| `POST` | `/d/<dataset>/ensemble` | `{model_names: ["Model 1", "Model 2", ...]}` | JSON result |
| `POST` | `/d/<dataset>/ablation` | `{model_name: "Model 1"}` | SSE stream |
| `POST` | `/d/<dataset>/predict` | `{image: "<b64>"}` or `{features: {...}}` | `{results: {name: {prediction, confidence, probs}}}` |
| `GET` | `/d/<dataset>/models` | — | `{name: {model_type, epochs, ..., eval_result}}` |
| `GET` | `/d/<dataset>/model-info/<type>` | — | `{html}` (rendered MODELS.md section) |
| `DELETE` | `/d/<dataset>/models/<name>` | — | `{ok: true}` |
| `POST` | `/d/<dataset>/models/<name>/export` | — | `{ok: true, filename}` |
| `GET` | `/d/<dataset>/models/disk` | — | `[{filename, name, model_type, ...}]` |
| `POST` | `/d/<dataset>/models/disk/<fn>/load` | — | `{ok: true, name, model_type, ...}` |

Training, evaluation and ablation stream newline-delimited JSON:

```
data: {"type": "status", "msg": "Epoch 1/3 - loss: 0.312"}\n\n
data: {"type": "history", "epoch": 1, "batch": 50, "train_loss": 0.312, "val_accuracy": 0.95}\n\n
data: {"type": "ablation_result", "layer": "conv1", "accuracy": 0.11, "accuracy_ci": [0.10, 0.12], "num_samples": 10000, "drop": 0.87}\n\n
data: {"type": "done", "name": "CNN", "model_type": "CNN", "history": [...], ...}\n\n
data: {"type": "error", "msg": "..."}\n\n
```

Evaluation results carry `accuracy_ci` and `num_samples` beside every `accuracy`.

### Optional training fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `patience` | `int` | — | Early-stopping patience, counted in validation checks (see `val_gap`). Applies only once the best validation accuracy passes 60%, so a model near chance trains for all its epochs |
| `val_gap` | `int` | `50` | Batches between validation checks |
| `teacher` | `string` | — | Name of a trained model to distil from |
| `distill_weight` | `float` | `0.5` | Blend weight: `(1-w)*true_loss + w*distill_loss` |
| `distill_temperature` | `float` | `4.0` | Softmax temperature for the distillation term, scaled by T² |
| `seed` | `int` | — | Seeds weight initialisation, shuffling and quantum sampling, and is echoed in the result. Without it the run is not repeatable |

## Custom layers

| Layer | Module | Description |
|-------|--------|-------------|
| `Quadratic` | `layers.py` | Quadratic expansion: `y = W * concat(x^T * x, x)` |
| `Polynomial` | `layers.py` | Polynomial basis: `y = exp(W * log(\|x\| + 1))` |
| `QiskitQLayer` | `qiskit_layers.py` | Trainable parametric quantum circuit with parameter-shift gradients; takes `num_heads`, and both MNIST models run one |

## Exports

`make export-web` writes the browser-served linear weights to `exports/web/`, `make export-qsvm` writes the QSVM decision rules beside them, and `make sync-web` copies both into the portfolio site's checkout. `tests/test_web_export.py` checks the committed files against what the code produces.

## Tests

```bash
python -m pytest tests/ -v
```

`tests/contract/` holds live-HTTP contract tests that run against a booted server with JSON-schema validation and pin the API the portal depends on; CI runs them in their own job.

## Configuration

| Setting | Env var | Default | Read in |
|---------|---------|---------|---------|
| Dev server port | `CLASSIFIERS_PORT` | random free port (logged at startup) | `classifiers/__main__.py` |
| Dev server host | `CLASSIFIERS_HOST` | `127.0.0.1` | `classifiers/__main__.py` |
| Debug mode (dev server) | `CLASSIFIERS_DEBUG` | on for `python -m classifiers`; `0` in the container | `classifiers/__main__.py` |
| Container port (gunicorn) | `PORT` | `8080` | `Dockerfile` CMD |
| Allowed CORS origins | `CLASSIFIERS_CORS_ORIGINS` | `^https?://localhost(:\d+)?$,https://andypeterson.dev` (comma-separated; anchor any pattern with `^…$`) | `classifiers/server.py` |
| Max request body size | `CLASSIFIERS_MAX_CONTENT_LENGTH` | 2 MB | `classifiers/server.py` |
| Gateway origin guard | `ORIGIN_SECRET` | unset (guard inactive) | `classifiers/server.py` |
| Concurrent heavy jobs | `CLASSIFIERS_MAX_JOBS` | `2` | `classifiers/server.py` |
| Models kept per dataset | `CLASSIFIERS_MAX_MODELS` | `20` | `classifiers/model_registry.py` |
| Saved checkpoints kept | `CLASSIFIERS_MAX_CHECKPOINTS` | `50` | `classifiers/routes/model_routes.py` |
| Max predict image side | `CLASSIFIERS_MAX_IMAGE_DIM` | `4096` px | `classifiers/routes/model_routes.py` |
| Concurrent SSE clients | `CLASSIFIERS_MAX_CLIENTS` | `8` | `classifiers/routes/connection_routes.py` |
| Heartbeat stream lifetime | `CLASSIFIERS_CONNECT_LIFETIME` | `1800` s | `classifiers/routes/connection_routes.py` |
| Job stream lifetime | `CLASSIFIERS_SSE_LIFETIME` | `3600` s | `classifiers/routes/sse.py` |
| Stream keepalive interval | `CLASSIFIERS_SSE_GET_TIMEOUT` | `30` s | `classifiers/routes/sse.py` |
| Dev TLS certificate dir | `DEV_CERT_DIR` | unset (plain HTTP) | `classifiers/__main__.py` |
| Checkpoint directory | — | `./models/` | `classifiers/server.py` |
| MNIST data directory | — | `classifiers/data/` | `classifiers/datasets/mnist/plugin.py` |

## Two pinned environments

The dev machine is an Intel Mac, where PyTorch's last wheel is 2.2.2. torch 2.2 needs numpy below 2, and numpy 1 caps pennylane below 0.45. Production has none of those limits.

| | File | Stack |
|---|---|---|
| Local dev | `requirements.txt` | torch 2.2.2, numpy 1.26, pennylane 0.44 |
| Linux image | `requirements/linux/{torch,requirements}.txt` | torch 2.14, numpy 2.5, pennylane 0.45 |

Both are exact pins, so rebuilding one commit installs the same versions twice. The image installs from the linux lock and then the package with `--no-deps`, so the lock decides what lands, not `pyproject.toml`'s ranges. Regenerate it with:

```bash
docker build -t qml-lock . && docker run --rm qml-lock pip freeze
```

Dependabot updates the linux lock but leaves torch and torchvision alone in both files: they come from PyTorch's CPU index and have to move as a pair. `numpy` and `pennylane` are held back in the dev file only. `tests/test_dependency_policy.py` enforces all of it.

## License

MIT.
