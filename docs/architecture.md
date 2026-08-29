# Architecture

## System Overview

The Multi-Dataset Classifier Platform is an API-only Flask service that decouples dataset-specific concerns from shared infrastructure using the plugin pattern. It serves no HTML or static assets: the browser frontend lives in the separate portfolio portal repository and is an external client of this service, consuming the HTTP + SSE contract (pinned by the live-HTTP contract tests in `tests/contract/`).

```
┌──────────────────────────────────────────────────────────┐
│           Portal Frontend  (external repo)                │
│  consumes the HTTP + SSE contract over CORS               │
└──────────────┬──────────────────────┬────────────────────┘
               │ REST API             │ SSE Streams
┌──────────────▼──────────────────────▼────────────────────┐
│                Flask API Server (this repo)               │
│  server.py → create_app() factory (static_folder=None)   │
│  routes/  → Blueprints (main, connection, dataset:       │
│             train, eval, model)                          │
└──────┬──────────┬──────────┬──────────┬──────────────────┘
       │          │          │          │
  ┌────▼───┐ ┌───▼────┐ ┌──▼───┐ ┌───▼──────────┐
  │Trainer │ │Evaluator│ │Predic│ │ModelRegistry  │
  │        │ │         │ │ tor  │ │+ Persistence  │
  └────┬───┘ └───┬────┘ └──┬───┘ └──────────────┘
       │         │         │
  ┌────▼─────────▼─────────▼───┐
  │      DatasetPlugin ABC      │
  │  (auto-discovered at boot)  │
  └────┬──────────────────┬────┘
  ┌────▼────┐        ┌───▼────┐
  │  MNIST  │        │  Iris  │
  │ Plugin  │        │ Plugin │
  └─────────┘        └────────┘
```

## SOLID Design Principles

### Single Responsibility (SRP)

Each module owns exactly one concern:

| Module | Responsibility |
|--------|---------------|
| `trainer.py` | Training loop — no data loading, no evaluation |
| `evaluator.py` | Test-set metrics — no training, no I/O |
| `predictor.py` | Single-sample inference — delegates preprocessing to plugin |
| `model_registry.py` | In-memory model storage — no file I/O |
| `persistence.py` | Disk checkpoint I/O — no in-memory state |
| `layers.py` | Reusable neural network layers — no model assembly |

### Open/Closed (OCP)

The `DatasetPlugin` ABC is the sole extension point. Adding a new dataset means creating a new subpackage under `classifiers/datasets/` — zero changes to any existing file. Auto-discovery finds and registers it at startup.

### Liskov Substitution (LSP)

All `DatasetPlugin` subclasses and all `BaseModel` subclasses are fully interchangeable. The shared infrastructure works identically regardless of which concrete plugin or model is active.

### Interface Segregation (ISP)

- `BaseModel` exposes only `forward()` and `loss_fn()`
- `DatasetPlugin` groups only dataset-specific concerns
- `StatusCallback` is a minimal type alias
- `TrainingConfig` is opt-in

### Dependency Inversion (DIP)

Route handlers access shared state through `current_app.extensions[...]` rather than importing concrete objects. The trainer depends on the `DataLoader` abstraction, and the evaluator depends on `BaseModel`. Qiskit and PennyLane are lazy-imported only when quantum models are instantiated.

## Request Flow

### Training

1. `POST /d/<dataset>/train` → `train_routes.py`
2. Plugin provides `DataLoader` via `get_train_loader()`
3. `Trainer` runs training in a daemon thread
4. Progress streamed as SSE events via a `queue.Queue`
5. On completion, model registered in `ModelRegistry`

### Prediction

1. `POST /d/<dataset>/predict` → `model_routes.py`
2. Plugin's `preprocess()` converts raw input to tensor
3. `Predictor` runs forward pass + softmax
4. Results returned as JSON for all registered models

### Evaluation

1. `POST /d/<dataset>/evaluate` → `eval_routes.py`
2. `Evaluator` iterates test set, computes accuracy + per-class metrics
3. Results streamed via SSE, then stored on registry entries
