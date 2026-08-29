# API Reference

Top-level endpoints (`/health`, `/api`, `/connect`, ...) live at the root; all dataset-scoped endpoints live under `/d/<dataset>/`. JSON schemas for the health, manifest, and error shapes live in `tests/contract/schemas/`, and the live-HTTP contract tests in `tests/contract/` pin this surface.

Streaming (SSE) routes each have a synchronous REST equivalent (`/train/sync`, `/evaluate/sync`) that runs the same work to completion and returns the final result directly in the response body — every operation is reachable with plain `curl`.

## Endpoints

### Health

```
GET /health
```

**Response** (see `routes/main.py`; `uptime` is a legacy alias of `uptime_s` for pre-contract clients):
```json
{
  "status": "ok",
  "service": "classifiers",
  "version": "0.2.0",
  "uptime_s": 123.4,
  "uptime": 123.4,
  "clients": 1,
  "timestamp": 1756400000.0
}
```

`/health` stays public even when the `ORIGIN_SECRET` gateway guard is enabled.

### Discovery Index

```
GET /api
```

Returns every registered HTTP endpoint plus the SSE streaming channels:

**Response:**
```json
{
  "service": "classifiers",
  "version": "0.2.0",
  "endpoints": [
    {"method": "GET", "path": "/health", "summary": "Return server health status, uptime, and connected-client count."},
    {"method": "POST", "path": "/d/<dataset>/train", "summary": "Train a new model and stream progress as Server-Sent Events."}
  ],
  "streaming": [
    {"protocol": "sse", "channel": "train", "description": "Live training metrics stream; live equivalent of the synchronous train route."},
    {"protocol": "sse", "channel": "evaluate", "description": "Live evaluation stream; live equivalent of the synchronous evaluate route."}
  ]
}
```

### List Datasets

```
GET /api/datasets
```

**Response:**
```json
[
  {"name": "mnist", "display_name": "MNIST Handwritten Digits", "input_type": "image"},
  {"name": "iris", "display_name": "Iris Flower Classification", "input_type": "tabular"}
]
```

### Dataset Configuration

```
GET /api/datasets/<name>/config
```

**Response:**
```json
{
  "ui_config": {
    "name": "mnist",
    "display_name": "MNIST Handwritten Digits",
    "input_type": "image",
    "num_classes": 10,
    "class_labels": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    "image_size": [28, 28],
    "image_channels": 1,
    "default_hyperparams": {"epochs": 3, "batch_size": 64, "lr": 0.001}
  },
  "model_types": ["CNN", "Linear", "SVM", "Quadratic", "Polynomial"]
}
```

### Train a Model

```
POST /d/<dataset>/train
Content-Type: application/json
```

**Request body:**
```json
{
  "model_type": "CNN",
  "epochs": 3,
  "batch_size": 64,
  "lr": 0.001,
  "name": "My CNN"
}
```

**Advanced training options:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `patience` | `int` | — | Early stopping patience (epochs without improvement) |
| `val_gap` | `int` | `50` | Batches between validation checks |
| `teacher` | `string` | — | Name of a trained model for distillation |
| `distill_weight` | `float` | `0.5` | Blend: `(1-w)*true_loss + w*distill_loss` |

**Response:** SSE stream with events:
```
data: {"type": "status", "msg": "Epoch 1/3 — batch 0/938 — loss: 2.3012"}

data: {"type": "history", "epoch": 0, "batch": 50, "train_loss": 0.45, "val_accuracy": 0.92}

data: {"type": "done", "name": "My CNN", "model_type": "CNN", "epochs": 3, "num_params": 1199882}
```

`history` events are emitted at validation checkpoints, which run when validation is enabled (i.e. `patience` or `teacher` is set). The `done` event also carries `epochs_completed`, `batch_size`, `lr`, `stopped_early`, and — when validation ran — `best_val_accuracy` and the accumulated `history`.

### Train a Model (Synchronous)

```
POST /d/<dataset>/train/sync
Content-Type: application/json
```

Same request body as `/train` (including the advanced options). Runs training to completion and returns the final result directly — the `done` event payload without its `type` field. Intended for scripts and CI.

**Response:**
```json
{
  "name": "My CNN",
  "model_type": "CNN",
  "epochs": 3,
  "epochs_completed": 3,
  "batch_size": 64,
  "lr": 0.001,
  "num_params": 1199882,
  "stopped_early": false
}
```

### Predict

```
POST /d/<dataset>/predict
Content-Type: application/json
```

**Request body (image dataset):**
```json
{"image": "<base64-encoded-PNG>"}
```

**Request body (tabular dataset):**
```json
{
  "features": {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }
}
```

**Response:**
```json
{
  "results": {
    "My CNN": {
      "prediction": "7",
      "confidence": 0.94,
      "probs": [0.01, 0.01, 0.02, 0.01, 0.01, 0.01, 0.01, 0.94, 0.01, 0.01]
    }
  }
}
```

### Evaluate All Models

```
POST /d/<dataset>/evaluate
```

**Response:** SSE stream ending with:
```json
{
  "type": "done",
  "results": {
    "My CNN": {
      "accuracy": 0.9912,
      "avg_loss": 0.0312,
      "per_class_accuracy": {"0": 0.99, "1": 0.99, ...},
      "num_params": 1199882
    }
  }
}
```

### Evaluate All Models (Synchronous)

```
POST /d/<dataset>/evaluate/sync
```

Runs the same evaluation inline and returns the results directly:

**Response:**
```json
{
  "results": {
    "My CNN": {
      "accuracy": 0.9912,
      "avg_loss": 0.0312,
      "per_class_accuracy": {"0": 0.99, "1": 0.99, ...},
      "num_params": 1199882
    }
  }
}
```

### Ensemble Evaluation

```
POST /d/<dataset>/ensemble
Content-Type: application/json

{"model_names": ["Model 1", "Model 2"]}
```

**Response:**
```json
{
  "accuracy": 0.9934,
  "avg_loss": 0.028,
  "per_class_accuracy": {"0": 0.99, ...}
}
```

### Ablation Study

```
POST /d/<dataset>/ablation
Content-Type: application/json

{"model_name": "My CNN"}
```

**Response:** SSE stream with per-layer results:
```json
{"type": "ablation_result", "layer": "conv1", "accuracy": 0.11, "drop": 0.88}
```

### Model Management

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/d/<dataset>/models` | List all session models (metadata, training history, eval results) |
| `GET` | `/d/<dataset>/model-info/<model_type>` | Rendered docs for a model type |
| `DELETE` | `/d/<dataset>/models/<name>` | Remove a model |
| `POST` | `/d/<dataset>/models/<name>/export` | Save to disk (.pt) |
| `GET` | `/d/<dataset>/models/disk` | List saved checkpoints |
| `POST` | `/d/<dataset>/models/disk/<filename>/load` | Load from disk |

### Model Info

```
GET /d/<dataset>/model-info/<model_type>
```

Returns the matching section of the dataset plugin's `MODELS.md`, rendered to HTML:

**Response:**
```json
{"html": "<h2>CNN (...)</h2>..."}
```

Returns a 404 error envelope when no section exists for that model type.

### Connection Lifecycle

The heartbeat channel lets the server track connected clients (reported by `/health` as `clients`).

```
GET /connect
```

**Response:** SSE stream. First a `welcome` event, then a `ping` every `heartbeat_interval` seconds:

```
data: {"type": "welcome", "client_id": "...", "heartbeat_interval": 25}

data: {"type": "ping", "ts": 1756400000.0}
```

```
POST /pong
Content-Type: application/json

{"client_id": "..."}
```

**Response:** `204` on success (updates the client's last-seen timestamp); `404` for an unknown or missing `client_id`. Clients that stop ponging are swept as stale after 90 seconds.

```
POST /disconnect
Content-Type: application/json

{"client_id": "..."}
```

**Response:** `204`. Gracefully removes the client from the tracker (suitable for `navigator.sendBeacon` on page unload).

### Example: cURL

The examples assume the server was started with `CLASSIFIERS_PORT=5001` (without it, the dev server picks a random free port and logs it at startup).

```bash
# Health check + endpoint discovery
curl http://localhost:5001/health
curl http://localhost:5001/api

# List datasets
curl http://localhost:5001/api/datasets

# Train a CNN on MNIST (SSE stream)
curl -N -X POST http://localhost:5001/d/mnist/train \
  -H "Content-Type: application/json" \
  -d '{"model_type": "CNN", "epochs": 3, "batch_size": 64, "lr": 0.001}'

# Train synchronously (result in the response body — no SSE client needed)
curl -X POST http://localhost:5001/d/mnist/train/sync \
  -H "Content-Type: application/json" \
  -d '{"model_type": "Linear", "epochs": 1}'

# Evaluate everything synchronously
curl -X POST http://localhost:5001/d/mnist/evaluate/sync

# Predict with Iris
curl -X POST http://localhost:5001/d/iris/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}}'
```
