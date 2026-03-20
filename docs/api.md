# API Reference

All dataset-scoped endpoints live under `/d/<dataset>/`.

## Endpoints

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
| `GET` | `/d/<dataset>/models` | List all session models |
| `DELETE` | `/d/<dataset>/models/<name>` | Remove a model |
| `POST` | `/d/<dataset>/models/<name>/export` | Save to disk (.pt) |
| `GET` | `/d/<dataset>/models/disk` | List saved checkpoints |
| `POST` | `/d/<dataset>/models/disk/<filename>/load` | Load from disk |

### Example: cURL

```bash
# List datasets
curl http://localhost:5001/api/datasets

# Train a CNN on MNIST
curl -X POST http://localhost:5001/d/mnist/train \
  -H "Content-Type: application/json" \
  -d '{"model_type": "CNN", "epochs": 3, "batch_size": 64, "lr": 0.001}'

# Predict with Iris
curl -X POST http://localhost:5001/d/iris/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}}'
```
