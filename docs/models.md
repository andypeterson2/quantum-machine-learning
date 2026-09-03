# Model Architectures

## MNIST Models

### CNN (`MNISTNet`)

Two-layer convolutional network for digit classification.

```
Conv2d(1→32, k=3) → ReLU
Conv2d(32→64, k=3) → ReLU → MaxPool2d(2)
Flatten
Linear(9216→128) → ReLU
Linear(128→10)
```

- **Parameters:** ~1.2M
- **Typical accuracy:** ~99%
- **Training time:** ~45s (3 epochs)

### Linear (`LinearNet`)

Multinomial logistic regression — a single fully-connected layer.

```
Flatten → Linear(784→10)
```

- **Parameters:** 7,850
- **Typical accuracy:** ~92%
- **Training time:** ~10s (3 epochs)

### SVM (`SVMNet`)

Same architecture as Linear but trained with multi-class Crammer-Singer hinge loss instead of cross-entropy.

```
Flatten → Linear(784→10)
```

- **Parameters:** 7,850
- **Typical accuracy:** ~91-92%
- **Loss function:** `multi_class_hinge_loss` from `losses.py`

### Quadratic (`MNISTQuadraticNet`)

CNN backbone with a quadratic feature expansion layer that computes all pairwise products of features.

```
Conv2d(1→6, k=5) → ReLU → MaxPool2d(2)
Conv2d(6→16, k=5) → ReLU → MaxPool2d(2)
Flatten → Linear(256→120) → ReLU
Linear(120→32) → ReLU
Quadratic(32→16) → ReLU
Linear(16→10)
```

The `Quadratic` layer expands `x` into `z = concat(x^T · x, x)`, producing all pairwise quadratic products plus linear terms.

### Polynomial (`MNISTPolynomialNet`)

CNN backbone with polynomial basis layers using log-linear-exp transformations.

```
Conv2d(1→6, k=5) → ReLU → MaxPool2d(2)
Conv2d(6→16, k=5) → ReLU → MaxPool2d(2)
Flatten → Linear(256→120) → ReLU
Polynomial(120→84) → ReLU
Linear(84→32) → ReLU
Polynomial(32→16) → ReLU
Linear(16→10)
```

The `Polynomial` layer computes `y = exp(W · log(|x| + 1))`, creating polynomial-like feature transformations without explicit polynomial expansion.

### Qiskit Models (optional)

**Qiskit-CNN** and **Qiskit-Linear** replace the final classification head with a parameterised quantum circuit simulated via Qiskit. Requires `qiskit` and `qiskit-aer` to be installed.

## Iris Models

### Linear (`IrisLinear`)

Logistic regression for 3-class flower species classification.

```
Linear(4→3)
```

- **Parameters:** 15
- **Typical accuracy:** 90% at the default hyper-parameters — the measured, committed number in `exports/web/iris.json` (drift-checked in CI)

### SVM (`IrisSVM`)

Same architecture trained with hinge loss.

```
Linear(4→3)
```

### QVC (`IrisQVC`, optional)

Quantum Variational Classifier using PennyLane's `default.qubit` backend.

```
AngleEmbedding(4 features → 4 qubits, rotation=Y)
StronglyEntanglingLayers(n_layers=2, n_wires=4)
Measure ⟨Z₀⟩, ⟨Z₁⟩, ⟨Z₂⟩ → 3 class scores
```

- **Parameters:** 24 trainable rotation angles
- **Requires:** `pennylane`

## Custom Layers

| Layer | Module | Formula |
|-------|--------|---------|
| `Quadratic` | `layers.py` | `y = W · concat(x^T · x, x)` |
| `Polynomial` | `layers.py` | `y = exp(W · log(\|x\| + 1))` |
| `QiskitQLayer` | `qiskit_layers.py` | Multi-headed parametric quantum circuit |
