# Custom Layer Building Blocks

Reusable neural network layers that can be composed into model architectures across any dataset plugin. These layers are defined in `classifiers/layers.py` and `classifiers/qiskit_layers.py`.

---

## Quadratic (`classifiers.layers.Quadratic`)

**Source:** `classifiers/layers.py`
**Dependencies:** PyTorch only

### What it does

Performs quadratic feature expansion followed by a learned linear projection:

```
z = concat(x^T * x, x)
y = W * z + b
```

Given an input vector `x` of dimension `d`, the expansion produces all `d * d` pairwise products (via the outer product `x^T * x`) concatenated with the `d` original features, yielding a `d * (d + 1)`-dimensional vector. A learned linear layer then projects this expanded representation to the desired output dimension.

### API

```python
from classifiers.layers import Quadratic

layer = Quadratic(input_dim=32, output_dim=16)
out = layer(x)  # (N, 32) -> (N, 16)
```

**Parameters:**
- `input_dim` (int): Dimension of the input vector
- `output_dim` (int): Dimension of the output vector

**Trainable parameters:** `input_dim * (input_dim + 1) * output_dim + output_dim` (the linear projection weight matrix and bias)

### Static method: `expand(x)`

You can use the expansion step independently without the learned projection:

```python
z = Quadratic.expand(x)  # (N, 32) -> (N, 1056)
```

### Mathematical detail

The expansion works by:
1. Reshaping `x` from `(N, d)` to `(N, 1, d)`
2. Computing the transpose `x^T` as `(N, d, 1)`
3. Appending a column of ones to `x`: `(N, 1, d+1)`
4. Computing `x^T @ x_augmented` to get `(N, d, d+1)`
5. Flattening to `(N, d*(d+1))`

The column of ones ensures the product includes terms like `x_i * 1 = x_i`, retaining the original linear features alongside the quadratic cross-terms.

### Design considerations

- **Dimensionality:** The expanded vector grows as O(d^2). For `d=32`, this produces 1056 features -- manageable. For large `d`, consider reducing dimensionality before the quadratic layer.
- **Expressivity:** Captures all second-order feature interactions explicitly, which stacked ReLU layers can only approximate.
- **No activation:** The layer itself applies no nonlinearity. Apply ReLU or another activation after the layer if desired.

---

## Polynomial (`classifiers.layers.Polynomial`)

**Source:** `classifiers/layers.py`
**Dependencies:** PyTorch only

### What it does

Computes polynomial-like feature transformations via the log-linear-exp trick:

```
y = exp(W * log(|x| + 1))
```

Working in log-space converts multiplication into addition, so the linear transformation `W` effectively computes weighted sums of logarithms. Exponentiating the result produces polynomial-like combinations of input features -- without the combinatorial explosion of explicit polynomial expansion.

### API

```python
from classifiers.layers import Polynomial

layer = Polynomial(input_dim=120, output_dim=84)
out = layer(x)  # (N, 120) -> (N, 84)
```

**Parameters:**
- `input_dim` (int): Dimension of the input vector
- `output_dim` (int): Dimension of the output vector

**Trainable parameters:** `input_dim * output_dim + output_dim` (standard linear layer weight and bias)

### Mathematical detail

The forward pass:
1. `x_log = log(|x| + 1)` -- maps features to log-space, with `abs()` handling negatives and `+1` ensuring stability near zero
2. `z = W * x_log + b` -- standard linear transformation in log-space
3. `y = exp(z)` -- back to feature-space

If `W` has a row `[2, 0, 1, 0, ...]`, the corresponding output is `exp(2*log(|x_0|+1) + log(|x_2|+1))` which approximates `(|x_0|+1)^2 * (|x_2|+1)` -- a polynomial in the input features, with the degree controlled by the learned weights.

### Design considerations

- **Parameter efficiency:** Same parameter count as a standard linear layer (no quadratic blowup), but captures polynomial-like interactions through the log-exp transformation.
- **Output range:** All outputs are strictly positive (due to `exp`). Apply an appropriate activation or normalisation downstream.
- **Numerical stability:** The `+1` inside `log` prevents log(0). The `abs()` handles negative activations from upstream ReLU layers, though it loses sign information.
- **Gradient flow:** Gradients flow through `exp`, `linear`, and `log`, which can lead to large gradients when outputs are large. Consider gradient clipping or layer normalisation if training becomes unstable.

---

## QiskitQLayer (`classifiers.qiskit_layers.QiskitQLayer`)

**Source:** `classifiers/qiskit_layers.py`
**Dependencies:** PyTorch, `qiskit`, `qiskit-aer` (optional -- only needed when instantiated)

### What it does

A trainable quantum circuit layer that runs on the Qiskit Aer QASM simulator. Each forward pass encodes input features as qubit rotations, applies trainable entangling gates, measures the circuit, and interprets the measurement statistics as output features. Gradients are estimated via finite differences.

### API

```python
from classifiers.qiskit_layers import QiskitQLayer

layer = QiskitQLayer(input_dim=3, num_heads=1)
out = layer(x)  # (N, 3) -> (N, 3)
```

**Parameters:**
- `input_dim` (int): Number of qubits and input/output features
- `num_heads` (int, default=1): Number of independent circuit heads. When >1, outputs are concatenated and reduced via a learned linear projection.

**Trainable parameters:** `2 * input_dim` per head (RXX and RZZ rotation angles) + linear projection if multi-headed

### Circuit architecture

For `input_dim=3`:

```
q0: ─ RX(x_0) ─ barrier ─ RXX(w_0, q0, q1) ─ RZZ(w_1, q0, q1) ─ barrier ─ measure
q1: ─ RX(x_1) ─ barrier ─ RXX(w_2, q1, q2) ─ RZZ(w_3, q1, q2) ─ barrier ─ measure
q2: ─ RX(x_2) ─ barrier ─ RXX(w_4, q2, q0) ─ RZZ(w_5, q2, q0) ─ barrier ─ measure
```

1. **Encoding:** Input features become RX rotation angles
2. **Entanglement:** Trainable RXX (XX-interaction) and RZZ (ZZ-interaction) gates on adjacent qubit pairs (cyclic)
3. **Measurement:** All qubits measured in the computational basis (8192 shots by default)
4. **Interpretation:** Per-qubit mean of '1' outcomes across all shots

### Gradient computation

Since measurement is non-differentiable, gradients are estimated via symmetric finite differences:

```
dL/dw_k ~= (f(w + delta * e_k) - f(w - delta * e_k)) / (2 * delta)
```

where `delta = 0.2` and `e_k` is the k-th basis vector. This requires `2 * num_params` additional circuit evaluations per backward pass.

### Internal components

| Class | Role |
|-------|------|
| `_QCSampler` | Runs circuits on the Aer QASM simulator |
| `_IndependentInterpret` | Converts measurement counts to per-qubit probabilities |
| `_ParametricCircuit` | Manages parameter binding and circuit execution |
| `_ExampleCircuit` | Builds the RX-encoding + RXX/RZZ-entangling circuit |
| `_RunCircuit` | Custom `torch.autograd.Function` for forward/backward through the circuit |
| `_Head` | Single-headed wrapper holding the trainable weight parameter |

### Design considerations

- **Speed:** Each forward pass runs a full QASM simulation (8192 shots). Training is orders of magnitude slower than classical layers. Keep batch sizes small.
- **Stochasticity:** Measurement outcomes are inherently stochastic. Results vary between runs even with the same parameters.
- **Input range:** RX rotations are periodic (2pi), so inputs should be bounded. The models using this layer apply `sigmoid * 0.8` before the quantum layer to keep inputs in [0, 0.8].
- **Lazy loading:** All Qiskit imports happen inside this module only. The rest of the codebase never imports Qiskit directly, so it remains an optional dependency.
- **Scalability:** The finite-difference gradient estimator scales linearly with the number of parameters. For the 6-parameter circuits used here, this is fine. For larger circuits, consider parameter-shift rules.
