# Iris Model Architectures

All models accept input tensors of shape `(N, 4)` (four standardised Iris features: sepal length, sepal width, petal length, petal width) and return raw logits of shape `(N, 3)` (one score per species: setosa, versicolor, virginica). Features are z-score standardised using training-set statistics before being passed to any model.

Every model extends `BaseModel` and can be trained, evaluated, and compared interchangeably through the shared infrastructure.

---

## Linear (`IrisLinear`)

**Type:** Multinomial Logistic Regression
**Loss:** Cross-entropy (default)
**Typical accuracy:** ~95-97%
**Trainable parameters:** 15

### Architecture

```
Input (N, 4)
  -> Linear(4 -> 3)   (N, 3)
```

### Description

A single linear layer mapping 4 standardised features directly to 3 class scores. This is multinomial logistic regression -- the simplest possible neural network classifier. Despite its simplicity, it achieves strong accuracy on Iris because the dataset is nearly linearly separable (setosa is perfectly separable; versicolor and virginica have a small overlap region).

The 15 trainable parameters consist of a 4x3 weight matrix (12 parameters) and a 3-element bias vector.

### When to use

- Default baseline for the Iris dataset
- Demonstrates that simple linear boundaries suffice for nearly-separable data
- Fast convergence (typically under 50 epochs with lr=0.01)
- Good reference point for comparing SVM and quantum approaches

---

## SVM (`IrisSVM`)

**Type:** Linear Support Vector Machine
**Loss:** Crammer-Singer multi-class hinge loss
**Typical accuracy:** ~94-96%
**Trainable parameters:** 15

### Architecture

```
Input (N, 4)
  -> Linear(4 -> 3)   (N, 3)
```

### Loss function

Overrides the default cross-entropy with multi-class hinge loss:

```
L = (1/N) * sum( max(0, s_j - s_y + margin) )  for all j != y
```

where `s_y` is the score for the correct class, `s_j` is the score for class `j`, and `margin = 1.0`. This encourages the correct class score to exceed all incorrect class scores by at least the margin.

### Description

Architecturally identical to `IrisLinear` -- same single linear layer, same 15 parameters. The only difference is the training objective. Hinge loss seeks maximum-margin decision boundaries rather than calibrated probabilities, producing a support vector machine trained via gradient descent.

On Iris, the accuracy is typically slightly lower than cross-entropy because hinge loss focuses on the decision boundary margin rather than fitting the full class-conditional distribution. However, the resulting model may generalise better to out-of-distribution samples near the decision boundary.

### When to use

- Controlled comparison of loss functions on the same architecture (SVM vs Linear)
- When decision margin matters more than probability calibration
- Ensemble diversity -- combining SVM and Linear produces a more diverse ensemble since they optimise different objectives
- Educational: demonstrates that the SVM vs logistic regression distinction is purely about the loss function, not the architecture

---

## QVC (`IrisQVC`)

**Type:** Quantum Variational Classifier (PennyLane)
**Loss:** Cross-entropy (default)
**Typical accuracy:** ~85-95% (depends on random initialisation)
**Trainable parameters:** 24 (quantum rotation angles)
**Requires:** `pip install pennylane`

### Architecture

```
Input (N, 4)
  -> AngleEmbedding(4 features -> 4 qubits, rotation=Y)
  -> StronglyEntanglingLayers(n_layers=2, n_wires=4)
  -> Measure <Z_0>, <Z_1>, <Z_2>
  -> Output (N, 3)
```

### Quantum circuit

The QVC is a pure quantum model -- no classical neural network layers. The full circuit:

1. **Angle embedding:** Each of the 4 standardised Iris features is encoded as a Y-rotation angle on its corresponding qubit. Standardised features (roughly in [-2, 2]) map naturally to rotation angles.

2. **Strongly entangling layers:** 2 variational layers, each applying:
   - 3 single-qubit rotations (RX, RY, RZ) per wire (4 wires)
   - CNOT entanglers covering all qubit pairs
   - Total: 2 layers x 4 wires x 3 rotations = **24 trainable parameters**

3. **Measurement:** Pauli-Z expectation values on qubits 0, 1, and 2 produce three real numbers in [-1, 1], used directly as class logits.

### Simulation backend

Uses PennyLane's `default.qubit` statevector simulator with `diff_method="backprop"`. Gradients propagate through the full quantum simulation via PyTorch autograd -- no parameter-shift rule or finite differences needed. This makes training efficient for small circuits.

### Training tips

- Converges well with the Iris plugin defaults: **50 epochs, lr=0.01, batch_size=16**
- Circuit evaluation is sample-by-sample, so training is slower than classical models
- Accuracy varies with random initialisation due to the non-convex quantum landscape
- The [-1, 1] output range of Pauli-Z measurements may cause cross-entropy to behave differently than with unbounded logits; this is expected

### When to use

- Research into quantum machine learning on tabular data
- Comparing quantum vs classical classifiers on a dataset where classical models already perform well
- Iris is an ideal testbed: small enough for statevector simulation, yet non-trivial (the versicolor/virginica overlap challenges the circuit's expressivity)
- Demonstrating that quantum circuits can be trained end-to-end with standard PyTorch optimisers
