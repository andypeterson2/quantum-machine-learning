# MNIST Model Architectures

All models accept input tensors of shape `(N, 1, 28, 28)` (grayscale 28x28 images) and return raw logits of shape `(N, 10)` (one score per digit class 0-9). Softmax is never applied inside the model -- it is handled by the loss function during training and by the `Predictor` during inference.

Every model extends `BaseModel` and can be trained, evaluated, and compared interchangeably through the shared infrastructure.

---

## CNN (`MNISTNet`)

**Type:** Convolutional Neural Network
**Loss:** Cross-entropy (default)
**Typical accuracy:** ~99%
**Trainable parameters:** ~1.2M

### Architecture

```
Input (N, 1, 28, 28)
  -> Conv2d(1 -> 32, kernel=3, stride=1)  -> ReLU
  -> Conv2d(32 -> 64, kernel=3, stride=1) -> ReLU -> MaxPool2d(2)
  -> Flatten                                          (N, 9216)
  -> Linear(9216 -> 128)                   -> ReLU
  -> Linear(128 -> 10)                                (N, 10)
```

### Description

The standard convolutional architecture for MNIST. Two convolutional layers extract spatial features (edges, curves, loops), max-pooling reduces spatial dimensions, and two fully connected layers map to class logits. This is the strongest classical model in the collection and serves as the default baseline.

### When to use

- Default choice for best accuracy
- Good teacher model for knowledge distillation experiments
- Useful baseline for ablation studies (convolutional layers contribute heavily to accuracy)

---

## Linear (`LinearNet`)

**Type:** Multinomial Logistic Regression
**Loss:** Cross-entropy (default)
**Typical accuracy:** ~92%
**Trainable parameters:** ~7.9K

### Architecture

```
Input (N, 1, 28, 28)
  -> Flatten           (N, 784)
  -> Linear(784 -> 10) (N, 10)
```

### Description

The simplest possible classifier -- a single linear transformation from pixel space to class scores. Equivalent to multinomial logistic regression. No hidden layers, no nonlinearities, no spatial awareness. Each of the 10 output neurons learns a weighted template over all 784 pixels.

### When to use

- Minimal baseline to compare against more complex architectures
- Fast training (seconds, not minutes)
- Good student model for distillation (can a linear model absorb a CNN's knowledge?)
- Useful for demonstrating that spatial features matter (7% accuracy gap vs CNN)

---

## SVM (`SVMNet`)

**Type:** Linear Support Vector Machine
**Loss:** Crammer-Singer multi-class hinge loss
**Typical accuracy:** ~91-92%
**Trainable parameters:** ~7.9K

### Architecture

```
Input (N, 1, 28, 28)
  -> Flatten           (N, 784)
  -> Linear(784 -> 10) (N, 10)
```

### Loss function

Unlike the other models which use cross-entropy, SVM overrides `loss_fn()` to use multi-class hinge loss:

```
L = (1/N) * sum( max(0, s_j - s_y + margin) )  for all j != y
```

where `s_y` is the score for the correct class, `s_j` is the score for class `j`, and `margin = 1.0`. This encourages the correct class score to exceed all others by at least the margin.

### Description

Architecturally identical to `LinearNet` (same single linear layer, same parameter count), but trained with a fundamentally different objective. Where cross-entropy optimises for calibrated probabilities, hinge loss optimises for maximum-margin separation between classes. The two models provide a controlled comparison of loss function impact on the same architecture.

### When to use

- Comparing loss functions: SVM vs Linear shows the effect of hinge loss vs cross-entropy on identical architecture
- When you care about decision margin rather than probability calibration
- Ensemble diversity -- combining SVM and Linear models can improve ensemble accuracy since they optimise different objectives

---

## Quadratic (`MNISTQuadraticNet`)

**Type:** CNN + Quadratic Expansion Layer
**Loss:** Cross-entropy (default)
**Typical accuracy:** ~98-99%
**Trainable parameters:** ~40K

### Architecture

```
Input (N, 1, 28, 28)
  -> Conv2d(1 -> 6, kernel=5)   -> ReLU -> MaxPool2d(2)
  -> Conv2d(6 -> 16, kernel=5)  -> ReLU -> MaxPool2d(2)
  -> Flatten                                (N, 256)
  -> Linear(256 -> 120)         -> ReLU
  -> Linear(120 -> 32)          -> ReLU
  -> Quadratic(32 -> 16)        -> ReLU
  -> Linear(16 -> 10)                       (N, 10)
```

### Quadratic expansion

The `Quadratic` layer (from `classifiers.layers`) expands input `x` into `z = concat(x^T * x, x)`, capturing all pairwise products between features plus the original linear terms. For a 32-dimensional input, this produces a `32 * (32 + 1) = 1056`-dimensional expanded vector, which a learned linear layer projects down to the output dimension.

This lets the network model second-order feature interactions explicitly, rather than relying on stacked ReLU layers to approximate them.

### When to use

- Exploring whether explicit quadratic feature interactions improve over standard FC layers
- Research comparison: quadratic expansion vs polynomial basis vs standard MLP
- Ablation studies -- zeroing the quadratic layer reveals its contribution vs the convolutional backbone

---

## Polynomial (`MNISTPolynomialNet`)

**Type:** CNN + Polynomial (Log-Linear-Exp) Layers
**Loss:** Cross-entropy (default)
**Typical accuracy:** ~98-99%
**Trainable parameters:** ~25K

### Architecture

```
Input (N, 1, 28, 28)
  -> Conv2d(1 -> 6, kernel=5)   -> ReLU -> MaxPool2d(2)
  -> Conv2d(6 -> 16, kernel=5)  -> ReLU -> MaxPool2d(2)
  -> Flatten                                (N, 256)
  -> Linear(256 -> 120)         -> ReLU
  -> Polynomial(120 -> 84)      -> ReLU
  -> Linear(84 -> 32)           -> ReLU
  -> Polynomial(32 -> 16)       -> ReLU
  -> Linear(16 -> 10)                       (N, 10)
```

### Polynomial basis

The `Polynomial` layer (from `classifiers.layers`) computes `y = exp(W * log(|x| + 1))`. Working in log-space means that the linear transformation `W` effectively computes weighted sums of logarithms, and exponentiating the result produces polynomial-like combinations of the input features -- without the combinatorial explosion of explicit polynomial expansion.

The `+1` inside the log ensures numerical stability for small inputs, and `abs()` handles negative activations.

### When to use

- Exploring polynomial feature transformations as an alternative to quadratic expansion
- The model uses two polynomial layers at different stages, allowing study of where in the network polynomial features help most
- Comparison with Quadratic: polynomial layers are more parameter-efficient (no quadratic blowup) but may capture different feature interactions

---

## Qiskit-CNN (`QiskitCNN`)

**Type:** CNN + Qiskit Quantum Circuit Layer
**Loss:** Cross-entropy (default)
**Typical accuracy:** Varies (quantum simulation is stochastic)
**Trainable parameters:** ~40K classical + 6 quantum
**Requires:** `pip install qiskit qiskit-aer`

### Architecture

```
Input (N, 1, 28, 28)
  -> Conv2d(1 -> 6, kernel=5)   -> ReLU -> MaxPool2d(2)
  -> Conv2d(6 -> 16, kernel=5)  -> ReLU -> MaxPool2d(2)
  -> Flatten                                (N, 256)
  -> Linear(256 -> 120)         -> ReLU
  -> Linear(120 -> 84)          -> ReLU
  -> Linear(84 -> 10)           -> Sigmoid * 0.8
  -> Linear(10 -> 3)
  -> QiskitQLayer(3)                        (N, 3)
  -> Linear(3 -> 10)                        (N, 10)
```

### Quantum layer

The `QiskitQLayer` (from `classifiers.qiskit_layers`) implements a 3-qubit parametric quantum circuit:

1. **Encoding:** Input features are encoded as RX rotation angles on 3 qubits
2. **Entanglement:** Trainable RXX and RZZ gates create entanglement between adjacent qubits
3. **Measurement:** 8192-shot sampling produces per-qubit expectation values
4. **Gradients:** Finite-difference estimation (shift +/- delta, measure, compute slope)

The sigmoid squashing before the quantum layer ensures inputs stay in a range where rotation angles are meaningful. The classical bottleneck (10 -> 3) reduces the problem to a dimensionality the quantum circuit can handle.

### When to use

- Research into quantum-classical hybrid neural networks
- Comparing quantum vs classical layers at the same network position
- Understanding the overhead and accuracy tradeoffs of quantum circuit simulation
- **Note:** Training is significantly slower than classical models due to circuit simulation (expect minutes per epoch, not seconds)

---

## Qiskit-Linear (`QiskitLinear`)

**Type:** Linear + Qiskit Quantum Circuit Layer
**Loss:** Cross-entropy (default)
**Typical accuracy:** Varies
**Trainable parameters:** ~66K classical + 6 quantum
**Requires:** `pip install qiskit qiskit-aer`

### Architecture

```
Input (N, 1, 28, 28)
  -> Flatten                     (N, 784)
  -> Linear(784 -> 84)  -> ReLU
  -> Linear(84 -> 10)   -> Sigmoid * 0.8
  -> Linear(10 -> 3)
  -> QiskitQLayer(3)             (N, 3)
  -> Linear(3 -> 10)            (N, 10)
```

### Description

A fully-connected (no convolution) version of the quantum hybrid architecture. The classical front-end compresses the 784-pixel input down to 3 dimensions, the quantum circuit processes these 3 features, and a final linear layer maps back to 10 class scores.

This provides a direct comparison with `QiskitCNN`: same quantum layer, but without convolutional feature extraction. The accuracy difference reveals how much the CNN backbone contributes vs the quantum circuit.

### When to use

- Ablation: comparing QiskitCNN vs QiskitLinear isolates the contribution of convolutional layers in quantum hybrids
- Research into whether quantum circuits can compensate for the lack of spatial feature extraction
- Faster than QiskitCNN (fewer parameters in the classical portion), but the quantum layer remains the bottleneck
