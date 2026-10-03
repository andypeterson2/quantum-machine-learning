# MNIST Model Architectures

All models accept input tensors of shape `(N, 1, 28, 28)` (grayscale 28x28 images) and return raw logits of shape `(N, 10)` (one score per digit class 0-9). Softmax is never applied inside the model -- it is handled by the loss function during training and by the `Predictor` during inference.

Every model extends `BaseModel` and can be trained, evaluated, and compared interchangeably through the shared infrastructure.

---

## CNN (`MNISTNet`)

**Type:** Convolutional Neural Network
**Loss:** Cross-entropy (default)
**Measured accuracy:** 98.8% (95% CI 98.6-99.0%, n=10000)
**Across seeds:** 98.8% mean, sd 0.07, range 98.7-98.8% over 3 seeds — the line above is seed 0's run.
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
**Measured accuracy:** 92.1% (95% CI 91.5-92.6%, n=10000)
**Across seeds:** 92.0% mean, sd 0.27, range 91.5-92.3% over 10 seeds — the line above is seed 0's run.
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
- The distillation student in `exports/distillation.json`. The answer there, at
  the default blend, is no: distilling the 98.8% CNN into this model costs it
  about a point (92.05% alone vs 91.09% distilled, negative on all three seeds)
- Useful for demonstrating that spatial features matter (7% accuracy gap vs CNN)

---

## SVM (`SVMNet`)

**Type:** Linear Support Vector Machine
**Loss:** Weston-Watkins multi-class hinge loss
**Measured accuracy:** 91.6% (95% CI 91.0-92.1%, n=10000)
**Across seeds:** 91.3% mean, sd 0.47, range 90.6-92.1% over 10 seeds — the line above is seed 0's run.
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
