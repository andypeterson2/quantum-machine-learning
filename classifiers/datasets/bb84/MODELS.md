# BB84 Model Architectures

All models accept input tensors of shape `(N, 2)` — two standardised session observables: the quantum bit error rate (QBER) and the sifted-key rate — and return raw logits of shape `(N, 2)`, one score per session class (clean, eavesdropped). Features are z-score standardised using training-set statistics before being passed to any model.

The sessions themselves are simulated with the channel physics of the quantum-video-chat project (Poisson photon source, fiber attenuation, detector efficiency, intercept-resend Eve), extended with channel noise and partial, lossy interception so the two regimes genuinely overlap near the protocol's 11% abort threshold — an eavesdropper is a statistical inference here, not a lookup.

Every model extends `BaseModel` and can be trained, evaluated, and compared interchangeably through the shared infrastructure.

---

## Linear (`BB84Linear`)

**Type:** Logistic Regression
**Loss:** Cross-entropy (default)
**Typical accuracy:** ~92-94%
**Trainable parameters:** 6

### Architecture

```
Input (N, 2)
  -> Linear(2 -> 2)   (N, 2)
```

### Description

A single linear layer mapping the two session observables directly to two class scores. The decision boundary it learns is, in effect, a data-driven QBER threshold tilted by the sifted-key rate — close to what a QKD operator would set by hand, but fitted rather than decreed. Accuracy tops out in the low 90s because the regimes overlap by construction: a noisy-but-clean channel and a lightly-tapped one can look alike over a finite session.

### When to use

The baseline. Fast, interpretable (two weights per class read directly as "how much each observable votes for eavesdropping"), and hard to beat on two features.

---

## SVM (`BB84SVM`)

**Type:** Linear Support Vector Machine
**Loss:** Multi-class hinge (Crammer-Singer)
**Typical accuracy:** ~92-94%
**Trainable parameters:** 6

### Architecture

```
Input (N, 2)
  -> Linear(2 -> 2)   (N, 2)
```

### Description

The same linear capacity as `BB84Linear`, trained with hinge loss instead of cross-entropy: it maximises the margin around the clean/eavesdropped boundary rather than fitting class probabilities. On overlapping regimes the two objectives place the boundary slightly differently — comparing them on the metrics table shows how much of the task is boundary placement versus raw capacity.

### When to use

When you care about the margin story, or want a second opinion on the same capacity.

---

## QVC (`BB84QVC`)

**Type:** Quantum Variational Classifier (simulated)
**Loss:** Cross-entropy (default)
**Typical accuracy:** ~90-94%
**Trainable parameters:** 12

### Architecture

```
Input (N, 2)
  -> AngleEmbedding(2 features -> 2 qubits, rotation=Y)
  -> StronglyEntanglingLayers(n_layers=2, n_wires=2)
  -> Measure <Z0>, <Z1>   (N, 2)
```

### Description

A 2-qubit parameterised quantum circuit simulated on PennyLane's `default.qubit` statevector backend, trained end-to-end with PyTorch backpropagation — the Iris QVC pattern sized down to two features. Quantum machine learning classifying the health of a quantum-cryptography session: the layered joke is intentional, the measured accuracy is not a joke.

Requires the optional `quantum` extra (PennyLane); the plugin only advertises this model when PennyLane is importable.

### When to use

For the demonstration that the platform's quantum model tier extends to a new dataset with zero changes to shared infrastructure — and for the narrative.
