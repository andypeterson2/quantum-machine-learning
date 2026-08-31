# QSVM on NISQ Computers — Iris recreation

A Qiskit recreation of Yang, Awan & Vall-Llosera, *Support Vector Machines on
Noisy Intermediate Scale Quantum Computers* ([arXiv:1909.11988](https://arxiv.org/abs/1909.11988)),
applied to the Iris dataset (setosa vs. versicolor) and to MNIST ("6" vs "9",
standing in for the paper's no-longer-distributed OCR corpus).

The notebook implements the paper's full pipeline:

1. **Preprocessing unit** — solved linear mapping onto the paper's fixed training
   geometry, L² normalization, and quadrant-aware rotation angles (Eq. 24).
2. **Kernel matrix generation** — the depth-1 training-data oracle with classical
   readout of raw counts (no state tomography).
3. **Optimized HHL solver** — the 4-qubit shallow circuit of Fig. 10, reconstructed
   from Section IV-C and verified against the classical LS-SVM solution, with the
   classical amplitude readout of Eqs. 34–35/31.
4. **Results** — 97% accuracy on Iris (paper: 97% simulated / 98% on IBMQX2)
   and 91% on MNIST 6-vs-9 with the paper's HR/VR pixel-ratio features (near the
   92.5% ceiling of an unconstrained linear SVM on the same features), reusing
   the same quantum solution — only the mapping coefficients change per dataset.
5. **Noise** — the paper's Jensen–Shannon divergence analysis repeated under a
   depolarizing + readout noise model in place of the retired IBMQX2 device.

## Run

```
make -C notebooks/qsvm-iris venv      # one-time: own venv (qiskit 2.x / numpy 2 —
                                      # deliberately separate from the service venv)
make -C notebooks/qsvm-iris execute   # re-execute in place (fixed seeds)
```

or open `qsvm_iris.ipynb` in JupyterLab (`make -C notebooks/qsvm-iris run`) and
run all cells. The first run downloads MNIST (~15 MB) via `fetch_openml`; it is
cached afterwards.

## Export to the website

```
make export-site        # from the repo root (or from this directory)
```

renders the executed notebook to CSP-clean HTML for the portfolio site's AI/ML
page: nbconvert's chrome-free `basic` template, LaTeX pre-rendered to native
MathML (zero JavaScript, no CDN), and a provenance comment stamped with this
repo's commit — the same convention as the browser weight exports
(`classifiers/web_export.py`).
