.PHONY: run test lint clean docker export-web sync-web export-qsvm export-qsvm-ovo benchmark distillation alpha-sensitivity alpha-ratio-sweep alpha-fit-noise qsvm-transfer hardware-mechanism

# Website checkout that consumes the browser model exports (override: make sync-web WEB=...)
# Relative to the working directory, so it resolves from the repo root. From a git
# worktree it points inside that worktree's parent, where there is no website checkout.
WEB ?= ../website

run:
	python -m classifiers

# Retrain the browser-served linear models through the real plugins/Trainer and
# write provenance-stamped weights to exports/web/ (drift-checked in CI).
export-web:
	python -m classifiers.web_export

# Exports the site does not serve. BB84's sessions come from this repo, and on
# them a fixed QBER threshold beats both fitted models, so an accuracy beside the
# other datasets would read as a result it is not. The files stay here: the alpha
# sensitivity sweep still scores them.
UNSERVED = bb84.json qsvm-bb84.json

# Copy the canonical exports into the website checkout's model directory.
sync-web:
	@test -d "$(WEB)/public/classifiers/models" || { \
	  echo "WEB=$(WEB) has no public/classifiers/models/ — pass an absolute path:"; \
	  echo "  make sync-web WEB=/path/to/website"; exit 1; }
	@for f in exports/web/*.json; do \
	  case " $(UNSERVED) " in *" $$(basename $$f) "*) continue;; esac; \
	  cp "$$f" $(WEB)/public/classifiers/models/; \
	done

# Derive the QSVM paper-recreation weights (closed-form, from the notebook's
# recorded solution) into exports/web/ (drift-checked in CI; ship via sync-web).
export-qsvm:
	python -m classifiers.qsvm_export

# Derive the three-class Iris rule (three pairwise maps over all four features,
# sharing the binary rule's alpha) into exports/web/ (ship via sync-web).
export-qsvm-ovo:
	python -m classifiers.qsvm_ovo_export

# Measure every model's accuracy (seeded, with a 95% interval) into
# exports/benchmarks.json — the numbers the MODELS.md files are held to.
# Add SLOW=--slow for the MNIST CNN-backbone models (minutes each).
benchmark:
	python tools/benchmark.py $(SLOW)

# Re-measure whether distillation helps the student (a few minutes per seed).
distillation:
	python tools/distillation_experiment.py

# Measure what the hardware's alpha readout is worth: how many deployed
# predictions change against the exact classical solution. Seconds.
alpha-sensitivity:
	python tools/alpha_sensitivity.py

# Sweep the one scalar the quantum step contributes and record where accuracy
# peaks, into exports/alpha-ratio-sweep.json.
alpha-ratio-sweep:
	python tools/alpha_ratio_sweep.py

# Redraw the fit sample and measure how much it alone moves accuracy, so the
# hardware alpha's effect can be read against it. exports/alpha-fit-noise.json.
alpha-fit-noise:
	python tools/alpha_fit_noise.py

# Apply the paper's rule to every Fashion-MNIST class pair, to see whether it
# transfers off the corpus it was fitted to. exports/qsvm-transfer.json.
qsvm-transfer:
	python tools/qsvm_transfer.py

# Account for the raw readout bias against the probes that ran beside the
# circuit, pool the ratio across every run, and range the hardware alphas
# downstream. Offline. exports/hardware/mechanism.json.
hardware-mechanism:
	python tools/hardware_mechanism.py

test:
	python -m pytest tests/ -v

lint:
	ruff check .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache/ .coverage htmlcov/

docker:
	docker compose up --build
