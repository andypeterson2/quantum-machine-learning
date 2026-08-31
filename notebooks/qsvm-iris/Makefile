.PHONY: venv run execute export-site

# Website checkout that embeds the exported notebook (override: make export-site WEB=...)
WEB ?= ../../../website

# The notebook needs its own venv: qiskit 2.x wants numpy 2, which the service
# venv can't have (torch 2.2 uses numpy's 1.x C-API).
venv:
	python3 -m venv .venv
	./.venv/bin/pip install -r requirements.txt

run:
	./.venv/bin/jupyter lab

# Re-execute the notebook in place (fixed seeds; downloads MNIST on first run).
execute:
	./.venv/bin/jupyter nbconvert --to notebook --execute --inplace qsvm_iris.ipynb

# Export CSP-clean HTML (MathML math, no JS) into the website checkout.
export-site:
	./.venv/bin/python export_site.py $(WEB)/src/content/notebooks/qsvm-iris.html
