.PHONY: venv run execute

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
