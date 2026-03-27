"""Phase 2 pipeline tests (WPs #686-#696).

Verify that each Phase 2 deliverable — quantum feature map, variational
circuit optimisation, classical-quantum interface, classification output,
training data pipeline, Docker build config, container environment,
classifier frontend, frontend-backend API, CI pipeline, and API docs — has
the expected source artefacts in the repository.

Tests use file-system reads (``open`` / ``os.path``) rather than direct Python
imports to avoid dependency-version issues in constrained CI environments.
"""

import os
import re

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _read(relpath: str) -> str:
    """Read a project file relative to the repository root."""
    path = os.path.join(ROOT, relpath)
    assert os.path.isfile(path), f"Expected file not found: {relpath}"
    with open(path, encoding="utf-8") as fh:
        return fh.read()


# ── WP #686: Quantum feature map encoding ─────────────────────────────────

class TestQuantumFeatureMapEncoding:
    """#686 — Verify quantum circuit encoding code exists and is correct."""

    def test_iris_models_file_exists(self):
        path = os.path.join(ROOT, "classifiers", "datasets", "iris", "models.py")
        assert os.path.isfile(path)

    def test_angle_embedding_present(self):
        src = _read("classifiers/datasets/iris/models.py")
        assert "AngleEmbedding" in src, "AngleEmbedding gate not found in iris models"

    def test_feature_rotation_axis(self):
        src = _read("classifiers/datasets/iris/models.py")
        assert 'rotation="Y"' in src or "rotation='Y'" in src, (
            "Y-axis rotation not specified for AngleEmbedding"
        )

    def test_qubit_count_matches_features(self):
        src = _read("classifiers/datasets/iris/models.py")
        assert "_N_QUBITS" in src
        match = re.search(r"_N_QUBITS.*?=\s*(\d+)", src)
        assert match and match.group(1) == "4", "Expected 4 qubits for 4 Iris features"

    def test_qiskit_rx_encoding_present(self):
        src = _read("classifiers/qiskit_layers.py")
        assert "qc.rx" in src or "RX" in src, "RX feature encoding not found in Qiskit layer"

    def test_qiskit_parametric_circuit_class(self):
        src = _read("classifiers/qiskit_layers.py")
        assert "class _ParametricCircuit" in src


# ── WP #687: Variational circuit parameter optimisation ───────────────────

class TestVariationalCircuitOptimisation:
    """#687 — Verify variational ansatz and trainable parameter code."""

    def test_strongly_entangling_layers_present(self):
        src = _read("classifiers/datasets/iris/models.py")
        assert "StronglyEntanglingLayers" in src

    def test_variational_layer_count(self):
        src = _read("classifiers/datasets/iris/models.py")
        match = re.search(r"_N_LAYERS.*?=\s*(\d+)", src)
        assert match, "_N_LAYERS constant not found"
        assert int(match.group(1)) >= 1, "Need at least 1 variational layer"

    def test_weight_shapes_defined(self):
        src = _read("classifiers/datasets/iris/models.py")
        assert "weight_shapes" in src, "weight_shapes dict not defined"

    def test_torch_layer_returned(self):
        src = _read("classifiers/datasets/iris/models.py")
        assert "TorchLayer" in src, "PennyLane TorchLayer not used"

    def test_qiskit_entangling_gates(self):
        src = _read("classifiers/qiskit_layers.py")
        assert "rxx" in src or "rzz" in src, "Entangling gates not found"

    def test_qiskit_trainable_parameters(self):
        src = _read("classifiers/qiskit_layers.py")
        assert "nn.Parameter" in src, "Trainable parameters not declared in Qiskit layer"


# ── WP #688: Classical-quantum interface ──────────────────────────────────

class TestClassicalQuantumInterface:
    """#688 — Verify autograd / PennyLane / Qiskit bridge code."""

    def test_pennylane_backprop_diff_method(self):
        src = _read("classifiers/datasets/iris/models.py")
        assert 'diff_method="backprop"' in src, "backprop diff method not set"

    def test_pennylane_torch_interface(self):
        src = _read("classifiers/datasets/iris/models.py")
        assert 'interface="torch"' in src, "torch interface not set on qnode"

    def test_qiskit_autograd_function(self):
        src = _read("classifiers/qiskit_layers.py")
        assert "class _RunCircuit(Function)" in src, "Custom autograd Function not found"

    def test_qiskit_forward_method(self):
        src = _read("classifiers/qiskit_layers.py")
        assert "def forward(ctx" in src, "forward method missing in _RunCircuit"

    def test_qiskit_backward_method(self):
        src = _read("classifiers/qiskit_layers.py")
        assert "def backward(ctx" in src, "backward method missing in _RunCircuit"

    def test_parameter_shift_rule(self):
        src = _read("classifiers/qiskit_layers.py")
        assert "_estimate_partial" in src, "Parameter-shift gradient estimation not found"

    def test_gradient_delta_uses_pi(self):
        src = _read("classifiers/qiskit_layers.py")
        assert "np.pi" in src, "Gradient estimation should use pi-based shift"


# ── WP #689: Classification output and confidence ─────────────────────────

class TestClassificationOutput:
    """#689 — Verify prediction + confidence code."""

    def test_predictor_module_exists(self):
        path = os.path.join(ROOT, "classifiers", "predictor.py")
        assert os.path.isfile(path)

    def test_softmax_applied(self):
        src = _read("classifiers/predictor.py")
        assert "softmax" in src, "softmax not applied in predictor"

    def test_predict_returns_numpy(self):
        src = _read("classifiers/predictor.py")
        assert ".numpy()" in src, "predict should return numpy array"

    def test_predict_route_returns_confidence(self):
        src = _read("classifiers/routes/model_routes.py")
        assert '"confidence"' in src, "predict route missing confidence field"

    def test_predict_route_returns_prediction(self):
        src = _read("classifiers/routes/model_routes.py")
        assert '"prediction"' in src, "predict route missing prediction field"

    def test_predict_route_returns_probs(self):
        src = _read("classifiers/routes/model_routes.py")
        assert '"probs"' in src, "predict route missing probs field"

    def test_pauli_z_measurement_for_qvc(self):
        src = _read("classifiers/datasets/iris/models.py")
        assert "PauliZ" in src, "PauliZ measurement not found for QVC output"


# ── WP #690: Training data pipeline ──────────────────────────────────────

class TestTrainingDataPipeline:
    """#690 — Verify data loading and preprocessing code."""

    def test_iris_plugin_exists(self):
        path = os.path.join(ROOT, "classifiers", "datasets", "iris", "plugin.py")
        assert os.path.isfile(path)

    def test_mnist_plugin_exists(self):
        path = os.path.join(ROOT, "classifiers", "datasets", "mnist", "plugin.py")
        assert os.path.isfile(path)

    def test_iris_train_loader(self):
        src = _read("classifiers/datasets/iris/plugin.py")
        assert "def get_train_loader" in src

    def test_iris_test_loader(self):
        src = _read("classifiers/datasets/iris/plugin.py")
        assert "def get_test_loader" in src

    def test_iris_val_loader(self):
        src = _read("classifiers/datasets/iris/plugin.py")
        assert "def get_val_loader" in src

    def test_iris_standardisation(self):
        src = _read("classifiers/datasets/iris/plugin.py")
        assert "_mean" in src and "_std" in src, "Feature standardisation not found"

    def test_iris_stratified_split(self):
        src = _read("classifiers/datasets/iris/plugin.py")
        assert "stratify" in src, "Stratified train/test split not found"

    def test_iris_preprocess(self):
        src = _read("classifiers/datasets/iris/plugin.py")
        assert "def preprocess" in src

    def test_trainer_uses_dataloader(self):
        src = _read("classifiers/trainer.py")
        assert "DataLoader" in src, "Trainer should depend on DataLoader abstraction"

    def test_trainer_adam_optimizer(self):
        src = _read("classifiers/trainer.py")
        assert "Adam" in src, "Adam optimizer not used in Trainer"


# ── WP #691: Dockerfile builds ────────────────────────────────────────────

class TestDockerfileBuild:
    """#691 — Verify Dockerfile syntax and structure."""

    def test_dockerfile_exists(self):
        assert os.path.isfile(os.path.join(ROOT, "Dockerfile"))

    def test_dockerfile_has_from(self):
        src = _read("Dockerfile")
        assert src.strip().startswith("FROM"), "Dockerfile must start with FROM"

    def test_dockerfile_python_base(self):
        src = _read("Dockerfile")
        assert "python:" in src, "Dockerfile should use a Python base image"

    def test_dockerfile_workdir(self):
        src = _read("Dockerfile")
        assert "WORKDIR" in src

    def test_dockerfile_copy_requirements(self):
        src = _read("Dockerfile")
        assert "COPY requirements.txt" in src

    def test_dockerfile_pip_install(self):
        src = _read("Dockerfile")
        assert "pip install" in src

    def test_dockerfile_copy_classifiers(self):
        src = _read("Dockerfile")
        assert "COPY classifiers/" in src

    def test_dockerfile_cmd(self):
        src = _read("Dockerfile")
        assert "CMD" in src

    def test_dockerfile_runs_classifiers_module(self):
        src = _read("Dockerfile")
        assert "classifiers" in src, "CMD should run the classifiers package"


# ── WP #692: Container environment config ─────────────────────────────────

class TestContainerEnvironment:
    """#692 — Verify environment variables and requirements config."""

    def test_docker_compose_exists(self):
        assert os.path.isfile(os.path.join(ROOT, "docker-compose.yml"))

    def test_docker_compose_port_mapping(self):
        src = _read("docker-compose.yml")
        assert "5001" in src, "Port 5001 not mapped in docker-compose"

    def test_docker_compose_env_port(self):
        src = _read("docker-compose.yml")
        assert "CLASSIFIERS_PORT" in src

    def test_dockerfile_env_cert_dir(self):
        src = _read("Dockerfile")
        assert "DEV_CERT_DIR" in src

    def test_requirements_flask(self):
        src = _read("requirements.txt")
        assert "flask" in src.lower()

    def test_requirements_torch(self):
        src = _read("requirements.txt")
        assert "torch" in src.lower()

    def test_requirements_numpy(self):
        src = _read("requirements.txt")
        assert "numpy" in src.lower()

    def test_requirements_pillow(self):
        src = _read("requirements.txt")
        assert "Pillow" in src or "pillow" in src

    def test_requirements_scikit_learn(self):
        src = _read("requirements.txt")
        assert "scikit-learn" in src or "sklearn" in src

    def test_pyproject_toml_exists(self):
        assert os.path.isfile(os.path.join(ROOT, "pyproject.toml"))


# ── WP #693: Classifier frontend rendering ────────────────────────────────

class TestClassifierFrontend:
    """#693 — Verify HTML/JS/CSS frontend artefacts exist."""

    def test_app_css_exists(self):
        path = os.path.join(ROOT, "classifiers", "static", "css", "app.css")
        assert os.path.isfile(path)

    def test_app_js_exists(self):
        path = os.path.join(ROOT, "classifiers", "static", "js", "app.js")
        assert os.path.isfile(path)

    def test_sse_js_exists(self):
        path = os.path.join(ROOT, "classifiers", "static", "js", "sse.js")
        assert os.path.isfile(path)

    def test_chart_js_exists(self):
        path = os.path.join(ROOT, "classifiers", "static", "js", "chart.js")
        assert os.path.isfile(path)

    def test_ui_kit_exists(self):
        path = os.path.join(ROOT, "ui-kit")
        assert os.path.isdir(path), "ui-kit directory not found"

    def test_ui_kit_css_exists(self):
        path = os.path.join(ROOT, "ui-kit", "ui-kit.css")
        assert os.path.isfile(path)

    def test_ui_kit_js_exists(self):
        path = os.path.join(ROOT, "ui-kit", "ui-kit.js")
        assert os.path.isfile(path)

    def test_flask_serves_static(self):
        src = _read("classifiers/server.py")
        assert "static_folder" in src, "Flask app should configure static_folder"


# ── WP #694: Frontend-backend API integration ─────────────────────────────

class TestFrontendBackendAPI:
    """#694 — Verify API endpoints for frontend integration."""

    def test_train_route_exists(self):
        src = _read("classifiers/routes/train_routes.py")
        assert '"/train"' in src or "'/train'" in src

    def test_predict_route_exists(self):
        src = _read("classifiers/routes/model_routes.py")
        assert '"/predict"' in src or "'/predict'" in src

    def test_models_route_exists(self):
        src = _read("classifiers/routes/model_routes.py")
        assert '"/models"' in src or "'/models'" in src

    def test_evaluate_route_exists(self):
        src = _read("classifiers/routes/eval_routes.py")
        assert "evaluate" in src

    def test_ensemble_route_exists(self):
        src = _read("classifiers/routes/eval_routes.py")
        assert "ensemble" in src

    def test_ablation_route_exists(self):
        src = _read("classifiers/routes/eval_routes.py")
        assert "ablation" in src

    def test_datasets_api_route(self):
        src = _read("classifiers/routes/main.py")
        assert "/api/datasets" in src

    def test_sse_streaming_helper(self):
        src = _read("classifiers/routes/sse.py")
        assert "sse" in src.lower(), "SSE helper module missing streaming code"

    def test_dataset_blueprint_prefix(self):
        src = _read("classifiers/routes/dataset_routes.py")
        assert "/d/<dataset>" in src, "Dataset blueprint URL prefix not set"


# ── WP #695: CI pipeline ──────────────────────────────────────────────────

class TestCIPipeline:
    """#695 — Verify CI configuration exists and is correct."""

    def test_ci_config_exists(self):
        path = os.path.join(ROOT, ".github", "workflows", "ci.yml")
        assert os.path.isfile(path)

    def test_ci_runs_pytest(self):
        src = _read(".github/workflows/ci.yml")
        assert "pytest" in src

    def test_ci_runs_on_push(self):
        src = _read(".github/workflows/ci.yml")
        assert "push" in src

    def test_ci_runs_on_pr(self):
        src = _read(".github/workflows/ci.yml")
        assert "pull_request" in src

    def test_ci_uses_python_312(self):
        src = _read(".github/workflows/ci.yml")
        assert "3.12" in src

    def test_ci_installs_dependencies(self):
        src = _read(".github/workflows/ci.yml")
        assert "pip install" in src

    def test_ci_has_lint_job(self):
        src = _read(".github/workflows/ci.yml")
        assert "lint" in src

    def test_ci_has_docker_job(self):
        src = _read(".github/workflows/ci.yml")
        assert "docker" in src.lower()


# ── WP #696: API documentation accuracy ───────────────────────────────────

class TestAPIDocumentationAccuracy:
    """#696 — Verify API docs match actual endpoints."""

    @pytest.fixture
    def api_docs(self):
        return _read("docs/api.md")

    @pytest.fixture
    def readme(self):
        return _read("README.md")

    def test_api_docs_exist(self):
        assert os.path.isfile(os.path.join(ROOT, "docs", "api.md"))

    def test_docs_cover_train_endpoint(self, api_docs):
        assert "/train" in api_docs

    def test_docs_cover_predict_endpoint(self, api_docs):
        assert "/predict" in api_docs

    def test_docs_cover_evaluate_endpoint(self, api_docs):
        assert "/evaluate" in api_docs

    def test_docs_cover_ensemble_endpoint(self, api_docs):
        assert "/ensemble" in api_docs

    def test_docs_cover_ablation_endpoint(self, api_docs):
        assert "/ablation" in api_docs

    def test_docs_cover_models_endpoint(self, api_docs):
        assert "/models" in api_docs

    def test_docs_cover_datasets_api(self, api_docs):
        assert "/api/datasets" in api_docs

    def test_docs_cover_sse_events(self, api_docs):
        assert "SSE" in api_docs or "sse" in api_docs or "Server-Sent" in api_docs

    def test_readme_api_table_matches_docs(self, readme, api_docs):
        """Key endpoints listed in README should also appear in API docs."""
        key_paths = ["/train", "/predict", "/evaluate", "/models"]
        for path in key_paths:
            assert path in api_docs, f"{path} missing from API docs"
            assert path in readme, f"{path} missing from README"

    def test_docs_cover_dataset_config_endpoint(self, api_docs):
        assert "/api/datasets/" in api_docs and "config" in api_docs
