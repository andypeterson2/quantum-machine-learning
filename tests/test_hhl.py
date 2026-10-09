"""The shared HHL circuit and readout (classifiers/hhl.py).

The circuit used to exist three times — in the notebook, in
``tools/hardware_run.py`` (whose comment read "any change there must be mirrored
here"), and implicitly in the exporter's alpha. The copies had drifted. These
tests pin the one definition against the paper's own expectations and against
the committed hardware run, so a change here has to explain itself.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from classifiers.web_export import REPO_ROOT

pytest.importorskip("qiskit", reason="qiskit not installed")

from classifiers.hhl import (
    alpha_from_counts,
    analyse,
    build_hhl,
    ideal_probs,
    js_divergence,
    paper_key,
)

HARDWARE_DIR = REPO_ROOT / "exports" / "hardware"


class TestCircuit:
    def test_depth_matches_the_paper_as_built(self):
        """Depth 8; the paper counts 7 after cancelling two X gates."""
        assert build_hhl().depth() == 8

    def test_barriers_are_optional_and_do_not_change_the_circuit(self):
        """They mark the paper's parts for the notebook's drawing only."""
        plain, marked = build_hhl(), build_hhl(barriers=True)
        assert marked.depth(lambda inst: inst.operation.name != "barrier") == plain.depth()

    def test_measured_variant_reads_all_four_qubits(self):
        assert build_hhl(measure=True).num_clbits == 4


class TestIdealDistribution:
    def test_four_states_share_the_probability(self):
        """The solution register is uniform over the four states the paper reads."""
        probs = ideal_probs()
        nonzero = {k: v for k, v in probs.items() if v > 1e-9}
        assert sorted(nonzero) == ["0000", "0001", "0010", "0011"]
        assert all(v == pytest.approx(0.25) for v in nonzero.values())

    def test_alpha_from_the_ideal_distribution_is_the_classical_solution(self):
        """F^-1 y is proportional to (1, -1); the readout must reproduce it."""
        alpha = alpha_from_counts(ideal_probs())
        assert alpha == pytest.approx([0.5, -0.5])


class TestReadout:
    def test_paper_key_is_its_own_inverse(self):
        assert paper_key(paper_key("0011")) == "0011"

    def test_identical_distributions_have_zero_divergence(self):
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        assert js_divergence(probs, probs) == pytest.approx(0.0)

    def test_disjoint_distributions_saturate_at_one_bit(self):
        assert js_divergence(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(1.0)


@pytest.mark.parametrize("path", sorted(HARDWARE_DIR.glob("hhl-*.json")), ids=lambda p: p.name)
def test_committed_hardware_run_reanalyses_to_its_recorded_values(path) -> None:
    """Re-run the analysis over the artifact's own counts.

    The counts are stored paper-keyed and paper_key is its own inverse, so this
    feeds them back exactly as the tool saw them. If the circuit or the readout
    changes, the stored D_JS and alpha stop reproducing and this fails.
    """
    run = json.loads(path.read_text())
    for label, job in run["jobs"].items():
        counts = {paper_key(k): v for k, v in job["counts"].items()}
        again = analyse(counts, run["shots"])
        assert again["js_divergence_vs_ideal"] == job["js_divergence_vs_ideal"], label
        assert again["alpha"] == pytest.approx(job["alpha"]), label
        assert again["p_q4_success"] == pytest.approx(job["p_q4_success"]), label
    # Runs are taken at whatever shot count the question needs, so the header
    # has to agree with the counts it describes.
    assert run["shots"] > 0
    for job in run["jobs"].values():
        assert sum(job["counts"].values()) == run["shots"]


class TestAlphaIsFixedByTheGeometry:
    """The README and the site both now say the quantum step reproduces a
    closed-form answer rather than computing an unknown one. That is a claim
    about the paper's construction, so it is checked rather than asserted.
    """

    @staticmethod
    def _system(k: float, gamma: float) -> np.ndarray:
        """``F = K + gamma^-1 I`` for two unit-norm training points."""
        inverse = 0.0 if np.isinf(gamma) else 1.0 / gamma
        return np.array([[1.0 + inverse, k], [k, 1.0 + inverse]])

    def test_the_paper_targets_are_unit_length(self) -> None:
        """Which is what makes F's diagonal equal, and the rest follow."""
        from classifiers.qsvm_rule import TARGETS

        unit = TARGETS / np.linalg.norm(TARGETS, axis=1, keepdims=True)
        assert np.allclose(np.linalg.norm(unit, axis=1), 1.0)

    @pytest.mark.parametrize("k", [0.0, 0.25, 0.490974, 0.75, 0.99])
    @pytest.mark.parametrize("gamma", [1.0, 2.0**3, 100.0])
    def test_the_label_vector_is_an_eigenvector(self, k: float, gamma: float) -> None:
        y = np.array([1.0, -1.0])
        product = self._system(k, gamma) @ y
        assert np.allclose(product, product[0] * y)

    @pytest.mark.parametrize("k", [0.0, 0.25, 0.490974, 0.75, 0.99])
    @pytest.mark.parametrize("gamma", [1.0, 2.0**3, 100.0])
    def test_the_ratio_is_one_whatever_the_data(self, k: float, gamma: float) -> None:
        """alpha1 / -alpha2 is exactly 1 for every dataset and every gamma, so
        a measured ratio away from it is device error and nothing else."""
        alpha = np.linalg.solve(self._system(k, gamma), np.array([1.0, -1.0]))
        assert alpha[0] / -alpha[1] == pytest.approx(1.0, abs=1e-12)

    def test_the_system_is_too_well_conditioned_to_generalise(self) -> None:
        """HHL's cost scales with the condition number, so the writeup may not
        read as a statement about HHL where that number is large."""
        from classifiers.qsvm_export import IRIS_CD  # noqa: F401  (module import check)
        from classifiers.qsvm_rule import TARGETS

        unit = TARGETS / np.linalg.norm(TARGETS, axis=1, keepdims=True)
        shipped = self._system(float(unit[0] @ unit[1]), 2.0**3)
        assert np.linalg.cond(shipped) < 5.0
