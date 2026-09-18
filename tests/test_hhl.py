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
    DEFAULT_SHOTS,
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
    assert run["shots"] == DEFAULT_SHOTS
