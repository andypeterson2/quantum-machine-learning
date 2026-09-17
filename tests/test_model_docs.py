"""The served model documents may only claim what was measured.

``/model-info`` renders these MODELS.md files, so their numbers are published.
They were written by hand, and two contradicted the committed exports: Iris
Linear claimed ~95-97% against a measured 90%, BB84 Linear claimed ~92-94%
against 95.4%. The rest had no source.

Each documented accuracy now has to match ``exports/benchmarks.json``
(``tools/benchmark.py``), and a model nobody measured has to say so instead of
quoting a number.
"""

from __future__ import annotations

import json
import re

import pytest

from classifiers.plugin_registry import discover_plugins, list_plugins
from classifiers.web_export import REPO_ROOT

BENCHMARKS = REPO_ROOT / "exports" / "benchmarks.json"
DATASETS_DIR = REPO_ROOT / "classifiers" / "datasets"

#: "**Measured accuracy:** 90.0% (95% CI 74.4-96.5%, n=30)" — the only shape a
#: documented number may take, so every claim carries its uncertainty.
MEASURED = re.compile(
    r"\*\*Measured accuracy:\*\* (\d+\.\d)% \(95% CI (\d+\.\d)-(\d+\.\d)%, n=(\d+)\)"
)
#: The alternative for a model no one has measured here.
UNMEASURED = "**Accuracy:** not measured in this repo"
#: What the old hand-written claims looked like.
BANNED = re.compile(r"\*\*Typical accuracy:\*\*")


@pytest.fixture(scope="module", autouse=True)
def _plugins() -> None:
    discover_plugins()


@pytest.fixture(scope="module")
def measured() -> dict[tuple[str, str], dict]:
    assert BENCHMARKS.is_file(), f"missing {BENCHMARKS} — run `python tools/benchmark.py --slow`"
    payload = json.loads(BENCHMARKS.read_text())
    return {(r["dataset"], r["model_type"]): r for r in payload["results"]}


def _docs() -> list[tuple[str, str]]:
    return [(path.parent.name, path.read_text()) for path in DATASETS_DIR.glob("*/MODELS.md")]


def _sections(text: str) -> dict[str, str]:
    """Split a MODELS.md into its per-model sections, keyed by heading."""
    parts = re.split(r"^## ", text, flags=re.MULTILINE)[1:]
    return {part.split("\n", 1)[0].strip(): part for part in parts}


@pytest.mark.parametrize("dataset", sorted(d for d, _ in _docs()))
def test_no_unsourced_accuracy_claims(dataset: str) -> None:
    text = dict(_docs())[dataset]
    assert not BANNED.search(text), (
        f"{dataset}/MODELS.md still states a hand-written 'Typical accuracy' — "
        "quote exports/benchmarks.json or say the model is not measured"
    )


def test_every_documented_number_matches_the_measurement(measured) -> None:
    """A number in a served document must be one that was actually measured."""
    checked = 0
    for dataset, text in _docs():
        for heading, section in _sections(text).items():
            match = MEASURED.search(section)
            if match is None:
                assert UNMEASURED in section, f"{dataset}/{heading}: no accuracy line"
                continue
            model_type = heading.split("(")[0].strip()
            record = measured.get((dataset, model_type))
            assert record is not None, f"{dataset}/{model_type} claims a number but is not measured"
            accuracy, low, high, n = match.groups()
            assert float(accuracy) == pytest.approx(record["accuracy"] * 100, abs=0.05)
            assert float(low) == pytest.approx(record["accuracy_ci"][0] * 100, abs=0.05)
            assert float(high) == pytest.approx(record["accuracy_ci"][1] * 100, abs=0.05)
            assert int(n) == record["n"]
            checked += 1
    assert checked >= 8, f"only {checked} documented measurements found — did the docs lose them?"


def test_benchmarks_cover_every_advertised_model(measured) -> None:
    """Whatever a plugin offers is either measured or explicitly exempt.

    The Qiskit models sample a circuit per prediction, which is hours on a test
    split, so they are the standing exemption.
    """
    exempt = {("mnist", "Qiskit-CNN"), ("mnist", "Qiskit-Linear")}
    advertised = {
        (name, model_type)
        for name, plugin in list_plugins().items()
        for model_type in plugin.get_model_types()
    }
    missing = advertised - set(measured) - exempt
    assert not missing, f"unmeasured and undocumented as such: {sorted(missing)}"


class TestBenchmarkArtifact:
    def test_intervals_bracket_their_accuracies(self, measured) -> None:
        for key, record in measured.items():
            low, high = record["accuracy_ci"]
            assert low <= record["accuracy"] <= high, key

    def test_records_a_seed_and_its_hyperparameters(self, measured) -> None:
        for key, record in measured.items():
            assert record["training"]["seed"] == 0, key
            assert {"epochs", "batch_size", "lr"} <= set(record["training"]), key

    def test_provenance_names_the_protocol(self) -> None:
        payload = json.loads(BENCHMARKS.read_text())
        assert "Wilson" in payload["provenance"]["training"]["protocol"]
        assert re.fullmatch(r"[0-9a-f]{40}", payload["provenance"]["source_sha"])
