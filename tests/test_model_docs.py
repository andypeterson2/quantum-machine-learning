"""The served model documents may only claim what was measured.

``/model-info`` renders these MODELS.md files, so their numbers are published.
They were written by hand, and two contradicted the committed exports: Iris
Linear claimed ~95-97% against a measured 90%, BB84 Linear claimed ~92-94%
against 95.4%. The rest had no source.

Each documented accuracy now has to match ``exports/benchmarks.json``
(``tools/benchmark.py``), and a model nobody measured has to say so instead of
quoting a number.

A documented accuracy also has to carry its seed spread. The Wilson interval
beside it is sampling error alone, and on Iris the seed moves the number
further: the published run is the best of ten on two of the three rows. That is
a missing line rather than a wrong one, so it needs a requirement rather than a
ban.
"""

from __future__ import annotations

import json
import re
import statistics

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

#: "**Across seeds:** 93.7% mean, sd 1.9, range 90.0-96.7% over 10 seeds" — the
#: second error component, required wherever a measured number is quoted.
ACROSS_SEEDS = re.compile(
    r"\*\*Across seeds:\*\* (\d+\.\d+)% mean, sd (\d+\.\d+), "
    r"range (\d+\.\d+)-(\d+\.\d+)% over (\d+) seeds"
)
#: The alternative for a row backed by a single run.
SINGLE_SEED = "**Across seeds:** not measured (single seed)"


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


#: "| CNN (`MNISTNet`) | ... | 98.8% (98.6-99.0%, n=10,000) |" — a README table row
#: quoting a measurement. The thousands separator is the README's own style.
README_ROW = re.compile(
    r"^\|\s*(?P<model>[^|(]+?)\s*\(`(?P<cls>\w+)`\)\s*\|[^|]*\|\s*"
    r"(?P<acc>\d+\.\d)%\s*\((?P<low>\d+\.\d)-(?P<high>\d+\.\d)%,\s*n=(?P<n>[\d,]+)\)",
    re.MULTILINE,
)

#: The heading above each accuracy table names the dataset the rows belong to.
README_DATASET = re.compile(r"^### (\w+)\s*$", re.MULTILINE)


def _readme_rows() -> list[tuple[str, str, tuple[str, str, str, str]]]:
    """(dataset, model type, quoted numbers) for every measured README row."""
    text = (REPO_ROOT / "README.md").read_text()
    sections = list(README_DATASET.finditer(text))
    rows = []
    for i, heading in enumerate(sections):
        end = sections[i + 1].start() if i + 1 < len(sections) else len(text)
        dataset = heading.group(1).lower()
        rows.extend(
            (
                dataset,
                match.group("model").strip(),
                (match.group("acc"), match.group("low"), match.group("high"), match.group("n")),
            )
            for match in README_ROW.finditer(text[heading.end() : end])
        )
    return rows


class TestTheReadmeTablesQuoteTheMeasurements:
    """The README's own claim, held.

    It says its numbers are "recorded in exports/benchmarks.json by make
    benchmark and held there by tests/test_model_docs.py" — which only globbed
    the MODELS.md files, so nothing held the README and its BB84 table quietly
    lost the measured QVC row.
    """

    def test_every_quoted_number_matches_the_measurement(self, measured) -> None:
        for dataset, model_type, (acc, low, high, n) in _readme_rows():
            record = measured.get((dataset, model_type))
            assert record is not None, f"README quotes {dataset}/{model_type}, which is unmeasured"
            assert float(acc) == pytest.approx(record["accuracy"] * 100, abs=0.05)
            assert float(low) == pytest.approx(record["accuracy_ci"][0] * 100, abs=0.05)
            assert float(high) == pytest.approx(record["accuracy_ci"][1] * 100, abs=0.05)
            assert int(n.replace(",", "")) == record["n"]

    def test_every_measured_model_has_a_row(self, measured) -> None:
        """A model measured but absent from the table is a number nobody sees."""
        quoted = {(dataset, model) for dataset, model, _ in _readme_rows()}
        missing = sorted(set(measured) - quoted)
        assert not missing, f"measured but missing from the README tables: {missing}"


class TestTheSeedSpread:
    """What the Wilson interval cannot say.

    A ten-seed sweep put seed 0 at the top of the distribution on two of the
    three Iris rows (SVM 96.7% against a 93.7 mean, Linear 90.0% against 88.3).
    Nothing was picked after the fact, since the seed is fixed in code, but the
    interval beside each number covers sampling error only and a reader had no
    way to see the rest. Both components now ship, and these hold them.
    """

    def test_every_result_reports_its_seeds(self, measured) -> None:
        for key, record in measured.items():
            seeds = [run["seed"] for run in record["runs"]]
            assert seeds, key
            assert len(set(seeds)) == len(seeds), key
            assert record["summary"]["seeds"] == len(seeds), key

    def test_the_headline_is_one_of_the_runs(self, measured) -> None:
        """The published number is a run in the file, not a fourth number."""
        for key, record in measured.items():
            headline = record["summary"]["headline_seed"]
            assert headline == record["training"]["seed"], key
            run = next(r for r in record["runs"] if r["seed"] == headline)
            assert run["accuracy"] == record["accuracy"], key
            assert run["accuracy_ci"] == record["accuracy_ci"], key

    def test_summary_matches_the_runs(self, measured) -> None:
        for key, record in measured.items():
            accuracies = [run["accuracy"] for run in record["runs"]]
            summary = record["summary"]
            assert summary["accuracy_mean"] == pytest.approx(
                statistics.fmean(accuracies), abs=5e-5
            ), key
            assert summary["accuracy_range"] == [min(accuracies), max(accuracies)], key
            assert summary["headline_is_best"] is (record["accuracy"] == max(accuracies)), key
            if len(accuracies) > 1:
                assert summary["accuracy_stdev"] == pytest.approx(
                    statistics.stdev(accuracies), abs=5e-5
                ), key
            else:
                assert summary["accuracy_stdev"] is None, key

    def test_a_single_seed_row_is_not_a_measured_zero(self, measured) -> None:
        """One run has no spread. Zero would claim the seed does not matter,
        which one run cannot establish."""
        for key, record in measured.items():
            if record["summary"]["seeds"] == 1:
                assert record["summary"]["accuracy_stdev"] is None, key
                assert record["summary"]["accuracy_range"] == [
                    record["accuracy"], record["accuracy"]
                ], key

    def test_every_run_scored_the_same_split(self, measured) -> None:
        """The split does not depend on the seed, which is what keeps the two
        error components separable. Every run's accuracy therefore lands on the
        same 1/n grid, and a split that changed size shows up here."""
        for key, record in measured.items():
            n = record["n"]
            tolerance = n * 5e-5 + 1e-6
            for run in record["runs"]:
                hits = run["accuracy"] * n
                assert hits == pytest.approx(round(hits), abs=tolerance), (key, run["seed"])


def test_every_measured_number_also_reports_its_seed_spread(measured) -> None:
    """A documented accuracy with no spread beside it is half a claim: the
    reader sees the sampling error and cannot see the seed error."""
    for dataset, text in _docs():
        for heading, section in _sections(text).items():
            if MEASURED.search(section) is None:
                continue
            model_type = heading.split("(")[0].strip()
            summary = measured[(dataset, model_type)]["summary"]
            match = ACROSS_SEEDS.search(section)
            if summary["seeds"] == 1:
                assert SINGLE_SEED in section, f"{dataset}/{model_type}: one seed, say so"
                continue
            assert match is not None, (
                f"{dataset}/{model_type}: quotes an accuracy with no '**Across seeds:**' line"
            )
            mean, deviation, low, high, seeds = match.groups()
            assert float(mean) == pytest.approx(summary["accuracy_mean"] * 100, abs=0.05)
            assert float(deviation) == pytest.approx(summary["accuracy_stdev"] * 100, abs=0.05)
            assert float(low) == pytest.approx(summary["accuracy_range"][0] * 100, abs=0.05)
            assert float(high) == pytest.approx(summary["accuracy_range"][1] * 100, abs=0.05)
            assert int(seeds) == summary["seeds"]


def test_the_readme_does_not_call_iris_unresolved_when_the_seeds_resolve_it(measured) -> None:
    """One 30-sample run cannot separate the Iris models. Ten runs can, and the
    README may not keep saying otherwise once the artifact shows it."""
    linear = measured[("iris", "Linear")]["summary"]
    svm = measured[("iris", "SVM")]["summary"]
    if svm["accuracy_range"][0] >= linear["accuracy_range"][1]:
        readme = (REPO_ROOT / "README.md").read_text()
        assert "does not separate these architectures" not in readme
