"""Where the raw readout bias comes from, and what every hardware alpha was worth.

Every raw run of the HHL circuit has read ``alpha1 / -alpha2`` above its exact
value of 1. The 2026-10-09 runs carried two probes in the same job as the
circuit, on the same physical qubits: ``readout`` prepares the solution qubit
in ``|1>`` and measures at once, ``decay`` waits first. Their flip rates say how
much of the bias the qubit's own decay and its measurement account for, and
whatever is left has to come from the circuit's operations.

The decay probe on those runs waited the circuit's whole length, measurement
window included, and then had a measurement window of its own; its rate covers
about 1.8 times the exposure the solution qubit sees in the circuit. The
artifacts now record the split (:func:`tools.hardware_run.exposure_split`), so
the probe's idle rate is scaled here to the circuit's pre-measurement time
before it is added to the readout rate. The stored rates are untouched; only
the arithmetic that reads them changed.

Three more things are gathered so the writeup can quote them from one file:
the raw and mitigated ratios pooled across every run, with the shot variance
known so the pooling needs no estimate of it; the same per device; and each
job's MNIST accuracy against the exact alpha's, read against the spread the fit
sample alone produces (``exports/alpha-fit-noise.json``).

Output: ``exports/hardware/mechanism.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from math import log, sqrt
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classifiers.web_export import provenance_base  # noqa: E402

logger = logging.getLogger(__name__)

HARDWARE_DIR = REPO_ROOT / "exports" / "hardware"
ARTIFACT = HARDWARE_DIR / "mechanism.json"
SENSITIVITY = REPO_ROOT / "exports" / "alpha-sensitivity.json"
FIT_NOISE = REPO_ROOT / "exports" / "alpha-fit-noise.json"

#: The two readout states alpha is read from, paper-keyed |q1q2q3q4>: the
#: solution qubit in |0> and |1>, with the ancilla flag set.
KEY_ALPHA1 = "0001"
KEY_ALPHA2 = "0011"

#: Residual size that counts as resolved; six jobs are tested, so two sigma is too low.
RESOLVED_SIGMA = 3.0

#: The mitigated arm's probes ran under the same twirling and decoupling as its
#: circuit, so the depletion model's fraction does not describe that arm.
MODELLED_ARMS = frozenset({"raw"})


def log_ratio_sd(n_a: int, n_b: int, total: int) -> float:
    """Shot-noise standard deviation of ``log sqrt(n_a / n_b)`` from one draw.

    The two counts are cells of the same multinomial draw, so they covary
    (``-p_a p_b / N``), and the ratio's variance is larger than two independent
    binomials would give. That covariance adds ``2 / N`` inside the bracket.

    Args:
        n_a:   Count of the first state.
        n_b:   Count of the second state.
        total: Shots in the draw.

    Returns:
        The standard deviation of the log ratio.

    Raises:
        ValueError: If either count is zero, since the log is then undefined.
    """
    if n_a <= 0 or n_b <= 0:
        raise ValueError(f"both counts must be positive (got {n_a}, {n_b})")
    p_a, p_b = n_a / total, n_b / total
    return 0.5 * sqrt((1 - p_a) / (total * p_a) + (1 - p_b) / (total * p_b) + 2 / total)


def depletion_ratio(delta: float) -> float:
    """The ratio a ``|1> -> |0>`` transfer of probability *delta* produces.

    With both readout states ideally at the same probability, moving a fraction
    *delta* of the ``|1>`` state's weight onto ``|0>`` gives
    ``sqrt((1 + delta) / (1 - delta))``.

    Args:
        delta: Transfer probability, in ``[0, 1)``.

    Returns:
        The biased ratio; 1.0 at ``delta = 0``.
    """
    return sqrt((1 + delta) / (1 - delta))


def rate_sd(rate: float, shots: int) -> float:
    """Binomial standard deviation of a probe's flip rate."""
    return sqrt(rate * (1 - rate) / shots)


def observed(job: dict) -> tuple[float, float]:
    """One job's ratio and the shot-noise sd of its log, from the counts.

    Args:
        job: A ``jobs`` entry of a hardware artifact.

    Returns:
        ``(ratio, log_sd)``.
    """
    counts = job["counts"]
    n_a, n_b = counts[KEY_ALPHA1], counts[KEY_ALPHA2]
    total = sum(counts.values())
    return sqrt(n_a / n_b), log_ratio_sd(n_a, n_b, total)


def account(run: dict, arm: str) -> dict:
    """One probe-carrying job: what the probes predict against what it read.

    Args:
        run: A hardware artifact.
        arm: ``"raw"`` or ``"mitigated"``.

    Returns:
        The observed ratio, the readout-only and exposure-corrected
        predictions, the residual in standard errors and, for the raw arm, the
        fraction of the bias the corrected prediction covers.

    Raises:
        KeyError: If the artifact has probes but no exposure split; run
            ``python tools/hardware_run.py exposure`` first.
    """
    job = run["jobs"][arm]
    probes = job["probes"]
    exposure = run["transpiled"]["exposure"]
    ratio, ratio_sd = observed(job)

    readout, decay = probes["readout"]["read_zero"], probes["decay"]["read_zero"]
    delay = probes["decay"]["delay_seconds"]
    circuit = exposure["circuit_seconds"]
    # The idle rate is per unit of wait; the circuit's own idle is its
    # pre-measurement time, so a wait longer than that is scaled down.
    fraction = min(1.0, circuit / delay)
    delta = readout + (decay - readout) * fraction
    predicted = depletion_ratio(delta)
    # The prediction's own noise, from the two probes' binomial rates. At these
    # rates ``d log(predicted) / d delta`` is 1 to within a percent.
    shots_ro, shots_decay = probes["readout"]["shots"], probes["decay"]["shots"]
    predicted_sd = sqrt(
        (rate_sd(readout, shots_ro) * (1 - fraction)) ** 2
        + (rate_sd(decay, shots_decay) * fraction) ** 2
    )
    residual = log(ratio) - log(predicted)
    residual_sigma = residual / sqrt(ratio_sd**2 + predicted_sd**2)
    return {
        "backend": run["backend"],
        "date": run["provenance"]["exported_at"],
        "shots": run["shots"],
        "arm": arm,
        "job_id": job["job_id"],
        "solution_qubit": run["transpiled"]["qubit_roles"]["solution_q3"],
        "observed_ratio": round(ratio, 6),
        "observed_log_sd": round(ratio_sd, 6),
        "readout_read_zero": readout,
        "decay_read_zero": decay,
        "delay_seconds": delay,
        "circuit_seconds": circuit,
        "measure_seconds": exposure["measure_seconds"],
        "delay_covers": probes["decay"].get("delay_covers"),
        "exposure_fraction": round(fraction, 6),
        "readout_only_ratio": round(depletion_ratio(readout), 6),
        "uncorrected_ratio": round(depletion_ratio(decay), 6),
        "corrected_delta": round(delta, 6),
        "corrected_ratio": round(predicted, 6),
        "corrected_log_sd": round(predicted_sd, 6),
        "residual_log": round(residual, 6),
        "residual_sigma": round(residual_sigma, 3),
        "fraction_explained": (
            round(log(predicted) / log(ratio), 4) if arm in MODELLED_ARMS else None
        ),
        "resolved": bool(abs(residual_sigma) >= RESOLVED_SIGMA),
    }


def pooled(rows: list[dict]) -> dict:
    """Combine jobs' log ratios with the shot variance taken as known.

    The inverse-variance mean is the estimate; Cochran's Q against its degrees
    of freedom says whether the jobs scatter more than their shots allow. The
    unweighted mean and its empirical standard error are kept beside it so a
    reader can see the two agree.

    Args:
        rows: Dicts carrying ``log_ratio`` and ``log_sd``.

    Returns:
        The pooled figures, or ``{"n": 0}`` for no rows.
    """
    if not rows:
        return {"n": 0}
    x = np.array([r["log_ratio"] for r in rows])
    w = 1.0 / np.array([r["log_sd"] for r in rows]) ** 2
    mean = float((w * x).sum() / w.sum())
    se = float(1.0 / sqrt(w.sum()))
    q = float((w * (x - mean) ** 2).sum())
    out = {
        "n": len(rows),
        "ratio": round(float(np.exp(mean)), 6),
        "log_se": round(se, 6),
        "z": round(mean / se, 3),
        "cochran_q": round(q, 3),
        "degrees_of_freedom": len(rows) - 1,
        "above_one": int((x > 0).sum()),
        "unweighted_ratio": round(float(np.exp(x.mean())), 6),
    }
    if len(rows) > 1:
        empirical = float(x.std(ddof=1) / sqrt(len(rows)))
        out["empirical_log_se"] = round(empirical, 6)
        out["t"] = round(float(x.mean()) / empirical, 3) if empirical else None
    return out


def every_job(runs: list[dict]) -> list[dict]:
    """One row per job across every artifact, with its ratio and log sd."""
    rows = []
    for run in runs:
        for arm, job in run["jobs"].items():
            ratio, sd = observed(job)
            rows.append(
                {
                    "backend": run["backend"],
                    "date": run["provenance"]["exported_at"],
                    "shots": run["shots"],
                    "arm": arm,
                    "job_id": job["job_id"],
                    "ratio": round(ratio, 6),
                    "log_ratio": log(ratio),
                    "log_sd": sd,
                    "mnist_accuracy": job["qsvm_accuracy"].get("mnist"),
                }
            )
    return rows


def pooled_by_arm(rows: list[dict]) -> dict:
    """Pooled figures for each arm, overall and per backend."""
    out = {}
    for arm in ("raw", "mitigated"):
        arm_rows = [r for r in rows if r["arm"] == arm]
        out[arm] = pooled(arm_rows)
        out[arm]["per_backend"] = {
            backend: pooled([r for r in arm_rows if r["backend"] == backend])
            for backend in sorted({r["backend"] for r in arm_rows})
        }
    return out


def downstream(rows: list[dict]) -> dict:
    """Every job's MNIST accuracy against the exact alpha's, in points.

    The exact alpha's accuracy comes from the sensitivity artifact and the band
    from the fit-noise artifact, so the three files agree by construction.

    Returns:
        The band, the range of deltas, whether every job sits inside it, and
        the per-job deltas.

    Raises:
        FileNotFoundError: If either upstream artifact is missing.
    """
    sensitivity = json.loads(SENSITIVITY.read_text())
    exact = next(c for c in sensitivity["comparisons"] if c["alpha"] == "exact")
    exact_accuracy = next(d for d in exact["datasets"] if d["dataset"] == "mnist")[
        "other_accuracy"
    ]
    fit_noise = json.loads(FIT_NOISE.read_text())
    band = next(d for d in fit_noise["datasets"] if d["dataset"] == "mnist")["fit_noise_sd"]

    jobs = []
    for row in rows:
        if row["mnist_accuracy"] is None:
            continue
        delta = row["mnist_accuracy"] - exact_accuracy
        jobs.append(
            {
                **{k: row[k] for k in ("backend", "date", "shots", "arm", "job_id")},
                "mnist_accuracy": row["mnist_accuracy"],
                "delta": round(delta, 6),
                "inside_band": bool(abs(delta) < band),
            }
        )
    deltas = [j["delta"] for j in jobs]
    return {
        "exact_alpha_accuracy": exact_accuracy,
        "band_sd": band,
        "band_source": str(FIT_NOISE.relative_to(REPO_ROOT)),
        "n_jobs": len(jobs),
        "n_without_mnist": len(rows) - len(jobs),
        "delta_min": round(min(deltas), 6) if deltas else None,
        "delta_max": round(max(deltas), 6) if deltas else None,
        "all_inside_band": all(j["inside_band"] for j in jobs),
        "jobs": jobs,
    }


def load_runs() -> list[dict]:
    """Every committed hardware artifact, oldest first."""
    return [json.loads(p.read_text()) for p in sorted(HARDWARE_DIR.glob("hhl-*.json"))]


def build() -> dict:
    """Assemble the mechanism accounting, the pooled ratios and the downstream range."""
    runs = load_runs()
    accounts = [
        account(run, arm)
        for run in runs
        for arm in ("raw", "mitigated")
        if "probes" in run["jobs"][arm]
    ]
    rows = every_job(runs)
    return {
        "kind": "hardware-mechanism",
        "resolved_sigma": RESOLVED_SIGMA,
        "exposure_note": (
            "the decay probe's wait on the 2026-10-09 runs was the circuit's whole length, "
            "measurement window included, so its idle rate is scaled by "
            "circuit_seconds / delay_seconds before it is added to the readout rate; the "
            "stored read_zero values are as measured"
        ),
        "probe_jobs": accounts,
        "pooled": pooled_by_arm(rows),
        "downstream": downstream(rows),
        "jobs": [{k: v for k, v in r.items() if k not in ("log_ratio", "log_sd")} for r in rows],
        "provenance": provenance_base(
            {
                "model": "HHL",
                "paper": "arXiv:1909.11988",
                "protocol": (
                    "ratios from the stored counts; shot variance of the log ratio from the "
                    "multinomial cells including their covariance; probes scaled to the "
                    "circuit's pre-measurement exposure"
                ),
            },
            {"numpy": np.__version__},
        ),
    }


def main(argv: list[str] | None = None) -> None:
    """Write the mechanism artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ARTIFACT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    artifact = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=1) + "\n")
    for row in artifact["probe_jobs"]:
        explained = row["fraction_explained"]
        logger.info(
            "%-14s %-9s observed %.4f  corrected %.4f  residual %+.2f sigma  explains %s",
            row["backend"],
            row["arm"],
            row["observed_ratio"],
            row["corrected_ratio"],
            row["residual_sigma"],
            f"{100 * explained:.0f}%" if explained is not None else "n/a",
        )
    for arm, figures in artifact["pooled"].items():
        logger.info(
            "%-9s pooled %.4f +/- %.4f (z %+.2f, Q %.1f on %d df, %d/%d above 1)",
            arm,
            figures["ratio"],
            figures["log_se"],
            figures["z"],
            figures["cochran_q"],
            figures["degrees_of_freedom"],
            figures["above_one"],
            figures["n"],
        )
    down = artifact["downstream"]
    logger.info(
        "downstream: %d jobs, delta %+.4f to %+.4f against a %.4f band, all inside: %s",
        down["n_jobs"],
        down["delta_min"],
        down["delta_max"],
        down["band_sd"],
        down["all_inside_band"],
    )
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
