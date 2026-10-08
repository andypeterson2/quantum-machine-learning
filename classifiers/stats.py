"""Uncertainty for the accuracies this platform reports.

Every accuracy here is a proportion measured on a finite test split, so it
carries sampling error — and the splits are small enough for that to matter:
Iris ships 30 test samples, where a single sample moves accuracy by 3.3 points
and two models three points apart are indistinguishable.

The Wilson score interval is used rather than the textbook normal approximation
because it stays inside ``[0, 1]`` and behaves at the extremes, where a small
split often sits (29/30 correct, say).

Sampling error is only half of it. The same model retrained under a different
seed lands somewhere else, and on Iris that spread is wider than the interval.
``spread`` measures the second component; both are reported together, because
one without the other tells the reader the measurement is steadier than it is.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from math import comb, sqrt

#: 1.96 standard deviations — the conventional 95% two-sided interval.
Z_95 = 1.959963984540054


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """Return the Wilson score interval for ``successes / total``.

    Args:
        successes: Number of correct predictions.
        total:     Number of predictions. ``0`` yields ``(0.0, 1.0)`` — no data,
                   so nothing is excluded.

    Returns:
        ``(low, high)`` at 95% confidence, each rounded to four decimals and
        clamped to ``[0, 1]``.

    Raises:
        ValueError: If *successes* is negative or exceeds *total*.
    """
    if total < 0 or successes < 0 or successes > total:
        raise ValueError(f"need 0 <= successes <= total (got {successes}/{total})")
    if total == 0:
        return (0.0, 1.0)

    p = successes / total
    z = Z_95
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4))


def spread(values: Sequence[float]) -> tuple[float, float | None, tuple[float, float]]:
    """Return the mean, standard deviation and range of a repeated measurement.

    The Wilson interval above describes one run's sampling error. This describes
    the other component: how far the same measurement moves when only the seed
    changes.

    Args:
        values: One measurement per repetition.

    Returns:
        ``(mean, stdev, (low, high))``, each rounded to four decimals. *stdev* is
        the sample standard deviation, and ``None`` for a single value: one run
        has no spread, which is not the same as a spread of zero.

    Raises:
        ValueError: If *values* is empty.
    """
    if not values:
        raise ValueError("need at least one value to describe")
    deviation = round(statistics.stdev(values), 4) if len(values) > 1 else None
    return (
        round(statistics.fmean(values), 4),
        deviation,
        (round(min(values), 4), round(max(values), 4)),
    )


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for two rules on one test split.

    Two rules scored on the same points agree on most of them, and the pairs
    where they agree carry no information about which is better. Only the
    discordant pairs do, and under the null each is a coin flip. Comparing the
    two accuracies' intervals instead treats them as independent samples, which
    throws away the pairing and almost all of the power.

    Args:
        b: Pairs the first rule got right and the second wrong.
        c: Pairs the second got right and the first wrong.

    Returns:
        The two-sided p-value; ``1.0`` when there are no discordant pairs.

    Raises:
        ValueError: If either count is negative.
    """
    if b < 0 or c < 0:
        raise ValueError(f"discordant counts must be non-negative (got {b}, {c})")
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(b, c) + 1))
    return min(1.0, 2.0 * tail / 2.0**n)


def paired_accuracy_delta(b: int, c: int, n: int, z: float = Z_95) -> tuple[float, float, float]:
    """Accuracy difference between two rules on one split, with its interval.

    The difference is ``(b - c) / n`` and its standard error follows from the
    discordant counts alone, so the interval is far tighter than either rule's
    own. Reporting it is what lets a reader see whether a difference is resolved
    or merely small.

    Args:
        b: Pairs the first rule got right and the second wrong.
        c: Pairs the second got right and the first wrong.
        n: Total paired predictions.
        z: Standard deviations for the interval (default: 95% two-sided).

    Returns:
        ``(delta, low, high)``, each rounded to six decimals.

    Raises:
        ValueError: If *n* is not positive or the counts exceed it.
    """
    if n <= 0 or b + c > n:
        raise ValueError(f"need 0 < b + c <= n (got {b} + {c}, n={n})")
    delta = (b - c) / n
    se = sqrt((b + c) - (b - c) ** 2 / n) / n if b + c else 0.0
    return (round(delta, 6), round(delta - z * se, 6), round(delta + z * se, 6))
