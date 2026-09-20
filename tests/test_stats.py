"""The Wilson interval, and the reason every accuracy now ships with one.

Iris is scored on 30 samples: 90% and 96.7% are one and a half samples apart,
and their intervals overlap almost entirely. Reporting the point estimate alone
invited comparisons the data cannot support.
"""

from __future__ import annotations

import pytest

from classifiers.stats import Z_95, wilson_interval


class TestWilsonInterval:
    # The four committed exports, so a change in the formula shows up as a
    # change in a published claim.
    @pytest.mark.parametrize(
        ("successes", "total", "expected"),
        [
            (27, 30, (0.7438, 0.9654)),
            (29, 30, (0.8333, 0.9941)),
            (891, 1000, (0.8702, 0.9088)),
            (9206, 10000, (0.9151, 0.9257)),
        ],
    )
    def test_matches_known_values(self, successes, total, expected):
        assert wilson_interval(successes, total) == pytest.approx(expected, abs=1e-4)

    def test_interval_brackets_the_estimate(self):
        low, high = wilson_interval(27, 30)
        assert low < 27 / 30 < high

    def test_small_splits_give_wider_intervals(self):
        """Same accuracy, ten times the data: the interval must shrink."""
        small = wilson_interval(9, 10)
        large = wilson_interval(900, 1000)
        assert (small[1] - small[0]) > (large[1] - large[0])

    def test_stays_inside_zero_one_at_the_extremes(self):
        """Where the normal approximation runs off the end of the scale."""
        assert wilson_interval(30, 30) == pytest.approx((0.8865, 1.0), abs=1e-4)
        # Mirror image of the line above, as the interval is symmetric in p.
        assert wilson_interval(0, 30) == pytest.approx((0.0, 0.1135), abs=1e-4)

    def test_no_data_excludes_nothing(self):
        assert wilson_interval(0, 0) == (0.0, 1.0)

    @pytest.mark.parametrize(("successes", "total"), [(-1, 10), (11, 10), (1, -5)])
    def test_impossible_counts_raise(self, successes, total):
        with pytest.raises(ValueError, match="successes <= total"):
            wilson_interval(successes, total)

    def test_z_95_is_the_conventional_quantile(self):
        assert pytest.approx(1.96, abs=5e-4) == Z_95
