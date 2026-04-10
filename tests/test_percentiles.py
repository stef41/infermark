"""Tests for compute_percentiles and percentile utilities."""

import pytest

from infermark._types import compute_percentiles, compute_stats, percentile


class TestComputePercentiles:
    def test_default_percentiles(self):
        values = list(range(1, 101))
        result = compute_percentiles(values)
        assert set(result.keys()) == {50, 75, 90, 95, 99}
        assert result[50] == pytest.approx(50.5, abs=0.5)
        assert result[95] >= 95

    def test_custom_percentiles(self):
        values = list(range(1, 101))
        result = compute_percentiles(values, [10, 25, 50])
        assert set(result.keys()) == {10, 25, 50}

    def test_single_value(self):
        result = compute_percentiles([42.0], [50, 99])
        assert result[50] == 42.0
        assert result[99] == 42.0

    def test_two_values(self):
        result = compute_percentiles([1.0, 9.0], [50])
        assert result[50] == pytest.approx(5.0)

    def test_empty_values(self):
        result = compute_percentiles([], [50, 95, 99])
        assert result == {50: 0.0, 95: 0.0, 99: 0.0}

    def test_empty_percentiles(self):
        result = compute_percentiles([1.0, 2.0, 3.0], [])
        assert result == {}

    def test_p0_and_p100(self):
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = compute_percentiles(values, [0, 100])
        assert result[0] == 10.0
        assert result[100] == 50.0

    def test_invalid_percentile_above_100(self):
        with pytest.raises(ValueError, match="percentile must be in"):
            compute_percentiles([1.0, 2.0], [101])

    def test_invalid_percentile_below_0(self):
        with pytest.raises(ValueError, match="percentile must be in"):
            compute_percentiles([1.0, 2.0], [-1])

    def test_unsorted_input(self):
        result = compute_percentiles([5, 1, 3, 2, 4], [50])
        assert result[50] == 3.0

    def test_duplicate_percentile_keys(self):
        result = compute_percentiles([1.0, 2.0, 3.0], [50, 50])
        assert result[50] == 2.0

    def test_identical_values(self):
        result = compute_percentiles([7.0] * 100, [25, 50, 75, 99])
        for v in result.values():
            assert v == 7.0

    def test_large_dataset(self):
        values = [float(i) for i in range(1, 10001)]
        result = compute_percentiles(values, [50, 95, 99])
        assert result[50] == pytest.approx(5000.5, abs=1)
        assert result[95] == pytest.approx(9500.5, abs=1)
        assert result[99] == pytest.approx(9900.5, abs=1)

    def test_matches_percentile_function(self):
        values = [3.5, 1.2, 7.8, 9.1, 0.4, 5.5]
        result = compute_percentiles(values, [50, 95])
        assert result[50] == pytest.approx(percentile(values, 50))
        assert result[95] == pytest.approx(percentile(values, 95))

    def test_return_type(self):
        result = compute_percentiles([1.0, 2.0], [50])
        assert isinstance(result, dict)
        assert isinstance(result[50], float)


class TestLatencyStatsFields:
    """Verify LatencyStats contains all expected percentile fields."""

    def test_all_percentile_fields_exist(self):
        stats = compute_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        assert hasattr(stats, "p50")
        assert hasattr(stats, "p75")
        assert hasattr(stats, "p90")
        assert hasattr(stats, "p95")
        assert hasattr(stats, "p99")

    def test_percentile_ordering(self):
        stats = compute_stats(list(range(1, 101)))
        assert stats.p50 <= stats.p75 <= stats.p90 <= stats.p95 <= stats.p99

    def test_compute_stats_empty(self):
        stats = compute_stats([])
        assert stats.p50 == 0
        assert stats.p99 == 0
        assert stats.mean == 0
