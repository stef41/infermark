"""Tests for infermark.histogram module."""

from __future__ import annotations

import math

import pytest

from infermark.histogram import (
    HistogramConfig,
    HistogramData,
    LatencyHistogram,
    compute_bins,
    format_histogram_report,
)


# ---------------------------------------------------------------------------
# HistogramConfig
# ---------------------------------------------------------------------------


class TestHistogramConfig:
    def test_defaults(self):
        cfg = HistogramConfig()
        assert cfg.bins == 20
        assert cfg.width == 60
        assert cfg.show_percentiles is True
        assert cfg.unit == "ms"

    def test_invalid_bins(self):
        with pytest.raises(ValueError, match="bins"):
            HistogramConfig(bins=0)

    def test_invalid_width(self):
        with pytest.raises(ValueError, match="width"):
            HistogramConfig(width=5)


# ---------------------------------------------------------------------------
# compute_bins
# ---------------------------------------------------------------------------


class TestComputeBins:
    def test_basic(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        edges, counts = compute_bins(values, 5)
        assert len(edges) == 6
        assert len(counts) == 5
        assert sum(counts) == len(values)

    def test_empty(self):
        edges, counts = compute_bins([], 10)
        assert edges == []
        assert counts == []

    def test_single_value(self):
        edges, counts = compute_bins([42.0], 5)
        assert len(edges) == 2
        assert counts == [1]

    def test_identical_values(self):
        edges, counts = compute_bins([3.0, 3.0, 3.0], 4)
        assert len(edges) == 2
        assert counts == [3]

    def test_all_in_one_bin(self):
        edges, counts = compute_bins([1.0, 1.0, 1.0, 1.0], 1)
        assert counts == [4]


# ---------------------------------------------------------------------------
# LatencyHistogram.compute
# ---------------------------------------------------------------------------


class TestHistogramCompute:
    def test_basic_compute(self):
        hist = LatencyHistogram()
        data = hist.compute([10.0, 20.0, 30.0, 40.0, 50.0])
        assert data.total == 5
        assert data.min_val == 10.0
        assert data.max_val == 50.0
        assert data.mean == pytest.approx(30.0)
        assert data.std > 0

    def test_empty(self):
        hist = LatencyHistogram()
        data = hist.compute([])
        assert data.total == 0
        assert data.counts == []


# ---------------------------------------------------------------------------
# LatencyHistogram.render_ascii
# ---------------------------------------------------------------------------


class TestRenderAscii:
    def test_renders_bars(self):
        hist = LatencyHistogram(HistogramConfig(bins=5, width=20))
        data = hist.compute([10, 20, 30, 40, 50, 10, 20, 30])
        text = hist.render_ascii(data)
        assert "\u2588" in text  # At least one bar character
        assert "total=" in text

    def test_empty_data(self):
        hist = LatencyHistogram()
        data = HistogramData()
        assert hist.render_ascii(data) == "(no data)"


# ---------------------------------------------------------------------------
# LatencyHistogram.percentiles
# ---------------------------------------------------------------------------


class TestPercentiles:
    def test_basic(self):
        hist = LatencyHistogram()
        pcts = hist.percentiles(list(range(1, 101)))
        assert pcts[50] == pytest.approx(50.0, abs=1)
        assert pcts[99] == pytest.approx(99.0, abs=1)

    def test_empty(self):
        hist = LatencyHistogram()
        pcts = hist.percentiles([])
        assert all(v == 0.0 for v in pcts.values())

    def test_single_value(self):
        hist = LatencyHistogram()
        pcts = hist.percentiles([42.0])
        assert pcts[50] == 42.0
        assert pcts[99] == 42.0

    def test_custom_pcts(self):
        hist = LatencyHistogram()
        pcts = hist.percentiles(list(range(100)), pcts=[25, 75])
        assert 25 in pcts
        assert 75 in pcts


# ---------------------------------------------------------------------------
# LatencyHistogram.render_cdf
# ---------------------------------------------------------------------------


class TestRenderCdf:
    def test_renders(self):
        hist = LatencyHistogram(HistogramConfig(width=30))
        text = hist.render_cdf([10, 20, 30, 40, 50])
        assert "CDF" in text
        assert "%" in text

    def test_empty(self):
        hist = LatencyHistogram()
        assert hist.render_cdf([]) == "(no data)"


# ---------------------------------------------------------------------------
# LatencyHistogram.summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_basic(self):
        hist = LatencyHistogram()
        s = hist.summary([10, 20, 30])
        assert s["count"] == 3
        assert s["min"] == 10
        assert s["max"] == 30
        assert s["mean"] == pytest.approx(20.0)
        assert "percentiles" in s

    def test_empty(self):
        hist = LatencyHistogram()
        s = hist.summary([])
        assert s["count"] == 0
        assert s["median"] == 0.0


# ---------------------------------------------------------------------------
# format_histogram_report
# ---------------------------------------------------------------------------


class TestFormatHistogramReport:
    def test_basic_report(self):
        hist = LatencyHistogram(HistogramConfig(bins=5))
        data = hist.compute([10, 20, 30, 40, 50])
        report = format_histogram_report(data, HistogramConfig(bins=5))
        assert "Latency Histogram Report" in report
        assert "Samples" in report
        assert "Mean" in report

    def test_empty_report(self):
        report = format_histogram_report(HistogramData())
        assert "0" in report
