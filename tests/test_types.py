"""Tests for infermark._types."""

import pytest

from infermark._types import (
    BenchmarkConfig,
    BenchmarkMode,
    BenchmarkReport,
    ConcurrencyResult,
    InfermarkError,
    LatencyStats,
    RequestResult,
    compute_stats,
    percentile,
)


class TestPercentile:
    def test_single_value(self):
        assert percentile([5.0], 50) == 5.0

    def test_two_values(self):
        assert percentile([1.0, 9.0], 50) == 5.0

    def test_p0(self):
        assert percentile([1, 2, 3, 4, 5], 0) == 1

    def test_p100(self):
        assert percentile([1, 2, 3, 4, 5], 100) == 5

    def test_p50(self):
        assert percentile([1, 2, 3, 4, 5], 50) == 3.0

    def test_p95_large(self):
        values = list(range(1, 101))
        p95 = percentile(values, 95)
        assert 95 <= p95 <= 96

    def test_empty(self):
        assert percentile([], 50) == 0.0

    def test_unsorted_input(self):
        assert percentile([5, 1, 3, 2, 4], 50) == 3.0


class TestComputeStats:
    def test_basic(self):
        stats = compute_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        assert stats.mean == 3.0
        assert stats.min == 1.0
        assert stats.max == 5.0
        assert stats.p50 == 3.0

    def test_empty(self):
        stats = compute_stats([])
        assert stats.mean == 0
        assert stats.p50 == 0

    def test_single(self):
        stats = compute_stats([42.0])
        assert stats.mean == 42.0
        assert stats.p50 == 42.0
        assert stats.min == 42.0
        assert stats.max == 42.0

    def test_std(self):
        stats = compute_stats([2, 4, 4, 4, 5, 5, 7, 9])
        assert abs(stats.mean - 5.0) < 0.01
        assert stats.std > 0

    def test_all_same(self):
        stats = compute_stats([3.0, 3.0, 3.0])
        assert stats.mean == 3.0
        assert stats.std == 0.0
        assert stats.p50 == 3.0
        assert stats.p99 == 3.0


class TestBenchmarkConfig:
    def test_defaults(self):
        c = BenchmarkConfig(url="http://localhost:8000/v1")
        assert c.url == "http://localhost:8000/v1"
        assert c.n_requests == 100
        assert c.mode == BenchmarkMode.STREAMING

    def test_trailing_slash_stripped(self):
        c = BenchmarkConfig(url="http://localhost:8000/v1/")
        assert c.url == "http://localhost:8000/v1"

    def test_empty_url_raises(self):
        with pytest.raises(ValueError, match="url must not be empty"):
            BenchmarkConfig(url="")

    def test_zero_requests_raises(self):
        with pytest.raises(ValueError, match="n_requests"):
            BenchmarkConfig(url="http://x", n_requests=0)

    def test_zero_timeout_raises(self):
        with pytest.raises(ValueError, match="timeout"):
            BenchmarkConfig(url="http://x", timeout=0)

    def test_empty_concurrency_raises(self):
        with pytest.raises(ValueError, match="concurrency_levels"):
            BenchmarkConfig(url="http://x", concurrency_levels=[])

    def test_negative_concurrency_raises(self):
        with pytest.raises(ValueError, match="concurrency level must be >= 1"):
            BenchmarkConfig(url="http://x", concurrency_levels=[0])

    def test_custom(self):
        c = BenchmarkConfig(
            url="http://example.com/v1",
            model="llama-3",
            n_requests=50,
            concurrency_levels=[1, 2, 4],
            mode=BenchmarkMode.NON_STREAMING,
        )
        assert c.model == "llama-3"
        assert c.n_requests == 50
        assert c.mode == BenchmarkMode.NON_STREAMING


class TestRequestResult:
    def test_tokens_per_second(self):
        r = RequestResult(success=True, latency=2.0, output_tokens=100)
        assert r.tokens_per_second == 50.0

    def test_tokens_per_second_zero_latency(self):
        r = RequestResult(success=True, latency=0.0, output_tokens=10)
        assert r.tokens_per_second == 0.0

    def test_tokens_per_second_zero_tokens(self):
        r = RequestResult(success=True, latency=1.0, output_tokens=0)
        assert r.tokens_per_second == 0.0

    def test_failed_request(self):
        r = RequestResult(success=False, latency=0.5, error="timeout")
        assert r.error == "timeout"


class TestBenchmarkReport:
    def _make_report(self):
        r1 = ConcurrencyResult(
            concurrency=1, n_requests=10, n_success=10, n_error=0,
            total_duration=5.0, requests_per_second=2.0, tokens_per_second=100.0,
            latency=LatencyStats(p50=0.4, p75=0.5, p90=0.6, p95=0.7, p99=0.8, mean=0.5, min=0.3, max=0.9, std=0.1),
        )
        r2 = ConcurrencyResult(
            concurrency=8, n_requests=10, n_success=10, n_error=0,
            total_duration=2.0, requests_per_second=5.0, tokens_per_second=500.0,
            latency=LatencyStats(p50=0.8, p75=0.9, p90=1.0, p95=1.1, p99=1.2, mean=0.9, min=0.5, max=1.5, std=0.2),
        )
        return BenchmarkReport(url="http://localhost:8000/v1", model="llama", timestamp="2026-01-01T00:00:00Z", results=[r1, r2])

    def test_best_throughput(self):
        report = self._make_report()
        best = report.best_throughput()
        assert best.concurrency == 8
        assert best.tokens_per_second == 500.0

    def test_lowest_latency(self):
        report = self._make_report()
        low = report.lowest_latency()
        assert low.concurrency == 1
        assert low.latency.p50 == 0.4


class TestBenchmarkMode:
    def test_values(self):
        assert BenchmarkMode.STREAMING.value == "streaming"
        assert BenchmarkMode.NON_STREAMING.value == "non_streaming"


class TestExceptions:
    def test_base(self):
        e = InfermarkError("fail")
        assert isinstance(e, Exception)
        assert str(e) == "fail"
