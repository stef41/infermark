"""Tests for infermark.compare."""

from infermark._types import (
    BenchmarkReport,
    ConcurrencyResult,
    LatencyStats,
)
from infermark.compare import compare_reports, format_comparison_text


def _stats(base: float = 0.1) -> LatencyStats:
    return LatencyStats(
        p50=base, p75=base*1.2, p90=base*1.5, p95=base*1.8,
        p99=base*2.5, mean=base*1.1, min=base*0.5, max=base*3, std=base*0.3,
    )


def _make_report(url: str, model: str, tps: float, lat: float) -> BenchmarkReport:
    return BenchmarkReport(
        url=url, model=model, timestamp="2026-01-01",
        results=[
            ConcurrencyResult(
                concurrency=1, n_requests=50, n_success=50, n_error=0,
                total_duration=5.0, requests_per_second=10.0,
                tokens_per_second=tps / 2,
                latency=_stats(lat),
                ttft=_stats(lat * 0.3),
            ),
            ConcurrencyResult(
                concurrency=8, n_requests=50, n_success=50, n_error=0,
                total_duration=2.0, requests_per_second=25.0,
                tokens_per_second=tps,
                latency=_stats(lat * 2),
                ttft=_stats(lat * 0.5),
            ),
        ],
    )


class TestCompareReports:
    def test_basic_comparison(self):
        r1 = _make_report("http://a", "llama", 500, 0.1)
        r2 = _make_report("http://b", "mistral", 800, 0.15)
        comp = compare_reports([r1, r2])
        assert "1" in comp
        assert "8" in comp
        assert len(comp["1"]) == 2
        assert len(comp["8"]) == 2

    def test_single_report(self):
        r1 = _make_report("http://a", "llama", 500, 0.1)
        comp = compare_reports([r1])
        assert len(comp["1"]) == 1

    def test_empty(self):
        assert compare_reports([]) == {}

    def test_different_concurrency_levels(self):
        r1 = BenchmarkReport(url="a", model="m1", timestamp="t", results=[
            ConcurrencyResult(concurrency=1, n_requests=10, n_success=10, n_error=0,
                            total_duration=1, requests_per_second=10, tokens_per_second=100,
                            latency=_stats()),
        ])
        r2 = BenchmarkReport(url="b", model="m2", timestamp="t", results=[
            ConcurrencyResult(concurrency=4, n_requests=10, n_success=10, n_error=0,
                            total_duration=1, requests_per_second=10, tokens_per_second=200,
                            latency=_stats()),
        ])
        comp = compare_reports([r1, r2])
        assert "1" in comp and "4" in comp
        assert len(comp["1"]) == 1
        assert len(comp["4"]) == 1

    def test_entries_have_correct_fields(self):
        r1 = _make_report("http://a", "llama", 500, 0.1)
        comp = compare_reports([r1])
        entry = comp["1"][0]
        assert "url" in entry
        assert "tokens_per_second" in entry
        assert "latency_p50" in entry
        assert "error_rate" in entry


class TestFormatComparisonText:
    def test_has_endpoint_headers(self):
        r1 = _make_report("http://vllm", "llama", 500, 0.1)
        r2 = _make_report("http://tgi", "llama", 600, 0.12)
        text = format_comparison_text([r1, r2])
        assert "vllm" in text
        assert "tgi" in text

    def test_has_throughput(self):
        r1 = _make_report("http://a", "m", 500, 0.1)
        text = format_comparison_text([r1])
        assert "Tok/s" in text

    def test_empty(self):
        assert "No reports" in format_comparison_text([])

    def test_has_winner(self):
        r1 = _make_report("http://a", "m", 500, 0.1)
        r2 = _make_report("http://b", "m", 800, 0.1)
        text = format_comparison_text([r1, r2])
        assert "Highest throughput" in text
