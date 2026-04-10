"""Tests for infermark.runner — aggregate logic (no real HTTP)."""

from infermark._types import (
    RequestResult,
)
from infermark.runner import _aggregate_results


class TestAggregateResults:
    def test_all_success(self):
        results = [
            RequestResult(success=True, latency=0.5, ttft=0.1, itl=[0.02, 0.03], output_tokens=50),
            RequestResult(success=True, latency=0.6, ttft=0.12, itl=[0.025, 0.035], output_tokens=55),
            RequestResult(success=True, latency=0.4, ttft=0.08, itl=[0.015, 0.025], output_tokens=45),
        ]
        cr = _aggregate_results(results, concurrency=2, total_duration=1.0)
        assert cr.concurrency == 2
        assert cr.n_requests == 3
        assert cr.n_success == 3
        assert cr.n_error == 0
        assert cr.tokens_per_second == 150.0  # (50+55+45) / 1.0
        assert cr.requests_per_second == 3.0
        assert cr.latency.mean > 0
        assert cr.ttft is not None
        assert cr.itl is not None

    def test_mixed_success_failure(self):
        results = [
            RequestResult(success=True, latency=0.5, output_tokens=50),
            RequestResult(success=False, latency=0.1, error="timeout"),
            RequestResult(success=False, latency=0.2, error="connection_error: refused"),
        ]
        cr = _aggregate_results(results, concurrency=1, total_duration=1.0)
        assert cr.n_success == 1
        assert cr.n_error == 2
        assert "timeout" in cr.errors
        assert "connection_error" in cr.errors

    def test_all_failures(self):
        results = [
            RequestResult(success=False, latency=0.1, error="timeout"),
            RequestResult(success=False, latency=0.2, error="timeout"),
        ]
        cr = _aggregate_results(results, concurrency=1, total_duration=1.0)
        assert cr.n_success == 0
        assert cr.n_error == 2
        assert cr.tokens_per_second == 0.0
        assert cr.latency.mean == 0  # no successful latencies

    def test_no_ttft(self):
        results = [
            RequestResult(success=True, latency=0.5, output_tokens=50),
        ]
        cr = _aggregate_results(results, concurrency=1, total_duration=1.0)
        assert cr.ttft is None  # no TTFT data

    def test_empty(self):
        cr = _aggregate_results([], concurrency=1, total_duration=1.0)
        assert cr.n_requests == 0
        assert cr.n_success == 0

    def test_tokens_per_second_calculation(self):
        results = [
            RequestResult(success=True, latency=1.0, output_tokens=100),
            RequestResult(success=True, latency=1.0, output_tokens=200),
        ]
        cr = _aggregate_results(results, concurrency=2, total_duration=2.0)
        assert cr.tokens_per_second == 150.0  # 300 tokens / 2 seconds

    def test_error_counting(self):
        results = [
            RequestResult(success=False, latency=0.1, error="timeout"),
            RequestResult(success=False, latency=0.1, error="timeout"),
            RequestResult(success=False, latency=0.1, error="connection_error: x"),
        ]
        cr = _aggregate_results(results, concurrency=1, total_duration=1.0)
        assert cr.errors["timeout"] == 2
        assert cr.errors["connection_error"] == 1

    def test_itl_aggregated(self):
        results = [
            RequestResult(success=True, latency=0.5, itl=[0.01, 0.02], output_tokens=3),
            RequestResult(success=True, latency=0.5, itl=[0.015, 0.025], output_tokens=3),
        ]
        cr = _aggregate_results(results, concurrency=1, total_duration=1.0)
        assert cr.itl is not None
        # 4 ITL values total
        assert cr.itl.mean > 0
