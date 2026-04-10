"""Tests for infermark.report."""

import json
import pytest

from infermark._types import (
    BenchmarkReport,
    ConcurrencyResult,
    LatencyStats,
)
from infermark.report import (
    format_markdown,
    format_report_text,
    report_to_dict,
    save_json,
    load_json,
)


def _make_stats(base: float = 0.1) -> LatencyStats:
    return LatencyStats(
        p50=base, p75=base * 1.2, p90=base * 1.5, p95=base * 1.8,
        p99=base * 2.5, mean=base * 1.1, min=base * 0.5, max=base * 3, std=base * 0.3,
    )


def _make_report() -> BenchmarkReport:
    results = [
        ConcurrencyResult(
            concurrency=1, n_requests=50, n_success=48, n_error=2,
            total_duration=10.0, requests_per_second=4.8, tokens_per_second=240.0,
            latency=_make_stats(0.2),
            ttft=_make_stats(0.05),
            itl=_make_stats(0.01),
            errors={"timeout": 2},
        ),
        ConcurrencyResult(
            concurrency=8, n_requests=50, n_success=50, n_error=0,
            total_duration=3.0, requests_per_second=16.7, tokens_per_second=835.0,
            latency=_make_stats(0.5),
            ttft=_make_stats(0.08),
            itl=_make_stats(0.015),
        ),
    ]
    return BenchmarkReport(
        url="http://localhost:8000/v1",
        model="llama-3-70b",
        timestamp="2026-04-10T12:00:00Z",
        results=results,
        config={"mode": "streaming", "max_tokens": 256, "n_requests": 50},
    )


class TestFormatReportText:
    def test_contains_url(self):
        text = format_report_text(_make_report())
        assert "localhost:8000" in text

    def test_contains_model(self):
        text = format_report_text(_make_report())
        assert "llama-3-70b" in text

    def test_contains_concurrency_levels(self):
        text = format_report_text(_make_report())
        assert "1" in text and "8" in text

    def test_contains_throughput(self):
        text = format_report_text(_make_report())
        assert "835.0" in text

    def test_contains_summary(self):
        text = format_report_text(_make_report())
        assert "Peak throughput" in text
        assert "Lowest P50" in text


class TestReportToDict:
    def test_roundtrip_json(self):
        d = report_to_dict(_make_report())
        s = json.dumps(d)
        loaded = json.loads(s)
        assert loaded["url"] == "http://localhost:8000/v1"
        assert len(loaded["results"]) == 2

    def test_has_all_fields(self):
        d = report_to_dict(_make_report())
        assert "url" in d
        assert "model" in d
        assert "timestamp" in d
        assert "config" in d
        assert "results" in d

    def test_result_structure(self):
        d = report_to_dict(_make_report())
        r = d["results"][0]
        assert "concurrency" in r
        assert "latency" in r
        assert "ttft" in r
        assert r["latency"]["p50"] > 0

    def test_none_ttft(self):
        report = _make_report()
        report.results[0].ttft = None
        d = report_to_dict(report)
        assert d["results"][0]["ttft"] is None


class TestSaveLoadJson:
    def test_save_and_load(self, tmp_path):
        report = _make_report()
        path = tmp_path / "report.json"
        save_json(report, path)

        loaded = load_json(path)
        assert loaded["url"] == report.url
        assert loaded["model"] == report.model
        assert len(loaded["results"]) == len(report.results)

    def test_creates_parent_dirs(self, tmp_path):
        report = _make_report()
        path = tmp_path / "sub" / "dir" / "report.json"
        save_json(report, path)
        assert path.exists()


class TestFormatMarkdown:
    def test_has_header(self):
        md = format_markdown(_make_report())
        assert "# Benchmark Report" in md

    def test_has_table(self):
        md = format_markdown(_make_report())
        assert "| Conc |" in md
        assert "| 1 |" in md
        assert "| 8 |" in md

    def test_has_url(self):
        md = format_markdown(_make_report())
        assert "localhost:8000" in md
