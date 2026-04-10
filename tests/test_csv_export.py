"""Tests for CSV export (save_csv / format_csv)."""

import csv
import io

import pytest

from infermark._types import (
    BenchmarkReport,
    ConcurrencyResult,
    LatencyStats,
)
from infermark.report import format_csv, save_csv


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


class TestFormatCsv:
    def test_has_header_and_rows(self):
        text = format_csv(_make_report())
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
        assert len(rows) == 2

    def test_header_fields(self):
        text = format_csv(_make_report())
        reader = csv.DictReader(io.StringIO(text))
        fields = reader.fieldnames
        assert "concurrency" in fields
        assert "tokens_per_second" in fields
        assert "latency_p50_ms" in fields
        assert "ttft_p50_ms" in fields
        assert "itl_p50_ms" in fields

    def test_row_values(self):
        text = format_csv(_make_report())
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
        assert rows[0]["concurrency"] == "1"
        assert rows[0]["url"] == "http://localhost:8000/v1"
        assert rows[0]["model"] == "llama-3-70b"
        assert float(rows[0]["tokens_per_second"]) == 240.0

    def test_latency_values_in_ms(self):
        text = format_csv(_make_report())
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
        # base=0.2 => p50=0.2s => 200.0 ms
        assert float(rows[0]["latency_p50_ms"]) == 200.0

    def test_none_ttft_itl(self):
        report = _make_report()
        report.results[0].ttft = None
        report.results[0].itl = None
        text = format_csv(report)
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
        assert rows[0]["ttft_p50_ms"] == ""
        assert rows[0]["itl_p50_ms"] == ""
        # Second row should still have values
        assert rows[1]["ttft_p50_ms"] != ""

    def test_single_result(self):
        report = _make_report()
        report.results = report.results[:1]
        text = format_csv(report)
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
        assert len(rows) == 1


class TestSaveCsv:
    def test_save_and_read(self, tmp_path):
        report = _make_report()
        path = tmp_path / "report.csv"
        save_csv(report, path)

        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["model"] == "llama-3-70b"

    def test_creates_parent_dirs(self, tmp_path):
        report = _make_report()
        path = tmp_path / "sub" / "dir" / "report.csv"
        save_csv(report, path)
        assert path.exists()

    def test_consistent_with_format_csv(self, tmp_path):
        report = _make_report()
        path = tmp_path / "report.csv"
        save_csv(report, path)
        with open(path) as f:
            reader = csv.DictReader(f)
            saved_rows = list(reader)
        formatted_rows = list(csv.DictReader(io.StringIO(format_csv(report))))
        assert saved_rows == formatted_rows
