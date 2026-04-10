"""Tests for infermark.cli."""

import pytest

try:
    from click.testing import CliRunner
    _HAS_CLICK = True
except ImportError:
    _HAS_CLICK = False

from infermark._types import (
    BenchmarkReport,
    ConcurrencyResult,
    LatencyStats,
)
from infermark.cli import _build_cli
from infermark.report import save_json

pytestmark = pytest.mark.skipif(not _HAS_CLICK, reason="click not installed")


@pytest.fixture
def cli():
    c = _build_cli()
    assert c is not None
    return c


@pytest.fixture
def runner():
    return CliRunner()


def _stats(base=0.1):
    return LatencyStats(
        p50=base, p75=base*1.2, p90=base*1.5, p95=base*1.8,
        p99=base*2.5, mean=base*1.1, min=base*0.5, max=base*3, std=base*0.3,
    )


def _make_json_report(path, url="http://localhost:8000/v1", model="llama"):
    report = BenchmarkReport(
        url=url, model=model, timestamp="2026-01-01",
        results=[
            ConcurrencyResult(
                concurrency=1, n_requests=10, n_success=10, n_error=0,
                total_duration=2.0, requests_per_second=5.0, tokens_per_second=250.0,
                latency=_stats(0.2), ttft=_stats(0.05), itl=_stats(0.01),
            ),
        ],
    )
    save_json(report, path)


class TestCLIGroup:
    def test_help(self, cli, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "infermark" in result.output

    def test_run_help(self, cli, runner):
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output
        assert "--concurrency" in result.output

    def test_compare_help(self, cli, runner):
        result = runner.invoke(cli, ["compare", "--help"])
        assert result.exit_code == 0


class TestCLICompare:
    def test_compare_two_reports(self, cli, runner, tmp_path):
        r1 = tmp_path / "r1.json"
        r2 = tmp_path / "r2.json"
        _make_json_report(r1, url="http://vllm:8000/v1", model="llama")
        _make_json_report(r2, url="http://tgi:8000/v1", model="llama")

        result = runner.invoke(cli, ["compare", str(r1), str(r2)])
        assert result.exit_code == 0, result.output

    def test_compare_single_report(self, cli, runner, tmp_path):
        r1 = tmp_path / "r1.json"
        _make_json_report(r1)
        result = runner.invoke(cli, ["compare", str(r1)])
        assert result.exit_code == 0


class TestCLIInit:
    def test_init_api(self):
        """The public API exports are correct."""
        import infermark
        assert infermark.__version__
        assert hasattr(infermark, "run_benchmark")
        assert hasattr(infermark, "BenchmarkConfig")
