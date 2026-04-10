"""Report formatting — rich terminal output, JSON, and markdown."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from infermark._types import BenchmarkReport, ConcurrencyResult, LatencyStats


def _fmt_ms(seconds: float) -> str:
    """Format seconds as milliseconds with 1 decimal."""
    return f"{seconds * 1000:.1f}"


def _fmt_tps(tps: float) -> str:
    """Format tokens per second."""
    return f"{tps:.1f}"


def _stats_row(label: str, stats: Optional[LatencyStats]) -> Dict[str, str]:
    if stats is None:
        return {label: "N/A"}
    return {
        "p50": _fmt_ms(stats.p50),
        "p95": _fmt_ms(stats.p95),
        "p99": _fmt_ms(stats.p99),
        "mean": _fmt_ms(stats.mean),
        "min": _fmt_ms(stats.min),
        "max": _fmt_ms(stats.max),
    }


def format_report_text(report: BenchmarkReport) -> str:
    """Format report as plain text table."""
    lines: list[str] = []
    lines.append(f"Benchmark: {report.url}")
    lines.append(f"Model: {report.model}")
    lines.append(f"Timestamp: {report.timestamp}")
    lines.append("")

    header = f"{'Conc':>5} {'Reqs':>5} {'OK':>5} {'Err':>4} {'RPS':>7} {'Tok/s':>8} {'P50ms':>8} {'P95ms':>8} {'P99ms':>8} {'TTFT-P50':>9} {'ITL-P50':>8}"
    lines.append(header)
    lines.append("-" * len(header))

    for r in report.results:
        ttft_p50 = _fmt_ms(r.ttft.p50) if r.ttft else "N/A"
        itl_p50 = _fmt_ms(r.itl.p50) if r.itl else "N/A"
        lines.append(
            f"{r.concurrency:>5} {r.n_requests:>5} {r.n_success:>5} {r.n_error:>4} "
            f"{r.requests_per_second:>7.1f} {r.tokens_per_second:>8.1f} "
            f"{_fmt_ms(r.latency.p50):>8} {_fmt_ms(r.latency.p95):>8} {_fmt_ms(r.latency.p99):>8} "
            f"{ttft_p50:>9} {itl_p50:>8}"
        )

    # Summary
    best = report.best_throughput()
    lines.append("")
    lines.append(f"Peak throughput: {best.tokens_per_second:.1f} tok/s at concurrency {best.concurrency}")
    low = report.lowest_latency()
    lines.append(f"Lowest P50 latency: {_fmt_ms(low.latency.p50)} ms at concurrency {low.concurrency}")

    return "\n".join(lines)


def format_report_rich(report: BenchmarkReport) -> str:
    """Format report using rich markup (for Console.print)."""
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:
        return format_report_text(report)

    console = Console(record=True, width=120)
    console.print(f"\n[bold]Benchmark:[/bold] {report.url}")
    console.print(f"[bold]Model:[/bold] {report.model}")
    console.print(f"[bold]Time:[/bold] {report.timestamp}\n")

    table = Table(title="Results by Concurrency", show_lines=True)
    table.add_column("Conc", justify="right", style="cyan bold")
    table.add_column("Reqs", justify="right")
    table.add_column("OK", justify="right", style="green")
    table.add_column("Err", justify="right", style="red")
    table.add_column("RPS", justify="right")
    table.add_column("Tok/s", justify="right", style="yellow bold")
    table.add_column("P50 (ms)", justify="right")
    table.add_column("P95 (ms)", justify="right")
    table.add_column("P99 (ms)", justify="right")
    table.add_column("TTFT P50", justify="right", style="magenta")
    table.add_column("ITL P50", justify="right", style="blue")

    for r in report.results:
        ttft = _fmt_ms(r.ttft.p50) if r.ttft else "-"
        itl = _fmt_ms(r.itl.p50) if r.itl else "-"
        err_style = "red" if r.n_error > 0 else "dim"
        table.add_row(
            str(r.concurrency),
            str(r.n_requests),
            str(r.n_success),
            f"[{err_style}]{r.n_error}[/{err_style}]",
            f"{r.requests_per_second:.1f}",
            f"{r.tokens_per_second:.1f}",
            _fmt_ms(r.latency.p50),
            _fmt_ms(r.latency.p95),
            _fmt_ms(r.latency.p99),
            ttft,
            itl,
        )

    console.print(table)

    best = report.best_throughput()
    low = report.lowest_latency()
    console.print(f"\n[bold green]Peak throughput:[/bold green] {best.tokens_per_second:.1f} tok/s at concurrency {best.concurrency}")
    console.print(f"[bold blue]Lowest P50 latency:[/bold blue] {_fmt_ms(low.latency.p50)} ms at concurrency {low.concurrency}")

    return console.export_text()


def report_to_dict(report: BenchmarkReport) -> Dict[str, Any]:
    """Convert a report to a serializable dict."""
    def _stats_dict(s: Optional[LatencyStats]) -> Optional[Dict[str, float]]:
        if s is None:
            return None
        return {
            "p50": s.p50, "p75": s.p75, "p90": s.p90, "p95": s.p95, "p99": s.p99,
            "mean": s.mean, "min": s.min, "max": s.max, "std": s.std,
        }

    return {
        "url": report.url,
        "model": report.model,
        "timestamp": report.timestamp,
        "config": report.config,
        "results": [
            {
                "concurrency": r.concurrency,
                "n_requests": r.n_requests,
                "n_success": r.n_success,
                "n_error": r.n_error,
                "total_duration": r.total_duration,
                "requests_per_second": r.requests_per_second,
                "tokens_per_second": r.tokens_per_second,
                "latency": _stats_dict(r.latency),
                "ttft": _stats_dict(r.ttft),
                "itl": _stats_dict(r.itl),
                "errors": r.errors,
            }
            for r in report.results
        ],
    }


def save_json(report: BenchmarkReport, path: Union[str, Path]) -> None:
    """Save report as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report_to_dict(report), f, indent=2)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a JSON report from disk."""
    with open(path) as f:
        return json.load(f)


def format_markdown(report: BenchmarkReport) -> str:
    """Format report as a Markdown table."""
    lines: list[str] = []
    lines.append(f"# Benchmark Report")
    lines.append(f"")
    lines.append(f"- **URL:** {report.url}")
    lines.append(f"- **Model:** {report.model}")
    lines.append(f"- **Time:** {report.timestamp}")
    lines.append("")
    lines.append("| Conc | Reqs | OK | Err | RPS | Tok/s | P50 (ms) | P95 (ms) | P99 (ms) | TTFT P50 | ITL P50 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for r in report.results:
        ttft = _fmt_ms(r.ttft.p50) if r.ttft else "-"
        itl = _fmt_ms(r.itl.p50) if r.itl else "-"
        lines.append(
            f"| {r.concurrency} | {r.n_requests} | {r.n_success} | {r.n_error} "
            f"| {r.requests_per_second:.1f} | {r.tokens_per_second:.1f} "
            f"| {_fmt_ms(r.latency.p50)} | {_fmt_ms(r.latency.p95)} | {_fmt_ms(r.latency.p99)} "
            f"| {ttft} | {itl} |"
        )

    return "\n".join(lines)
