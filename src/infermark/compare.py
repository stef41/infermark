"""Compare multiple benchmark reports side by side."""

from __future__ import annotations

from typing import Any

from infermark._types import BenchmarkReport, ConcurrencyResult


def _find_matching_concurrency(
    report: BenchmarkReport, concurrency: int
) -> ConcurrencyResult | None:
    """Find the result for a specific concurrency level."""
    for r in report.results:
        if r.concurrency == concurrency:
            return r
    return None


def compare_reports(reports: list[BenchmarkReport]) -> dict[str, Any]:
    """Compare multiple reports, returning a structured comparison.

    Returns a dict keyed by concurrency level, with per-endpoint metrics.
    """
    if not reports:
        return {}

    all_levels: set[int] = set()
    for report in reports:
        for r in report.results:
            all_levels.add(r.concurrency)

    comparison: dict[str, Any] = {}
    for level in sorted(all_levels):
        entries: list[dict[str, Any]] = []
        for report in reports:
            result = _find_matching_concurrency(report, level)
            if result is None:
                continue
            entries.append({
                "url": report.url,
                "model": report.model,
                "tokens_per_second": result.tokens_per_second,
                "requests_per_second": result.requests_per_second,
                "latency_p50": result.latency.p50,
                "latency_p95": result.latency.p95,
                "latency_p99": result.latency.p99,
                "ttft_p50": result.ttft.p50 if result.ttft else None,
                "itl_p50": result.itl.p50 if result.itl else None,
                "error_rate": result.n_error / result.n_requests if result.n_requests > 0 else 0,
            })
        comparison[str(level)] = entries

    return comparison


def format_comparison_text(reports: list[BenchmarkReport]) -> str:
    """Format a side-by-side comparison as text."""
    if not reports:
        return "No reports to compare."

    comp = compare_reports(reports)
    lines: list[str] = []
    lines.append("=== Endpoint Comparison ===\n")

    for level, entries in sorted(comp.items(), key=lambda x: int(x[0])):
        lines.append(f"Concurrency: {level}")
        lines.append(f"{'Endpoint':<50} {'Tok/s':>8} {'P50ms':>8} {'P95ms':>8} {'TTFT':>8} {'Err%':>6}")
        lines.append("-" * 90)
        for e in sorted(entries, key=lambda x: -x["tokens_per_second"]):
            label = f"{e['url']} ({e['model']})"[:50]
            ttft = f"{e['ttft_p50']*1000:.1f}" if e['ttft_p50'] is not None else "N/A"
            lines.append(
                f"{label:<50} {e['tokens_per_second']:>8.1f} "
                f"{e['latency_p50']*1000:>8.1f} {e['latency_p95']*1000:>8.1f} "
                f"{ttft:>8} {e['error_rate']*100:>5.1f}%"
            )
        lines.append("")

    # Overall winner
    best_tps = 0.0
    best_label = ""
    for entries in comp.values():
        for e in entries:
            if e["tokens_per_second"] > best_tps:
                best_tps = e["tokens_per_second"]
                best_label = f"{e['url']} ({e['model']})"

    if best_label:
        lines.append(f"Highest throughput: {best_label} at {best_tps:.1f} tok/s")

    return "\n".join(lines)


def format_comparison_rich(reports: list[BenchmarkReport]) -> str:
    """Format comparison using rich (returns rendered text)."""
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:
        return format_comparison_text(reports)

    if not reports:
        return "No reports to compare."

    console = Console(record=True, width=140)
    comp = compare_reports(reports)

    for level, entries in sorted(comp.items(), key=lambda x: int(x[0])):
        table = Table(title=f"Concurrency {level}", show_lines=True)
        table.add_column("Endpoint", style="cyan")
        table.add_column("Model", style="white")
        table.add_column("Tok/s", justify="right", style="yellow bold")
        table.add_column("P50 (ms)", justify="right")
        table.add_column("P95 (ms)", justify="right")
        table.add_column("TTFT P50", justify="right", style="magenta")
        table.add_column("Error %", justify="right")

        for e in sorted(entries, key=lambda x: -x["tokens_per_second"]):
            ttft = f"{e['ttft_p50']*1000:.1f}" if e['ttft_p50'] is not None else "-"
            err_style = "red" if e["error_rate"] > 0.01 else "green"
            table.add_row(
                e["url"],
                e["model"],
                f"{e['tokens_per_second']:.1f}",
                f"{e['latency_p50']*1000:.1f}",
                f"{e['latency_p95']*1000:.1f}",
                ttft,
                f"[{err_style}]{e['error_rate']*100:.1f}%[/{err_style}]",
            )
        console.print(table)

    return console.export_text()
