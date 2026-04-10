"""CLI for infermark."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

try:
    import click
    _HAS_CLICK = True
except ImportError:
    _HAS_CLICK = False

try:
    from rich.console import Console
    _console = Console()
except ImportError:
    _console = None  # type: ignore[assignment]


def _build_cli() -> Any:
    if not _HAS_CLICK:
        return None

    from infermark._types import BenchmarkConfig, BenchmarkMode
    from infermark.report import format_markdown, format_report_rich, format_report_text, save_json
    from infermark.runner import run_benchmark

    @click.group()
    @click.version_option(package_name="infermark")
    def cli() -> None:
        """infermark — benchmark any OpenAI-compatible LLM endpoint."""

    @cli.command()
    @click.argument("url")
    @click.option("-m", "--model", default="", help="Model name to send in requests.")
    @click.option("-n", "--requests", default=50, type=int, help="Requests per concurrency level.")
    @click.option("-c", "--concurrency", default="1,4,8,16,32", help="Comma-separated concurrency levels.")
    @click.option("--max-tokens", default=256, type=int, help="Max output tokens per request.")
    @click.option("--prompt", default=None, help="Custom prompt text.")
    @click.option("--mode", default="streaming", type=click.Choice(["streaming", "non_streaming"]))
    @click.option("--warmup", default=3, type=int, help="Warmup requests before benchmarking.")
    @click.option("--timeout", default=120.0, type=float, help="Request timeout in seconds.")
    @click.option("--api-key", default="", help="API key (or set via OPENAI_API_KEY env).")
    @click.option("-o", "--output", default=None, help="Save JSON report to file.")
    @click.option("--markdown", default=None, help="Save Markdown report to file.")
    def run(
        url: str,
        model: str,
        requests: int,
        concurrency: str,
        max_tokens: int,
        prompt: str | None,
        mode: str,
        warmup: int,
        timeout: float,
        api_key: str,
        output: str | None,
        markdown: str | None,
    ) -> None:
        """Run a benchmark against an endpoint."""
        levels = [int(x.strip()) for x in concurrency.split(",")]
        config = BenchmarkConfig(
            url=url,
            model=model,
            prompt=prompt or "Explain the theory of relativity in simple terms.",
            max_tokens=max_tokens,
            concurrency_levels=levels,
            n_requests=requests,
            timeout=timeout,
            mode=BenchmarkMode(mode),
            warmup=warmup,
            api_key=api_key,
        )

        def on_progress(level: int, result) -> None:  # type: ignore[no-untyped-def]
            tps = result.tokens_per_second
            p50 = result.latency.p50 * 1000
            click.echo(
                f"  concurrency={level:>3}  "
                f"tok/s={tps:>8.1f}  "
                f"p50={p50:>7.1f}ms  "
                f"ok={result.n_success}/{result.n_requests}"
            )

        click.echo(f"Benchmarking {url}")
        click.echo(f"  model={model}  requests={requests}  levels={levels}")
        click.echo(f"  mode={mode}  max_tokens={max_tokens}  warmup={warmup}")
        click.echo("")

        report = run_benchmark(config, on_progress=on_progress)

        click.echo("")
        text = format_report_rich(report) if _console is not None else format_report_text(report)
        click.echo(text)

        if output:
            save_json(report, output)
            click.echo(f"\nJSON report saved to {output}")

        if markdown:
            Path(markdown).write_text(format_markdown(report))
            click.echo(f"Markdown report saved to {markdown}")

    @cli.command()
    @click.argument("files", nargs=-1, required=True, type=click.Path(exists=True))
    def compare(files: tuple) -> None:  # type: ignore[type-arg]
        """Compare multiple JSON reports."""
        from infermark.compare import (
            format_comparison_rich,
            format_comparison_text,
        )
        from infermark.report import load_json

        reports = []
        for f in files:
            data = load_json(f)
            # Reconstruct minimal BenchmarkReport for comparison
            from infermark._types import BenchmarkReport, ConcurrencyResult, LatencyStats
            results = []
            for r in data.get("results", []):
                lat = r["latency"]
                latency = LatencyStats(**lat)
                ttft = LatencyStats(**r["ttft"]) if r.get("ttft") else None
                itl = LatencyStats(**r["itl"]) if r.get("itl") else None
                results.append(ConcurrencyResult(
                    concurrency=r["concurrency"],
                    n_requests=r["n_requests"],
                    n_success=r["n_success"],
                    n_error=r["n_error"],
                    total_duration=r["total_duration"],
                    requests_per_second=r["requests_per_second"],
                    tokens_per_second=r["tokens_per_second"],
                    latency=latency,
                    ttft=ttft,
                    itl=itl,
                    errors=r.get("errors", {}),
                ))
            report = BenchmarkReport(
                url=data["url"],
                model=data["model"],
                timestamp=data["timestamp"],
                results=results,
                config=data.get("config", {}),
            )
            reports.append(report)

        if _console is not None:
            text = format_comparison_rich(reports)
        else:
            text = format_comparison_text(reports)
        click.echo(text)

    return cli


cli = _build_cli()


def main() -> None:
    if cli is None:
        print(
            "The CLI requires extra dependencies. Install with:\n"
            "  pip install infermark[cli]",
            file=sys.stderr,
        )
        sys.exit(1)
    cli()


if __name__ == "__main__":
    main()
