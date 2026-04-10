#!/usr/bin/env python3
"""Generate SVG terminal screenshots for README."""

from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.text import Text


ASSETS = Path(__file__).parent.parent / "assets"
ASSETS.mkdir(exist_ok=True)


def generate_demo():
    console = Console(record=True, width=110)

    console.print()
    console.print("[bold white]$ infermark run http://localhost:8000/v1 --model meta-llama/Llama-3-70B -n 50[/bold white]")
    console.print()
    console.print("[bold]Benchmark:[/bold] http://localhost:8000/v1")
    console.print("[bold]Model:[/bold] meta-llama/Llama-3-70B-Instruct")
    console.print("[bold]Time:[/bold] 2026-04-10T14:32:01Z")
    console.print()

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

    data = [
        ("1",  "50", "50", "[dim]0[/dim]", "3.2",  "162.4",  "308.2", "412.5", "498.1", "45.2",  "8.3"),
        ("4",  "50", "50", "[dim]0[/dim]", "11.8", "589.6",  "337.1", "478.3", "562.0", "51.8",  "9.1"),
        ("8",  "50", "50", "[dim]0[/dim]", "21.4", "1071.2", "372.4", "534.8", "641.2", "58.4",  "9.7"),
        ("16", "50", "50", "[dim]0[/dim]", "35.7", "1784.5", "445.2", "687.3", "823.1", "72.1",  "11.2"),
        ("32", "50", "49", "[red]1[/red]", "42.1", "2103.8", "756.3", "1245.7","1802.4","128.5", "14.8"),
    ]
    for row in data:
        table.add_row(*row)

    console.print(table)
    console.print()
    console.print("[bold green]Peak throughput:[/bold green] 2103.8 tok/s at concurrency 32")
    console.print("[bold blue]Lowest P50 latency:[/bold blue] 308.2 ms at concurrency 1")
    console.print()

    svg = console.export_svg(title="infermark — LLM Inference Benchmarking")
    (ASSETS / "demo.svg").write_text(svg)
    print(f"  ✓ demo.svg ({len(svg)} bytes)")


def generate_comparison():
    console = Console(record=True, width=120)

    console.print()
    console.print("[bold white]$ infermark compare vllm_report.json tgi_report.json ollama_report.json[/bold white]")
    console.print()

    table = Table(title="Concurrency 16 — Endpoint Comparison", show_lines=True)
    table.add_column("Endpoint", style="cyan")
    table.add_column("Model", style="white")
    table.add_column("Tok/s", justify="right", style="yellow bold")
    table.add_column("P50 (ms)", justify="right")
    table.add_column("P95 (ms)", justify="right")
    table.add_column("TTFT P50", justify="right", style="magenta")
    table.add_column("Error %", justify="right")

    table.add_row("http://gpu1:8000/v1", "Llama-3-70B (vLLM)", "[bold]1784.5[/bold]", "445.2", "687.3", "72.1", "[green]0.0%[/green]")
    table.add_row("http://gpu2:8080/v1", "Llama-3-70B (TGI)",  "1523.1", "521.8", "812.4", "89.3", "[green]0.0%[/green]")
    table.add_row("http://gpu3:11434/v1", "Llama-3-70B (Ollama)", "847.2", "943.6", "1521.3", "156.7", "[red]2.0%[/red]")

    console.print(table)
    console.print()
    console.print("[bold green]Winner:[/bold green] vLLM — 17% faster than TGI, 2.1x faster than Ollama")
    console.print()

    svg = console.export_svg(title="infermark — Endpoint Comparison")
    (ASSETS / "comparison.svg").write_text(svg)
    print(f"  ✓ comparison.svg ({len(svg)} bytes)")


if __name__ == "__main__":
    print("Generating infermark assets...")
    generate_demo()
    generate_comparison()
    print("Done!")
