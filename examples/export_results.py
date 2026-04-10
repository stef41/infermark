"""Export benchmark results to text, CSV, and JSON.

Demonstrates format_report_text, format_csv, save_json, save_csv,
and report round-tripping via load_json.
"""

import tempfile
from pathlib import Path

from infermark import (
    BenchmarkConfig,
    format_csv,
    format_report_text,
    load_json,
    run_benchmark,
    save_csv,
    save_json,
)


def main() -> None:
    # Run a quick benchmark (low request count for demo purposes)
    config = BenchmarkConfig(
        url="http://localhost:8000",
        model="test-model",
        prompt="Hello!",
        max_tokens=32,
        concurrency_levels=[1, 2],
        n_requests=5,
        warmup=0,
        timeout=30.0,
    )

    print("Running benchmark...")
    report = run_benchmark(config)

    # -- Plain-text report ------------------------------------------------
    print("\n=== Text Report ===")
    print(format_report_text(report))

    # -- CSV output -------------------------------------------------------
    print("\n=== CSV Output ===")
    csv_text = format_csv(report)
    print(csv_text[:500])

    # -- Save to files ----------------------------------------------------
    out_dir = Path(tempfile.mkdtemp(prefix="infermark_"))

    json_path = out_dir / "report.json"
    save_json(report, json_path)
    print(f"\nSaved JSON: {json_path}")

    csv_path = out_dir / "report.csv"
    save_csv(report, csv_path)
    print(f"Saved CSV:  {csv_path}")

    # -- Round-trip: load JSON back ---------------------------------------
    loaded = load_json(json_path)
    print(f"\nReloaded JSON — {len(loaded['results'])} concurrency levels")
    for r in loaded["results"]:
        print(f"  concurrency={r['concurrency']}  rps={r['requests_per_second']:.1f}")


if __name__ == "__main__":
    main()
