"""Benchmark a local LLM endpoint at different concurrency levels.

Demonstrates BenchmarkConfig and run_benchmark() against an
OpenAI-compatible server (e.g., vLLM, Ollama, llama.cpp).
"""

from infermark import BenchmarkConfig, BenchmarkMode, run_benchmark, format_report_text


def main() -> None:
    # Configure the benchmark — adjust url/model for your setup
    config = BenchmarkConfig(
        url="http://localhost:8000",
        model="meta-llama/Llama-3-8B-Instruct",
        prompt="Explain the theory of relativity in simple terms.",
        max_tokens=128,
        concurrency_levels=[1, 4, 8, 16],
        n_requests=50,
        mode=BenchmarkMode.STREAMING,
        warmup=2,
        timeout=60.0,
    )

    print(f"Benchmarking {config.url} (model={config.model})")
    print(f"Concurrency levels: {config.concurrency_levels}")
    print(f"Requests per level: {config.n_requests}")
    print()

    # Progress callback
    def on_progress(concurrency: int, result) -> None:
        print(
            f"  concurrency={concurrency:>3}  "
            f"rps={result.requests_per_second:.1f}  "
            f"tok/s={result.tokens_per_second:.1f}  "
            f"p50={result.latency.p50 * 1000:.0f}ms  "
            f"errors={result.n_error}"
        )

    report = run_benchmark(config, on_progress=on_progress)

    # Print full report
    print()
    print(format_report_text(report))


if __name__ == "__main__":
    main()
