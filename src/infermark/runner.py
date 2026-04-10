"""Benchmark orchestration — run concurrent requests and aggregate results."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from infermark._types import (
    BenchmarkConfig,
    BenchmarkReport,
    ConcurrencyResult,
    RequestResult,
    compute_stats,
)
from infermark.client import send_request


def _aggregate_results(
    results: list[RequestResult],
    concurrency: int,
    total_duration: float,
) -> ConcurrencyResult:
    """Aggregate individual request results into a ConcurrencyResult."""
    successes = [r for r in results if r.success]
    failures = [r for r in results if not r.success]

    # Error breakdown
    error_counts: dict[str, int] = {}
    for r in failures:
        key = (r.error or "unknown").split(":")[0]
        error_counts[key] = error_counts.get(key, 0) + 1

    # Latency stats (successful only)
    latencies = [r.latency for r in successes]
    latency_stats = compute_stats(latencies)

    # TTFT stats (streaming, successful with TTFT)
    ttfts = [r.ttft for r in successes if r.ttft is not None]
    ttft_stats = compute_stats(ttfts) if ttfts else None

    # ITL stats (all ITL values across all requests)
    all_itl: list[float] = []
    for r in successes:
        all_itl.extend(r.itl)
    itl_stats = compute_stats(all_itl) if all_itl else None

    # Throughput
    total_output_tokens = sum(r.output_tokens for r in successes)
    rps = len(successes) / total_duration if total_duration > 0 else 0.0
    tps = total_output_tokens / total_duration if total_duration > 0 else 0.0

    return ConcurrencyResult(
        concurrency=concurrency,
        n_requests=len(results),
        n_success=len(successes),
        n_error=len(failures),
        total_duration=total_duration,
        requests_per_second=rps,
        tokens_per_second=tps,
        latency=latency_stats,
        ttft=ttft_stats,
        itl=itl_stats,
        errors=error_counts,
    )


async def _run_concurrency_level(
    config: BenchmarkConfig,
    concurrency: int,
    client: httpx.AsyncClient,
) -> ConcurrencyResult:
    """Run benchmark at a single concurrency level."""
    sem = asyncio.Semaphore(concurrency)
    results: list[RequestResult] = []

    async def _worker() -> RequestResult:
        async with sem:
            return await send_request(
                client=client,
                url=config.url,
                model=config.model,
                prompt=config.prompt,
                max_tokens=config.max_tokens,
                mode=config.mode,
                api_key=config.api_key,
                timeout=config.timeout,
                extra_body=config.extra_body,
            )

    start = time.perf_counter()
    tasks = [asyncio.create_task(_worker()) for _ in range(config.n_requests)]
    results = await asyncio.gather(*tasks)
    total_duration = time.perf_counter() - start

    return _aggregate_results(list(results), concurrency, total_duration)


async def _warmup(config: BenchmarkConfig, client: httpx.AsyncClient) -> None:
    """Send warmup requests to prime the server."""
    for _ in range(config.warmup):
        await send_request(
            client=client,
            url=config.url,
            model=config.model,
            prompt=config.prompt,
            max_tokens=min(config.max_tokens, 16),
            mode=config.mode,
            api_key=config.api_key,
            timeout=config.timeout,
        )


async def run_benchmark_async(
    config: BenchmarkConfig,
    on_progress: Any | None = None,
) -> BenchmarkReport:
    """Run a full benchmark across all concurrency levels.

    Parameters
    ----------
    config:
        Benchmark configuration.
    on_progress:
        Optional callback(concurrency: int, result: ConcurrencyResult) called
        after each concurrency level completes.
    """
    async with httpx.AsyncClient() as client:
        if config.warmup > 0:
            await _warmup(config, client)

        results: list[ConcurrencyResult] = []
        for level in sorted(config.concurrency_levels):
            result = await _run_concurrency_level(config, level, client)
            results.append(result)
            if on_progress is not None:
                on_progress(level, result)

    return BenchmarkReport(
        url=config.url,
        model=config.model,
        timestamp=datetime.now(timezone.utc).isoformat(),
        results=results,
        config={
            "prompt_length": len(config.prompt),
            "max_tokens": config.max_tokens,
            "n_requests": config.n_requests,
            "mode": config.mode.value,
            "warmup": config.warmup,
        },
    )


def run_benchmark(
    config: BenchmarkConfig,
    on_progress: Any | None = None,
) -> BenchmarkReport:
    """Synchronous wrapper for run_benchmark_async."""
    return asyncio.run(run_benchmark_async(config, on_progress))
