"""Load testing with configurable ramp-up profiles.

Executes a user-supplied callable at increasing request rates and
collects latency / success metrics.  No external dependencies — uses
only ``time``, ``threading``, and ``statistics`` from the stdlib.
"""

from __future__ import annotations

import statistics
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class LoadProfile:
    """Describes how request rate changes over time."""

    initial_rps: float
    target_rps: float
    ramp_duration_s: float
    hold_duration_s: float
    cooldown_s: float = 0.0

    def __post_init__(self) -> None:
        if self.initial_rps < 0:
            raise ValueError("initial_rps must be >= 0")
        if self.target_rps < 0:
            raise ValueError("target_rps must be >= 0")
        if self.ramp_duration_s < 0:
            raise ValueError("ramp_duration_s must be >= 0")
        if self.hold_duration_s < 0:
            raise ValueError("hold_duration_s must be >= 0")
        if self.cooldown_s < 0:
            raise ValueError("cooldown_s must be >= 0")


@dataclass
class LoadResult:
    """Result of a single request during a load test."""

    timestamp: float
    rps: float
    latency_ms: float
    success: bool
    error: str | None = None


class LoadTestPlan:
    """Generates the RPS schedule for a load test from a :class:`LoadProfile`."""

    def __init__(self, profile: LoadProfile) -> None:
        self.profile = profile

    @property
    def total_duration(self) -> float:
        return (
            self.profile.ramp_duration_s
            + self.profile.hold_duration_s
            + self.profile.cooldown_s
        )

    def rps_at(self, t: float) -> float:
        """Return the target RPS at time *t* seconds from the start."""
        p = self.profile
        if t < 0:
            return p.initial_rps

        # Phase 1 — ramp
        if t <= p.ramp_duration_s:
            if p.ramp_duration_s == 0:
                return p.target_rps
            frac = t / p.ramp_duration_s
            return p.initial_rps + (p.target_rps - p.initial_rps) * frac

        # Phase 2 — hold
        hold_end = p.ramp_duration_s + p.hold_duration_s
        if t <= hold_end:
            return p.target_rps

        # Phase 3 — cooldown
        cooldown_end = hold_end + p.cooldown_s
        if t <= cooldown_end:
            if p.cooldown_s == 0:
                return p.initial_rps
            elapsed = t - hold_end
            frac = elapsed / p.cooldown_s
            return p.target_rps + (p.initial_rps - p.target_rps) * frac

        return p.initial_rps

    def schedule(self) -> list[tuple[float, float]]:
        """Return ``(time_offset, target_rps)`` pairs at 0.1 s granularity."""
        step = 0.1
        total = self.total_duration
        points: list[tuple[float, float]] = []
        t = 0.0
        while t <= total:
            points.append((round(t, 3), self.rps_at(t)))
            t += step
        # Ensure the final point is included
        if not points or points[-1][0] < total:
            points.append((round(total, 3), self.rps_at(total)))
        return points


class LoadTestRunner:
    """Execute a load test plan against a callable and collect results.

    Parameters
    ----------
    plan:
        The :class:`LoadTestPlan` describing how RPS evolves.
    fn:
        A callable to invoke for each request.  It should take no
        arguments and raise on failure.
    """

    def __init__(self, plan: LoadTestPlan, fn: Callable[[], Any]) -> None:
        self.plan = plan
        self.fn = fn
        self._results: list[LoadResult] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def run_sync(self) -> list[LoadResult]:
        """Execute the load test synchronously and return all results."""
        self._results = []
        total = self.plan.total_duration
        if total <= 0:
            return self._results

        start = time.perf_counter()
        t = 0.0
        step = 0.1  # scheduling granularity in seconds

        while t < total:
            current_rps = self.plan.rps_at(t)
            # How many requests to fire in this step?
            n_requests = max(0, round(current_rps * step))

            threads: list[threading.Thread] = []
            for _ in range(n_requests):
                th = threading.Thread(target=self._fire, args=(current_rps,))
                th.start()
                threads.append(th)
            for th in threads:
                th.join()

            # Advance wall-clock
            elapsed = time.perf_counter() - start
            t = elapsed
            # Sleep until next step if ahead of schedule
            next_slot = t + step
            remaining_sleep = (start + next_slot) - time.perf_counter()
            if remaining_sleep > 0 and t < total:
                time.sleep(remaining_sleep)
            t = time.perf_counter() - start

        return list(self._results)

    # ------------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        """Aggregate results into a summary dict."""
        results = self._results
        if not results:
            return {
                "total_requests": 0,
                "success_rate": 0.0,
                "p50_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "p99_latency_ms": 0.0,
                "peak_rps": 0.0,
                "mean_latency_ms": 0.0,
                "min_latency_ms": 0.0,
                "max_latency_ms": 0.0,
            }

        latencies = [r.latency_ms for r in results]
        successes = sum(1 for r in results if r.success)
        peak_rps = max(r.rps for r in results) if results else 0.0

        sorted_lat = sorted(latencies)
        return {
            "total_requests": len(results),
            "success_rate": successes / len(results) if results else 0.0,
            "p50_latency_ms": _percentile(sorted_lat, 50),
            "p95_latency_ms": _percentile(sorted_lat, 95),
            "p99_latency_ms": _percentile(sorted_lat, 99),
            "peak_rps": peak_rps,
            "mean_latency_ms": statistics.mean(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
        }

    # ------------------------------------------------------------------
    def _fire(self, current_rps: float) -> None:
        ts = time.perf_counter()
        try:
            self.fn()
            latency = (time.perf_counter() - ts) * 1000
            result = LoadResult(
                timestamp=ts,
                rps=current_rps,
                latency_ms=latency,
                success=True,
            )
        except Exception as exc:
            latency = (time.perf_counter() - ts) * 1000
            result = LoadResult(
                timestamp=ts,
                rps=current_rps,
                latency_ms=latency,
                success=False,
                error=str(exc),
            )
        with self._lock:
            self._results.append(result)


# ======================================================================
# Percentile helper (stdlib only)
# ======================================================================

def _percentile(sorted_data: list[float], pct: float) -> float:
    """Return the *pct*-th percentile (nearest-rank) from sorted data."""
    if not sorted_data:
        return 0.0
    k = max(0, min(len(sorted_data) - 1, int(len(sorted_data) * pct / 100)))
    return sorted_data[k]


# ======================================================================
# Report formatting
# ======================================================================

def format_load_report(summary: dict[str, Any]) -> str:
    """Format a load-test summary dict into a human-readable string."""
    if not summary or summary.get("total_requests", 0) == 0:
        return "No load test results."

    lines = [
        "Load Test Report",
        "=" * 40,
        f"  Total requests:   {summary['total_requests']}",
        f"  Success rate:     {summary['success_rate']:.2%}",
        f"  Peak RPS:         {summary['peak_rps']:.1f}",
        "-" * 40,
        f"  Mean latency:     {summary['mean_latency_ms']:.2f} ms",
        f"  Min  latency:     {summary['min_latency_ms']:.2f} ms",
        f"  Max  latency:     {summary['max_latency_ms']:.2f} ms",
        f"  p50  latency:     {summary['p50_latency_ms']:.2f} ms",
        f"  p95  latency:     {summary['p95_latency_ms']:.2f} ms",
        f"  p99  latency:     {summary['p99_latency_ms']:.2f} ms",
    ]
    return "\n".join(lines)
