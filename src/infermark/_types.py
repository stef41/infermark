"""Core types for infermark."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence


class BenchmarkMode(str, Enum):
    """How to send requests."""

    STREAMING = "streaming"
    NON_STREAMING = "non_streaming"


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    url: str
    model: str = ""
    prompt: str = "Explain the theory of relativity in simple terms."
    max_tokens: int = 256
    concurrency_levels: List[int] = field(default_factory=lambda: [1, 4, 8, 16, 32])
    n_requests: int = 100
    timeout: float = 120.0
    mode: BenchmarkMode = BenchmarkMode.STREAMING
    warmup: int = 3
    api_key: str = ""
    extra_body: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.url = self.url.rstrip("/")
        if not self.url:
            raise ValueError("url must not be empty")
        if self.n_requests < 1:
            raise ValueError("n_requests must be >= 1")
        if self.timeout <= 0:
            raise ValueError("timeout must be > 0")
        if not self.concurrency_levels:
            raise ValueError("concurrency_levels must not be empty")
        for c in self.concurrency_levels:
            if c < 1:
                raise ValueError(f"concurrency level must be >= 1, got {c}")


@dataclass
class RequestResult:
    """Result of a single benchmark request."""

    success: bool
    latency: float  # total end-to-end seconds
    ttft: Optional[float] = None  # time to first token (streaming only)
    itl: List[float] = field(default_factory=list)  # inter-token latencies
    output_tokens: int = 0
    input_tokens: int = 0
    error: Optional[str] = None

    @property
    def tokens_per_second(self) -> float:
        if self.latency <= 0 or self.output_tokens <= 0:
            return 0.0
        return self.output_tokens / self.latency


@dataclass
class LatencyStats:
    """Percentile-based latency statistics."""

    p50: float
    p75: float
    p90: float
    p95: float
    p99: float
    mean: float
    min: float
    max: float
    std: float


@dataclass
class ConcurrencyResult:
    """Aggregated results for one concurrency level."""

    concurrency: int
    n_requests: int
    n_success: int
    n_error: int
    total_duration: float
    requests_per_second: float
    tokens_per_second: float
    latency: LatencyStats
    ttft: Optional[LatencyStats] = None
    itl: Optional[LatencyStats] = None
    errors: Dict[str, int] = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    """Full benchmark report."""

    url: str
    model: str
    timestamp: str
    results: List[ConcurrencyResult]
    config: Dict[str, Any] = field(default_factory=dict)

    def best_throughput(self) -> ConcurrencyResult:
        """Return the concurrency level with the highest throughput."""
        return max(self.results, key=lambda r: r.tokens_per_second)

    def lowest_latency(self) -> ConcurrencyResult:
        """Return the concurrency level with the lowest p50 latency."""
        return min(self.results, key=lambda r: r.latency.p50)


def percentile(values: Sequence[float], p: float) -> float:
    """Calculate the p-th percentile of a sorted list of values."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    n = len(sorted_v)
    k = (p / 100.0) * (n - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_v[int(k)]
    return sorted_v[f] * (c - k) + sorted_v[c] * (k - f)


def compute_percentiles(
    values: Sequence[float],
    percentiles: Sequence[float] = (50, 75, 90, 95, 99),
) -> Dict[float, float]:
    """Compute arbitrary percentiles from a sequence of values.

    Parameters
    ----------
    values:
        Raw latency (or other metric) values.
    percentiles:
        Which percentiles to compute.  Each value must be in [0, 100].

    Returns
    -------
    Dict mapping each requested percentile to its computed value.
    """
    if not values:
        return {p: 0.0 for p in percentiles}
    for p in percentiles:
        if not 0 <= p <= 100:
            raise ValueError(f"percentile must be in [0, 100], got {p}")
    return {p: percentile(values, p) for p in percentiles}


def compute_stats(values: Sequence[float]) -> LatencyStats:
    """Compute latency statistics from a list of values."""
    if not values:
        return LatencyStats(
            p50=0, p75=0, p90=0, p95=0, p99=0,
            mean=0, min=0, max=0, std=0,
        )
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n if n > 1 else 0.0
    return LatencyStats(
        p50=percentile(values, 50),
        p75=percentile(values, 75),
        p90=percentile(values, 90),
        p95=percentile(values, 95),
        p99=percentile(values, 99),
        mean=mean,
        min=min(values),
        max=max(values),
        std=math.sqrt(variance),
    )


class InfermarkError(Exception):
    """Base exception for infermark."""


class ConnectionError(InfermarkError):
    """Failed to connect to the endpoint."""


class TimeoutError(InfermarkError):
    """Request timed out."""
