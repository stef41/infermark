"""Latency histogram visualization.

Renders ASCII histograms and CDF plots for latency distributions.
Pure Python — uses only ``math`` and ``statistics`` from the stdlib.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass
class HistogramConfig:
    """Configuration for histogram rendering."""

    bins: int = 20
    width: int = 60
    show_percentiles: bool = True
    unit: str = "ms"

    def __post_init__(self) -> None:
        if self.bins < 1:
            raise ValueError("bins must be >= 1")
        if self.width < 10:
            raise ValueError("width must be >= 10")


@dataclass
class HistogramData:
    """Computed histogram data."""

    bin_edges: list[float] = field(default_factory=list)
    counts: list[int] = field(default_factory=list)
    total: int = 0
    min_val: float = 0.0
    max_val: float = 0.0
    mean: float = 0.0
    std: float = 0.0


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------


def compute_bins(
    values: Sequence[float],
    n_bins: int,
) -> tuple[list[float], list[int]]:
    """Compute histogram bin edges and counts.

    Returns ``(edges, counts)`` where ``len(edges) == n_bins + 1`` and
    ``len(counts) == n_bins``.
    """
    if not values:
        return [], []

    v_min = min(values)
    v_max = max(values)

    # If all values are identical, create a single-width bin
    if v_min == v_max:
        return [v_min, v_min + 1.0], [len(values)]

    step = (v_max - v_min) / n_bins
    edges = [v_min + i * step for i in range(n_bins)] + [v_max]
    counts = [0] * n_bins

    for v in values:
        idx = int((v - v_min) / step)
        if idx >= n_bins:
            idx = n_bins - 1
        counts[idx] += 1

    return edges, counts


# ---------------------------------------------------------------------------
# LatencyHistogram
# ---------------------------------------------------------------------------


class LatencyHistogram:
    """Compute and render latency histograms.

    Parameters
    ----------
    config:
        Optional :class:`HistogramConfig`.
    """

    def __init__(self, config: HistogramConfig | None = None) -> None:
        self.config = config or HistogramConfig()

    # -- public API ---------------------------------------------------------

    def compute(self, latencies: Sequence[float]) -> HistogramData:
        """Compute histogram bins and basic statistics."""
        if not latencies:
            return HistogramData()

        vals = list(latencies)
        edges, counts = compute_bins(vals, self.config.bins)
        mean = statistics.mean(vals)
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0

        return HistogramData(
            bin_edges=edges,
            counts=counts,
            total=len(vals),
            min_val=min(vals),
            max_val=max(vals),
            mean=mean,
            std=std,
        )

    def render_ascii(self, data: HistogramData) -> str:
        """Render an ASCII horizontal bar chart."""
        if not data.counts:
            return "(no data)"

        max_count = max(data.counts)
        bar_width = self.config.width
        lines: list[str] = []

        for i, count in enumerate(data.counts):
            lo = data.bin_edges[i]
            hi = data.bin_edges[i + 1]
            bar_len = int(count / max_count * bar_width) if max_count else 0
            bar = "\u2588" * bar_len
            label = f"[{lo:8.2f}, {hi:8.2f})"
            lines.append(f"{label} | {bar} {count}")

        unit = self.config.unit
        lines.append("")
        lines.append(
            f"  total={data.total}  min={data.min_val:.2f}{unit}"
            f"  max={data.max_val:.2f}{unit}  mean={data.mean:.2f}{unit}"
            f"  std={data.std:.2f}{unit}"
        )
        return "\n".join(lines)

    def percentiles(
        self,
        latencies: Sequence[float],
        pcts: Sequence[int] = (50, 90, 95, 99),
    ) -> dict[int, float]:
        """Compute latency percentiles.

        Uses linear interpolation.
        """
        if not latencies:
            return {p: 0.0 for p in pcts}

        sorted_vals = sorted(latencies)
        n = len(sorted_vals)
        result: dict[int, float] = {}

        for p in pcts:
            if n == 1:
                result[p] = sorted_vals[0]
                continue
            rank = p / 100.0 * (n - 1)
            lo = int(math.floor(rank))
            hi = int(math.ceil(rank))
            frac = rank - lo
            if hi >= n:
                hi = n - 1
            result[p] = sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])

        return result

    def render_cdf(self, latencies: Sequence[float]) -> str:
        """Render an ASCII cumulative-distribution-function plot."""
        if not latencies:
            return "(no data)"

        sorted_vals = sorted(latencies)
        n = len(sorted_vals)
        width = self.config.width
        unit = self.config.unit

        # Sample up to `width` points along the CDF
        n_steps = min(width, n)
        lines: list[str] = []
        lines.append(f"CDF ({unit})")
        lines.append("-" * (width + 20))

        for step in range(n_steps):
            idx = int(step / max(n_steps - 1, 1) * (n - 1))
            val = sorted_vals[idx]
            cum_pct = (idx + 1) / n * 100.0
            bar_len = int(cum_pct / 100.0 * width)
            bar = "\u2588" * bar_len
            lines.append(f"{val:10.2f}{unit} | {bar} {cum_pct:.0f}%")

        return "\n".join(lines)

    def summary(self, latencies: Sequence[float]) -> dict[str, Any]:
        """Return a statistical summary as a plain dict."""
        if not latencies:
            return {
                "count": 0,
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "median": 0.0,
                "percentiles": {},
            }

        vals = list(latencies)
        pcts = self.percentiles(vals)
        return {
            "count": len(vals),
            "min": min(vals),
            "max": max(vals),
            "mean": statistics.mean(vals),
            "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "median": statistics.median(vals),
            "percentiles": pcts,
        }


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def format_histogram_report(
    data: HistogramData,
    config: HistogramConfig | None = None,
) -> str:
    """Return a full histogram report as a string."""
    cfg = config or HistogramConfig()
    hist = LatencyHistogram(cfg)
    lines = [
        "Latency Histogram Report",
        "=" * 40,
        "",
        hist.render_ascii(data),
    ]

    if cfg.show_percentiles and data.total > 0:
        # Reconstruct approximate latencies from bins for percentiles
        lines.append("")
        lines.append("Note: percentiles require raw latency values;")
        lines.append("use LatencyHistogram.percentiles() for exact values.")

    lines.extend([
        "",
        f"Samples : {data.total}",
        f"Min     : {data.min_val:.2f} {cfg.unit}",
        f"Max     : {data.max_val:.2f} {cfg.unit}",
        f"Mean    : {data.mean:.2f} {cfg.unit}",
        f"Std Dev : {data.std:.2f} {cfg.unit}",
    ])

    return "\n".join(lines)
