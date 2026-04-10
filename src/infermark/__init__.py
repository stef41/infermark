"""infermark — benchmark any OpenAI-compatible LLM endpoint."""

from infermark._types import (
    BenchmarkConfig,
    BenchmarkMode,
    BenchmarkReport,
    ConcurrencyResult,
    InfermarkError,
    LatencyStats,
    RequestResult,
    compute_percentiles,
    compute_stats,
    percentile,
)
from infermark.compare import compare_reports, format_comparison_text
from infermark.report import (
    format_csv,
    format_markdown,
    format_report_text,
    load_json,
    report_to_dict,
    save_csv,
    save_json,
)
from infermark.runner import run_benchmark, run_benchmark_async
from infermark.backends import (
    Backend,
    OpenAIBackend,
    VLLMBackend,
    TGIBackend,
    detect_backend,
)
from infermark.load import (
    LoadProfile,
    LoadResult,
    LoadTestPlan,
    LoadTestRunner,
    format_load_report,
)
from infermark.histogram import (
    HistogramConfig,
    HistogramData,
    LatencyHistogram,
    compute_bins,
    format_histogram_report,
)

__version__ = "0.2.0"

__all__ = [
    "__version__",
    # Types
    "BenchmarkConfig",
    "BenchmarkMode",
    "BenchmarkReport",
    "ConcurrencyResult",
    "LatencyStats",
    "RequestResult",
    "InfermarkError",
    # Stats
    "compute_percentiles",
    "compute_stats",
    "percentile",
    # Runner
    "run_benchmark",
    "run_benchmark_async",
    # Report
    "format_report_text",
    "format_markdown",
    "format_csv",
    "report_to_dict",
    "save_json",
    "load_json",
    "save_csv",
    # Compare
    "compare_reports",
    "format_comparison_text",
    # Backends
    "Backend",
    "OpenAIBackend",
    "VLLMBackend",
    "TGIBackend",
    "detect_backend",
    # Load testing
    "LoadProfile",
    "LoadResult",
    "LoadTestPlan",
    "LoadTestRunner",
    "format_load_report",
    # Histogram
    "HistogramConfig",
    "HistogramData",
    "LatencyHistogram",
    "compute_bins",
    "format_histogram_report",
]
