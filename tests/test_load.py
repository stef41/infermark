"""Tests for infermark.load module."""

from __future__ import annotations

import time

import pytest

from infermark.load import (
    LoadProfile,
    LoadResult,
    LoadTestPlan,
    LoadTestRunner,
    format_load_report,
)

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture()
def profile() -> LoadProfile:
    return LoadProfile(
        initial_rps=1.0,
        target_rps=10.0,
        ramp_duration_s=1.0,
        hold_duration_s=1.0,
        cooldown_s=1.0,
    )


@pytest.fixture()
def plan(profile: LoadProfile) -> LoadTestPlan:
    return LoadTestPlan(profile)


def _noop() -> None:
    """A fast no-op function for load testing."""
    pass


def _slow() -> None:
    time.sleep(0.001)


def _failing() -> None:
    raise RuntimeError("boom")


# ------------------------------------------------------------------
# LoadProfile validation
# ------------------------------------------------------------------

def test_load_profile_rejects_negative_initial_rps():
    with pytest.raises(ValueError, match="initial_rps"):
        LoadProfile(initial_rps=-1, target_rps=10, ramp_duration_s=1, hold_duration_s=1)


def test_load_profile_rejects_negative_ramp():
    with pytest.raises(ValueError, match="ramp_duration_s"):
        LoadProfile(initial_rps=1, target_rps=10, ramp_duration_s=-1, hold_duration_s=1)


def test_load_profile_rejects_negative_hold():
    with pytest.raises(ValueError, match="hold_duration_s"):
        LoadProfile(initial_rps=1, target_rps=10, ramp_duration_s=1, hold_duration_s=-1)


def test_load_profile_defaults():
    p = LoadProfile(initial_rps=0, target_rps=5, ramp_duration_s=2, hold_duration_s=3)
    assert p.cooldown_s == 0.0


# ------------------------------------------------------------------
# LoadResult
# ------------------------------------------------------------------

def test_load_result_fields():
    r = LoadResult(timestamp=1.0, rps=5.0, latency_ms=12.3, success=True)
    assert r.error is None
    assert r.success is True


def test_load_result_with_error():
    r = LoadResult(timestamp=1.0, rps=5.0, latency_ms=12.3, success=False, error="timeout")
    assert r.error == "timeout"
    assert r.success is False


# ------------------------------------------------------------------
# LoadTestPlan.rps_at
# ------------------------------------------------------------------

def test_rps_at_start(plan: LoadTestPlan):
    assert plan.rps_at(0.0) == pytest.approx(1.0)


def test_rps_at_mid_ramp(plan: LoadTestPlan):
    # Midpoint of ramp: 0.5s into a 1s ramp from 1→10
    assert plan.rps_at(0.5) == pytest.approx(5.5)


def test_rps_at_end_of_ramp(plan: LoadTestPlan):
    assert plan.rps_at(1.0) == pytest.approx(10.0)


def test_rps_at_hold_phase(plan: LoadTestPlan):
    assert plan.rps_at(1.5) == pytest.approx(10.0)


def test_rps_at_cooldown_midpoint(plan: LoadTestPlan):
    # cooldown starts at t=2.0, ends at t=3.0, goes 10→1
    assert plan.rps_at(2.5) == pytest.approx(5.5)


def test_rps_at_end(plan: LoadTestPlan):
    assert plan.rps_at(3.0) == pytest.approx(1.0)


def test_rps_at_negative_time(plan: LoadTestPlan):
    assert plan.rps_at(-1.0) == pytest.approx(1.0)


def test_rps_at_beyond_end(plan: LoadTestPlan):
    assert plan.rps_at(100.0) == pytest.approx(1.0)


# ------------------------------------------------------------------
# LoadTestPlan.schedule
# ------------------------------------------------------------------

def test_schedule_non_empty(plan: LoadTestPlan):
    sched = plan.schedule()
    assert len(sched) > 0
    assert sched[0][0] == 0.0


def test_schedule_ends_near_total_duration(plan: LoadTestPlan):
    sched = plan.schedule()
    assert sched[-1][0] == pytest.approx(plan.total_duration, abs=0.11)


def test_total_duration(plan: LoadTestPlan):
    assert plan.total_duration == pytest.approx(3.0)


# ------------------------------------------------------------------
# LoadTestPlan with zero ramp
# ------------------------------------------------------------------

def test_zero_ramp_jumps_to_target():
    p = LoadProfile(initial_rps=1, target_rps=50, ramp_duration_s=0, hold_duration_s=1, cooldown_s=0)
    plan = LoadTestPlan(p)
    assert plan.rps_at(0.0) == pytest.approx(50.0)


# ------------------------------------------------------------------
# LoadTestRunner — sync execution
# ------------------------------------------------------------------

def test_runner_basic():
    p = LoadProfile(initial_rps=10, target_rps=10, ramp_duration_s=0, hold_duration_s=0.3, cooldown_s=0)
    plan = LoadTestPlan(p)
    runner = LoadTestRunner(plan, _noop)
    results = runner.run_sync()
    assert len(results) > 0
    assert all(r.success for r in results)


def test_runner_captures_failures():
    p = LoadProfile(initial_rps=10, target_rps=10, ramp_duration_s=0, hold_duration_s=0.3, cooldown_s=0)
    plan = LoadTestPlan(p)
    runner = LoadTestRunner(plan, _failing)
    results = runner.run_sync()
    assert len(results) > 0
    assert all(not r.success for r in results)
    assert all(r.error == "boom" for r in results)


def test_runner_empty_plan():
    p = LoadProfile(initial_rps=0, target_rps=0, ramp_duration_s=0, hold_duration_s=0, cooldown_s=0)
    plan = LoadTestPlan(p)
    runner = LoadTestRunner(plan, _noop)
    results = runner.run_sync()
    assert results == []


# ------------------------------------------------------------------
# LoadTestRunner.summary
# ------------------------------------------------------------------

def test_summary_structure():
    p = LoadProfile(initial_rps=20, target_rps=20, ramp_duration_s=0, hold_duration_s=0.3, cooldown_s=0)
    plan = LoadTestPlan(p)
    runner = LoadTestRunner(plan, _noop)
    runner.run_sync()
    s = runner.summary()
    assert s["total_requests"] > 0
    assert 0.0 <= s["success_rate"] <= 1.0
    assert s["p50_latency_ms"] >= 0
    assert s["p95_latency_ms"] >= s["p50_latency_ms"]
    assert s["p99_latency_ms"] >= s["p95_latency_ms"]
    assert s["peak_rps"] > 0


def test_summary_empty():
    p = LoadProfile(initial_rps=0, target_rps=0, ramp_duration_s=0, hold_duration_s=0, cooldown_s=0)
    plan = LoadTestPlan(p)
    runner = LoadTestRunner(plan, _noop)
    s = runner.summary()
    assert s["total_requests"] == 0
    assert s["success_rate"] == 0.0


# ------------------------------------------------------------------
# format_load_report
# ------------------------------------------------------------------

def test_format_load_report_nonempty():
    summary = {
        "total_requests": 100,
        "success_rate": 0.95,
        "peak_rps": 50.0,
        "mean_latency_ms": 10.5,
        "min_latency_ms": 1.0,
        "max_latency_ms": 120.0,
        "p50_latency_ms": 8.0,
        "p95_latency_ms": 50.0,
        "p99_latency_ms": 100.0,
    }
    report = format_load_report(summary)
    assert "Load Test Report" in report
    assert "100" in report
    assert "95.00%" in report


def test_format_load_report_empty():
    assert "No load test" in format_load_report({})
    assert "No load test" in format_load_report({"total_requests": 0})
