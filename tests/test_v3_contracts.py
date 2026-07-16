from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch
import torch.nn as nn

from custom_dl_optimizer import (
    BuiltPlan,
    ExecutionTarget,
    FunctionExecutionProvider,
    InferenceOptimizer,
    MeasurementPolicy,
    OptimizationPolicy,
    PlanCache,
    ProviderAvailability,
    ValidationPolicy,
    WorkloadCase,
    WorkloadProfile,
)
from custom_dl_optimizer.config import OptimizationConfig
from custom_dl_optimizer.core.engine import AutoOptimizer, _populate_candidate_metrics
from custom_dl_optimizer.report import CandidateReport


def _policy(**changes):
    policy = OptimizationPolicy(
        measurement=MeasurementPolicy(
            warmup=0,
            iterations=2,
            repeats=1,
            bootstrap_resamples=100,
            random_seed=11,
            measure_peak_memory=False,
        ),
        validation=ValidationPolicy(rtol=1e-5, atol=1e-5),
        enable_profiling=False,
        enable_fx=False,
        enable_triton=False,
        enable_amp=False,
        channels_last=False,
    )
    return replace(policy, **changes)


def _candidate(name: str, samples: list[float], *, setup_s: float = 0.0):
    return CandidateReport(
        name=name,
        latency_ms=sum(samples) / len(samples),
        latency_mean_ms=sum(samples) / len(samples),
        latency_samples_ms=samples,
        setup_time_s=setup_s,
        first_call_time_s=samples[0] / 1000.0,
        parity=True,
    )


def test_confidence_gate_accepts_only_a_clear_challenger():
    reports = [
        _candidate("native", [10.0, 10.1, 9.9, 10.0]),
        _candidate("provider", [5.0, 5.1, 4.9, 5.0]),
    ]
    _populate_candidate_metrics(
        reports,
        expected_calls=None,
        confidence_level=0.95,
        bootstrap_resamples=200,
        random_seed=7,
    )
    engine = AutoOptimizer(
        nn.Identity(),
        config=OptimizationConfig(
            enable_profiling=False,
            min_speedup=1.05,
            bootstrap_resamples=100,
        ),
    )

    selected, reason = engine._select_candidate(reports)

    assert selected == "provider"
    assert reports[1].confidence_gate_passed is True
    assert "confidence bound" in reason


def test_confidence_gate_retains_baseline_for_overlapping_evidence():
    reports = [
        _candidate("native", [9.8, 10.2, 9.9, 10.1]),
        _candidate("provider", [8.0, 11.0, 8.5, 10.5]),
    ]
    _populate_candidate_metrics(
        reports,
        expected_calls=None,
        confidence_level=0.95,
        bootstrap_resamples=500,
        random_seed=7,
    )
    engine = AutoOptimizer(
        nn.Identity(),
        config=OptimizationConfig(
            enable_profiling=False,
            min_speedup=1.02,
            bootstrap_resamples=100,
        ),
    )

    selected, reason = engine._select_candidate(reports)

    assert selected == "native"
    assert reports[1].confidence_gate_passed is False
    assert reports[1].rejection_reason
    assert "overlapped" in reason


def test_selector_chooses_the_lowest_cost_challenger_that_passes_the_gate():
    reports = [
        _candidate("native", [10.0] * 8),
        _candidate("noisy", [1.0, 1.0, 1.0, 17.0]),
        _candidate("stable", [6.0] * 8),
    ]
    _populate_candidate_metrics(
        reports,
        expected_calls=None,
        confidence_level=0.95,
        bootstrap_resamples=500,
        random_seed=9,
    )
    engine = AutoOptimizer(
        nn.Identity(),
        config=OptimizationConfig(
            enable_profiling=False,
            min_speedup=1.02,
            bootstrap_resamples=100,
        ),
    )

    selected, _ = engine._select_candidate(reports)

    assert selected == "stable"
    assert reports[1].confidence_gate_passed is False
    assert reports[2].confidence_gate_passed is True


def test_lifecycle_cost_accounts_for_provider_setup():
    reports = [
        _candidate("native", [10.0] * 4),
        _candidate("provider", [1.0] * 4, setup_s=1.0),
    ]
    _populate_candidate_metrics(
        reports,
        expected_calls=10,
        bootstrap_resamples=100,
    )
    short_engine = AutoOptimizer(
        nn.Identity(),
        config=OptimizationConfig(
            enable_profiling=False,
            expected_calls=10,
            bootstrap_resamples=100,
        ),
    )
    short_selected, _ = short_engine._select_candidate(reports)

    reports = [
        _candidate("native", [10.0] * 4),
        _candidate("provider", [1.0] * 4, setup_s=1.0),
    ]
    _populate_candidate_metrics(
        reports,
        expected_calls=1_000,
        bootstrap_resamples=100,
    )
    long_engine = AutoOptimizer(
        nn.Identity(),
        config=OptimizationConfig(
            enable_profiling=False,
            expected_calls=1_000,
            bootstrap_resamples=100,
        ),
    )
    long_selected, _ = long_engine._select_candidate(reports)

    assert short_selected == "native"
    assert long_selected == "provider"


def test_v3_provider_runner_report_and_bundle_contract(tmp_path):
    class CallableRunner:
        def __init__(self, module):
            self.module = module
            self.closed = False

        def __call__(self, value):
            return self.module(value)

        def close(self):
            self.closed = True

    provider = FunctionExecutionProvider(
        name="callable_backend",
        availability=lambda context: ProviderAvailability.supported(),
        builder=lambda model, context: BuiltPlan(
            runner=CallableRunner(model),
            artifacts=(str(context.artifact_dir or "memory"),),
            metadata={"backend": "callable-test"},
        ),
    )
    optimizer = InferenceOptimizer(
        target=ExecutionTarget("cpu"),
        policy=_policy(),
        providers=(provider,),
    )
    model = nn.Linear(4, 2).eval()
    value = torch.randn(3, 4)

    decision = optimizer.select_signature(model, value)
    bundle = decision.save_bundle(tmp_path / "bundle")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))

    assert torch.allclose(decision(value), model(value))
    assert decision.report.schema_version == 3
    assert decision.report.confidence_level == 0.95
    assert decision.report.candidate_order
    assert manifest["schema_version"] == 3
    provider_report = next(
        candidate
        for candidate in decision.report.candidates
        if candidate.name == "callable_backend"
    )
    assert provider_report.provider_metadata["backend"] == "callable-test"


def test_selected_callable_runner_close_is_propagated():
    class SlowIdentity(nn.Module):
        def forward(self, value):
            import time

            time.sleep(0.004)
            return value

    class FastRunner:
        def __init__(self):
            self.closed = False

        def __call__(self, value):
            return value

        def close(self):
            self.closed = True

    runner = FastRunner()
    provider = FunctionExecutionProvider(
        name="fast_runner",
        builder=lambda model, context: BuiltPlan(runner=runner),
    )
    decision = InferenceOptimizer(
        policy=_policy(),
        providers=(provider,),
    ).select_signature(SlowIdentity(), torch.ones(2))

    assert decision.selected_plan == "fast_runner"
    decision.close()
    assert runner.closed is True


def test_unavailable_provider_is_reported_without_aborting_selection():
    provider = FunctionExecutionProvider(
        name="missing_npu",
        availability=lambda context: ProviderAvailability.unsupported(
            "vendor SDK is not installed"
        ),
        builder=lambda model, context: model,
    )
    decision = InferenceOptimizer(
        policy=_policy(),
        providers=(provider,),
    ).select_signature(nn.Identity(), torch.ones(2))

    failed = next(
        candidate for candidate in decision.report.candidates if candidate.name == "missing_npu"
    )
    assert failed.error == "vendor SDK is not installed"
    assert decision.selected_plan in {"eager_fp32", "native"}


def test_candidate_order_is_deterministic_for_a_seed():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU()).eval()
    value = torch.randn(2, 4)
    first = InferenceOptimizer(policy=_policy()).select_signature(model, value)
    second = InferenceOptimizer(policy=_policy()).select_signature(model, value)

    assert first.report.candidate_order == second.report.candidate_order


def test_v3_cache_rejects_an_old_schema_record(tmp_path):
    cache = PlanCache(tmp_path)
    path = cache.record_path("old")
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "key": "old",
                "selected_plan": "native",
                "latency_ms": 1.0,
                "report": {},
                "created_at": "2026-01-01T00:00:00+00:00",
                "package_version": "2.2.0",
            }
        ),
        encoding="utf-8",
    )

    assert cache.load("old") is None


def test_lifecycle_policy_requires_calls_at_selection_time():
    optimizer = InferenceOptimizer(
        policy=_policy(objective="lifecycle_latency"),
    )

    with pytest.raises(ValueError, match="expected_calls"):
        optimizer.select_signature(nn.Identity(), torch.ones(1))


def test_steady_state_objective_ignores_workload_call_horizon():
    profile = WorkloadProfile(
        name="finite-traffic",
        expected_calls=100,
        cases=(WorkloadCase("default", args=(torch.ones(1),)),),
    )
    decision = InferenceOptimizer(policy=_policy()).select(nn.Identity(), profile)

    assert decision.report.expected_calls is None
    assert decision.report.selection_basis == "steady_state_latency"
