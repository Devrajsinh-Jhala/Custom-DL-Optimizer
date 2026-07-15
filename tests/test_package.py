import json
import time
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import custom_dl_optimizer
from custom_dl_optimizer import (
    AutoOptimizer,
    CandidateReport,
    FunctionCandidateProvider,
    ONNXRuntimeProvider,
    OptimizationAgentToolkit,
    OptimizationConfig,
    OptimizationResult,
    Optimizer,
    PlanCache,
    TorchAOQuantizationProvider,
    TorchTensorRTProvider,
    WorkloadCase,
    WorkloadProfile,
    create_plan_cache_key,
    export_paper_artifacts,
    inspect_runtime,
)
from custom_dl_optimizer.cli import main as cli_main
from custom_dl_optimizer.core.engine import _populate_candidate_metrics
from custom_dl_optimizer.core.graph_surgeon import optimize_graph
from custom_dl_optimizer.core.triton_kernels import TritonReLU

FAST_CONFIG = OptimizationConfig(
    enable_profiling=False,
    enable_compile=False,
    selection_warmup=0,
    selection_iterations=1,
    selection_repeats=1,
    min_speedup=1.0,
    copy_model=False,
)


def test_version_exported():
    assert custom_dl_optimizer.__version__ == "2.2.0"


def test_cpu_optimizer_runs_and_exposes_report():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU()).eval()
    inputs = torch.randn(2, 4)
    optimizer = AutoOptimizer(model, device="cpu", config=FAST_CONFIG)

    optimized = optimizer.optimize(inputs)

    with torch.inference_mode():
        baseline = model(inputs)
        actual = optimized(inputs)
    assert torch.allclose(baseline, actual)
    assert optimizer.last_report is not None
    assert optimizer.last_report.selected_plan in {"eager_fp32", "native", "fx"}
    assert optimizer.last_report.selection_basis == "steady_state_latency"
    assert optimizer.last_report.expected_calls is None
    assert any(candidate.selected for candidate in optimizer.last_report.candidates)
    assert optimizer.last_report.runtime is not None
    eager = next(
        candidate
        for candidate in optimizer.last_report.candidates
        if candidate.name == "eager_fp32"
    )
    assert eager.calls_per_second is not None
    assert eager.speedup_vs_eager == 1.0


def test_multi_input_model_is_supported():
    class AddModel(nn.Module):
        def forward(self, left, right, scale=1.0):
            return (left + right) * scale

    model = AddModel().eval()
    left = torch.randn(2, 4)
    right = torch.randn(2, 4)
    optimized = AutoOptimizer(model, device="cpu", config=FAST_CONFIG).optimize(
        left,
        right,
        scale=0.5,
    )
    assert torch.allclose(optimized(left, right, scale=0.5), model(left, right, scale=0.5))


def test_nested_relu_is_replaced_in_fx_graph():
    class Nested(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = nn.Sequential(nn.Linear(4, 4), nn.ReLU())

        def forward(self, x):
            return self.block(x)

    optimized, report = optimize_graph(Nested().eval(), enable_conv_bn_folding=False)
    assert report.module_relu_replacements == 1
    assert any(isinstance(module, TritonReLU) for module in optimized.modules())


def test_functional_relu_is_replaced():
    class FunctionalRelu(nn.Module):
        def forward(self, x):
            return F.relu(x)

    optimized, report = optimize_graph(FunctionalRelu().eval(), enable_conv_bn_folding=False)
    assert report.functional_relu_replacements == 1
    assert any(isinstance(module, TritonReLU) for module in optimized.modules())


def test_inplace_relu_is_not_replaced():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(inplace=True)).eval()
    optimized, report = optimize_graph(model, enable_conv_bn_folding=False)
    assert report.skipped_inplace_relu == 1
    assert not any(isinstance(module, TritonReLU) for module in optimized.modules())


def test_conv_batchnorm_is_folded_with_parity():
    model = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(8),
        nn.ReLU(),
    ).eval()
    inputs = torch.randn(2, 3, 8, 8)
    with torch.inference_mode():
        expected = model(inputs)
    optimized, report = optimize_graph(model, enable_triton=False)
    with torch.inference_mode():
        actual = optimized(inputs)
    assert report.conv_bn_fusions == 1
    assert torch.allclose(expected, actual, rtol=1e-5, atol=1e-5)


def test_fx_trace_failure_falls_back_to_native():
    class DataDependentModel(nn.Module):
        def forward(self, x):
            if x.sum() > 0:
                return x * 2
            return x - 2

    inputs = torch.ones(2, 4)
    optimizer = AutoOptimizer(
        DataDependentModel().eval(),
        device="cpu",
        config=FAST_CONFIG,
    )
    optimized = optimizer.optimize(inputs)
    assert optimizer.last_report is not None
    assert not optimizer.last_report.graph.traced
    assert optimizer.last_report.selected_plan in {"eager_fp32", "native"}
    assert torch.equal(optimized(inputs), inputs * 2)


def test_v2_optimizer_returns_callable_result_and_saves_report(tmp_path):
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU()).eval()
    inputs = torch.randn(2, 4)
    result = Optimizer(device="cpu", config=FAST_CONFIG).optimize(model, inputs)

    assert isinstance(result, OptimizationResult)
    assert result(inputs).shape == (2, 4)
    destination = result.save_report(tmp_path / "report.json")
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["selected_plan"] == result.selected_plan
    assert payload["runtime"]["device_type"] == "cpu"


def test_custom_candidate_provider_is_measured():
    provider = FunctionCandidateProvider(
        name="identity_backend",
        builder=lambda model, context: model,
    )
    optimizer = Optimizer(
        device="cpu",
        config=FAST_CONFIG,
        providers=(provider,),
    )
    result = optimizer.optimize(nn.Linear(4, 4).eval(), torch.randn(2, 4))

    candidate = next(
        candidate
        for candidate in result.report.candidates
        if candidate.name == "identity_backend"
    )
    assert candidate.parity
    assert candidate.latency_ms is not None


def test_agent_toolkit_only_operates_on_registered_workloads():
    toolkit = OptimizationAgentToolkit(Optimizer(device="cpu", config=FAST_CONFIG))
    toolkit.register_workload(
        "linear",
        nn.Linear(4, 4).eval(),
        torch.randn(2, 4),
        description="Small test workload",
    )

    names = {schema["name"] for schema in toolkit.tool_schemas()}
    assert "custom_dl_optimize" in names
    listed = toolkit.invoke("custom_dl_list_workloads")
    assert listed == {
        "workloads": [{"name": "linear", "description": "Small test workload"}]
    }
    optimized = toolkit.invoke("custom_dl_optimize", {"workload": "linear"})
    report = toolkit.invoke("custom_dl_get_report", {"workload": "linear"})
    assert report == optimized


def test_runtime_inspection_is_serializable():
    capabilities = inspect_runtime("cpu")
    assert capabilities.device_type == "cpu"
    json.dumps(capabilities.as_dict())


def test_selector_never_prefers_a_slower_builtin_plan():
    optimizer = AutoOptimizer(nn.Identity(), device="cpu", config=FAST_CONFIG)
    selected, _ = optimizer._select_candidate(
        [
            CandidateReport(name="eager_fp32", latency_ms=1.0, parity=True),
            CandidateReport(name="native", latency_ms=1.5, parity=True),
            CandidateReport(name="provider", latency_ms=1.1, parity=True),
        ]
    )
    assert selected == "eager_fp32"


def test_output_parity_rejects_different_mapping_keys():
    class DictionaryModel(nn.Module):
        def forward(self, value):
            return {"expected": value}

    class WrongDictionaryModel(nn.Module):
        def forward(self, value):
            return {"unexpected": value}

    provider = FunctionCandidateProvider(
        name="wrong_structure",
        builder=lambda model, context: WrongDictionaryModel(),
    )
    result = Optimizer(
        device="cpu",
        config=FAST_CONFIG,
        providers=(provider,),
    ).optimize(DictionaryModel(), torch.randn(2, 4))
    candidate = next(
        candidate
        for candidate in result.report.candidates
        if candidate.name == "wrong_structure"
    )
    assert not candidate.parity
    assert candidate.error == "Output parity check failed"


def test_expected_calls_must_be_positive():
    with pytest.raises(ValueError, match="expected_calls"):
        OptimizationConfig(expected_calls=0)


def test_report_includes_latency_distribution_and_total_optimization_time():
    config = replace(FAST_CONFIG, selection_repeats=3, expected_calls=100)
    result = Optimizer(device="cpu", config=config).optimize(
        nn.Sequential(nn.Linear(4, 4), nn.ReLU()).eval(),
        torch.randn(2, 4),
    )
    valid = [candidate for candidate in result.report.candidates if candidate.latency_ms]

    assert result.report.selection_basis == "projected_total_time"
    assert result.report.expected_calls == 100
    assert result.report.optimization_time_s > 0
    assert valid
    for candidate in valid:
        assert len(candidate.latency_samples_ms) == 3
        assert candidate.latency_min_ms == min(candidate.latency_samples_ms)
        assert candidate.latency_p90_ms is not None
        assert candidate.latency_stdev_ms is not None
        assert candidate.first_call_time_s is not None
        assert candidate.projected_total_ms is not None


def test_provider_lazy_first_call_is_measured_separately():
    class LazyFirstCall(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.cold = True

        def forward(self, value):
            if self.cold:
                time.sleep(0.02)
                self.cold = False
            return self.model(value)

    provider = FunctionCandidateProvider(
        name="lazy_provider",
        builder=lambda model, context: LazyFirstCall(model),
    )
    result = Optimizer(
        device="cpu",
        config=FAST_CONFIG,
        providers=(provider,),
    ).optimize(nn.Linear(4, 4).eval(), torch.randn(2, 4))
    candidate = next(
        candidate
        for candidate in result.report.candidates
        if candidate.name == "lazy_provider"
    )

    assert candidate.first_call_time_s is not None
    assert candidate.first_call_time_s >= 0.015
    assert candidate.setup_time_s >= 0


def test_amortized_selection_changes_after_break_even():
    baseline = CandidateReport(
        name="eager_fp32",
        latency_ms=2.0,
        first_call_time_s=0.002,
        parity=True,
    )
    provider = CandidateReport(
        name="compiled_provider",
        latency_ms=1.0,
        first_call_time_s=0.2,
        parity=True,
    )
    reports = [baseline, provider]

    _populate_candidate_metrics(reports, expected_calls=50)
    short_lived = AutoOptimizer(
        nn.Identity(),
        device="cpu",
        config=replace(FAST_CONFIG, expected_calls=50),
    )
    short_selection, _ = short_lived._select_candidate(reports)

    _populate_candidate_metrics(reports, expected_calls=500)
    long_lived = AutoOptimizer(
        nn.Identity(),
        device="cpu",
        config=replace(FAST_CONFIG, expected_calls=500),
    )
    long_selection, reason = long_lived._select_candidate(reports)

    assert provider.break_even_calls_vs_baseline == 199
    assert short_selection == "eager_fp32"
    assert long_selection == "compiled_provider"
    assert "500 calls" in reason


def test_weighted_workload_profile_measures_every_case():
    profile = WorkloadProfile(
        name="mixed-batches",
        expected_calls=100,
        cases=(
            WorkloadCase(name="batch-1", args=(torch.randn(1, 4),), weight=1),
            WorkloadCase(name="batch-8", args=(torch.randn(8, 4),), weight=3),
        ),
    )
    config = replace(
        FAST_CONFIG,
        selection_iterations=2,
        selection_repeats=2,
    )
    result = Optimizer(device="cpu", config=config).optimize_workload(
        nn.Linear(4, 4).eval(),
        profile,
    )

    assert result.report.workload_name == "mixed-batches"
    assert result.report.expected_calls == 100
    valid = [candidate for candidate in result.report.candidates if candidate.latency_ms]
    assert valid
    for candidate in valid:
        assert len(candidate.workload_cases) == 2
        assert sum(case.weight for case in candidate.workload_cases) == pytest.approx(1.0)
        assert len(candidate.latency_samples_ms) == 4
        assert candidate.latency_p95_ms is not None
        assert candidate.latency_p99_ms is not None
        assert candidate.latency_ci95_low_ms is not None
        assert candidate.latency_ci95_high_ms is not None
    assert result(torch.randn(3, 4)).shape == (3, 4)


def test_plan_cache_reuses_a_validated_winner(tmp_path):
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU()).eval()
    inputs = torch.randn(2, 4)
    config = replace(
        FAST_CONFIG,
        plan_cache_dir=str(tmp_path / "cache"),
        cache_validation_iterations=1,
        cache_max_latency_regression=1000.0,
    )

    first = Optimizer(device="cpu", config=config).optimize(model, inputs)
    second = Optimizer(device="cpu", config=config).optimize(model, inputs)

    assert not first.report.cache_hit
    assert second.report.cache_hit
    assert second.selected_plan == first.selected_plan
    assert Path(second.report.cache_record_path).is_file()
    assert "Reused cached plan" in second.report.selection_reason


def test_plan_cache_key_changes_with_weights(tmp_path):
    model = nn.Linear(4, 4).eval()
    profile = WorkloadProfile.single(torch.randn(2, 4))
    runtime = inspect_runtime("cpu")
    key_before = create_plan_cache_key(model, profile, FAST_CONFIG, runtime)
    with torch.no_grad():
        model.weight.add_(1)
    key_after = create_plan_cache_key(model, profile, FAST_CONFIG, runtime)

    assert key_before != key_after
    cache = PlanCache(tmp_path / "cache")
    assert cache.records() == []


def test_resource_constraint_rejects_slow_setup_provider():
    def build_slow(model, context):
        time.sleep(0.02)
        return model

    provider = FunctionCandidateProvider(name="slow_setup", builder=build_slow)
    config = replace(FAST_CONFIG, max_setup_time_s=0.005)
    result = Optimizer(
        device="cpu",
        config=config,
        providers=(provider,),
    ).optimize(nn.Linear(4, 4).eval(), torch.randn(2, 4))
    candidate = next(
        candidate for candidate in result.report.candidates if candidate.name == "slow_setup"
    )

    assert candidate.constraint_violations == ["setup_time_s>0.005"]
    assert not candidate.selected


def test_result_bundle_and_cli_report(tmp_path, capsys):
    result = Optimizer(device="cpu", config=FAST_CONFIG).optimize(
        nn.Linear(4, 4).eval(),
        torch.randn(2, 4),
    )
    bundle = result.save_bundle(tmp_path / "bundle")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["selected_plan"] == result.selected_plan
    assert not manifest["executable_serialized"]
    assert cli_main(["report", str(bundle / "report.json")]) == 0
    output = capsys.readouterr().out
    assert f"selected: {result.selected_plan}" in output
    assert "p99 ms" in output


def test_paper_export_writes_auditable_tables(tmp_path, capsys):
    result = Optimizer(device="cpu", config=FAST_CONFIG).optimize(
        nn.Linear(4, 4).eval(),
        torch.randn(2, 4),
    )
    report_path = result.save_report(tmp_path / "report.json")
    output_dir = tmp_path / "paper"

    outputs = export_paper_artifacts([report_path], output_dir, plots=False)
    assert {path.name for path in outputs} == {
        "paper_candidates.csv",
        "paper_workload_cases.csv",
        "paper_results.tex",
        "paper_artifacts.json",
    }
    manifest = json.loads((output_dir / "paper_artifacts.json").read_text(encoding="utf-8"))
    assert manifest["candidate_rows"] == len(result.report.candidates)
    assert manifest["latency_semantics"] == "serial invocation measurements"
    assert cli_main(
        [
            "paper-export",
            str(report_path),
            "--output-dir",
            str(tmp_path / "cli-paper"),
            "--no-plots",
        ]
    ) == 0
    assert "paper_candidates.csv" in capsys.readouterr().out


def test_first_party_provider_identities_are_serializable():
    providers = (
        TorchTensorRTProvider(),
        ONNXRuntimeProvider(),
        TorchAOQuantizationProvider(),
    )
    for provider in providers:
        json.dumps(provider.cache_identity())
    assert providers[2].name == "torchao_int8_weight_only"


def test_plan_cache_clear_preserves_unrelated_files(tmp_path):
    root = tmp_path / "shared-directory"
    root.mkdir()
    unrelated = root / "keep.txt"
    unrelated.write_text("keep", encoding="utf-8")
    cache = PlanCache(root)
    cache.save(
        key="a" * 64,
        selected_plan="native",
        latency_ms=1.0,
        report={},
    )

    assert cache.clear() == 1
    assert unrelated.read_text(encoding="utf-8") == "keep"


def test_agent_toolkit_accepts_weighted_profile():
    toolkit = OptimizationAgentToolkit(Optimizer(device="cpu", config=FAST_CONFIG))
    profile = WorkloadProfile(
        name="linear-traffic",
        cases=(
            WorkloadCase("small", args=(torch.randn(1, 4),), weight=3),
            WorkloadCase("large", args=(torch.randn(4, 4),), weight=1),
        ),
    )
    toolkit.register_workload_profile("linear-profile", nn.Linear(4, 4), profile)

    report = toolkit.invoke("custom_dl_optimize", {"workload": "linear-profile"})
    assert report["workload_name"] == "linear-traffic"


def test_resource_and_cache_config_validation():
    with pytest.raises(ValueError, match="max_setup_time_s"):
        OptimizationConfig(max_setup_time_s=-1)
    with pytest.raises(ValueError, match="cache_validation_iterations"):
        OptimizationConfig(cache_validation_iterations=0)
    with pytest.raises(ValueError, match="cache_max_latency_regression"):
        OptimizationConfig(cache_max_latency_regression=0.9)
    with pytest.raises(ValueError, match="measure_peak_memory"):
        OptimizationConfig(max_peak_memory_mb=10, measure_peak_memory=False)


@pytest.mark.parametrize("weight", [float("nan"), float("inf"), 0.0, -1.0])
def test_workload_weight_must_be_finite_and_positive(weight):
    with pytest.raises(ValueError, match="finite and positive"):
        WorkloadCase("invalid", args=(torch.ones(1),), weight=weight)
