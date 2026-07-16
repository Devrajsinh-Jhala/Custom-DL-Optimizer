import copy
import hashlib
import logging
import math
import random
import statistics
import time
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from custom_dl_optimizer.config import OptimizationConfig
from custom_dl_optimizer.policy import ExecutionTarget, OptimizationPolicy
from custom_dl_optimizer.providers import (
    BuiltPlan,
    ExecutionProvider,
    ProviderContext,
)
from custom_dl_optimizer.report import (
    CandidateReport,
    OptimizationReport,
    WorkloadCaseReport,
)
from custom_dl_optimizer.runtime import inspect_runtime
from custom_dl_optimizer.workload import WorkloadCase, WorkloadProfile

from .graph_surgeon import optimize_graph
from .profiler import analyze_bottlenecks

LOGGER = logging.getLogger("custom_dl_optimizer")


def _walk_values(value: Any) -> Iterable[Any]:
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk_values(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _walk_values(item)
    else:
        yield value


def _map_values(value: Any, transform) -> Any:
    if isinstance(value, dict):
        return {key: _map_values(item, transform) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_map_values(item, transform) for item in value)
    if isinstance(value, list):
        return [_map_values(item, transform) for item in value]
    return transform(value)


def _prepare_tensor(
    value: Any,
    *,
    device: torch.device,
    channels_last: bool,
) -> Any:
    if not isinstance(value, torch.Tensor):
        return value
    value = value.to(device)
    if channels_last and value.dim() == 4:
        return value.contiguous(memory_format=torch.channels_last)
    return value


def _input_signature(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    parts: list[str] = []
    for value in _walk_values((args, kwargs)):
        if isinstance(value, torch.Tensor):
            parts.append(f"{tuple(value.shape)}:{value.dtype}:{value.device.type}")
        else:
            parts.append(type(value).__name__)
    return "|".join(parts)


class _InferenceWrapper(nn.Module):
    def __init__(
        self,
        core_model: nn.Module,
        *,
        device: torch.device,
        enable_amp: bool,
        channels_last: bool,
        amp_dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        self.core_model = core_model
        self.device = device
        self.enable_amp = enable_amp and device.type == "cuda"
        self.channels_last = channels_last and device.type == "cuda"
        self.amp_dtype = amp_dtype

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        def transform(value: Any) -> Any:
            return _prepare_tensor(
                value,
                device=self.device,
                channels_last=self.channels_last,
            )

        prepared_args = _map_values(args, transform)
        prepared_kwargs = _map_values(kwargs, transform)
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.enable_amp,
        ):
            return self.core_model(*prepared_args, **prepared_kwargs)

    def close(self) -> None:
        close = getattr(self.core_model, "close", None)
        if callable(close):
            close()


def _compare_outputs(
    reference: Any,
    candidate: Any,
    *,
    rtol: float,
    atol: float,
) -> tuple[bool, float, float]:
    max_errors: list[float] = []
    mean_errors: list[float] = []

    def compare(ref: Any, cand: Any) -> bool:
        if isinstance(ref, torch.Tensor) or isinstance(cand, torch.Tensor):
            if not isinstance(ref, torch.Tensor) or not isinstance(cand, torch.Tensor):
                return False
            if ref.shape != cand.shape:
                return False
            ref_float = ref.detach().float()
            cand_float = cand.detach().float()
            diff = (ref_float - cand_float).abs()
            max_errors.append(float(diff.max().item()) if diff.numel() else 0.0)
            mean_errors.append(float(diff.mean().item()) if diff.numel() else 0.0)
            return bool(torch.allclose(ref_float, cand_float, rtol=rtol, atol=atol))
        if type(ref) is not type(cand):
            return False
        if isinstance(ref, dict):
            if ref.keys() != cand.keys():
                return False
            return all(compare(ref[key], cand[key]) for key in ref)
        if isinstance(ref, (tuple, list)):
            return len(ref) == len(cand) and all(
                compare(ref_item, cand_item)
                for ref_item, cand_item in zip(ref, cand, strict=True)
            )
        try:
            return bool(ref == cand)
        except (TypeError, ValueError):
            return False

    allclose = compare(reference, candidate)
    return (
        bool(allclose),
        max(max_errors, default=0.0) if allclose or max_errors else float("inf"),
        statistics.mean(mean_errors) if mean_errors else (0.0 if allclose else float("inf")),
    )


@dataclass(frozen=True)
class _BenchmarkStats:
    median_ms: float
    mean_ms: float
    minimum_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    stdev_ms: float
    ci95_low_ms: float
    ci95_high_ms: float
    samples_ms: tuple[float, ...]


@dataclass(frozen=True)
class _PreparedWorkloadCase:
    name: str
    weight: float
    signature: str
    base_args: tuple[Any, ...]
    base_kwargs: dict[str, Any]
    selection_args: tuple[Any, ...]
    selection_kwargs: dict[str, Any]
    reference: Any


def _percentile(samples: list[float], percentile: int) -> float:
    if len(samples) == 1:
        return float(samples[0])
    return float(
        statistics.quantiles(samples, n=100, method="inclusive")[percentile - 1]
    )


def _quantile(samples: list[float], probability: float) -> float:
    ordered = sorted(samples)
    if len(ordered) == 1:
        return float(ordered[0])
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def _bootstrap_mean_interval(
    samples: list[float],
    *,
    confidence_level: float,
    resamples: int,
    seed: int,
) -> tuple[float, float, float]:
    if not samples:
        raise ValueError("at least one selection-cost sample is required")
    point = float(statistics.mean(samples))
    if len(samples) == 1:
        return point, point, point
    rng = random.Random(seed)
    sample_count = len(samples)
    means = [
        statistics.fmean(samples[rng.randrange(sample_count)] for _ in range(sample_count))
        for _ in range(resamples)
    ]
    alpha = (1.0 - confidence_level) / 2.0
    return point, _quantile(means, alpha), _quantile(means, 1.0 - alpha)


def _stats_from_samples(samples: list[float]) -> _BenchmarkStats:
    if not samples:
        raise ValueError("at least one latency sample is required")
    mean = float(statistics.mean(samples))
    stdev = float(statistics.stdev(samples)) if len(samples) > 1 else 0.0
    margin = 1.96 * stdev / math.sqrt(len(samples)) if len(samples) > 1 else 0.0
    return _BenchmarkStats(
        median_ms=float(statistics.median(samples)),
        mean_ms=mean,
        minimum_ms=float(min(samples)),
        p90_ms=_percentile(samples, 90),
        p95_ms=_percentile(samples, 95),
        p99_ms=_percentile(samples, 99),
        stdev_ms=stdev,
        ci95_low_ms=max(0.0, mean - margin),
        ci95_high_ms=mean + margin,
        samples_ms=tuple(float(sample) for sample in samples),
    )


def _benchmark_latency(
    model: nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
    repeats: int,
) -> _BenchmarkStats:
    samples: list[float] = []
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(*args, **kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        for _ in range(repeats):
            if device.type == "cuda":
                events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
                for _ in range(iterations):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    model(*args, **kwargs)
                    end.record()
                    events.append((start, end))
                events[-1][1].synchronize()
                samples.extend(float(start.elapsed_time(end)) for start, end in events)
            else:
                for _ in range(iterations):
                    started = time.perf_counter()
                    model(*args, **kwargs)
                    samples.append((time.perf_counter() - started) * 1000.0)
    return _stats_from_samples(samples)


def _measure_peak_memory_mb(
    model: nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    device: torch.device,
) -> float | None:
    if device.type != "cuda":
        return None
    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        model(*args, **kwargs)
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    return max(0.0, float(peak - baseline) / (1024.0 * 1024.0))


def _elapsed_s(started: float, device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter() - started


def _copy_candidate_model(
    model: nn.Module,
    *,
    warnings: list[str],
    candidate_name: str,
) -> nn.Module:
    try:
        return copy.deepcopy(model)
    except Exception as exc:
        warnings.append(
            f"Could not isolate candidate {candidate_name!r}; reused the working model: {exc!r}"
        )
        return model


def _first_call_ms(candidate: CandidateReport) -> float:
    if candidate.first_call_time_s is not None:
        return candidate.first_call_time_s * 1000.0
    return candidate.latency_ms or 0.0


def _apply_latency_stats(report: CandidateReport | WorkloadCaseReport, stats: _BenchmarkStats) -> None:
    report.latency_ms = stats.median_ms
    report.latency_mean_ms = stats.mean_ms
    report.latency_min_ms = stats.minimum_ms
    report.latency_p90_ms = stats.p90_ms
    report.latency_p95_ms = stats.p95_ms
    report.latency_p99_ms = stats.p99_ms
    report.latency_stdev_ms = stats.stdev_ms
    report.latency_ci95_low_ms = stats.ci95_low_ms
    report.latency_ci95_high_ms = stats.ci95_high_ms
    report.latency_samples_ms = list(stats.samples_ms)


def _aggregate_workload_cases(candidate: CandidateReport) -> None:
    cases = candidate.workload_cases
    if not cases:
        return
    candidate.parity = all(case.parity for case in cases)
    candidate.first_call_time_s = sum(case.first_call_time_s or 0.0 for case in cases)
    candidate.max_abs_error = max(
        (case.max_abs_error or 0.0 for case in cases),
        default=0.0,
    )
    candidate.mean_abs_error = sum(
        case.weight * (case.mean_abs_error or 0.0) for case in cases
    )
    measured_memory = [
        case.peak_memory_mb for case in cases if case.peak_memory_mb is not None
    ]
    candidate.peak_memory_mb = max(measured_memory) if measured_memory else None
    if not candidate.parity:
        candidate.error = next(
            (case.error for case in cases if case.error),
            "Output parity check failed",
        )
        return
    sample_count = min(len(case.latency_samples_ms) for case in cases)
    weighted_samples = [
        sum(case.weight * case.latency_samples_ms[index] for case in cases)
        for index in range(sample_count)
    ]
    _apply_latency_stats(candidate, _stats_from_samples(weighted_samples))


def _apply_candidate_constraints(
    candidate: CandidateReport,
    config: OptimizationConfig,
) -> None:
    if (
        config.max_setup_time_s is not None
        and candidate.setup_time_s > config.max_setup_time_s
    ):
        candidate.constraint_violations.append(
            f"setup_time_s>{config.max_setup_time_s:g}"
        )
    if (
        config.max_first_call_time_s is not None
        and candidate.first_call_time_s is not None
        and candidate.first_call_time_s > config.max_first_call_time_s
    ):
        candidate.constraint_violations.append(
            f"first_call_time_s>{config.max_first_call_time_s:g}"
        )
    if (
        config.max_peak_memory_mb is not None
        and candidate.peak_memory_mb is not None
        and candidate.peak_memory_mb > config.max_peak_memory_mb
    ):
        candidate.constraint_violations.append(
            f"peak_memory_mb>{config.max_peak_memory_mb:g}"
        )


def _first_case_steady_ms(candidate: CandidateReport) -> float:
    if candidate.workload_cases:
        return sum(case.latency_ms or 0.0 for case in candidate.workload_cases)
    return candidate.latency_ms or 0.0


def _populate_candidate_metrics(
    candidate_reports: list[CandidateReport],
    *,
    expected_calls: int | None,
    confidence_level: float = 0.95,
    bootstrap_resamples: int = 1_000,
    random_seed: int = 17,
) -> None:
    eager = next(
        (
            candidate
            for candidate in candidate_reports
            if candidate.name == "eager_fp32" and candidate.latency_ms
        ),
        None,
    )
    native = next(
        (
            candidate
            for candidate in candidate_reports
            if candidate.name == "native" and candidate.latency_ms
        ),
        None,
    )
    built_in_baselines = [
        candidate
        for candidate in candidate_reports
        if candidate.name in {"eager_fp32", "native"}
        and candidate.parity
        and candidate.latency_ms is not None
        and not candidate.constraint_violations
    ]
    steady_state_baseline = (
        min(
            built_in_baselines,
            key=lambda candidate: candidate.latency_ms or float("inf"),
        )
        if built_in_baselines
        else None
    )

    for candidate in candidate_reports:
        if not candidate.latency_ms:
            continue
        candidate.calls_per_second = 1000.0 / candidate.latency_ms
        if eager is not None and eager.latency_ms:
            candidate.speedup_vs_eager = eager.latency_ms / candidate.latency_ms
        if native is not None and native.latency_ms:
            candidate.speedup_vs_native = native.latency_ms / candidate.latency_ms
        if expected_calls is not None:
            first_call_count = max(1, len(candidate.workload_cases))
            candidate.projected_total_ms = (
                candidate.setup_time_s * 1000.0
                + _first_call_ms(candidate)
                + (candidate.latency_mean_ms or candidate.latency_ms)
                * max(expected_calls - first_call_count, 0)
            )

        latency_samples = candidate.latency_samples_ms or [
            candidate.latency_mean_ms or candidate.latency_ms
        ]
        if expected_calls is None:
            selection_samples = list(latency_samples)
        else:
            first_call_count = max(1, len(candidate.workload_cases))
            fixed_cost_ms = candidate.setup_time_s * 1000.0 + _first_call_ms(candidate)
            remaining_calls = max(expected_calls - first_call_count, 0)
            selection_samples = [
                fixed_cost_ms + sample * remaining_calls for sample in latency_samples
            ]
        digest = hashlib.sha256(
            f"{random_seed}:{candidate.name}".encode()
        ).digest()
        seed = int.from_bytes(digest[:8], "big")
        point, lower, upper = _bootstrap_mean_interval(
            selection_samples,
            confidence_level=confidence_level,
            resamples=bootstrap_resamples,
            seed=seed,
        )
        candidate.selection_cost_ms = point
        candidate.selection_cost_ci_low_ms = lower
        candidate.selection_cost_ci_high_ms = upper

        if (
            steady_state_baseline is None
            or candidate is steady_state_baseline
            or steady_state_baseline.latency_ms is None
            or candidate.latency_ms >= steady_state_baseline.latency_ms
        ):
            continue
        baseline_cold_overhead = (
            steady_state_baseline.setup_time_s * 1000.0
            + _first_call_ms(steady_state_baseline)
            - _first_case_steady_ms(steady_state_baseline)
        )
        candidate_cold_overhead = (
            candidate.setup_time_s * 1000.0
            + _first_call_ms(candidate)
            - _first_case_steady_ms(candidate)
        )
        latency_saved = steady_state_baseline.latency_ms - candidate.latency_ms
        candidate.break_even_calls_vs_baseline = max(
            1,
            math.ceil(
                max(0.0, candidate_cold_overhead - baseline_cold_overhead)
                / latency_saved
            ),
        )


class AutoOptimizer:
    """Profile candidate PyTorch inference plans and return the fastest safe one."""

    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device | None = None,
        enable_amp: bool | None = None,
        channels_last: bool | None = None,
        *,
        config: OptimizationConfig | None = None,
        providers: tuple[ExecutionProvider, ...] = (),
        policy: OptimizationPolicy | None = None,
        target: ExecutionTarget | None = None,
        cache_key: str = "",
        artifact_root: str | Path | None = None,
    ) -> None:
        initialization_started = time.perf_counter()
        resolved_device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._warnings: list[str] = []
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            self._warnings.append("CUDA was requested but is unavailable; using CPU.")
            resolved_device = torch.device("cpu")

        resolved_config = config or OptimizationConfig()
        if enable_amp is not None:
            resolved_config = replace(resolved_config, enable_amp=enable_amp)
        if channels_last is not None:
            resolved_config = replace(resolved_config, channels_last=channels_last)

        if resolved_config.copy_model:
            try:
                model = copy.deepcopy(model)
            except Exception as exc:
                self._warnings.append(
                    f"Model copy failed; optimizing the supplied instance in place: {exc!r}"
                )

        self.model = model.to(resolved_device).eval()
        self.device = resolved_device
        self.config = resolved_config
        self.policy = policy or OptimizationPolicy.from_engine_config(resolved_config)
        self.target = target or ExecutionTarget(device=resolved_device)
        self.providers = providers
        self.cache_key = cache_key
        self.artifact_root = (
            Path(artifact_root).expanduser().resolve()
            if artifact_root is not None
            else None
        )
        self.last_report: OptimizationReport | None = None
        self._initialization_time_s = _elapsed_s(
            initialization_started,
            self.device,
        )

    def prepare_inputs(
        self,
        *args: Any,
        channels_last: bool | None = None,
        **kwargs: Any,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Move example inputs to the target device and apply the selected layout."""

        use_channels_last = (
            self.config.channels_last if channels_last is None else channels_last
        ) and self.device.type == "cuda"

        def transform(value: Any) -> Any:
            return _prepare_tensor(
                value,
                device=self.device,
                channels_last=use_channels_last,
            )

        return _map_values(args, transform), _map_values(kwargs, transform)

    def optimize(self, *example_args: Any, **example_kwargs: Any) -> nn.Module:
        """Return the fastest numerically valid plan for the example input signature."""

        optimized, report = self.optimize_with_report(*example_args, **example_kwargs)
        self.last_report = report
        return optimized

    def optimize_workload(self, profile: WorkloadProfile) -> nn.Module:
        """Return the fastest valid plan across a weighted workload profile."""

        optimized, report = self.optimize_workload_with_report(profile)
        self.last_report = report
        return optimized

    def optimize_with_report(
        self,
        *example_args: Any,
        **example_kwargs: Any,
    ) -> tuple[nn.Module, OptimizationReport]:
        profile = WorkloadProfile(
            name="single-signature",
            cases=(
                WorkloadCase(
                    name="default",
                    args=example_args,
                    kwargs=example_kwargs,
                ),
            ),
            expected_calls=self.config.expected_calls,
        )
        return self.optimize_workload_with_report(profile)

    def optimize_workload_with_report(
        self,
        profile: WorkloadProfile,
        *,
        preferred_plan: str | None = None,
        cached_latency_ms: float | None = None,
        _cache_retry_reason: str = "",
    ) -> tuple[nn.Module, OptimizationReport]:
        """Select one plan across a weighted distribution of serving inputs."""

        optimization_started = time.perf_counter()
        use_channels_last = self.config.channels_last and self.device.type == "cuda"
        expected_calls = (
            profile.expected_calls
            if profile.expected_calls is not None
            else self.config.expected_calls
        )
        if expected_calls is not None and expected_calls < len(profile.cases):
            raise ValueError("expected_calls must cover every workload case at least once")
        weights = profile.normalized_weights

        prepared_cases: list[_PreparedWorkloadCase] = []
        with torch.inference_mode():
            for case in profile.cases:
                base_args, base_kwargs = self.prepare_inputs(
                    *case.args,
                    channels_last=False,
                    **case.kwargs,
                )
                selection_args, selection_kwargs = self.prepare_inputs(
                    *case.args,
                    channels_last=use_channels_last,
                    **case.kwargs,
                )
                prepared_cases.append(
                    _PreparedWorkloadCase(
                        name=case.name,
                        weight=weights[case.name],
                        signature=_input_signature(base_args, base_kwargs),
                        base_args=base_args,
                        base_kwargs=base_kwargs,
                        selection_args=selection_args,
                        selection_kwargs=selection_kwargs,
                        reference=self.model(*base_args, **base_kwargs),
                    )
                )

        report = OptimizationReport(
            device=str(self.device),
            workload_name=profile.name,
            input_signature=";".join(
                f"{case.name}={case.signature}" for case in prepared_cases
            ),
            channels_last=use_channels_last,
            amp=self.config.enable_amp and self.device.type == "cuda",
            expected_calls=expected_calls,
            selection_basis=(
                "projected_total_time"
                if expected_calls is not None
                else "steady_state_latency"
            ),
            confidence_level=self.config.confidence_level,
            bootstrap_resamples=self.config.bootstrap_resamples,
            random_seed=self.config.random_seed,
            cache_key=self.cache_key,
            runtime=inspect_runtime(self.device),
            warnings=list(self._warnings),
        )
        if _cache_retry_reason:
            report.warnings.append(_cache_retry_reason)
        if self.config.max_peak_memory_mb is not None and self.device.type != "cuda":
            report.warnings.append(
                "max_peak_memory_mb applies to incremental CUDA allocation and is "
                "not enforced on this device."
            )

        if self.config.enable_profiling:
            try:
                representative = prepared_cases[0]
                report.operator_profile = analyze_bottlenecks(
                    self.model,
                    representative.base_args,
                    representative.base_kwargs,
                )
            except Exception as exc:
                report.warnings.append(f"Operator profiling failed: {exc!r}")

        candidates: dict[str, nn.Module] = {}
        candidate_uses_selection_inputs: dict[str, bool] = {}
        candidate_setup_times: dict[str, float] = {}
        candidate_plan_details: dict[str, BuiltPlan] = {}
        failed_candidates: list[CandidateReport] = []

        def requested(name: str) -> bool:
            return preferred_plan is None or preferred_plan == name

        if self.config.benchmark_eager and requested("eager_fp32"):
            started = time.perf_counter()
            eager_core = _copy_candidate_model(
                self.model,
                warnings=report.warnings,
                candidate_name="eager_fp32",
            )
            candidates["eager_fp32"] = _InferenceWrapper(
                eager_core,
                device=self.device,
                enable_amp=False,
                channels_last=False,
            ).eval()
            candidate_uses_selection_inputs["eager_fp32"] = False
            candidate_setup_times["eager_fp32"] = _elapsed_s(
                started,
                self.device,
            )

        if requested("native"):
            started = time.perf_counter()
            native_core = _copy_candidate_model(
                self.model,
                warnings=report.warnings,
                candidate_name="native",
            )
            if use_channels_last:
                native_core = native_core.to(memory_format=torch.channels_last)
            candidates["native"] = _InferenceWrapper(
                native_core,
                device=self.device,
                enable_amp=self.config.enable_amp,
                channels_last=use_channels_last,
            ).eval()
            candidate_uses_selection_inputs["native"] = True
            candidate_setup_times["native"] = _elapsed_s(started, self.device)

        if self.config.enable_fx and requested("fx"):
            started = time.perf_counter()
            graph_source = _copy_candidate_model(
                self.model,
                warnings=report.warnings,
                candidate_name="fx",
            )
            graph_model, graph_report = optimize_graph(
                graph_source,
                enable_conv_bn_folding=self.config.enable_conv_bn_folding,
                enable_triton=self.config.enable_triton,
            )
            report.graph = graph_report
            if graph_report.traced:
                if use_channels_last:
                    graph_model = graph_model.to(memory_format=torch.channels_last)
                candidates["fx"] = _InferenceWrapper(
                    graph_model,
                    device=self.device,
                    enable_amp=self.config.enable_amp,
                    channels_last=use_channels_last,
                ).eval()
                candidate_uses_selection_inputs["fx"] = True
                candidate_setup_times["fx"] = _elapsed_s(
                    started,
                    self.device,
                )
            elif graph_report.error:
                report.warnings.append(f"FX tracing failed: {graph_report.error}")

        if (
            self.config.enable_compile
            and hasattr(torch, "compile")
            and self.config.enable_fx
            and requested("fx_inductor")
        ):
            started = time.perf_counter()
            compiler_source = _copy_candidate_model(
                self.model,
                warnings=report.warnings,
                candidate_name="fx_inductor",
            )
            compiler_graph, compiler_report = optimize_graph(
                compiler_source,
                enable_conv_bn_folding=self.config.enable_conv_bn_folding,
                enable_triton=False,
            )
            if compiler_report.traced:
                try:
                    if use_channels_last:
                        compiler_graph = compiler_graph.to(
                            memory_format=torch.channels_last
                        )
                    compiler_wrapper = _InferenceWrapper(
                        compiler_graph,
                        device=self.device,
                        enable_amp=self.config.enable_amp,
                        channels_last=use_channels_last,
                    ).eval()
                    compiled = torch.compile(
                        compiler_wrapper,
                        mode=self.config.compile_mode,
                        fullgraph=False,
                        dynamic=self.config.dynamic_shapes,
                    )
                    candidates["fx_inductor"] = compiled
                    candidate_uses_selection_inputs["fx_inductor"] = True
                    candidate_setup_times["fx_inductor"] = _elapsed_s(
                        started,
                        self.device,
                    )
                except Exception as exc:
                    report.warnings.append(f"TorchInductor candidate failed: {exc!r}")
                    failed_candidates.append(
                        CandidateReport(name="fx_inductor", error=repr(exc)[:1000])
                    )

        provider_profile = WorkloadProfile(
            name=profile.name,
            cases=tuple(
                WorkloadCase(
                    name=case.name,
                    args=case.selection_args,
                    kwargs=case.selection_kwargs,
                    weight=case.weight,
                )
                for case in prepared_cases
            ),
            expected_calls=expected_calls,
        )
        representative = prepared_cases[0]
        provider_context = ProviderContext(
            device=self.device,
            target=self.target,
            policy=self.policy,
            example_args=representative.selection_args,
            example_kwargs=representative.selection_kwargs,
            workload_profile=provider_profile,
            cache_key=self.cache_key,
        )
        for provider in self.providers:
            name = provider.name.strip()
            if not requested(name):
                continue
            if not name or name in candidates:
                failed_candidates.append(
                    CandidateReport(
                        name=name or "unnamed_provider",
                        error="Provider name is empty or conflicts with another candidate",
                    )
                )
                continue
            try:
                availability = provider.probe(provider_context)
                if not availability.available:
                    failed_candidates.append(
                        CandidateReport(
                            name=name,
                            error=(
                                availability.reason
                                or "Provider is unavailable in this runtime"
                            ),
                        )
                    )
                    continue
                started = time.perf_counter()
                provider_model = _copy_candidate_model(
                    self.model,
                    warnings=report.warnings,
                    candidate_name=name,
                )
                artifact_dir = (
                    self.artifact_root / name if self.artifact_root is not None else None
                )
                context = replace(provider_context, artifact_dir=artifact_dir)
                built_plan = provider.build(provider_model, context)
                if not isinstance(built_plan, BuiltPlan):
                    built_plan = BuiltPlan(runner=built_plan)
                provider_candidate = built_plan.as_module().eval()
                candidate = _InferenceWrapper(
                    provider_candidate,
                    device=self.device,
                    enable_amp=False,
                    channels_last=use_channels_last,
                ).eval()
                candidate_setup_times[name] = _elapsed_s(started, self.device)
                candidates[name] = candidate
                candidate_plan_details[name] = built_plan
                candidate_uses_selection_inputs[name] = True
            except Exception as exc:
                failed_candidates.append(CandidateReport(name=name, error=repr(exc)[:1000]))

        candidate_models: dict[str, nn.Module] = {}
        candidate_items = list(candidates.items())
        if (
            preferred_plan is None
            and self.config.randomize_candidate_order
            and len(candidate_items) > 1
        ):
            random.Random(self.config.random_seed).shuffle(candidate_items)
        report.candidate_order = [name for name, _ in candidate_items]
        for name, candidate in candidate_items:
            candidate_report = CandidateReport(name=name)
            candidate_report.setup_time_s = candidate_setup_times.get(name, 0.0)
            if name in candidate_plan_details:
                candidate_report.artifacts = list(candidate_plan_details[name].artifacts)
                candidate_report.provider_metadata = dict(
                    candidate_plan_details[name].metadata
                )
            for prepared in prepared_cases:
                case_report = WorkloadCaseReport(
                    name=prepared.name,
                    weight=prepared.weight,
                    input_signature=prepared.signature,
                )
                if candidate_uses_selection_inputs[name]:
                    benchmark_args = prepared.selection_args
                    benchmark_kwargs = prepared.selection_kwargs
                else:
                    benchmark_args = prepared.base_args
                    benchmark_kwargs = prepared.base_kwargs
                try:
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                    first_call_started = time.perf_counter()
                    with torch.inference_mode():
                        output = candidate(*benchmark_args, **benchmark_kwargs)
                    case_report.first_call_time_s = _elapsed_s(
                        first_call_started,
                        self.device,
                    )
                    parity, max_error, mean_error = _compare_outputs(
                        prepared.reference,
                        output,
                        rtol=self.config.rtol,
                        atol=self.config.atol,
                    )
                    case_report.parity = parity or not self.config.verify_outputs
                    case_report.max_abs_error = max_error
                    case_report.mean_abs_error = mean_error
                    if not case_report.parity:
                        case_report.error = "Output parity check failed"
                        candidate_report.workload_cases.append(case_report)
                        break
                    if self.config.measure_peak_memory:
                        case_report.peak_memory_mb = _measure_peak_memory_mb(
                            candidate,
                            benchmark_args,
                            benchmark_kwargs,
                            device=self.device,
                        )
                    benchmark = _benchmark_latency(
                        candidate,
                        benchmark_args,
                        benchmark_kwargs,
                        device=self.device,
                        warmup=(0 if preferred_plan is not None else self.config.selection_warmup),
                        iterations=(
                            self.config.cache_validation_iterations
                            if preferred_plan is not None
                            else self.config.selection_iterations
                        ),
                        repeats=(1 if preferred_plan is not None else self.config.selection_repeats),
                    )
                    _apply_latency_stats(case_report, benchmark)
                except Exception as exc:
                    case_report.error = repr(exc)[:1000]
                candidate_report.workload_cases.append(case_report)
                if case_report.error:
                    break
            _aggregate_workload_cases(candidate_report)
            _apply_candidate_constraints(candidate_report, self.config)
            if (
                candidate_report.parity
                and candidate_report.latency_ms is not None
                and not candidate_report.constraint_violations
            ):
                candidate_models[name] = candidate
            report.candidates.append(candidate_report)
        report.candidates.extend(failed_candidates)

        if preferred_plan is not None:
            cached_candidate = next(
                (
                    candidate
                    for candidate in report.candidates
                    if candidate.name == preferred_plan
                ),
                None,
            )
            cache_valid = preferred_plan in candidate_models and cached_candidate is not None
            if (
                cache_valid
                and cached_latency_ms is not None
                and cached_latency_ms > 0
                and cached_candidate is not None
                and cached_candidate.latency_ms is not None
                and cached_candidate.latency_ms
                > cached_latency_ms * self.config.cache_max_latency_regression
            ):
                cache_valid = False
                retry_reason = (
                    f"Cached plan {preferred_plan!r} regressed beyond "
                    f"{self.config.cache_max_latency_regression:.3f}x; reran full selection."
                )
            else:
                retry_reason = (
                    f"Cached plan {preferred_plan!r} failed validation; reran full selection."
                )
            if cache_valid and cached_candidate is not None:
                cached_candidate.selected = True
                report.selected_plan = preferred_plan
                report.selection_reason = (
                    f"Reused cached plan {preferred_plan!r} after parity, constraint, "
                    "and latency-regression validation."
                )
                report.cache_hit = True
                report.optimization_time_s = (
                    self._initialization_time_s + time.perf_counter() - optimization_started
                )
                self.last_report = report
                return candidate_models[preferred_plan], report
            failed_validation_time_s = time.perf_counter() - optimization_started
            module, retry_report = self.optimize_workload_with_report(
                profile,
                _cache_retry_reason=retry_reason,
            )
            retry_report.optimization_time_s += failed_validation_time_s
            self.last_report = retry_report
            return module, retry_report

        _populate_candidate_metrics(
            report.candidates,
            expected_calls=expected_calls,
            confidence_level=self.config.confidence_level,
            bootstrap_resamples=self.config.bootstrap_resamples,
            random_seed=self.config.random_seed,
        )
        selected_name, reason = self._select_candidate(
            report.candidates,
            expected_calls=expected_calls,
        )
        if selected_name not in candidate_models:
            selected_name = "eager_fp32"
            reason = "No optimized candidate passed; returned FP32 eager fallback."
            if selected_name not in candidate_models:
                fallback = _InferenceWrapper(
                    self.model,
                    device=self.device,
                    enable_amp=False,
                    channels_last=False,
                ).eval()
                candidate_models[selected_name] = fallback
                report.candidates.append(CandidateReport(name=selected_name, parity=True))

        for candidate_report in report.candidates:
            candidate_report.selected = candidate_report.name == selected_name
        report.selected_plan = selected_name
        report.selection_reason = reason
        baseline = next(
            (
                candidate
                for candidate in report.candidates
                if candidate.baseline_reference
            ),
            None,
        )
        report.baseline_plan = baseline.name if baseline is not None else ""
        selected_report = next(
            (
                candidate
                for candidate in report.candidates
                if candidate.name == selected_name
            ),
            None,
        )
        report.confidence_gate_passed = bool(
            selected_report is not None and selected_report.confidence_gate_passed
        )
        report.optimization_time_s = (
            self._initialization_time_s + time.perf_counter() - optimization_started
        )
        self.last_report = report

        if self.config.verbose:
            LOGGER.info("Selected %s: %s", selected_name, reason)
        return candidate_models[selected_name], report

    def _select_candidate(
        self,
        candidate_reports: list[CandidateReport],
        *,
        expected_calls: int | None = None,
    ) -> tuple[str, str]:
        effective_expected_calls = (
            expected_calls if expected_calls is not None else self.config.expected_calls
        )
        valid = [
            candidate
            for candidate in candidate_reports
            if candidate.parity
            and candidate.latency_ms is not None
            and not candidate.constraint_violations
        ]
        if not valid:
            return "", "No candidate completed successfully."

        def selection_cost(candidate: CandidateReport) -> float:
            if candidate.selection_cost_ms is not None:
                return candidate.selection_cost_ms
            if effective_expected_calls is None:
                return candidate.latency_mean_ms or candidate.latency_ms or float("inf")
            if candidate.projected_total_ms is not None:
                return candidate.projected_total_ms
            return (
                candidate.setup_time_s * 1000.0
                + _first_call_ms(candidate)
                + (candidate.latency_ms or float("inf"))
                * max(
                    effective_expected_calls - max(1, len(candidate.workload_cases)),
                    0,
                )
            )

        def lower_bound(candidate: CandidateReport) -> float:
            return candidate.selection_cost_ci_low_ms or selection_cost(candidate)

        def upper_bound(candidate: CandidateReport) -> float:
            return candidate.selection_cost_ci_high_ms or selection_cost(candidate)

        baselines = [
            candidate
            for candidate in valid
            if candidate.name in {"eager_fp32", "native"}
        ]
        custom = [
            candidate
            for candidate in valid
            if candidate.name not in {"eager_fp32", "native"}
        ]
        if not baselines:
            if not custom:
                return "", "No selectable candidate completed successfully."
            fastest = min(custom, key=selection_cost)
            fastest.confidence_gate_passed = None
            return fastest.name, "Built-in plans were invalid; selected fastest valid provider."

        baseline = min(baselines, key=selection_cost)
        baseline.baseline_reference = True
        if not custom:
            qualifier = (
                "lowest projected-cost"
                if effective_expected_calls is not None
                else "fastest valid"
            )
            return baseline.name, f"Selected {qualifier} built-in plan: {baseline.name}."

        baseline_cost = selection_cost(baseline)
        required_upper_bound = lower_bound(baseline) / self.config.min_speedup
        for candidate in custom:
            candidate.confidence_gate_passed = (
                upper_bound(candidate) <= required_upper_bound
            )
            if not candidate.confidence_gate_passed:
                candidate.rejection_reason = (
                    f"upper confidence cost {upper_bound(candidate):.6g} ms did not clear "
                    f"required bound {required_upper_bound:.6g} ms"
                )
        passing_custom = [
            candidate for candidate in custom if candidate.confidence_gate_passed
        ]
        if passing_custom:
            fastest_custom = min(passing_custom, key=selection_cost)
            custom_cost = selection_cost(fastest_custom)
            speedup = baseline_cost / custom_cost
            if effective_expected_calls is not None:
                return (
                    fastest_custom.name,
                    f"Projected lifecycle cost is {speedup:.3f}x lower than "
                    f"{baseline.name} over {effective_expected_calls} calls; the "
                    f"{self.config.confidence_level:.1%} upper confidence bound cleared "
                    f"the {self.config.min_speedup:.3f}x replacement threshold.",
                )
            return (
                fastest_custom.name,
                f"Mean steady-state cost is {speedup:.3f}x lower than {baseline.name}; "
                f"the {self.config.confidence_level:.1%} upper confidence bound cleared "
                f"the {self.config.min_speedup:.3f}x replacement threshold.",
            )
        return (
            baseline.name,
            (
                f"Challenger confidence bounds did not clear the lifecycle threshold "
                f"over {effective_expected_calls} calls; retained {baseline.name}."
                if effective_expected_calls is not None
                else "Challenger confidence bounds overlapped the steady-state "
                f"replacement threshold; retained {baseline.name}."
            ),
        )
