import copy
import logging
import math
import statistics
import time
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.nn as nn

from custom_dl_optimizer.config import OptimizationConfig
from custom_dl_optimizer.providers import CandidateContext, CandidateProvider
from custom_dl_optimizer.report import CandidateReport, OptimizationReport
from custom_dl_optimizer.runtime import inspect_runtime

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
    minimum_ms: float
    p90_ms: float
    stdev_ms: float
    samples_ms: tuple[float, ...]


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
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(iterations):
                    model(*args, **kwargs)
                end.record()
                end.synchronize()
                samples.append(float(start.elapsed_time(end) / iterations))
            else:
                started = time.perf_counter()
                for _ in range(iterations):
                    model(*args, **kwargs)
                samples.append((time.perf_counter() - started) * 1000.0 / iterations)
    p90 = (
        statistics.quantiles(samples, n=10, method="inclusive")[8]
        if len(samples) > 1
        else samples[0]
    )
    return _BenchmarkStats(
        median_ms=float(statistics.median(samples)),
        minimum_ms=float(min(samples)),
        p90_ms=float(p90),
        stdev_ms=float(statistics.stdev(samples)) if len(samples) > 1 else 0.0,
        samples_ms=tuple(float(sample) for sample in samples),
    )


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


def _populate_candidate_metrics(
    candidate_reports: list[CandidateReport],
    *,
    expected_calls: int | None,
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
            candidate.projected_total_ms = (
                candidate.setup_time_s * 1000.0
                + _first_call_ms(candidate)
                + candidate.latency_ms * max(expected_calls - 1, 0)
            )

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
            - steady_state_baseline.latency_ms
        )
        candidate_cold_overhead = (
            candidate.setup_time_s * 1000.0
            + _first_call_ms(candidate)
            - candidate.latency_ms
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
        providers: tuple[CandidateProvider, ...] = (),
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
        self.providers = providers
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

    def optimize_with_report(
        self,
        *example_args: Any,
        **example_kwargs: Any,
    ) -> tuple[nn.Module, OptimizationReport]:
        optimization_started = time.perf_counter()
        if not example_args and not example_kwargs:
            raise ValueError("At least one example input is required")

        base_args, base_kwargs = self.prepare_inputs(
            *example_args,
            channels_last=False,
            **example_kwargs,
        )
        use_channels_last = self.config.channels_last and self.device.type == "cuda"
        selection_args, selection_kwargs = self.prepare_inputs(
            *example_args,
            channels_last=use_channels_last,
            **example_kwargs,
        )

        report = OptimizationReport(
            device=str(self.device),
            input_signature=_input_signature(base_args, base_kwargs),
            channels_last=use_channels_last,
            amp=self.config.enable_amp and self.device.type == "cuda",
            expected_calls=self.config.expected_calls,
            selection_basis=(
                "projected_total_time"
                if self.config.expected_calls is not None
                else "steady_state_latency"
            ),
            runtime=inspect_runtime(self.device),
            warnings=list(self._warnings),
        )

        with torch.inference_mode():
            reference = self.model(*base_args, **base_kwargs)

        if self.config.enable_profiling:
            try:
                report.operator_profile = analyze_bottlenecks(
                    self.model,
                    base_args,
                    base_kwargs,
                )
            except Exception as exc:
                report.warnings.append(f"Operator profiling failed: {exc!r}")

        candidates: dict[str, nn.Module] = {}
        candidate_inputs: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {}
        candidate_setup_times: dict[str, float] = {}
        failed_candidates: list[CandidateReport] = []

        if self.config.benchmark_eager:
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
            candidate_inputs["eager_fp32"] = (base_args, base_kwargs)
            candidate_setup_times["eager_fp32"] = _elapsed_s(
                started,
                self.device,
            )

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
        candidate_inputs["native"] = (selection_args, selection_kwargs)
        candidate_setup_times["native"] = _elapsed_s(started, self.device)

        if self.config.enable_fx:
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
                candidate_inputs["fx"] = (selection_args, selection_kwargs)
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
                    candidate_inputs["fx_inductor"] = (
                        selection_args,
                        selection_kwargs,
                    )
                    candidate_setup_times["fx_inductor"] = _elapsed_s(
                        started,
                        self.device,
                    )
                except Exception as exc:
                    report.warnings.append(f"TorchInductor candidate failed: {exc!r}")
                    failed_candidates.append(
                        CandidateReport(name="fx_inductor", error=repr(exc)[:1000])
                    )

        provider_context = CandidateContext(
            device=self.device,
            config=self.config,
            example_args=selection_args,
            example_kwargs=selection_kwargs,
        )
        for provider in self.providers:
            name = provider.name.strip()
            if not name or name in candidates:
                failed_candidates.append(
                    CandidateReport(
                        name=name or "unnamed_provider",
                        error="Provider name is empty or conflicts with another candidate",
                    )
                )
                continue
            try:
                if not provider.is_available(provider_context):
                    failed_candidates.append(
                        CandidateReport(name=name, error="Provider is unavailable in this runtime")
                    )
                    continue
                started = time.perf_counter()
                provider_model = _copy_candidate_model(
                    self.model,
                    warnings=report.warnings,
                    candidate_name=name,
                )
                provider_candidate = provider.build(provider_model, provider_context).eval()
                candidate = _InferenceWrapper(
                    provider_candidate,
                    device=self.device,
                    enable_amp=False,
                    channels_last=use_channels_last,
                ).eval()
                candidate_setup_times[name] = _elapsed_s(started, self.device)
                candidates[name] = candidate
                candidate_inputs[name] = (selection_args, selection_kwargs)
            except Exception as exc:
                failed_candidates.append(CandidateReport(name=name, error=repr(exc)[:1000]))

        candidate_models: dict[str, nn.Module] = {}
        for name, candidate in candidates.items():
            candidate_report = CandidateReport(name=name)
            candidate_report.setup_time_s = candidate_setup_times.get(name, 0.0)
            benchmark_args, benchmark_kwargs = candidate_inputs[name]
            try:
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                first_call_started = time.perf_counter()
                with torch.inference_mode():
                    output = candidate(*benchmark_args, **benchmark_kwargs)
                candidate_report.first_call_time_s = _elapsed_s(
                    first_call_started,
                    self.device,
                )
                parity, max_error, mean_error = _compare_outputs(
                    reference,
                    output,
                    rtol=self.config.rtol,
                    atol=self.config.atol,
                )
                candidate_report.parity = parity or not self.config.verify_outputs
                candidate_report.max_abs_error = max_error
                candidate_report.mean_abs_error = mean_error
                if candidate_report.parity:
                    benchmark = _benchmark_latency(
                        candidate,
                        benchmark_args,
                        benchmark_kwargs,
                        device=self.device,
                        warmup=self.config.selection_warmup,
                        iterations=self.config.selection_iterations,
                        repeats=self.config.selection_repeats,
                    )
                    candidate_report.latency_ms = benchmark.median_ms
                    candidate_report.latency_min_ms = benchmark.minimum_ms
                    candidate_report.latency_p90_ms = benchmark.p90_ms
                    candidate_report.latency_stdev_ms = benchmark.stdev_ms
                    candidate_report.latency_samples_ms = list(benchmark.samples_ms)
                    candidate_models[name] = candidate
                else:
                    candidate_report.error = "Output parity check failed"
            except Exception as exc:
                candidate_report.error = repr(exc)[:1000]
            report.candidates.append(candidate_report)
        report.candidates.extend(failed_candidates)

        _populate_candidate_metrics(
            report.candidates,
            expected_calls=self.config.expected_calls,
        )
        selected_name, reason = self._select_candidate(report.candidates)
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
    ) -> tuple[str, str]:
        valid = [
            candidate
            for candidate in candidate_reports
            if candidate.parity and candidate.latency_ms is not None
        ]
        if not valid:
            return "", "No candidate completed successfully."

        def selection_cost(candidate: CandidateReport) -> float:
            if self.config.expected_calls is None:
                return candidate.latency_ms or float("inf")
            if candidate.projected_total_ms is not None:
                return candidate.projected_total_ms
            return (
                candidate.setup_time_s * 1000.0
                + _first_call_ms(candidate)
                + (candidate.latency_ms or float("inf"))
                * max(self.config.expected_calls - 1, 0)
            )

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
            return fastest.name, "Built-in plans were invalid; selected fastest valid provider."

        baseline = min(baselines, key=selection_cost)
        if not custom:
            qualifier = (
                "lowest projected-cost"
                if self.config.expected_calls is not None
                else "fastest valid"
            )
            return baseline.name, f"Selected {qualifier} built-in plan: {baseline.name}."

        fastest_custom = min(custom, key=selection_cost)
        baseline_cost = selection_cost(baseline)
        custom_cost = selection_cost(fastest_custom)
        required_cost = baseline_cost / self.config.min_speedup
        if custom_cost <= required_cost:
            speedup = baseline_cost / custom_cost
            if self.config.expected_calls is not None:
                return (
                    fastest_custom.name,
                    f"Projected total cost is {speedup:.3f}x faster than {baseline.name} "
                    f"over {self.config.expected_calls} calls and cleared the "
                    f"{self.config.min_speedup:.3f}x selection threshold.",
                )
            return (
                fastest_custom.name,
                f"Measured {speedup:.3f}x over {baseline.name} and cleared the "
                f"{self.config.min_speedup:.3f}x selection threshold.",
            )
        return (
            baseline.name,
            (
                f"Custom plans did not clear the projected-total threshold over "
                f"{self.config.expected_calls} calls; using {baseline.name}."
                if self.config.expected_calls is not None
                else "Custom plans did not clear the steady-state speedup threshold; "
                f"using {baseline.name}."
            ),
        )
