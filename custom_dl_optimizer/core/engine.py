import copy
import logging
import statistics
import time
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

import torch
import torch.nn as nn

from custom_dl_optimizer.config import OptimizationConfig
from custom_dl_optimizer.report import CandidateReport, OptimizationReport

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
    reference_tensors = list(_walk_values(reference))
    candidate_tensors = list(_walk_values(candidate))
    if len(reference_tensors) != len(candidate_tensors):
        return False, float("inf"), float("inf")

    allclose = True
    max_errors: list[float] = []
    mean_errors: list[float] = []
    for ref, cand in zip(reference_tensors, candidate_tensors, strict=True):
        if isinstance(ref, torch.Tensor) != isinstance(cand, torch.Tensor):
            return False, float("inf"), float("inf")
        if not isinstance(ref, torch.Tensor):
            allclose = allclose and ref == cand
            continue
        if ref.shape != cand.shape:
            return False, float("inf"), float("inf")
        ref_float = ref.detach().float()
        cand_float = cand.detach().float()
        diff = (ref_float - cand_float).abs()
        max_errors.append(float(diff.max().item()) if diff.numel() else 0.0)
        mean_errors.append(float(diff.mean().item()) if diff.numel() else 0.0)
        allclose = allclose and torch.allclose(
            ref_float,
            cand_float,
            rtol=rtol,
            atol=atol,
        )
    return (
        bool(allclose),
        max(max_errors, default=0.0),
        statistics.mean(mean_errors) if mean_errors else 0.0,
    )


def _benchmark_ms(
    model: nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
    repeats: int,
) -> float:
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
    return float(statistics.median(samples))


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
    ) -> None:
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
        self.last_report: OptimizationReport | None = None

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
        native_core = self.model
        if use_channels_last:
            native_core = native_core.to(memory_format=torch.channels_last)
        candidates["native"] = _InferenceWrapper(
            native_core,
            device=self.device,
            enable_amp=self.config.enable_amp,
            channels_last=use_channels_last,
        ).eval()

        if self.config.enable_fx:
            graph_model, graph_report = optimize_graph(
                self.model,
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
            elif graph_report.error:
                report.warnings.append(f"FX tracing failed: {graph_report.error}")

        if (
            self.config.enable_compile
            and hasattr(torch, "compile")
            and self.config.enable_fx
        ):
            compiler_graph, compiler_report = optimize_graph(
                self.model,
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
                    started = time.perf_counter()
                    compiled = torch.compile(
                        compiler_wrapper,
                        mode=self.config.compile_mode,
                        fullgraph=False,
                        dynamic=self.config.dynamic_shapes,
                    )
                    with torch.inference_mode():
                        compiled(*selection_args, **selection_kwargs)
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                    candidates["fx_inductor"] = compiled
                    compile_setup_time = time.perf_counter() - started
                except Exception as exc:
                    report.warnings.append(f"TorchInductor candidate failed: {exc!r}")
                    compile_setup_time = 0.0
            else:
                compile_setup_time = 0.0
        else:
            compile_setup_time = 0.0

        candidate_models: dict[str, nn.Module] = {}
        for name, candidate in candidates.items():
            candidate_report = CandidateReport(name=name)
            if name == "fx_inductor":
                candidate_report.setup_time_s = compile_setup_time
            try:
                with torch.inference_mode():
                    output = candidate(*selection_args, **selection_kwargs)
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
                    candidate_report.latency_ms = _benchmark_ms(
                        candidate,
                        selection_args,
                        selection_kwargs,
                        device=self.device,
                        warmup=self.config.selection_warmup,
                        iterations=self.config.selection_iterations,
                        repeats=self.config.selection_repeats,
                    )
                    candidate_models[name] = candidate
                else:
                    candidate_report.error = "Output parity check failed"
            except Exception as exc:
                candidate_report.error = repr(exc)[:1000]
            report.candidates.append(candidate_report)

        selected_name, reason = self._select_candidate(report.candidates)
        if selected_name not in candidate_models:
            selected_name = "eager_fp32"
            reason = "No optimized candidate passed; returned FP32 eager fallback."
            fallback = _InferenceWrapper(
                self.model,
                device=self.device,
                enable_amp=False,
                channels_last=False,
            ).eval()
            candidate_models[selected_name] = fallback
            report.candidates.append(
                CandidateReport(name=selected_name, parity=True, selected=True)
            )

        for candidate_report in report.candidates:
            candidate_report.selected = candidate_report.name == selected_name
        report.selected_plan = selected_name
        report.selection_reason = reason
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

        native = next((candidate for candidate in valid if candidate.name == "native"), None)
        custom = [candidate for candidate in valid if candidate.name != "native"]
        if native is None:
            fastest = min(custom, key=lambda candidate: candidate.latency_ms or float("inf"))
            return fastest.name, "Native plan was invalid; selected fastest valid plan."
        if not custom:
            return native.name, "No custom plan completed successfully."

        fastest_custom = min(
            custom,
            key=lambda candidate: candidate.latency_ms or float("inf"),
        )
        required_latency = (native.latency_ms or float("inf")) / self.config.min_speedup
        if (fastest_custom.latency_ms or float("inf")) <= required_latency:
            speedup = (native.latency_ms or 0.0) / (
                fastest_custom.latency_ms or float("inf")
            )
            return (
                fastest_custom.name,
                f"Measured {speedup:.3f}x over native and cleared the "
                f"{self.config.min_speedup:.3f}x selection threshold.",
            )
        return (
            native.name,
            "Custom plans did not clear the measured speedup threshold; using native.",
        )
