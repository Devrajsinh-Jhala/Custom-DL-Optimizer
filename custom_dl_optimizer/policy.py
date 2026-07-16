from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

from .config import OptimizationConfig

SelectionObjective = Literal["steady_state_latency", "lifecycle_latency"]


@dataclass(frozen=True)
class ExecutionTarget:
    """Hardware target for candidate discovery and execution."""

    device: str | torch.device | None = None
    accelerator: str = "auto"

    def resolve_device(self) -> torch.device:
        requested = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        resolved = torch.device(requested)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return resolved


@dataclass(frozen=True)
class MeasurementPolicy:
    """Controls repeatable candidate timing and uncertainty estimation."""

    warmup: int = 5
    iterations: int = 20
    repeats: int = 3
    confidence_level: float = 0.95
    bootstrap_resamples: int = 1_000
    random_seed: int = 17
    randomize_candidate_order: bool = True
    measure_peak_memory: bool = True

    def __post_init__(self) -> None:
        if self.warmup < 0:
            raise ValueError("warmup must be non-negative")
        if self.iterations < 1 or self.repeats < 1:
            raise ValueError("iterations and repeats must be at least 1")
        if not 0.5 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between 0.5 and 1.0")
        if self.bootstrap_resamples < 100:
            raise ValueError("bootstrap_resamples must be at least 100")


@dataclass(frozen=True)
class ValidationPolicy:
    """Numerical acceptance policy applied before a plan can be selected."""

    verify_outputs: bool = True
    rtol: float = 5e-2
    atol: float = 5e-2

    def __post_init__(self) -> None:
        if self.rtol < 0 or self.atol < 0:
            raise ValueError("output tolerances must be non-negative")


@dataclass(frozen=True)
class DeploymentConstraints:
    """Hard limits and the minimum evidence required to replace a baseline."""

    expected_calls: int | None = None
    min_speedup: float = 1.02
    max_setup_time_s: float | None = None
    max_first_call_time_s: float | None = None
    max_peak_memory_mb: float | None = None

    def __post_init__(self) -> None:
        if self.expected_calls is not None and self.expected_calls < 1:
            raise ValueError("expected_calls must be at least 1 when provided")
        if self.min_speedup < 1.0:
            raise ValueError("min_speedup must be at least 1.0")
        for name in (
            "max_setup_time_s",
            "max_first_call_time_s",
            "max_peak_memory_mb",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class OptimizationPolicy:
    """Complete v3 policy for plan generation, measurement, and selection."""

    objective: SelectionObjective = "steady_state_latency"
    measurement: MeasurementPolicy = field(default_factory=MeasurementPolicy)
    validation: ValidationPolicy = field(default_factory=ValidationPolicy)
    constraints: DeploymentConstraints = field(default_factory=DeploymentConstraints)
    enable_profiling: bool = True
    enable_fx: bool = True
    enable_conv_bn_folding: bool = True
    enable_triton: bool = True
    enable_amp: bool = True
    channels_last: bool = True
    enable_compile: bool = False
    compile_mode: str = "default"
    dynamic_shapes: bool = False
    benchmark_eager: bool = True
    plan_cache_dir: str | None = None
    reuse_cached_plan: bool = True
    cache_validation_iterations: int = 2
    cache_max_latency_regression: float = 1.25
    copy_model: bool = True
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.objective not in {"steady_state_latency", "lifecycle_latency"}:
            raise ValueError(f"Unsupported objective: {self.objective!r}")
        if self.cache_validation_iterations < 1:
            raise ValueError("cache_validation_iterations must be at least 1")
        if self.cache_max_latency_regression < 1.0:
            raise ValueError("cache_max_latency_regression must be at least 1.0")
        if (
            self.constraints.max_peak_memory_mb is not None
            and not self.measurement.measure_peak_memory
        ):
            raise ValueError(
                "measure_peak_memory must be enabled when max_peak_memory_mb is set"
            )

    @classmethod
    def from_engine_config(cls, config: OptimizationConfig) -> OptimizationPolicy:
        """Adapt the v2 flat config for internal and migration-only callers."""

        return cls(
            objective=(
                "lifecycle_latency"
                if config.expected_calls is not None
                else "steady_state_latency"
            ),
            measurement=MeasurementPolicy(
                warmup=config.selection_warmup,
                iterations=config.selection_iterations,
                repeats=config.selection_repeats,
                confidence_level=config.confidence_level,
                bootstrap_resamples=config.bootstrap_resamples,
                random_seed=config.random_seed,
                randomize_candidate_order=config.randomize_candidate_order,
                measure_peak_memory=config.measure_peak_memory,
            ),
            validation=ValidationPolicy(
                verify_outputs=config.verify_outputs,
                rtol=config.rtol,
                atol=config.atol,
            ),
            constraints=DeploymentConstraints(
                expected_calls=config.expected_calls,
                min_speedup=config.min_speedup,
                max_setup_time_s=config.max_setup_time_s,
                max_first_call_time_s=config.max_first_call_time_s,
                max_peak_memory_mb=config.max_peak_memory_mb,
            ),
            enable_profiling=config.enable_profiling,
            enable_fx=config.enable_fx,
            enable_conv_bn_folding=config.enable_conv_bn_folding,
            enable_triton=config.enable_triton,
            enable_amp=config.enable_amp,
            channels_last=config.channels_last,
            enable_compile=config.enable_compile,
            compile_mode=config.compile_mode,
            dynamic_shapes=config.dynamic_shapes,
            benchmark_eager=config.benchmark_eager,
            plan_cache_dir=config.plan_cache_dir,
            reuse_cached_plan=config.reuse_cached_plan,
            cache_validation_iterations=config.cache_validation_iterations,
            cache_max_latency_regression=config.cache_max_latency_regression,
            copy_model=config.copy_model,
            verbose=config.verbose,
        )

    def to_engine_config(self) -> OptimizationConfig:
        """Translate the public policy into the internal measurement engine contract."""

        return OptimizationConfig(
            enable_profiling=self.enable_profiling,
            enable_fx=self.enable_fx,
            enable_conv_bn_folding=self.enable_conv_bn_folding,
            enable_triton=self.enable_triton,
            enable_amp=self.enable_amp,
            channels_last=self.channels_last,
            enable_compile=self.enable_compile,
            compile_mode=self.compile_mode,
            dynamic_shapes=self.dynamic_shapes,
            verify_outputs=self.validation.verify_outputs,
            benchmark_eager=self.benchmark_eager,
            rtol=self.validation.rtol,
            atol=self.validation.atol,
            selection_warmup=self.measurement.warmup,
            selection_iterations=self.measurement.iterations,
            selection_repeats=self.measurement.repeats,
            expected_calls=self.constraints.expected_calls,
            min_speedup=self.constraints.min_speedup,
            max_setup_time_s=self.constraints.max_setup_time_s,
            max_first_call_time_s=self.constraints.max_first_call_time_s,
            max_peak_memory_mb=self.constraints.max_peak_memory_mb,
            measure_peak_memory=self.measurement.measure_peak_memory,
            confidence_level=self.measurement.confidence_level,
            bootstrap_resamples=self.measurement.bootstrap_resamples,
            random_seed=self.measurement.random_seed,
            randomize_candidate_order=self.measurement.randomize_candidate_order,
            plan_cache_dir=self.plan_cache_dir,
            reuse_cached_plan=self.reuse_cached_plan,
            cache_validation_iterations=self.cache_validation_iterations,
            cache_max_latency_regression=self.cache_max_latency_regression,
            copy_model=self.copy_model,
            verbose=self.verbose,
        )
