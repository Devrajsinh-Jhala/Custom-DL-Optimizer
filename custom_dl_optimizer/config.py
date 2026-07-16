from dataclasses import dataclass


@dataclass(frozen=True)
class OptimizationConfig:
    """Controls graph rewriting, plan selection, and correctness checks."""

    enable_profiling: bool = True
    enable_fx: bool = True
    enable_conv_bn_folding: bool = True
    enable_triton: bool = True
    enable_amp: bool = True
    channels_last: bool = True
    enable_compile: bool = False
    compile_mode: str = "default"
    dynamic_shapes: bool = False
    verify_outputs: bool = True
    benchmark_eager: bool = True
    rtol: float = 5e-2
    atol: float = 5e-2
    selection_warmup: int = 3
    selection_iterations: int = 10
    selection_repeats: int = 3
    expected_calls: int | None = None
    min_speedup: float = 1.02
    confidence_level: float = 0.95
    bootstrap_resamples: int = 1_000
    random_seed: int = 17
    randomize_candidate_order: bool = True
    max_setup_time_s: float | None = None
    max_first_call_time_s: float | None = None
    max_peak_memory_mb: float | None = None
    measure_peak_memory: bool = True
    plan_cache_dir: str | None = None
    reuse_cached_plan: bool = True
    cache_validation_iterations: int = 2
    cache_max_latency_regression: float = 1.25
    copy_model: bool = True
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.selection_warmup < 0:
            raise ValueError("selection_warmup must be non-negative")
        if self.selection_iterations < 1:
            raise ValueError("selection_iterations must be at least 1")
        if self.selection_repeats < 1:
            raise ValueError("selection_repeats must be at least 1")
        if self.expected_calls is not None and self.expected_calls < 1:
            raise ValueError("expected_calls must be at least 1 when provided")
        if self.min_speedup < 1.0:
            raise ValueError("min_speedup must be at least 1.0")
        if not 0.5 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between 0.5 and 1.0")
        if self.bootstrap_resamples < 100:
            raise ValueError("bootstrap_resamples must be at least 100")
        if self.max_setup_time_s is not None and self.max_setup_time_s < 0:
            raise ValueError("max_setup_time_s must be non-negative")
        if self.max_first_call_time_s is not None and self.max_first_call_time_s < 0:
            raise ValueError("max_first_call_time_s must be non-negative")
        if self.max_peak_memory_mb is not None and self.max_peak_memory_mb < 0:
            raise ValueError("max_peak_memory_mb must be non-negative")
        if self.max_peak_memory_mb is not None and not self.measure_peak_memory:
            raise ValueError(
                "measure_peak_memory must be enabled when max_peak_memory_mb is set"
            )
        if self.cache_validation_iterations < 1:
            raise ValueError("cache_validation_iterations must be at least 1")
        if self.cache_max_latency_regression < 1.0:
            raise ValueError("cache_max_latency_regression must be at least 1.0")
        if self.rtol < 0 or self.atol < 0:
            raise ValueError("output tolerances must be non-negative")
