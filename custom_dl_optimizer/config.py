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
        if self.rtol < 0 or self.atol < 0:
            raise ValueError("output tolerances must be non-negative")
