import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .runtime import RuntimeCapabilities


@dataclass
class OperatorProfile:
    name: str
    self_time_us: float
    calls: int


@dataclass
class GraphSurgeryReport:
    traced: bool = False
    conv_bn_fusions: int = 0
    module_relu_replacements: int = 0
    functional_relu_replacements: int = 0
    skipped_inplace_relu: int = 0
    error: str = ""

    @property
    def total_rewrites(self) -> int:
        return (
            self.conv_bn_fusions
            + self.module_relu_replacements
            + self.functional_relu_replacements
        )


@dataclass
class WorkloadCaseReport:
    name: str
    weight: float
    input_signature: str
    latency_ms: float | None = None
    latency_mean_ms: float | None = None
    latency_min_ms: float | None = None
    latency_p90_ms: float | None = None
    latency_p95_ms: float | None = None
    latency_p99_ms: float | None = None
    latency_stdev_ms: float | None = None
    latency_ci95_low_ms: float | None = None
    latency_ci95_high_ms: float | None = None
    latency_samples_ms: list[float] = field(default_factory=list)
    first_call_time_s: float | None = None
    peak_memory_mb: float | None = None
    parity: bool = False
    max_abs_error: float | None = None
    mean_abs_error: float | None = None
    error: str = ""


@dataclass
class CandidateReport:
    name: str
    latency_ms: float | None = None
    latency_mean_ms: float | None = None
    latency_min_ms: float | None = None
    latency_p90_ms: float | None = None
    latency_p95_ms: float | None = None
    latency_p99_ms: float | None = None
    latency_stdev_ms: float | None = None
    latency_ci95_low_ms: float | None = None
    latency_ci95_high_ms: float | None = None
    latency_samples_ms: list[float] = field(default_factory=list)
    parity: bool = False
    max_abs_error: float | None = None
    mean_abs_error: float | None = None
    setup_time_s: float = 0.0
    first_call_time_s: float | None = None
    projected_total_ms: float | None = None
    selection_cost_ms: float | None = None
    selection_cost_ci_low_ms: float | None = None
    selection_cost_ci_high_ms: float | None = None
    confidence_gate_passed: bool | None = None
    baseline_reference: bool = False
    rejection_reason: str = ""
    break_even_calls_vs_baseline: int | None = None
    selected: bool = False
    calls_per_second: float | None = None
    speedup_vs_eager: float | None = None
    speedup_vs_native: float | None = None
    peak_memory_mb: float | None = None
    constraint_violations: list[str] = field(default_factory=list)
    workload_cases: list[WorkloadCaseReport] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    provider_metadata: dict[str, Any] = field(default_factory=dict)
    error: str = ""


@dataclass
class OptimizationReport:
    device: str
    schema_version: int = 3
    workload_name: str = "workload"
    selected_plan: str = "native"
    selection_reason: str = ""
    input_signature: str = ""
    channels_last: bool = False
    amp: bool = False
    expected_calls: int | None = None
    selection_basis: str = "steady_state_latency"
    selection_estimator: str = "bootstrap_mean"
    confidence_level: float = 0.95
    bootstrap_resamples: int = 0
    random_seed: int = 0
    candidate_order: list[str] = field(default_factory=list)
    baseline_plan: str = ""
    confidence_gate_passed: bool = False
    optimization_time_s: float = 0.0
    cache_key: str = ""
    cache_hit: bool = False
    cache_record_path: str = ""
    cache_lookup_time_s: float = 0.0
    runtime: RuntimeCapabilities | None = None
    operator_profile: list[OperatorProfile] = field(default_factory=list)
    graph: GraphSurgeryReport = field(default_factory=GraphSurgeryReport)
    candidates: list[CandidateReport] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def selected_candidate(self) -> CandidateReport | None:
        return next((candidate for candidate in self.candidates if candidate.selected), None)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, sort_keys=True)

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.to_json() + "\n", encoding="utf-8")
        return destination
