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
class CandidateReport:
    name: str
    latency_ms: float | None = None
    parity: bool = False
    max_abs_error: float | None = None
    mean_abs_error: float | None = None
    setup_time_s: float = 0.0
    selected: bool = False
    calls_per_second: float | None = None
    speedup_vs_eager: float | None = None
    speedup_vs_native: float | None = None
    error: str = ""


@dataclass
class OptimizationReport:
    device: str
    selected_plan: str = "native"
    selection_reason: str = ""
    input_signature: str = ""
    channels_last: bool = False
    amp: bool = False
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
