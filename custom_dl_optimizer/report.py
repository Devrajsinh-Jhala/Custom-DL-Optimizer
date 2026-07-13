from dataclasses import asdict, dataclass, field
from typing import Any


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
    error: str = ""


@dataclass
class OptimizationReport:
    device: str
    selected_plan: str = "native"
    selection_reason: str = ""
    input_signature: str = ""
    channels_last: bool = False
    amp: bool = False
    operator_profile: list[OperatorProfile] = field(default_factory=list)
    graph: GraphSurgeryReport = field(default_factory=GraphSurgeryReport)
    candidates: list[CandidateReport] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
