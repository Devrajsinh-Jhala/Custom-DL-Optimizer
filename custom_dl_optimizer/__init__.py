from importlib.metadata import PackageNotFoundError, version

from .agent import OptimizationAgentToolkit
from .cache import PlanCache, PlanCacheRecord, create_plan_cache_key
from .config import OptimizationConfig
from .core.engine import AutoOptimizer
from .integrations import (
    ONNXRuntimeProvider,
    TorchAOQuantizationProvider,
    TorchTensorRTProvider,
)
from .optimizer import OptimizationResult, Optimizer
from .providers import CandidateContext, CandidateProvider, FunctionCandidateProvider
from .report import (
    CandidateReport,
    GraphSurgeryReport,
    OptimizationReport,
    WorkloadCaseReport,
)
from .research import export_paper_artifacts
from .runtime import RuntimeCapabilities, inspect_runtime
from .workload import WorkloadCase, WorkloadProfile

try:
    __version__ = version("custom-dl-optimizer")
except PackageNotFoundError:  # Source checkout before installation.
    __version__ = "2.2.0"

__all__ = [
    "AutoOptimizer",
    "CandidateContext",
    "CandidateProvider",
    "CandidateReport",
    "FunctionCandidateProvider",
    "GraphSurgeryReport",
    "ONNXRuntimeProvider",
    "OptimizationAgentToolkit",
    "OptimizationConfig",
    "OptimizationReport",
    "OptimizationResult",
    "Optimizer",
    "PlanCache",
    "PlanCacheRecord",
    "RuntimeCapabilities",
    "TorchAOQuantizationProvider",
    "TorchTensorRTProvider",
    "WorkloadCase",
    "WorkloadCaseReport",
    "WorkloadProfile",
    "__version__",
    "create_plan_cache_key",
    "export_paper_artifacts",
    "inspect_runtime",
]
