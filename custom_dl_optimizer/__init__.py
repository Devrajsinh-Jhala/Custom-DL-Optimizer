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
from .optimizer import (
    InferenceOptimizer,
    OptimizationDecision,
    OptimizationResult,
    Optimizer,
)
from .policy import (
    DeploymentConstraints,
    ExecutionTarget,
    MeasurementPolicy,
    OptimizationPolicy,
    ValidationPolicy,
)
from .providers import (
    BuiltPlan,
    CandidateContext,
    CandidateProvider,
    ExecutionProvider,
    FunctionCandidateProvider,
    FunctionExecutionProvider,
    InferenceRunner,
    ProviderAvailability,
    ProviderContext,
)
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
    __version__ = "3.0.0"

__all__ = [
    "AutoOptimizer",
    "BuiltPlan",
    "CandidateContext",
    "CandidateProvider",
    "CandidateReport",
    "DeploymentConstraints",
    "ExecutionProvider",
    "ExecutionTarget",
    "FunctionCandidateProvider",
    "FunctionExecutionProvider",
    "GraphSurgeryReport",
    "InferenceOptimizer",
    "InferenceRunner",
    "MeasurementPolicy",
    "ONNXRuntimeProvider",
    "OptimizationAgentToolkit",
    "OptimizationConfig",
    "OptimizationDecision",
    "OptimizationPolicy",
    "OptimizationReport",
    "OptimizationResult",
    "Optimizer",
    "PlanCache",
    "PlanCacheRecord",
    "ProviderAvailability",
    "ProviderContext",
    "RuntimeCapabilities",
    "TorchAOQuantizationProvider",
    "TorchTensorRTProvider",
    "ValidationPolicy",
    "WorkloadCase",
    "WorkloadCaseReport",
    "WorkloadProfile",
    "__version__",
    "create_plan_cache_key",
    "export_paper_artifacts",
    "inspect_runtime",
]
