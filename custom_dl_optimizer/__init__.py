from importlib.metadata import PackageNotFoundError, version

from .agent import OptimizationAgentToolkit
from .config import OptimizationConfig
from .core.engine import AutoOptimizer
from .optimizer import OptimizationResult, Optimizer
from .providers import CandidateContext, CandidateProvider, FunctionCandidateProvider
from .report import CandidateReport, GraphSurgeryReport, OptimizationReport
from .runtime import RuntimeCapabilities, inspect_runtime

try:
    __version__ = version("custom-dl-optimizer")
except PackageNotFoundError:  # Source checkout before installation.
    __version__ = "2.1.0"

__all__ = [
    "AutoOptimizer",
    "CandidateContext",
    "CandidateProvider",
    "CandidateReport",
    "FunctionCandidateProvider",
    "GraphSurgeryReport",
    "OptimizationAgentToolkit",
    "OptimizationConfig",
    "OptimizationReport",
    "OptimizationResult",
    "Optimizer",
    "RuntimeCapabilities",
    "__version__",
    "inspect_runtime",
]
