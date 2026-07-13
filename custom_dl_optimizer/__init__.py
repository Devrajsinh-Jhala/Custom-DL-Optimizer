from importlib.metadata import PackageNotFoundError, version

from .config import OptimizationConfig
from .core.engine import AutoOptimizer
from .report import CandidateReport, GraphSurgeryReport, OptimizationReport

try:
    __version__ = version("custom-dl-optimizer")
except PackageNotFoundError:  # Source checkout before installation.
    __version__ = "1.1.0"

__all__ = [
    "AutoOptimizer",
    "CandidateReport",
    "GraphSurgeryReport",
    "OptimizationConfig",
    "OptimizationReport",
    "__version__",
]
