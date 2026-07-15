from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
import torch.nn as nn

from .config import OptimizationConfig
from .workload import WorkloadProfile


@dataclass(frozen=True)
class CandidateContext:
    """Inputs and runtime details supplied to a candidate provider."""

    device: torch.device
    config: OptimizationConfig
    example_args: tuple[Any, ...]
    example_kwargs: dict[str, Any]
    workload_profile: WorkloadProfile | None = None
    cache_key: str = ""
    artifact_dir: Path | None = None


class CandidateProvider(Protocol):
    """Extension point for TensorRT, ONNX Runtime, or private compiler plans."""

    name: str

    def is_available(self, context: CandidateContext) -> bool: ...

    def build(self, model: nn.Module, context: CandidateContext) -> nn.Module: ...


@dataclass(frozen=True)
class FunctionCandidateProvider:
    """Create a provider from ordinary Python callables."""

    name: str
    builder: Callable[[nn.Module, CandidateContext], nn.Module]
    availability: Callable[[CandidateContext], bool] | None = None

    def is_available(self, context: CandidateContext) -> bool:
        return self.availability(context) if self.availability is not None else True

    def build(self, model: nn.Module, context: CandidateContext) -> nn.Module:
        return self.builder(model, context)

    def cache_identity(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "builder": (
                f"{getattr(self.builder, '__module__', '')}."
                f"{getattr(self.builder, '__qualname__', type(self.builder).__qualname__)}"
            ),
        }
