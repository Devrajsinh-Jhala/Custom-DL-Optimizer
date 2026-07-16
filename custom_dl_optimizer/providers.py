from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn as nn

from .policy import ExecutionTarget, OptimizationPolicy
from .workload import WorkloadProfile


@runtime_checkable
class InferenceRunner(Protocol):
    """Smallest execution contract accepted by the v3 optimizer."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


class _CallableModule(nn.Module):
    def __init__(self, runner: InferenceRunner) -> None:
        super().__init__()
        self.runner = runner

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.runner(*args, **kwargs)

    def close(self) -> None:
        close = getattr(self.runner, "close", None)
        if callable(close):
            close()


@dataclass(frozen=True)
class BuiltPlan:
    """Executable runner plus provider-owned artifact and provenance metadata."""

    runner: InferenceRunner
    artifacts: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not callable(self.runner):
            raise TypeError("BuiltPlan.runner must be callable")
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_module(self) -> nn.Module:
        return self.runner if isinstance(self.runner, nn.Module) else _CallableModule(self.runner)

    def close(self) -> None:
        close = getattr(self.runner, "close", None)
        if callable(close):
            close()


@dataclass(frozen=True)
class ProviderAvailability:
    available: bool
    reason: str = ""

    @classmethod
    def supported(cls) -> ProviderAvailability:
        return cls(available=True)

    @classmethod
    def unsupported(cls, reason: str) -> ProviderAvailability:
        return cls(available=False, reason=reason)


@dataclass(frozen=True)
class ProviderContext:
    """Target, policy, representative inputs, and cache paths for a provider."""

    device: torch.device
    target: ExecutionTarget
    policy: OptimizationPolicy
    example_args: tuple[Any, ...]
    example_kwargs: dict[str, Any]
    workload_profile: WorkloadProfile | None = None
    cache_key: str = ""
    artifact_dir: Path | None = None


class ExecutionProvider(Protocol):
    """Build one complete inference plan for a target runtime."""

    name: str

    def probe(self, context: ProviderContext) -> ProviderAvailability: ...

    def build(self, model: nn.Module, context: ProviderContext) -> BuiltPlan: ...


@dataclass(frozen=True)
class FunctionExecutionProvider:
    """Create a provider from ordinary Python callables."""

    name: str
    builder: Callable[[nn.Module, ProviderContext], BuiltPlan | InferenceRunner]
    availability: Callable[[ProviderContext], ProviderAvailability | bool] | None = None

    def probe(self, context: ProviderContext) -> ProviderAvailability:
        if self.availability is None:
            return ProviderAvailability.supported()
        result = self.availability(context)
        if isinstance(result, ProviderAvailability):
            return result
        return (
            ProviderAvailability.supported()
            if result
            else ProviderAvailability.unsupported("Provider availability check returned false")
        )

    def build(self, model: nn.Module, context: ProviderContext) -> BuiltPlan:
        result = self.builder(model, context)
        return result if isinstance(result, BuiltPlan) else BuiltPlan(runner=result)

    def cache_identity(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "builder": (
                f"{getattr(self.builder, '__module__', '')}."
                f"{getattr(self.builder, '__qualname__', type(self.builder).__qualname__)}"
            ),
        }


# Import-path compatibility for v2 applications. These names are intentionally not
# part of the v3 top-level API; see docs/migration-v3.md.
CandidateContext = ProviderContext
CandidateProvider = ExecutionProvider
FunctionCandidateProvider = FunctionExecutionProvider
